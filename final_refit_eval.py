"""Refit selected SNN configurations with validation-aligned optimizer steps.

Selection uses the prior train/validation experiments only. This script uses
fixed epochs/configurations, trains on final_windows_* X_train/y_train, and
evaluates the official test set once at the end. It does not select models or
hyperparameters on test results.
"""
from __future__ import annotations

import csv
import json
import random
import argparse
import shutil
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from train_bilstm_teacher import AttentionBiLSTM
from train_spike_models import DATA, DEVICE, DistilledSNN, regression_metrics, seed_all

ROOT = Path(__file__).resolve().parent
TEACHER_HISTORY = ROOT / "results" / "bilstm_teacher_tuned"
OUT = ROOT / "results" / "final_refit_evaluation_aligned"
REFERENCE = ROOT / "results" / "final_refit_evaluation"
SEEDS = (42, 43, 44)
# Match the validation refinement's optimizer-update frequency. The teacher
# checkpoints are reused from the completed full-data run.
BATCH_TEACHER, BATCH_STUDENT = 256, 128


def teacher_epoch(tag: str) -> int:
    h = json.loads((TEACHER_HISTORY / f"attention_bilstm_teacher_{tag}_history.json").read_text(encoding="utf-8"))
    return int(min(h["history"], key=lambda r: r["val_mse"])["epoch"])


def student_recipe(tag: str) -> dict[str, object]:
    if tag == "FD001":
        path = ROOT / "results/spike_models/rmse_refinement/FD001/label_mse_h96_seed42_history.json"
        hist = json.loads(path.read_text(encoding="utf-8"))
        epochs = int(min(hist, key=lambda r: r["val_rmse"])["epoch"])
        return {"hidden_size": 96, "lr": .0015, "teacher_weight": 0.0, "leak_init": .8, "epochs": epochs}
    if tag == "FD002":
        path = ROOT / "results/spike_models/rmse_refinement/FD002/output_mse_h192_seed42_history.json"
        hist = json.loads(path.read_text(encoding="utf-8"))
        epochs = int(min(hist, key=lambda r: r["val_rmse"])["epoch"])
        return {"hidden_size": 192, "lr": .0015, "teacher_weight": .2, "leak_init": .6, "epochs": epochs}
    if tag == "FD003":
        chosen = json.loads((ROOT / "results/spike_models/fd003_refinement/selected_h192_lr0.0003_kd0.1_best_epoch.json").read_text(encoding="utf-8"))
        return {"hidden_size": 192, "lr": float(chosen["config"]["lr"]), "teacher_weight": .1, "leak_init": .8,
                "epochs": int(chosen["best_epoch"])}
    path = ROOT / "results/spike_models/rmse_refinement/FD004/output_mse_h192_seed42_history.json"
    hist = json.loads(path.read_text(encoding="utf-8"))
    epochs = int(min(hist, key=lambda r: r["val_rmse"])["epoch"])
    return {"hidden_size": 192, "lr": .0015, "teacher_weight": .2, "leak_init": .8, "epochs": epochs}


def fit_teacher(x: np.ndarray, y: np.ndarray, input_size: int, epochs: int, seed: int):
    seed_all(seed)
    model = AttentionBiLSTM(input_size).to(DEVICE)
    weight = np.where(y < 30, 1.4, np.where(y < 80, 1.0, .85)).astype(np.float64)
    sampler = WeightedRandomSampler(torch.as_tensor(weight), len(weight), replacement=True,
                                    generator=torch.Generator().manual_seed(seed))
    loader = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
                        batch_size=BATCH_TEACHER, sampler=sampler)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    losses = []
    for epoch in range(epochs):
        model.train(); total = 0.; n = 0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)[0]
            loss = nn.functional.mse_loss(pred, yb)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            total += float(loss.item())*len(xb); n += len(xb)
        losses.append(total/max(1,n))
    model.eval()
    with torch.no_grad():
        teacher_y = model(torch.from_numpy(x).to(DEVICE))[0].cpu().numpy().astype(np.float32)
    return model, teacher_y, losses


def fit_student(x: np.ndarray, y: np.ndarray, teacher_y: np.ndarray, recipe: dict[str, object], seed: int, tag: str):
    seed_all(seed)
    model = DistilledSNN(x.shape[2], hidden_size=int(recipe["hidden_size"]), leak_init=float(recipe["leak_init"])).to(DEVICE)
    weight = np.where(y < 30, 1.4, np.where(y < 80, 1.0, .85)).astype(np.float64)
    sampler = WeightedRandomSampler(torch.as_tensor(weight), len(weight), replacement=True,
                                    generator=torch.Generator().manual_seed(seed))
    loader = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y),
                                      torch.from_numpy(teacher_y)),
                        batch_size=BATCH_STUDENT, sampler=sampler)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(recipe["lr"]), weight_decay=1e-5)
    loss_history = []
    alpha = float(recipe["teacher_weight"])
    for epoch in range(int(recipe["epochs"])):
        model.train(); total = 0.; n = 0
        for xb, yb, tb in loader:
            xb, yb, tb = xb.to(DEVICE), yb.to(DEVICE), tb.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)[0]
            loss = (1-alpha)*nn.functional.mse_loss(pred, yb) + alpha*nn.functional.mse_loss(pred, tb)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            total += float(loss.item())*len(xb); n += len(xb)
        loss_history.append(total/max(1,n))
        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch + 1 == int(recipe["epochs"]):
            print(f"{tag} student seed={seed}: epoch {epoch + 1}/{recipe['epochs']}, train loss={loss_history[-1]:.2f}", flush=True)
    model.eval()
    return model, loss_history


def infer_teacher(model, x: np.ndarray) -> np.ndarray:
    model.eval(); result=[]
    with torch.no_grad():
        for start in range(0,len(x),128):
            result.append(model(torch.from_numpy(x[start:start+128]).to(DEVICE))[0].cpu().numpy().astype(np.float32))
    return np.concatenate(result)


def infer_student(model, x: np.ndarray) -> tuple[np.ndarray,float]:
    model.eval(); result=[];spikes=[]
    with torch.no_grad():
        for start in range(0,len(x),128):
            out=model(torch.from_numpy(x[start:start+128]).to(DEVICE))
            result.append(out[0].cpu().numpy().astype(np.float32));spikes.append(float(out[3].float().mean().cpu()))
    return np.concatenate(result),float(np.mean(spikes))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", choices=[f"FD{i:03d}" for i in range(1, 5)],
                        help="Run only one subset; useful for resuming/repeating a refit.")
    args = parser.parse_args()
    torch.set_num_threads(min(8,torch.get_num_threads()))
    OUT.mkdir(parents=True,exist_ok=True)
    report={"evaluation_policy":"configurations and epochs fixed using validation; full official train used for final refit; official test evaluated after fitting and not used for any choices",
            "subsets":{}}
    pending=[]
    tags = [args.subset] if args.subset else [f"FD{fd:03d}" for fd in range(1,5)]
    for tag in tags:
        # Access only training fields during fitting. Test fields are loaded in
        # a later loop after every final checkpoint has been saved.
        z=np.load(DATA/f"final_windows_{tag}.npz")
        x=z["X_train"].astype(np.float32);y=z["y_train"].astype(np.float32)
        recipe=student_recipe(tag); te=teacher_epoch(tag)
        subset_dir=OUT/tag;subset_dir.mkdir(parents=True,exist_ok=True)
        teacher_path=subset_dir/"full_train_teacher.pt"
        reference_teacher=REFERENCE/tag/"full_train_teacher.pt"
        if not teacher_path.exists() and reference_teacher.exists():
            shutil.copy2(reference_teacher, teacher_path)
            reference_history=REFERENCE/tag/"teacher_train_history.json"
            if reference_history.exists():
                shutil.copy2(reference_history, subset_dir/"teacher_train_history.json")
            print(f"{tag} reusing teacher checkpoint from prior completed full-train run",flush=True)
        if teacher_path.exists():
            state=torch.load(teacher_path,map_location=DEVICE,weights_only=False)
            teacher=AttentionBiLSTM(x.shape[2]).to(DEVICE)
            teacher.load_state_dict(state["state_dict"])
            teacher.eval()
            with torch.no_grad():
                ty=infer_teacher(teacher,x)
            th=json.loads((subset_dir/"teacher_train_history.json").read_text(encoding="utf-8"))
            print(f"{tag} reusing saved full-train teacher",flush=True)
        else:
            teacher,ty,th=fit_teacher(x,y,x.shape[2],te,42)
            torch.save({"state_dict":teacher.state_dict(),"input_size":x.shape[2],
                        "teacher_epochs":te},teacher_path)
            (subset_dir/"teacher_train_history.json").write_text(json.dumps(th,indent=2),encoding="utf-8")
        seed_cfg=[]
        for seed in SEEDS:
            model_path=subset_dir/f"full_train_snn_seed{seed}.pt"
            if model_path.exists():
                print(f"{tag} seed={seed}: reusing saved full-train SNN",flush=True)
            else:
                model,sh=fit_student(x,y,ty,recipe,seed,tag)
                torch.save({"state_dict":model.state_dict(),"input_size":x.shape[2],
                            "hidden_size":recipe["hidden_size"],"seed":seed,"recipe":recipe},model_path)
                (subset_dir/f"snn_seed{seed}_train_history.json").write_text(json.dumps(sh,indent=2),encoding="utf-8")
            seed_cfg.append(seed)
        recipe.update({"teacher_epochs":te,"teacher_seed":42,"student_seeds":seed_cfg,
                       "feature_count":int(x.shape[2]),"train_windows":int(len(x))})
        (subset_dir/"locked_recipe.json").write_text(json.dumps(recipe,indent=2),encoding="utf-8")
        pending.append((tag,recipe,subset_dir))
        print(f"{tag} full official train refit complete: teacher epochs={te}; SNN recipe={recipe}",flush=True)

    # Official test arrays are first accessed after every fitted checkpoint is locked.
    rows=[]
    for tag,recipe,subset_dir in pending:
        z=np.load(DATA/f"final_windows_{tag}.npz")
        xt=z["X_test"].astype(np.float32);yt=z["y_test"].astype(np.float32)
        teacher=AttentionBiLSTM(xt.shape[2]).to(DEVICE)
        teacher.load_state_dict(torch.load(subset_dir/"full_train_teacher.pt",map_location=DEVICE,weights_only=False)["state_dict"])
        teacher_pred=infer_teacher(teacher,xt)
        seed_pred=[];seed_rows=[]
        for seed in SEEDS:
            student=DistilledSNN(xt.shape[2],hidden_size=int(recipe["hidden_size"]),leak_init=float(recipe["leak_init"])).to(DEVICE)
            student.load_state_dict(torch.load(subset_dir/f"full_train_snn_seed{seed}.pt",map_location=DEVICE,weights_only=False)["state_dict"])
            pred,rate=infer_student(student,xt);seed_pred.append(pred)
            metric=regression_metrics(yt,pred)
            record={"subset":tag,"model":f"full_train_snn_seed{seed}","seed":seed,**metric,"spike_rate":rate}
            rows.append(record);seed_rows.append(record)
            np.savez_compressed(subset_dir/f"test_predictions_seed{seed}.npz",y_true=yt,y_pred=pred)
        ensemble=np.mean(np.stack(seed_pred),axis=0).astype(np.float32)
        teacher_metrics=regression_metrics(yt,teacher_pred)
        ensemble_metrics=regression_metrics(yt,ensemble)
        teacher_row={"subset":tag,"model":"full_train_attention_bilstm_teacher","seed":42,**teacher_metrics,"spike_rate":float("nan")}
        ensemble_row={"subset":tag,"model":"full_train_snn_ensemble","seed":"ensemble",**ensemble_metrics,"spike_rate":float(np.mean([r["spike_rate"] for r in seed_rows]))}
        rows.extend([teacher_row,ensemble_row])
        np.savez_compressed(subset_dir/"test_predictions_teacher.npz",y_true=yt,y_pred=teacher_pred)
        np.savez_compressed(subset_dir/"test_predictions_snn_ensemble.npz",y_true=yt,y_pred=ensemble)
        report["subsets"][tag]={"recipe":recipe,"teacher":teacher_row,"snn_seed_metrics":seed_rows,
                               "snn_ensemble":ensemble_row,
                               "snn_beats_teacher_rmse":ensemble_metrics["RMSE"]<teacher_metrics["RMSE"],
                               "snn_beats_teacher_cmapss_score":ensemble_metrics["CMAPSS_score"]<teacher_metrics["CMAPSS_score"]}
        with (OUT/"metrics.csv").open("w",newline="",encoding="utf-8") as f:
            writer=csv.DictWriter(f,fieldnames=["subset","model","seed","MAE","MSE","RMSE","R2","CMAPSS_score","spike_rate"])
            writer.writeheader();writer.writerows(rows)
        (OUT/"metrics.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
        print(f"{tag} final test: SNN ensemble RMSE={ensemble_metrics['RMSE']:.3f}; teacher RMSE={teacher_metrics['RMSE']:.3f}",flush=True)
    print("Full-training refit and final test evaluation complete:",OUT,flush=True)


if __name__=="__main__":main()
