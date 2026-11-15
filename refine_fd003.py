"""Focused FD003 SNN capacity/distillation search (train/validation only)."""
from __future__ import annotations

import csv
import gc
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from train_spike_models import DATA, DEVICE, DistilledSNN, regression_metrics, seed_all

ROOT = Path(__file__).resolve().parent
TEACHER = ROOT / "results" / "bilstm_teacher_tuned"
OUT = ROOT / "results" / "spike_models" / "fd003_refinement"
MAX_EPOCHS, PATIENCE, BATCH = 60, 12, 128
SEEDS = (42, 43, 44)


def fit(data, teacher, cfg, seed):
    seed_all(seed)
    model = DistilledSNN(data["x_train"].shape[2], leak_init=0.8,
                         hidden_size=cfg["hidden_size"]).to(DEVICE)
    opt = torch.optim.AdamW([p for n,p in model.named_parameters()
                             if n != "teacher_projection.weight"],
                            lr=cfg["lr"], weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=5, min_lr=1e-5)
    y = data["y_train"]
    sw = np.where(y < 30, 1.4, np.where(y < 80, 1., .85)).astype(np.float64)
    sampler = WeightedRandomSampler(torch.as_tensor(sw), len(sw), replacement=True,
                                    generator=torch.Generator().manual_seed(seed))
    loader = DataLoader(TensorDataset(torch.from_numpy(data["x_train"]),
                                      torch.from_numpy(y), torch.from_numpy(teacher)),
                        batch_size=BATCH, sampler=sampler)
    xv = torch.from_numpy(data["x_val"]).to(DEVICE)
    best, stale, best_state, best_pred, best_metrics = float("inf"), 0, None, None, None
    history = []
    for epoch in range(1, MAX_EPOCHS+1):
        model.train()
        for xb, yb, tb in loader:
            xb, yb, tb = xb.to(DEVICE), yb.to(DEVICE), tb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)[0]
            alpha = cfg["teacher_weight"]
            loss = (1-alpha)*nn.functional.mse_loss(pred, yb) + alpha*nn.functional.mse_loss(pred, tb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
        model.eval()
        with torch.no_grad(): pred = model(xv)[0].cpu().numpy().astype(np.float32)
        metrics = regression_metrics(data["y_val"], pred)
        sched.step(metrics["MSE"])
        history.append({"epoch":epoch,"RMSE":metrics["RMSE"],"CMAPSS_score":metrics["CMAPSS_score"]})
        if metrics["RMSE"] < best - 1e-4:
            best, stale = metrics["RMSE"], 0
            best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            best_pred, best_metrics = pred.copy(), metrics
        else:
            stale += 1
            if stale >= PATIENCE: break
    if best_state is None: raise RuntimeError("No checkpoint")
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad(): rate=float(model(xv)[3].mean().cpu())
    row={"subset":"FD003",**cfg,"seed":seed,**best_metrics,"spike_rate":rate,"epochs_run":len(history)}
    return model, row, best_pred, history, best_state


def main():
    torch.set_num_threads(min(8,torch.get_num_threads()))
    OUT.mkdir(parents=True,exist_ok=True)
    d=np.load(DATA/"windows_FD003.npz")
    data={"x_train":d["X_train"].astype(np.float32),"y_train":d["y_train"].astype(np.float32),
          "x_val":d["X_val"].astype(np.float32),"y_val":d["y_val"].astype(np.float32)}
    tr=np.load(TEACHER/"teacher_signals_FD003_train.npz")["y_teacher"].astype(np.float32)
    tv=np.load(TEACHER/"teacher_signals_FD003_val.npz")["y_teacher"].astype(np.float32)
    teacher_metrics=regression_metrics(data["y_val"],tv)
    configs=[
        {"hidden_size":256,"lr":0.0001,"teacher_weight":0.2},
        {"hidden_size":256,"lr":0.0003,"teacher_weight":0.2},
        {"hidden_size":256,"lr":0.0008,"teacher_weight":0.2},
        {"hidden_size":192,"lr":0.0003,"teacher_weight":0.0},
        {"hidden_size":192,"lr":0.0003,"teacher_weight":0.1},
    ]
    rows=[]; preds={}; history_out={}
    for i,cfg in enumerate(configs,1):
        name=f"h{cfg['hidden_size']}_lr{cfg['lr']}_kd{cfg['teacher_weight']}"
        model,row,pred,hist,state=fit(data,tr,cfg,42)
        rows.append(row); preds[name]={42:pred}; history_out[name]={42:hist}
        torch.save({"state_dict":state,"input_size":data["x_train"].shape[2],
                    "hidden_size":cfg["hidden_size"],"config":cfg,"seed":42},OUT/f"{name}_seed42.pt")
        np.savez_compressed(OUT/f"{name}_seed42.npz",y_true=data["y_val"],y_pred=pred)
        print(f"screen {i}/{len(configs)} {name}: RMSE={row['RMSE']:.3f}, score={row['CMAPSS_score']:.1f}",flush=True)
        del model;gc.collect()
    best=min(rows,key=lambda r:(float(r["RMSE"]),float(r["CMAPSS_score"])))
    winner={k:best[k] for k in ("hidden_size","lr","teacher_weight")}
    name=f"h{winner['hidden_size']}_lr{winner['lr']}_kd{winner['teacher_weight']}"
    selected=[best]
    for seed in (43,44):
        model,row,pred,hist,state=fit(data,tr,winner,seed)
        selected.append(row);preds[name][seed]=pred;history_out[name][seed]=hist
        torch.save({"state_dict":state,"input_size":data["x_train"].shape[2],
                    "hidden_size":winner["hidden_size"],"config":winner,"seed":seed},OUT/f"{name}_seed{seed}.pt")
        np.savez_compressed(OUT/f"{name}_seed{seed}.npz",y_true=data["y_val"],y_pred=pred)
        print(f"confirm seed={seed}: RMSE={row['RMSE']:.3f}",flush=True)
        del model;gc.collect()
    ensemble=np.mean(np.stack([preds[name][s] for s in SEEDS]),axis=0)
    ensemble_metrics=regression_metrics(data["y_val"],ensemble)
    np.savez_compressed(OUT/"selected_ensemble.npz",y_true=data["y_val"],y_pred=ensemble)
    summary={"teacher_validation":teacher_metrics,"selected_config":winner,
             "selected_seed_metrics":[{"seed":r["seed"],"RMSE":r["RMSE"],"CMAPSS_score":r["CMAPSS_score"]} for r in selected],
             "selected_ensemble":ensemble_metrics,
             "beats_teacher_rmse":ensemble_metrics["RMSE"]<teacher_metrics["RMSE"]}
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    with (OUT/"screen.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys()));w.writeheader();w.writerows(rows)
    print(f"FD003 teacher RMSE={teacher_metrics['RMSE']:.3f}; new SNN ensemble RMSE={ensemble_metrics['RMSE']:.3f}",flush=True)


if __name__=="__main__":main()
