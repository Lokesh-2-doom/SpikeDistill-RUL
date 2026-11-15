"""One-shot official C-MAPSS test evaluation after model selection.

No model, threshold, or hyperparameter selection is performed here. All
predictions are saved alongside metrics for auditability.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from train_baselines import DeepNN
from train_bilstm_teacher import AttentionBiLSTM
from train_sequence_baselines import build_model
from train_spike_models import (DATA, DEVICE, DirectSNN, DistilledSNN, SmallLSTM,
                                convert_ann_to_snn, regression_metrics)

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
OUT = RESULTS / "official_test_evaluation"
SEEDS = (42, 43, 44)


def load_state(path: Path, model: torch.nn.Module) -> dict:
    payload = torch.load(path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(payload["state_dict"])
    return payload


def infer(model: torch.nn.Module, x: np.ndarray, mode: str = "regression") -> np.ndarray:
    model = model.to(DEVICE).eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(x), 128):
            xb = torch.from_numpy(x[start:start+128]).to(DEVICE)
            out = model(xb)
            if mode in {"distilled_snn", "direct_snn", "teacher"}: out = out[0]
            preds.append(out.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(preds)


def main() -> None:
    torch.set_num_threads(min(8, torch.get_num_threads()))
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    report: dict[str, object] = {"policy": "official test used once for final reporting only; no tuning/model selection",
                                 "subsets": {}}
    for fd in range(1, 5):
        tag = f"FD{fd:03d}"
        z = np.load(DATA / f"windows_{tag}.npz")
        xtr = z["X_train"].astype(np.float32)
        ytr = z["y_train"].astype(np.float32)
        xt = z["X_test"].astype(np.float32)
        yt = z["y_test"].astype(np.float32)
        flat_tr, flat_test = xtr.reshape(len(xtr), -1), xt.reshape(len(xt), -1)
        predictions: dict[str, np.ndarray] = {}
        predictions["mean"] = np.full(len(yt), float(ytr.mean()), dtype=np.float32)
        ridge = Ridge(alpha=10.0).fit(flat_tr, ytr)
        predictions["ridge"] = ridge.predict(flat_test).astype(np.float32)
        forest = RandomForestRegressor(n_estimators=75, max_depth=20,
                                       min_samples_leaf=2, max_features=.7,
                                       random_state=42, n_jobs=-1).fit(flat_tr, ytr)
        predictions["random_forest"] = forest.predict(flat_test).astype(np.float32)
        deep = DeepNN(flat_tr.shape[1])
        load_state(RESULTS / "baselines" / f"deepnn_{tag}.pt", deep)
        predictions["deepnn"] = infer(deep, flat_test)

        for kind in ("rnn", "lstm", "gru", "bilstm", "cnn_lstm"):
            model = build_model(kind, xtr.shape[2])
            load_state(RESULTS / "sequence_baselines" / f"{kind}_{tag}.pt", model)
            predictions[kind] = infer(model, xt)

        teacher = AttentionBiLSTM(xtr.shape[2])
        load_state(RESULTS / "bilstm_teacher_tuned" / f"attention_bilstm_teacher_{tag}.pt", teacher)
        predictions["attention_bilstm_teacher"] = infer(teacher, xt, "teacher")

        small_kd = SmallLSTM(xtr.shape[2])
        load_state(RESULTS / "spike_models" / "vanilla_kd" / f"vanilla_kd_lstm_{tag}.pt", small_kd)
        predictions["vanilla_kd_lstm"] = infer(small_kd, xt)

        direct = DirectSNN(xtr.shape[2])
        load_state(RESULTS / "spike_models" / "snn_baselines" / f"surrogate_snn_{tag}.pt", direct)
        predictions["direct_surrogate_snn"] = infer(direct, xt, "direct_snn")
        ann_snn, _ = convert_ann_to_snn(tag, xt)
        predictions["ann_to_snn"] = ann_snn

        full_snn = DistilledSNN(xtr.shape[2])
        load_state(RESULTS / "spike_models" / "spikedistill_rul" / f"spikedistill_rul_{tag}.pt", full_snn)
        predictions["full_loss_spikedistill"] = infer(full_snn, xt, "distilled_snn")

        if tag == "FD001":
            folder, stem, width = RESULTS / "spike_models" / "rmse_refinement" / tag, "label_mse_h96", 96
        elif tag == "FD002":
            folder, stem, width = RESULTS / "spike_models" / "rmse_refinement" / tag, "output_mse_h192", 192
        elif tag == "FD003":
            folder, stem, width = RESULTS / "spike_models" / "fd003_refinement", "h192_lr0.0003_kd0.1", 192
        else:
            folder, stem, width = RESULTS / "spike_models" / "rmse_refinement" / tag, "output_mse_h192", 192
        seed_preds = []
        for seed in SEEDS:
            student = DistilledSNN(xtr.shape[2], hidden_size=width)
            load_state(folder / f"{stem}_seed{seed}.pt", student)
            p = infer(student, xt, "distilled_snn")
            predictions[f"refined_snn_seed{seed}"] = p
            seed_preds.append(p)
        predictions["refined_snn_ensemble"] = np.mean(np.stack(seed_preds), axis=0).astype(np.float32)

        # Attach the validation-selected SNN recipe predictions so the final
        # ranking and report plots compare the same student artifact.
        selected_path = (RESULTS / "snn_candidate_ensemble_evaluation" /
                         f"{tag}_snn_blend_test_predictions.npz")
        if selected_path.exists():
            predictions["selected_snn_student"] = np.load(selected_path)["y_pred"].astype(np.float32)

        subrows = []
        for name, pred in predictions.items():
            metric = regression_metrics(yt, pred)
            row = {"subset": tag, "model": name, **metric}
            rows.append(row); subrows.append(row)
            np.savez_compressed(OUT / f"{tag}_{name}_test_predictions.npz", y_true=yt, y_pred=pred)
        report["subsets"][tag] = {r["model"]: {k: r[k] for k in ("MAE", "MSE", "RMSE", "R2", "CMAPSS_score")} for r in subrows}
        with (OUT / "official_test_metrics.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["subset", "model", "MAE", "MSE", "RMSE", "R2", "CMAPSS_score"])
            writer.writeheader(); writer.writerows(rows)
        (OUT / "official_test_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"{tag}: refined SNN ensemble RMSE={regression_metrics(yt, predictions['refined_snn_ensemble'])['RMSE']:.2f}; "
              f"teacher RMSE={regression_metrics(yt, predictions['attention_bilstm_teacher'])['RMSE']:.2f}", flush=True)
    print("Official-test evaluation complete:", OUT, flush=True)


if __name__ == "__main__":
    main()
