"""Evaluate a validation-defined, equal-weight ensemble of saved SNN candidates.

Candidate membership is fixed from existing train/validation experiments. This
script only runs official-test inference and writes metrics/predictions; it does
not fit or tune any weights using official-test labels.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from train_spike_models import DATA, DEVICE, DistilledSNN, regression_metrics

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "snn_candidate_ensemble_evaluation"
REFINE = ROOT / "results" / "spike_models" / "rmse_refinement"
FD003 = ROOT / "results" / "spike_models" / "fd003_refinement"
VAL_CANDIDATES = {
    "FD001": [
        (REFINE / "FD001" / f"label_mse_h96_seed{s}", 96, .8) for s in (42, 43, 44)
    ] + [
        (REFINE / "FD001" / "output_mse_h96_seed42", 96, .8),
        (REFINE / "FD001" / "output_mse_h192_seed42", 192, .8),
    ],
    "FD002": [],
    "FD003": [],
    "FD004": [],
}

# Fill variable-length candidate lists explicitly for clarity.
VAL_CANDIDATES["FD002"] = [
    (REFINE / "FD002" / "label_mse_h96_seed42", 96, .6),
    (REFINE / "FD002" / "output_mse_h96_seed42", 96, .6),
    *((REFINE / "FD002" / f"output_mse_h192_seed{s}", 192, .6) for s in (42, 43, 44)),
]
VAL_CANDIDATES["FD003"] = [
    (FD003 / f"h192_lr0.0003_kd0.0_seed42", 192, .8),
    *((FD003 / f"h192_lr0.0003_kd0.1_seed{s}", 192, .8) for s in (42, 43, 44)),
    (FD003 / "h256_lr0.0001_kd0.2_seed42", 256, .8),
    (FD003 / "h256_lr0.0003_kd0.2_seed42", 256, .8),
    (FD003 / "h256_lr0.0008_kd0.2_seed42", 256, .8),
]
VAL_CANDIDATES["FD004"] = [
    (REFINE / "FD004" / "label_mse_h96_seed42", 96, .8),
    (REFINE / "FD004" / "output_mse_h96_seed42", 96, .8),
    *((REFINE / "FD004" / f"output_mse_h192_seed{s}", 192, .8) for s in (42, 43, 44)),
]


def predict(model: torch.nn.Module, x: np.ndarray) -> np.ndarray:
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(x), 128):
            pred = model(torch.from_numpy(x[start:start + 128]).to(DEVICE))[0]
            outputs.append(pred.cpu().numpy().astype(np.float32))
    return np.concatenate(outputs)


def main() -> None:
    torch.set_num_threads(min(8, torch.get_num_threads()))
    OUT.mkdir(parents=True, exist_ok=True)
    weights_path = OUT / "selected_weights.json"
    learned_weights = json.loads(weights_path.read_text(encoding="utf-8")) if weights_path.exists() else None
    rows = []
    report = {"selection": "engine-grouped validation-CV weights" if learned_weights else "equal weights over validation candidates",
              "official_test_used_for_selection": False, "subsets": {}}
    for tag, candidates in VAL_CANDIDATES.items():
        z = np.load(DATA / f"windows_{tag}.npz")
        x = z["X_test"].astype(np.float32)
        y = z["y_test"].astype(np.float32)
        val_preds = []
        test_preds = []
        candidate_names = []
        for stem, width, leak in candidates:
            checkpoint = Path(f"{stem}.pt")
            payload = torch.load(checkpoint, map_location=DEVICE, weights_only=False)
            model = DistilledSNN(x.shape[2], hidden_size=width, leak_init=leak).to(DEVICE)
            model.load_state_dict(payload["state_dict"])
            p_test = predict(model, x)
            test_preds.append(p_test)
            val_path = Path(f"{stem}_validation.npz")
            if not val_path.exists():
                val_path = Path(f"{stem}.npz")
            zv = np.load(val_path)
            val_preds.append(zv["y_pred"].astype(np.float32))
            candidate_names.append(stem.name)
        # Validation ensemble score is recorded for context; equal weights are fixed.
        val_y = np.load((REFINE / tag / "selected_three_seed_ensemble_validation.npz")
                        if tag != "FD003" else FD003 / "selected_ensemble.npz")["y_true"]
        if learned_weights:
            weight_map = learned_weights["subsets"][tag]["weights"]
            weights = np.asarray([weight_map[name] for name in candidate_names], dtype=np.float64)
        else:
            weights = np.full(len(candidates), 1.0 / len(candidates))
        pv = np.average(np.stack(val_preds), axis=0, weights=weights)
        test_pred = np.average(np.stack(test_preds), axis=0, weights=weights).astype(np.float32)
        val_metrics = regression_metrics(val_y, pv)
        test_metrics = regression_metrics(y, test_pred)
        np.savez_compressed(OUT / f"{tag}_candidate_ensemble_test_predictions.npz",
                            y_true=y, y_pred=test_pred)
        row = {"subset": tag, "model": "snn_multi_config_equal_ensemble", **test_metrics}
        rows.append(row)
        report["subsets"][tag] = {"candidate_count": len(candidates), "candidate_names": candidate_names,
                                  "validation_metrics": val_metrics, "official_test_metrics": test_metrics}
        print(f"{tag}: validation RMSE={val_metrics['RMSE']:.3f}; test RMSE={test_metrics['RMSE']:.3f}", flush=True)
    with (OUT / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["subset", "model", "MAE", "MSE", "RMSE", "R2", "CMAPSS_score"])
        w.writeheader(); w.writerows(rows)
    (OUT / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Saved validation-defined SNN ensemble evaluation:", OUT, flush=True)


if __name__ == "__main__":
    main()
