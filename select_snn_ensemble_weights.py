"""Select regularized SNN-ensemble weights with engine-grouped validation CV."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from sklearn.model_selection import GroupKFold

from evaluate_snn_candidate_ensemble import VAL_CANDIDATES

ROOT = Path(__file__).resolve().parent
TEACHER = ROOT / "results" / "bilstm_teacher_tuned"
OUT = ROOT / "results" / "snn_candidate_ensemble_evaluation"
REGULARIZATIONS = (0.0, 0.001, 0.01, 0.1, 1.0)


def fit_weights(pred: np.ndarray, target: np.ndarray, regularization: float) -> np.ndarray:
    k = pred.shape[1]
    uniform = np.full(k, 1.0 / k)
    scale = float(np.var(target))

    def objective(w: np.ndarray) -> float:
        err = pred @ w - target
        return float(np.mean(err**2) + regularization * scale * np.sum((w - uniform)**2))

    result = minimize(objective, uniform, method="SLSQP", bounds=[(0.0, 1.0)] * k,
                      constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}],
                      options={"maxiter": 300, "ftol": 1e-10})
    if not result.success:
        return uniform
    w = np.maximum(result.x, 0.0)
    return w / w.sum()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"method": "5-fold GroupKFold by engine; nonnegative simplex weights; regularized toward equal weights",
              "official_test_used_for_selection": False, "subsets": {}}
    for tag, candidates in VAL_CANDIDATES.items():
        teacher = np.load(TEACHER / f"teacher_signals_{tag}_val.npz")
        target = teacher["y_true"].astype(np.float64)
        groups = teacher["unit"]
        pred_cols = []
        names = []
        for stem, _, _ in candidates:
            val_path = Path(f"{stem}_validation.npz")
            if not val_path.exists():
                val_path = Path(f"{stem}.npz")
            z = np.load(val_path)
            if not np.array_equal(z["y_true"], target):
                raise ValueError(f"Validation targets mismatch for {stem}")
            pred_cols.append(z["y_pred"].astype(np.float64))
            names.append(stem.name)
        pred = np.stack(pred_cols, axis=1)
        splitter = GroupKFold(n_splits=5)
        uniform = np.full(pred.shape[1], 1.0 / pred.shape[1])
        uniform_errors = []
        cv_by_reg = {}
        fold_weights_by_reg = {}
        for reg in REGULARIZATIONS:
            errors, fold_weights = [], []
            for tr, va in splitter.split(pred, target, groups):
                w = fit_weights(pred[tr], target[tr], reg)
                fold_weights.append(w)
                errors.extend((pred[va] @ w - target[va]).tolist())
            cv_by_reg[str(reg)] = float(np.sqrt(np.mean(np.square(errors))))
            fold_weights_by_reg[str(reg)] = np.mean(np.stack(fold_weights), axis=0)
        for tr, va in splitter.split(pred, target, groups):
            uniform_errors.extend((pred[va] @ uniform - target[va]).tolist())
        uniform_rmse = float(np.sqrt(np.mean(np.square(uniform_errors))))
        selected_reg = min(REGULARIZATIONS, key=lambda reg: cv_by_reg[str(reg)])
        selected_rmse = cv_by_reg[str(selected_reg)]
        if selected_rmse >= uniform_rmse:
            selected_reg = 0.0
            weights = uniform
            selected_rmse = uniform_rmse
        else:
            weights = fold_weights_by_reg[str(selected_reg)]
            weights = np.maximum(weights, 0.0)
            weights /= weights.sum()
        report["subsets"][tag] = {"candidate_names": names,
            "weights": {n: float(w) for n, w in zip(names, weights)},
            "selected_regularization": float(selected_reg),
            "group_cv_RMSE_equal_weights": uniform_rmse,
            "group_cv_RMSE_selected_weights": selected_rmse,
            "group_cv_RMSE_by_regularization": cv_by_reg}
        print(f"{tag}: grouped-CV RMSE {uniform_rmse:.3f} -> {selected_rmse:.3f}; lambda={selected_reg}; weights={np.round(weights,3)}", flush=True)
    path = OUT / "selected_weights.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Saved validation-only SNN ensemble weights:", path, flush=True)


if __name__ == "__main__":
    main()
