"""Validation-only SpikeDistill hyperparameter sweep and 3-seed confirmation.

Run: python tune_spikedistill.py
The official test split is not read. Existing trial rows are reused on reruns.
"""

from __future__ import annotations

import csv
import gc
import itertools
import json
import random
from pathlib import Path

import numpy as np
import torch

from train_spike_models import (
    DATA,
    DEVICE,
    OUT as MODEL_OUT,
    DistilledSNN,
    fit_distilled_snn,
    regression_metrics,
    seed_all,
)


ROOT = Path(__file__).resolve().parent
TEACHER = ROOT / "results" / "bilstm_teacher_tuned"
OUT = MODEL_OUT / "tuning"
LEARNING_RATES = [3e-4, 8e-4, 1.5e-3]
SPARSITY_WEIGHTS = [0.001, 0.01, 0.05]
LEAK_INITIALIZATIONS = [0.60, 0.80]
SEEDS = [42, 43, 44]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, rows: list[dict[str, object]]) -> None:
    fields = ["subset", "learning_rate", "sparsity_weight", "leak_init", "seed", "RMSE", "MAE", "CMAPSS_score", "spike_rate", "epochs_run", "best_val_mse"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_subset(tag: str) -> dict[str, np.ndarray]:
    windows = np.load(DATA / f"windows_{tag}.npz")
    teacher_train = np.load(TEACHER / f"teacher_signals_{tag}_train.npz")
    teacher_val = np.load(TEACHER / f"teacher_signals_{tag}_val.npz")
    return {
        "x_train": windows["X_train"].astype(np.float32),
        "y_train": windows["y_train"].astype(np.float32),
        "x_val": windows["X_val"].astype(np.float32),
        "y_val": windows["y_val"].astype(np.float32),
        "teacher_train_y": teacher_train["y_teacher"].astype(np.float32),
        "teacher_train_hidden": teacher_train["hidden_states"].astype(np.float16),
        "teacher_train_attention": teacher_train["attention"].astype(np.float32),
        "teacher_val_y": teacher_val["y_teacher"].astype(np.float32),
        "teacher_val_hidden": teacher_val["hidden_states"].astype(np.float16),
        "teacher_val_attention": teacher_val["attention"].astype(np.float32),
    }


def train_config(tag: str, data: dict[str, np.ndarray], lr: float, sparsity: float, leak: float, seed: int):
    seed_all(seed)
    model, prediction, info = fit_distilled_snn(
        DistilledSNN(data["x_train"].shape[2], leak_init=leak),
        data["x_train"], data["y_train"],
        data["teacher_train_y"], data["teacher_train_hidden"], data["teacher_train_attention"],
        data["x_val"], data["y_val"],
        data["teacher_val_y"], data["teacher_val_hidden"], data["teacher_val_attention"],
        learning_rate=lr, sparsity_weight=sparsity, seed=seed,
    )
    metrics = regression_metrics(data["y_val"], prediction)
    record = {
        "subset": tag, "learning_rate": lr, "sparsity_weight": sparsity,
        "leak_init": leak, "seed": seed, **metrics,
        "spike_rate": info["validation_spike_rate"],
        "epochs_run": info["epochs_run"], "best_val_mse": info["best_val_mse"],
        "learned_threshold_mean": info["learned_threshold_mean"],
        "learned_leak_mean": info["learned_leak_mean"],
    }
    return model, prediction, info, record


def main() -> None:
    torch.set_num_threads(min(8, torch.get_num_threads()))
    OUT.mkdir(parents=True, exist_ok=True)
    sweep_path = OUT / "sweep_validation.csv"
    existing = read_csv_rows(sweep_path)
    done = {
        (r["subset"], float(r["learning_rate"]), float(r["sparsity_weight"]), float(r["leak_init"]))
        for r in existing
    }
    sweep_rows: list[dict[str, object]] = list(existing)
    final_rows: list[dict[str, object]] = read_csv_rows(OUT / "selected_three_seed_validation.csv")
    best_config: dict[str, dict[str, object]] = {}

    for fd in range(1, 5):
        tag = f"FD{fd:03d}"
        data = load_subset(tag)
        configs = list(itertools.product(LEARNING_RATES, SPARSITY_WEIGHTS, LEAK_INITIALIZATIONS))
        for index, (lr, sparsity, leak) in enumerate(configs, start=1):
            key = (tag, float(lr), float(sparsity), float(leak))
            if key in done:
                continue
            model, prediction, info, record = train_config(tag, data, lr, sparsity, leak, 42)
            sweep_rows.append(record)
            write_csv_rows(sweep_path, sweep_rows)
            print(
                f"{tag} sweep {index}/{len(configs)} lr={lr:g} sparsity={sparsity:g} leak={leak:.2f} "
                f"score={record['CMAPSS_score']:.1f} RMSE={record['RMSE']:.2f} spikes={record['spike_rate']:.3f}",
                flush=True,
            )
            del model, prediction, info
            gc.collect()

        candidates = [r for r in sweep_rows if r["subset"] == tag]
        winner = min(candidates, key=lambda r: (float(r["CMAPSS_score"]), float(r["RMSE"])))
        config = {
            "learning_rate": float(winner["learning_rate"]),
            "sparsity_weight": float(winner["sparsity_weight"]),
            "leak_init": float(winner["leak_init"]),
            "selection_metric": "minimum validation asymmetric C-MAPSS score; RMSE breaks ties",
            "seed42_sweep_result": winner,
        }
        best_config[tag] = config

        # Rerun the selected settings with three seeds and save those checkpoints.
        final_dir = OUT / tag
        final_dir.mkdir(parents=True, exist_ok=True)
        seed_records: list[dict[str, object]] = []
        seed_predictions: list[np.ndarray] = []
        for seed in SEEDS:
            pred_path = final_dir / f"seed_{seed}_validation_predictions.npz"
            model_path = final_dir / f"spikedistill_{tag}_seed_{seed}.pt"
            saved_row = next((r for r in final_rows if r["subset"] == tag and int(float(r["seed"])) == seed), None)
            if saved_row is not None and pred_path.exists() and model_path.exists():
                seed_records.append(saved_row)
                seed_predictions.append(np.load(pred_path)["y_pred"])
                continue
            model, prediction, info, record = train_config(
                tag, data, config["learning_rate"], config["sparsity_weight"], config["leak_init"], seed
            )
            record["subset"] = tag
            seed_records.append(record)
            final_rows.append(record)
            seed_predictions.append(prediction)
            torch.save(
                {"state_dict": model.state_dict(), "input_size": int(data["x_train"].shape[2]), "hidden_size": int(model.encoder.raw_threshold.numel()), "seed": seed, "config": config},
                model_path,
            )
            np.savez_compressed(pred_path, y_true=data["y_val"], y_pred=prediction)
            (final_dir / f"seed_{seed}_history.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
            write_csv_rows(OUT / "selected_three_seed_validation.csv", final_rows)
            print(f"{tag} selected config seed {seed}: score={record['CMAPSS_score']:.1f}, RMSE={record['RMSE']:.2f}", flush=True)
            del model, prediction, info
            gc.collect()

        ensemble = np.mean(np.stack(seed_predictions), axis=0)
        ensemble_metrics = regression_metrics(data["y_val"], ensemble)
        np.savez_compressed(final_dir / "three_seed_ensemble_validation_predictions.npz", y_true=data["y_val"], y_pred=ensemble)
        config["three_seed_mean"] = {
            metric: float(np.mean([float(r[metric]) for r in seed_records]))
            for metric in ("RMSE", "MAE", "CMAPSS_score", "spike_rate")
        }
        config["three_seed_std"] = {
            metric: float(np.std([float(r[metric]) for r in seed_records], ddof=1))
            for metric in ("RMSE", "MAE", "CMAPSS_score", "spike_rate")
        }
        config["three_seed_ensemble_prediction"] = ensemble_metrics
        (final_dir / "selected_config_and_summary.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        (OUT / "selected_configs.json").write_text(json.dumps(best_config, indent=2), encoding="utf-8")
        print(f"{tag} three-seed ensemble: score={ensemble_metrics['CMAPSS_score']:.1f}, RMSE={ensemble_metrics['RMSE']:.2f}", flush=True)
        del data
        gc.collect()

    print("Tuning and three-seed validation complete:", OUT, flush=True)


if __name__ == "__main__":
    main()
