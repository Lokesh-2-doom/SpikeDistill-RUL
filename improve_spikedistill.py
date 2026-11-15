"""Longer, focused SpikeDistill SNN comparison against its BiLSTM teacher.

Only train/validation artifacts are loaded. The official test split is untouched.
Run: python improve_spikedistill.py [--subset FD001]
"""

from __future__ import annotations

import argparse
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
PREVIOUS = ROOT / "results" / "spike_models" / "tuning" / "selected_configs.json"
OUT = ROOT / "results" / "spike_models" / "improved"
SEEDS = (42, 43, 44)
MAX_EPOCHS = 48
PATIENCE = 9
BATCH = 128


def save_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = ["subset", "method", "seed", "learning_rate", "sparsity_weight", "leak_init",
              "RMSE", "MAE", "MSE", "R2", "CMAPSS_score", "spike_rate", "epochs_run"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_one(tag: str, data: dict[str, np.ndarray], teacher_train: np.ndarray,
            teacher_val: np.ndarray, config: dict[str, object], method: str,
            seed: int, out_dir: Path) -> tuple[dict[str, object], np.ndarray]:
    seed_all(seed)
    hidden_size = int(config.get("hidden_size", 96))
    model = DistilledSNN(data["x_train"].shape[2], leak_init=float(config["leak_init"]),
                         hidden_size=hidden_size).to(DEVICE)
    trainable = [p for name, p in model.named_parameters() if name != "teacher_projection.weight"]
    optimizer = torch.optim.AdamW(trainable, lr=float(config["learning_rate"]), weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4, min_lr=1e-5
    )
    weights = np.where(data["y_train"] < 30, 1.4,
                       np.where(data["y_train"] < 80, 1.0, 0.85)).astype(np.float64)
    sampler = WeightedRandomSampler(torch.as_tensor(weights), len(weights), replacement=True,
                                    generator=torch.Generator().manual_seed(seed))
    loader = DataLoader(
        TensorDataset(torch.from_numpy(data["x_train"]), torch.from_numpy(data["y_train"]),
                      torch.from_numpy(teacher_train)), batch_size=BATCH, sampler=sampler,
    )
    xv = torch.from_numpy(data["x_val"]).to(DEVICE)
    yv = torch.from_numpy(data["y_val"]).to(DEVICE)
    best_key = (float("inf"), float("inf"))
    best_state = None
    best_prediction = None
    best_metrics = None
    stale = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total = 0.0
        count = 0
        for xb, yb, tb in loader:
            xb, yb, tb = xb.to(DEVICE), yb.to(DEVICE), tb.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            pred, _, _, emitted, _ = model(xb)
            label_loss = nn.functional.smooth_l1_loss(pred, yb, beta=10.0)
            if method == "label_only":
                loss = label_loss
            else:
                teacher_loss = nn.functional.smooth_l1_loss(pred, tb, beta=10.0)
                loss = 0.8 * label_loss + 0.2 * teacher_loss
            if float(config["sparsity_weight"]):
                loss = loss + float(config["sparsity_weight"]) * emitted.mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.item()) * len(xb)
            count += len(xb)

        model.eval()
        with torch.no_grad():
            prediction = model(xv)[0].cpu().numpy().astype(np.float32)
        metrics = regression_metrics(data["y_val"], prediction)
        scheduler.step(metrics["MSE"])
        history.append({"epoch": epoch, "train_loss": total / max(count, 1),
                        "val_rmse": metrics["RMSE"], "val_cmapss_score": metrics["CMAPSS_score"]})
        key = (metrics["CMAPSS_score"], metrics["RMSE"])
        if key < best_key:
            best_key = key
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_prediction = prediction.copy()
            best_metrics = metrics
            stale = 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break

    if best_state is None or best_prediction is None or best_metrics is None:
        raise RuntimeError(f"No checkpoint produced for {tag}/{method}/seed={seed}")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        spike_rate = float(model(xv)[3].mean().cpu())
    record: dict[str, object] = {
        "subset": tag, "method": method, "seed": seed,
        "learning_rate": config["learning_rate"], "sparsity_weight": config["sparsity_weight"],
        "leak_init": config["leak_init"], **best_metrics,
        "spike_rate": spike_rate, "epochs_run": len(history),
    }
    torch.save({"state_dict": best_state, "input_size": data["x_train"].shape[2],
                "hidden_size": int(model.encoder.raw_threshold.numel()), "seed": seed,
                "config": {**config, "method": method}},
               out_dir / f"{method}_seed_{seed}.pt")
    np.savez_compressed(out_dir / f"{method}_seed_{seed}_validation.npz",
                        y_true=data["y_val"], y_pred=best_prediction)
    (out_dir / f"{method}_seed_{seed}_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8")
    del model, optimizer, loader
    gc.collect()
    return record, best_prediction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", choices=[f"FD{i:03d}" for i in range(1, 5)])
    args = parser.parse_args()
    torch.set_num_threads(min(8, torch.get_num_threads()))
    OUT.mkdir(parents=True, exist_ok=True)
    selected = json.loads(PREVIOUS.read_text(encoding="utf-8"))
    tags = [args.subset] if args.subset else [f"FD{i:03d}" for i in range(1, 5)]
    metrics_path = OUT / "seed_validation_metrics.csv"
    if metrics_path.exists():
        with metrics_path.open("r", newline="", encoding="utf-8") as f:
            all_rows: list[dict[str, object]] = list(csv.DictReader(f))
    else:
        all_rows = []
    summary_path = OUT / "summary.json"
    all_summaries: dict[str, object] = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}

    for tag in tags:
        print(f"Preparing {tag}; training label-only and teacher-output students", flush=True)
        z = np.load(DATA / f"windows_{tag}.npz")
        data = {"x_train": z["X_train"].astype(np.float32), "y_train": z["y_train"].astype(np.float32),
                "x_val": z["X_val"].astype(np.float32), "y_val": z["y_val"].astype(np.float32)}
        tr = np.load(TEACHER / f"teacher_signals_{tag}_train.npz")
        va = np.load(TEACHER / f"teacher_signals_{tag}_val.npz")
        teacher_train = tr["y_teacher"].astype(np.float32)
        teacher_val = va["y_teacher"].astype(np.float32)
        teacher_metrics = regression_metrics(data["y_val"], teacher_val)
        config = selected[tag]
        subset_out = OUT / tag
        subset_out.mkdir(parents=True, exist_ok=True)
        subset_rows: list[dict[str, object]] = []
        methods = ["label_only", "output_distillation"]
        if tag in {"FD003", "FD004"}:
            methods.append("wide_output_distillation")
        ensemble_predictions: dict[str, list[np.ndarray]] = {method: [] for method in methods}

        for method in ensemble_predictions:
            for seed in SEEDS:
                pred_path = subset_out / f"{method}_seed_{seed}_validation.npz"
                model_path = subset_out / f"{method}_seed_{seed}.pt"
                if pred_path.exists() and model_path.exists():
                    p = np.load(pred_path)
                    prediction = p["y_pred"].astype(np.float32)
                    # Recover prediction and spike-rate metrics from the saved
                    # checkpoint, so resumed runs still have complete records.
                    m = regression_metrics(data["y_val"], prediction)
                    history = json.loads((subset_out / f"{method}_seed_{seed}_history.json").read_text(encoding="utf-8"))
                    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
                    width = int(checkpoint.get("hidden_size", 96))
                    saved_config = checkpoint.get("config", config)
                    student = DistilledSNN(data["x_train"].shape[2],
                                           leak_init=float(saved_config.get("leak_init", config["leak_init"])),
                                           hidden_size=width).to(DEVICE)
                    student.load_state_dict(checkpoint["state_dict"])
                    student.eval()
                    with torch.no_grad():
                        spike_rate = float(student(torch.from_numpy(data["x_val"]).to(DEVICE))[3].mean().cpu())
                    del student
                    row = {"subset": tag, "method": method, "seed": seed,
                           "learning_rate": config["learning_rate"], "sparsity_weight": config["sparsity_weight"],
                           "leak_init": config["leak_init"], **m, "spike_rate": spike_rate,
                           "epochs_run": len(history)}
                else:
                    run_config = dict(config)
                    if method == "wide_output_distillation":
                        run_config["hidden_size"] = 192
                    row, prediction = run_one(tag, data, teacher_train, teacher_val, run_config,
                                              method, seed, subset_out)
                subset_rows.append(row)
                ensemble_predictions[method].append(prediction)
                all_rows = [r for r in all_rows if not (r["subset"] == tag and r["method"] == method and r["seed"] == seed)]
                all_rows.append(row)
                save_csv(metrics_path, all_rows)
                print(f"{tag} {method} seed={seed}: RMSE={row['RMSE']:.2f}, score={row['CMAPSS_score']:.1f}, epochs={row['epochs_run']}", flush=True)

            ensemble = np.mean(np.stack(ensemble_predictions[method]), axis=0)
            ensemble_metrics = regression_metrics(data["y_val"], ensemble)
            np.savez_compressed(subset_out / f"{method}_three_seed_ensemble_validation.npz",
                                y_true=data["y_val"], y_pred=ensemble)
            method_seed_rows = [r for r in subset_rows if r["method"] == method]
            ens_row = {"subset": tag, "method": f"{method}_ensemble", "seed": "ensemble",
                       "learning_rate": config["learning_rate"], "sparsity_weight": config["sparsity_weight"],
                       "leak_init": config["leak_init"], **ensemble_metrics,
                       "spike_rate": float(np.mean([r["spike_rate"] for r in method_seed_rows])),
                       "epochs_run": float(np.mean([r["epochs_run"] for r in method_seed_rows]))}
            subset_rows.append(ens_row)
            all_rows = [r for r in all_rows if not (r["subset"] == tag and r["method"] == ens_row["method"])]
            all_rows.append(ens_row)
            save_csv(metrics_path, all_rows)

        ensembles = [r for r in subset_rows if str(r["method"]).endswith("ensemble")]
        best_by_score = min(ensembles, key=lambda r: (float(r["CMAPSS_score"]), float(r["RMSE"])))
        best_by_rmse = min(ensembles, key=lambda r: (float(r["RMSE"]), float(r["CMAPSS_score"])))
        comparison = {"teacher_validation": teacher_metrics,
                      "best_student_ensemble_by_cmapss_score": best_by_score,
                      "best_student_ensemble_by_rmse": best_by_rmse,
                      "student_beats_teacher_rmse": float(best_by_rmse["RMSE"]) < teacher_metrics["RMSE"],
                      "student_beats_teacher_cmapss_score": float(best_by_score["CMAPSS_score"]) < teacher_metrics["CMAPSS_score"],
                      "training_data_only": True}
        all_summaries[tag] = comparison
        (subset_out / "teacher_comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
        (OUT / "summary.json").write_text(json.dumps(all_summaries, indent=2), encoding="utf-8")
        print(f"{tag} teacher RMSE={teacher_metrics['RMSE']:.2f}; best SNN ensemble RMSE={best_by_rmse['RMSE']:.2f}", flush=True)
        del z, tr, va, data, teacher_train, teacher_val
        gc.collect()

    print("Focused SNN comparison complete:", OUT, flush=True)


if __name__ == "__main__":
    main()
