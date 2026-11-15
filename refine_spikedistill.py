"""Refine SpikeDistill by RMSE-aligned checkpoint selection.

Trains label-MSE and teacher-output-MSE candidates at 96/192 units. The
official test artifacts are never loaded. Run: python refine_spikedistill.py
"""

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
CONFIGS = ROOT / "results" / "spike_models" / "tuning" / "selected_configs.json"
OUT = ROOT / "results" / "spike_models" / "rmse_refinement"
SEEDS = (42, 43, 44)
MAX_EPOCHS = 50
PATIENCE = 10
BATCH = 128


def write_rows(rows: list[dict[str, object]]) -> None:
    fields = ["subset", "candidate", "seed", "hidden_size", "learning_rate",
              "sparsity_weight", "leak_init", "RMSE", "MAE", "MSE",
              "R2", "CMAPSS_score", "spike_rate", "epochs_run"]
    with (OUT / "validation_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def train(tag: str, data: dict[str, np.ndarray], teacher_train: np.ndarray,
          cfg: dict[str, float], candidate: str, width: int, seed: int,
          directory: Path) -> dict[str, object]:
    seed_all(seed)
    model = DistilledSNN(data["x_train"].shape[2], leak_init=cfg["leak_init"],
                         hidden_size=width).to(DEVICE)
    trainable = [p for name, p in model.named_parameters() if name != "teacher_projection.weight"]
    optimizer = torch.optim.AdamW(trainable, lr=cfg["learning_rate"], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4, min_lr=1e-5)
    weights = np.where(data["y_train"] < 30, 1.4,
                       np.where(data["y_train"] < 80, 1.0, 0.85)).astype(np.float64)
    sampler = WeightedRandomSampler(torch.as_tensor(weights), len(weights), replacement=True,
                                    generator=torch.Generator().manual_seed(seed))
    loader = DataLoader(
        TensorDataset(torch.from_numpy(data["x_train"]), torch.from_numpy(data["y_train"]),
                      torch.from_numpy(teacher_train)), batch_size=BATCH, sampler=sampler)
    xv = torch.from_numpy(data["x_val"]).to(DEVICE)
    yv = torch.from_numpy(data["y_val"]).to(DEVICE)
    best_rmse = float("inf")
    best_state = None
    best_prediction = None
    best_metrics = None
    history: list[dict[str, float]] = []
    stale = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total = 0.0
        n = 0
        for xb, yb, tb in loader:
            xb, yb, tb = xb.to(DEVICE), yb.to(DEVICE), tb.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)[0]
            if candidate == "label_mse":
                loss = nn.functional.mse_loss(pred, yb)
            else:
                loss = 0.8 * nn.functional.mse_loss(pred, yb) + 0.2 * nn.functional.mse_loss(pred, tb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.item()) * len(xb)
            n += len(xb)
        model.eval()
        with torch.no_grad():
            prediction = model(xv)[0].cpu().numpy().astype(np.float32)
        metrics = regression_metrics(data["y_val"], prediction)
        scheduler.step(metrics["MSE"])
        history.append({"epoch": epoch, "train_mse": total / max(n, 1),
                        "val_rmse": metrics["RMSE"], "val_cmapss_score": metrics["CMAPSS_score"]})
        if metrics["RMSE"] < best_rmse - 1e-4:
            best_rmse = metrics["RMSE"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_prediction = prediction.copy()
            best_metrics = metrics
            stale = 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    if best_state is None or best_prediction is None or best_metrics is None:
        raise RuntimeError(f"No best checkpoint for {tag} {candidate} seed {seed}")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        spike_rate = float(model(xv)[3].mean().cpu())
    row: dict[str, object] = {
        "subset": tag, "candidate": candidate, "seed": seed, "hidden_size": width,
        **cfg, **best_metrics, "spike_rate": spike_rate, "epochs_run": len(history),
    }
    stem = f"{candidate}_h{width}_seed{seed}"
    torch.save({"state_dict": best_state, "input_size": data["x_train"].shape[2],
                "hidden_size": width, "seed": seed,
                "config": {**cfg, "candidate": candidate}}, directory / f"{stem}.pt")
    np.savez_compressed(directory / f"{stem}_validation.npz",
                        y_true=data["y_val"], y_pred=best_prediction)
    (directory / f"{stem}_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    del model, loader, optimizer
    gc.collect()
    return row


def main() -> None:
    torch.set_num_threads(min(8, torch.get_num_threads()))
    OUT.mkdir(parents=True, exist_ok=True)
    config_by_subset = json.loads(CONFIGS.read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []
    summaries: dict[str, object] = {}
    for index in range(1, 5):
        tag = f"FD{index:03d}"
        print(f"Starting RMSE-aligned refinement: {tag}", flush=True)
        z = np.load(DATA / f"windows_{tag}.npz")
        data = {"x_train": z["X_train"].astype(np.float32),
                "y_train": z["y_train"].astype(np.float32),
                "x_val": z["X_val"].astype(np.float32),
                "y_val": z["y_val"].astype(np.float32)}
        train_teacher = np.load(TEACHER / f"teacher_signals_{tag}_train.npz")
        val_teacher = np.load(TEACHER / f"teacher_signals_{tag}_val.npz")
        ty = train_teacher["y_teacher"].astype(np.float32)
        teacher_pred = val_teacher["y_teacher"].astype(np.float32)
        teacher_metrics = regression_metrics(data["y_val"], teacher_pred)
        cfg = {k: float(v) for k, v in config_by_subset[tag].items()
               if k in {"learning_rate", "sparsity_weight", "leak_init"}}
        directory = OUT / tag
        directory.mkdir(parents=True, exist_ok=True)
        candidates = [("label_mse", 96), ("output_mse", 96), ("output_mse", 192)]
        seed42: list[dict[str, object]] = []
        for candidate, width in candidates:
            row = train(tag, data, ty, cfg, candidate, width, 42, directory)
            rows.append(row); seed42.append(row); write_rows(rows)
            print(f"{tag} {candidate} width={width}: RMSE={row['RMSE']:.2f}, score={row['CMAPSS_score']:.1f}", flush=True)
        winner = min(seed42, key=lambda r: (float(r["RMSE"]), float(r["CMAPSS_score"])))
        candidate, width = str(winner["candidate"]), int(winner["hidden_size"])
        predictions = [np.load(directory / f"{candidate}_h{width}_seed42_validation.npz")["y_pred"]]
        selected_rows = [winner]
        for seed in (43, 44):
            row = train(tag, data, ty, cfg, candidate, width, seed, directory)
            rows.append(row); selected_rows.append(row); write_rows(rows)
            predictions.append(np.load(directory / f"{candidate}_h{width}_seed{seed}_validation.npz")["y_pred"])
            print(f"{tag} selected {candidate} width={width} seed={seed}: RMSE={row['RMSE']:.2f}", flush=True)
        ensemble = np.mean(np.stack(predictions), axis=0)
        ensemble_metrics = regression_metrics(data["y_val"], ensemble)
        np.savez_compressed(directory / "selected_three_seed_ensemble_validation.npz",
                            y_true=data["y_val"], y_pred=ensemble)
        summary = {
            "teacher_validation": teacher_metrics,
            "selected_candidate": {"candidate": candidate, "hidden_size": width, **cfg},
            "selected_seed_metrics": [{k: r[k] for k in ("seed", "RMSE", "CMAPSS_score", "spike_rate")} for r in selected_rows],
            "selected_three_seed_mean_rmse": float(np.mean([float(r["RMSE"]) for r in selected_rows])),
            "selected_ensemble_validation": ensemble_metrics,
            "student_beats_teacher_rmse": ensemble_metrics["RMSE"] < teacher_metrics["RMSE"],
            "student_beats_teacher_cmapss_score": ensemble_metrics["CMAPSS_score"] < teacher_metrics["CMAPSS_score"],
        }
        summaries[tag] = summary
        (directory / "rmse_refinement_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (OUT / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
        print(f"{tag} teacher RMSE={teacher_metrics['RMSE']:.2f}; selected SNN ensemble RMSE={ensemble_metrics['RMSE']:.2f}", flush=True)
        del z, train_teacher, val_teacher, data, ty, teacher_pred
        gc.collect()
    print("RMSE refinement complete:", OUT, flush=True)


if __name__ == "__main__":
    main()
