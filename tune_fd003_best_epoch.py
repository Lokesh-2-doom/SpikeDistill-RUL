"""Recover the selected FD003 SNN's best validation epoch for final refit."""
from __future__ import annotations

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
BATCH, MAX_EPOCHS, PATIENCE = 128, 60, 12


def main() -> None:
    torch.set_num_threads(min(8, torch.get_num_threads()))
    seed = 42
    cfg = {"hidden_size": 192, "lr": 0.0003, "teacher_weight": 0.1}
    d = np.load(DATA / "windows_FD003.npz")
    x = d["X_train"].astype(np.float32)
    y = d["y_train"].astype(np.float32)
    xv = torch.from_numpy(d["X_val"].astype(np.float32)).to(DEVICE)
    yv = d["y_val"].astype(np.float32)
    teacher = np.load(TEACHER / "teacher_signals_FD003_train.npz")["y_teacher"].astype(np.float32)

    seed_all(seed)
    model = DistilledSNN(x.shape[2], hidden_size=cfg["hidden_size"], leak_init=0.8).to(DEVICE)
    optimizer = torch.optim.AdamW(
        [p for n, p in model.named_parameters() if n != "teacher_projection.weight"],
        lr=cfg["lr"], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-5)
    sampling_weights = np.where(y < 30, 1.4, np.where(y < 80, 1.0, 0.85)).astype(np.float64)
    sampler = WeightedRandomSampler(torch.as_tensor(sampling_weights), len(y), replacement=True,
                                    generator=torch.Generator().manual_seed(seed))
    loader = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y),
                                      torch.from_numpy(teacher)), batch_size=BATCH, sampler=sampler)

    best_rmse, best_epoch, stale = float("inf"), 0, 0
    history = []
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for xb, yb, tb in loader:
            xb, yb, tb = xb.to(DEVICE), yb.to(DEVICE), tb.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)[0]
            alpha = cfg["teacher_weight"]
            loss = (1-alpha) * nn.functional.mse_loss(pred, yb) + alpha * nn.functional.mse_loss(pred, tb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            pred = model(xv)[0].cpu().numpy().astype(np.float32)
        metrics = regression_metrics(yv, pred)
        scheduler.step(metrics["MSE"])
        history.append({"epoch": epoch, **metrics})
        if metrics["RMSE"] < best_rmse - 1e-4:
            best_rmse, best_epoch, stale = metrics["RMSE"], epoch, 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break
        print(f"FD003 epoch {epoch}/{MAX_EPOCHS}: val RMSE={metrics['RMSE']:.3f}; best={best_rmse:.3f}@{best_epoch}", flush=True)

    result = {"config": cfg, "seed": seed, "best_epoch": best_epoch,
              "best_val_RMSE": best_rmse, "history": history,
              "batch_size": BATCH, "early_stopping_patience": PATIENCE}
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "selected_h192_lr0.0003_kd0.1_best_epoch.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved {path}; selected best epoch={best_epoch}, val RMSE={best_rmse:.3f}", flush=True)


if __name__ == "__main__":
    main()
