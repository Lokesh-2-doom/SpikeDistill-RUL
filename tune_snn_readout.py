"""Validation-only experiment: add a nonlinear head to the recurrent SNN."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from train_spike_models import DATA, DEVICE, DistilledSNN, regression_metrics, seed_all

ROOT = Path(__file__).resolve().parent
TEACHER = ROOT / "results" / "bilstm_teacher_tuned"
OUT = ROOT / "results" / "spike_models" / "nonlinear_readout_snn"
RECIPES = {
    "FD001": {"hidden_size": 96, "lr": .0015, "alpha": 0., "leak": .8},
    "FD002": {"hidden_size": 192, "lr": .0015, "alpha": .2, "leak": .6},
    "FD003": {"hidden_size": 192, "lr": .0003, "alpha": .1, "leak": .8},
    "FD004": {"hidden_size": 192, "lr": .0015, "alpha": .2, "leak": .8},
}
BATCH, MAX_EPOCHS, PATIENCE = 128, 60, 10


class NonlinearReadoutSNN(DistilledSNN):
    def __init__(self, input_size: int, hidden_size: int, leak_init: float):
        super().__init__(input_size, hidden_size=hidden_size, leak_init=leak_init)
        self.readout = nn.Sequential(
            nn.Linear(hidden_size, 64), nn.GELU(), nn.Dropout(.1), nn.Linear(64, 1)
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", choices=list(RECIPES), action="append")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    torch.set_num_threads(min(8, torch.get_num_threads()))
    OUT.mkdir(parents=True, exist_ok=True)
    tags = args.subset or list(RECIPES)
    summary_path = OUT / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    for tag in tags:
        cfg = RECIPES[tag]
        seed = args.seed
        seed_all(seed)
        z = np.load(DATA / f"windows_{tag}.npz")
        x, y = z["X_train"].astype(np.float32), z["y_train"].astype(np.float32)
        xv, yv = z["X_val"].astype(np.float32), z["y_val"].astype(np.float32)
        signals = np.load(TEACHER / f"teacher_signals_{tag}_train.npz")
        teacher_y = signals["y_teacher"].astype(np.float32)
        model = NonlinearReadoutSNN(x.shape[2], cfg["hidden_size"], cfg["leak"]).to(DEVICE)
        optimizer = torch.optim.AdamW(
            [p for name, p in model.named_parameters() if name != "teacher_projection.weight"],
            lr=cfg["lr"], weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=.5, patience=5, min_lr=1e-5)
        sw = np.where(y < 30, 1.4, np.where(y < 80, 1., .85)).astype(np.float64)
        sampler = WeightedRandomSampler(torch.as_tensor(sw), len(sw), replacement=True,
                                        generator=torch.Generator().manual_seed(seed))
        loader = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y),
            torch.from_numpy(teacher_y)), batch_size=BATCH, sampler=sampler)
        xvt = torch.from_numpy(xv).to(DEVICE)
        best_rmse, best_state, best_pred, best_epoch, stale = float("inf"), None, None, 0, 0
        history = []
        for epoch in range(1, MAX_EPOCHS + 1):
            model.train()
            for xb, yb, tb in loader:
                xb, yb, tb = xb.to(DEVICE), yb.to(DEVICE), tb.to(DEVICE)
                optimizer.zero_grad(set_to_none=True)
                pred = model(xb)[0]
                loss = (1-cfg["alpha"]) * nn.functional.mse_loss(pred, yb) + cfg["alpha"] * nn.functional.mse_loss(pred, tb)
                loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            model.eval()
            with torch.no_grad():
                pred = model(xvt)[0].cpu().numpy().astype(np.float32)
            metrics = regression_metrics(yv, pred)
            scheduler.step(metrics["MSE"])
            history.append({"epoch": epoch, **metrics})
            if metrics["RMSE"] < best_rmse - 1e-4:
                best_rmse, best_state, best_pred, best_epoch, stale = metrics["RMSE"], {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, pred.copy(), epoch, 0
            else:
                stale += 1
                if stale >= PATIENCE:
                    break
            if epoch == 1 or epoch % 5 == 0:
                print(f"{tag} nonlinear readout epoch={epoch} val_RMSE={metrics['RMSE']:.3f} best={best_rmse:.3f}@{best_epoch}", flush=True)
        if best_state is None:
            raise RuntimeError(f"No checkpoint for {tag}")
        name = f"{tag}_nonlinear_readout_seed{seed}"
        torch.save({"state_dict": best_state, "input_size": x.shape[2],
                    "hidden_size": cfg["hidden_size"], "config": cfg, "seed": seed, "best_epoch": best_epoch}, OUT / f"{name}.pt")
        np.savez_compressed(OUT / f"{name}_validation.npz", y_true=yv, y_pred=best_pred)
        (OUT / f"{name}_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        baseline_path = (ROOT / "results" / "spike_models" / "rmse_refinement" / tag /
                         ("label_mse_h96_seed42_validation.npz" if tag == "FD001" else "output_mse_h192_seed42_validation.npz"))
        baseline = np.load(baseline_path)
        summary.setdefault(tag, {})[str(seed)] = {"recipe": cfg, "seed": seed, "best_epoch": best_epoch,
                        "nonlinear_readout_validation": regression_metrics(yv, best_pred)}
        # Keep comparison explicit without changing the existing linear model artifacts.
        e = baseline["y_pred"] - baseline["y_true"]
        summary[tag]["linear_readout_validation"] = {"RMSE": float(np.mean(e**2)**.5), "MAE": float(np.mean(np.abs(e)))}
        print(f"{tag}: best nonlinear validation RMSE={best_rmse:.3f} at epoch {best_epoch}", flush=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
