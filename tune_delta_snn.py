"""Validation-only test of delta-event encoding for the recurrent LIF student."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from train_spike_models import DATA, DEVICE, RecurrentLIF, regression_metrics, seed_all

ROOT = Path(__file__).resolve().parent
TEACHER = ROOT / "results" / "bilstm_teacher_tuned"
OUT = ROOT / "results" / "spike_models" / "delta_snn"
RECIPES = {
    "FD003": {"hidden_size": 192, "lr": .0003, "alpha": .1, "leak": .8},
    "FD004": {"hidden_size": 192, "lr": .0015, "alpha": .2, "leak": .8},
}
BATCH, MAX_EPOCHS, PATIENCE = 128, 55, 10
DELTA_THRESHOLD = .05


class DeltaEventSNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, leak_init: float):
        super().__init__()
        self.input_size = input_size
        self.encoder = RecurrentLIF(input_size * 3, hidden_size, leak_init)
        self.temporal_attention = nn.Sequential(nn.Linear(hidden_size, 64), nn.Tanh(), nn.Linear(64, 1, bias=False))
        self.readout = nn.Linear(hidden_size, 1)
        self.teacher_projection = nn.Linear(192, hidden_size, bias=False)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        delta = torch.zeros_like(x)
        delta[:, 1:] = x[:, 1:] - x[:, :-1]
        positive = (delta >= DELTA_THRESHOLD).to(x.dtype)
        negative = (delta <= -DELTA_THRESHOLD).to(x.dtype)
        return torch.cat((x, positive, negative), dim=-1)

    def forward(self, x: torch.Tensor):
        membrane, spikes, threshold = self.encoder(self.encode(x))
        attention = torch.softmax(self.temporal_attention(membrane).squeeze(-1), dim=1)
        context = torch.sum(membrane * attention.unsqueeze(-1), dim=1)
        pred = self.readout(context).squeeze(-1)
        return pred, membrane, attention, spikes, threshold


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", choices=list(RECIPES), action="append")
    args = parser.parse_args()
    torch.set_num_threads(min(8, torch.get_num_threads()))
    OUT.mkdir(parents=True, exist_ok=True)
    report = {}
    for tag in args.subset or list(RECIPES):
        cfg = RECIPES[tag]
        seed = 42
        seed_all(seed)
        z = np.load(DATA / f"windows_{tag}.npz")
        x, y = z["X_train"].astype(np.float32), z["y_train"].astype(np.float32)
        xv, yv = z["X_val"].astype(np.float32), z["y_val"].astype(np.float32)
        ty = np.load(TEACHER / f"teacher_signals_{tag}_train.npz")["y_teacher"].astype(np.float32)
        model = DeltaEventSNN(x.shape[2], cfg["hidden_size"], cfg["leak"]).to(DEVICE)
        optimizer = torch.optim.AdamW([p for n,p in model.named_parameters() if n!="teacher_projection.weight"],
                                      lr=cfg["lr"], weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=.5, patience=5, min_lr=1e-5)
        weights = np.where(y < 30, 1.4, np.where(y < 80, 1., .85)).astype(np.float64)
        sampler = WeightedRandomSampler(torch.as_tensor(weights), len(y), replacement=True,
                                        generator=torch.Generator().manual_seed(seed))
        loader = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(ty)),
                            batch_size=BATCH, sampler=sampler)
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
                loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.); optimizer.step()
            model.eval()
            with torch.no_grad(): pred = model(xvt)[0].cpu().numpy().astype(np.float32)
            metrics = regression_metrics(yv, pred); scheduler.step(metrics["MSE"])
            history.append({"epoch": epoch, **metrics})
            if metrics["RMSE"] < best_rmse - 1e-4:
                best_rmse, best_pred, best_epoch, stale = metrics["RMSE"], pred.copy(), epoch, 0
                best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            else:
                stale += 1
                if stale >= PATIENCE: break
            if epoch == 1 or epoch % 5 == 0:
                print(f"{tag} delta-SNN epoch={epoch} val_RMSE={metrics['RMSE']:.3f} best={best_rmse:.3f}@{best_epoch}",flush=True)
        name=f"{tag}_delta_event_seed42"
        torch.save({"state_dict":best_state,"input_size":x.shape[2],"hidden_size":cfg["hidden_size"],
                    "config":{**cfg,"delta_threshold":DELTA_THRESHOLD},"best_epoch":best_epoch},OUT/f"{name}.pt")
        np.savez_compressed(OUT/f"{name}_validation.npz",y_true=yv,y_pred=best_pred)
        (OUT/f"{name}_history.json").write_text(json.dumps(history,indent=2),encoding="utf-8")
        baseline_name = "output_mse_h192_seed42" if tag in ("FD003","FD004") else "label_mse_h96_seed42"
        base=np.load(ROOT/"results"/"spike_models"/"rmse_refinement"/tag/f"{baseline_name}_validation.npz")
        report[tag]={"config":{**cfg,"delta_threshold":DELTA_THRESHOLD},"best_epoch":best_epoch,
                     "delta_validation":regression_metrics(yv,best_pred),
                     "linear_validation":regression_metrics(base["y_true"],base["y_pred"])}
        print(f"{tag}: delta SNN RMSE={best_rmse:.3f}@{best_epoch}; baseline={report[tag]['linear_validation']['RMSE']:.3f}",flush=True)
    (OUT/"summary.json").write_text(json.dumps(report,indent=2),encoding="utf-8")


if __name__ == "__main__":
    main()
