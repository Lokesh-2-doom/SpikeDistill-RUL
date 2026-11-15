"""Train SNN baselines, vanilla LSTM KD, and SpikeDistill-RUL students.

Stages, in order:
  1. Direct surrogate-gradient recurrent LIF SNN and ANN-to-SNN conversion.
  2. Small LSTM with output-only teacher distillation.
  3. Recurrent LIF student with output, membrane, attention, and sparsity losses.

Run all stages: python train_spike_models.py
Optional: python train_spike_models.py --stage 1|2|3|all --subset FD001
Uses only training/validation data for fitting and selection. Official test labels
are not loaded. Requires numpy and PyTorch; sklearn is not used.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.autograd import Function
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "CMAPSSData" / "processed"
BASELINES = ROOT / "results" / "baselines"
TEACHER = ROOT / "results" / "bilstm_teacher_tuned"
OUT = ROOT / "results" / "spike_models"
SEED = 42
BATCH = 256
EPOCHS_SNN = 12
EPOCHS_KD = 30
PATIENCE = 5
SNN_HIDDEN = 96
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_all(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SurrogateSpike(Function):
    @staticmethod
    def forward(ctx, value: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(value)
        return (value >= 0).to(value.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor]:
        (value,) = ctx.saved_tensors
        # Fast-sigmoid surrogate derivative for the hard threshold.
        slope = 5.0
        derivative = 1.0 / (1.0 + slope * value.abs()).pow(2)
        return (grad_output * derivative,)


def spike(x: torch.Tensor) -> torch.Tensor:
    return SurrogateSpike.apply(x)


class RecurrentLIF(nn.Module):
    """Recurrent LIF encoder with trainable per-unit threshold and leak."""

    def __init__(self, input_size: int, hidden_size: int = SNN_HIDDEN, leak_init: float = 0.75) -> None:
        super().__init__()
        self.input = nn.Linear(input_size, hidden_size)
        self.recurrent = nn.Linear(hidden_size, hidden_size, bias=False)
        self.raw_threshold = nn.Parameter(torch.full((hidden_size,), 0.5413))  # softplus ~= 1
        leak_fraction = min(max((leak_init - 0.5) / 0.49, 1e-4), 1 - 1e-4)
        leak_logit = float(np.log(leak_fraction / (1 - leak_fraction)))
        self.raw_leak = nn.Parameter(torch.full((hidden_size,), leak_logit))

    def forward(self, x: torch.Tensor, ablate_spikes_at: int | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, steps, _ = x.shape
        membrane = x.new_zeros((batch, self.raw_threshold.numel()))
        previous_spike = torch.zeros_like(membrane)
        threshold = torch.nn.functional.softplus(self.raw_threshold) + 1e-3
        leak = 0.5 + 0.49 * torch.sigmoid(self.raw_leak)
        membranes: list[torch.Tensor] = []
        spikes: list[torch.Tensor] = []
        for t in range(steps):
            current = self.input(x[:, t]) + self.recurrent(previous_spike)
            membrane = leak * membrane + current
            pre_reset = membrane
            emitted = spike(pre_reset - threshold)
            if ablate_spikes_at == t:
                emitted = torch.zeros_like(emitted)
            membrane = pre_reset - emitted * threshold
            membranes.append(pre_reset)
            spikes.append(emitted)
            previous_spike = emitted
        return torch.stack(membranes, dim=1), torch.stack(spikes, dim=1), threshold


class DirectSNN(nn.Module):
    def __init__(self, input_size: int) -> None:
        super().__init__()
        self.encoder = RecurrentLIF(input_size)
        self.readout = nn.Linear(SNN_HIDDEN, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        membrane, spikes, _ = self.encoder(x)
        pred = self.readout(membrane[:, -1]).squeeze(-1)
        return pred, spikes


class DistilledSNN(nn.Module):
    def __init__(self, input_size: int, teacher_hidden: int = 192, teacher_attention: int = 96,
                 leak_init: float = 0.75, hidden_size: int = SNN_HIDDEN) -> None:
        super().__init__()
        self.encoder = RecurrentLIF(input_size, hidden_size=hidden_size, leak_init=leak_init)
        self.temporal_attention = nn.Sequential(
            nn.Linear(hidden_size, 64), nn.Tanh(), nn.Linear(64, 1, bias=False)
        )
        self.readout = nn.Linear(hidden_size, 1)
        self.teacher_projection = nn.Linear(teacher_hidden, hidden_size, bias=False)

    def forward(self, x: torch.Tensor, ablate_spikes_at: int | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        membrane, spikes, threshold = self.encoder(x, ablate_spikes_at=ablate_spikes_at)
        attention = torch.softmax(self.temporal_attention(membrane).squeeze(-1), dim=1)
        context = torch.sum(membrane * attention.unsqueeze(-1), dim=1)
        pred = self.readout(context).squeeze(-1)
        return pred, membrane, attention, spikes, threshold


class SmallLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 32) -> None:
        super().__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.regressor = nn.Sequential(nn.Dropout(0.1), nn.Linear(hidden_size, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence, _ = self.encoder(x)
        return self.regressor(sequence[:, -1]).squeeze(-1)


def regression_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    error = pred - y
    mse = float(np.mean(error**2))
    total = float(np.sum((y - y.mean()) ** 2))
    d = error
    asym = np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
    return {
        "MAE": float(np.mean(np.abs(error))),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "R2": float(1.0 - np.sum(error**2) / total) if total else float("nan"),
        "CMAPSS_score": float(asym.sum()),
    }


def fit_supervised(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int,
) -> tuple[nn.Module, np.ndarray, dict[str, object]]:
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-5)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=BATCH, shuffle=True, generator=torch.Generator().manual_seed(SEED),
    )
    xv = torch.from_numpy(x_val).to(DEVICE)
    yv = torch.from_numpy(y_val).to(DEVICE)
    best = float("inf")
    best_state = None
    stale = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            if isinstance(model, DirectSNN):
                pred, _ = model(xb)
            else:
                pred = model(xb)
            loss = nn.functional.mse_loss(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.item()) * len(xb)
            count += len(xb)
        model.eval()
        with torch.no_grad():
            if isinstance(model, DirectSNN):
                val_pred, _ = model(xv)
            else:
                val_pred = model(xv)
            val_loss = float(nn.functional.mse_loss(val_pred, yv).item())
        scheduler.step(val_loss)
        history.append({"epoch": epoch, "train_mse": total / count, "val_mse": val_loss})
        if val_loss < best - 1e-5:
            best = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    if best_state is None:
        raise RuntimeError("Training did not produce a best checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        if isinstance(model, DirectSNN):
            pred, spikes = model(xv)
            spike_rate = float(spikes.mean().cpu())
        else:
            pred = model(xv)
            spike_rate = float("nan")
    return model, pred.cpu().numpy().astype(np.float32), {
        "device": str(DEVICE), "epochs_run": len(history), "best_val_mse": best,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "history": history, "validation_spike_rate": spike_rate,
    }


def fit_output_kd(
    model: SmallLSTM,
    x_train: np.ndarray,
    y_train: np.ndarray,
    teacher_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    teacher_val: np.ndarray,
) -> tuple[SmallLSTM, np.ndarray, dict[str, object]]:
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-5)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train), torch.from_numpy(teacher_train)),
        batch_size=BATCH, shuffle=True, generator=torch.Generator().manual_seed(SEED),
    )
    xv = torch.from_numpy(x_val).to(DEVICE)
    yv = torch.from_numpy(y_val).to(DEVICE)
    tv = torch.from_numpy(teacher_val).to(DEVICE)
    best = float("inf")
    best_state = None
    stale = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, EPOCHS_KD + 1):
        model.train()
        total = 0.0
        count = 0
        for xb, yb, tb in loader:
            xb, yb, tb = xb.to(DEVICE), yb.to(DEVICE), tb.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = 0.5 * nn.functional.mse_loss(pred, yb) + 0.5 * nn.functional.mse_loss(pred, tb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.item()) * len(xb)
            count += len(xb)
        model.eval()
        with torch.no_grad():
            pred_val = model(xv)
            val_loss = float((0.5 * nn.functional.mse_loss(pred_val, yv) + 0.5 * nn.functional.mse_loss(pred_val, tv)).item())
        scheduler.step(val_loss)
        history.append({"epoch": epoch, "train_kd_mse": total / count, "val_combined_mse": val_loss})
        if val_loss < best - 1e-5:
            best = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    if best_state is None:
        raise RuntimeError("KD training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(xv).cpu().numpy().astype(np.float32)
    return model, pred, {"device": str(DEVICE), "epochs_run": len(history), "best_val_combined_mse": best, "history": history}


def fit_distilled_snn(
    model: DistilledSNN,
    x_train: np.ndarray,
    y_train: np.ndarray,
    teacher_y: np.ndarray,
    teacher_hidden: np.ndarray,
    teacher_attention: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    teacher_y_val: np.ndarray,
    teacher_hidden_val: np.ndarray,
    teacher_attention_val: np.ndarray,
    learning_rate: float = 8e-4,
    sparsity_weight: float = 0.01,
    seed: int = SEED,
    loss_ablation: str | None = None,
    epochs_override: int | None = None,
) -> tuple[DistilledSNN, np.ndarray, dict[str, object]]:
    seed_all(seed)
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-5)
    # Teacher hidden trajectories are stored float16 to keep artifacts compact.
    tensors = TensorDataset(
        torch.from_numpy(x_train), torch.from_numpy(y_train), torch.from_numpy(teacher_y),
        torch.from_numpy(teacher_hidden), torch.from_numpy(teacher_attention),
    )
    loader = DataLoader(tensors, batch_size=128, shuffle=True, generator=torch.Generator().manual_seed(seed))
    xv = torch.from_numpy(x_val).to(DEVICE)
    yv = torch.from_numpy(y_val).to(DEVICE)
    tyv = torch.from_numpy(teacher_y_val).to(DEVICE)
    thv = torch.from_numpy(teacher_hidden_val).to(device=DEVICE, dtype=torch.float32)
    tav = torch.from_numpy(teacher_attention_val).to(device=DEVICE, dtype=torch.float32)
    best = float("inf")
    best_state = None
    stale = 0
    history: list[dict[str, float]] = []
    epochs = epochs_override or 16
    weights = {"label": 0.50, "output": 0.25, "membrane": 0.20, "attention": 0.05, "sparsity": sparsity_weight}
    if loss_ablation is not None:
        if loss_ablation not in weights:
            raise ValueError(f"Unknown loss ablation: {loss_ablation}")
        weights[loss_ablation] = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        totals = np.zeros(6, dtype=float)
        batches = 0
        for xb, yb, tb, hb, ab in loader:
            xb, yb, tb = xb.to(DEVICE), yb.to(DEVICE), tb.to(DEVICE)
            hb, ab = hb.to(device=DEVICE, dtype=torch.float32), ab.to(device=DEVICE, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            pred, membrane, attention, emitted, _ = model(xb)
            projected_teacher = model.teacher_projection(hb)
            loss_label = nn.functional.mse_loss(pred, yb)
            loss_output = nn.functional.mse_loss(pred, tb)
            loss_membrane = nn.functional.smooth_l1_loss(membrane, projected_teacher)
            loss_attention = nn.functional.mse_loss(attention, ab)
            loss_sparse = emitted.mean()
            loss = (weights["label"] * loss_label + weights["output"] * loss_output +
                    weights["membrane"] * loss_membrane + weights["attention"] * loss_attention +
                    weights["sparsity"] * loss_sparse)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            totals += [float(loss.item()), float(loss_label.item()), float(loss_output.item()), float(loss_membrane.item()), float(loss_attention.item()), float(loss_sparse.item())]
            batches += 1

        model.eval()
        with torch.no_grad():
            pv, mv, av, sv, _ = model(xv)
            projv = model.teacher_projection(thv)
            val_label_mse = nn.functional.mse_loss(pv, yv)
            val_objective = (
                weights["label"] * val_label_mse +
                weights["output"] * nn.functional.mse_loss(pv, tyv) +
                weights["membrane"] * nn.functional.smooth_l1_loss(mv, projv) +
                weights["attention"] * nn.functional.mse_loss(av, tav) +
                weights["sparsity"] * sv.mean()
            )
            val_objective_value = float(val_objective.item())
            val_loss = float(val_label_mse.item())
        scheduler.step(val_loss)
        means = totals / max(batches, 1)
        history.append({"epoch": epoch, "train_total": means[0], "train_label": means[1], "train_output": means[2], "train_membrane": means[3], "train_attention": means[4], "train_spike_rate": means[5], "val_label_mse": val_loss, "val_objective": val_objective_value})
        if val_loss < best - 1e-5:
            best = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= 5:
                break
    if best_state is None:
        raise RuntimeError("SpikeDistill training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred, _, attention, spikes, threshold = model(xv)
    return model, pred.cpu().numpy().astype(np.float32), {
        "device": str(DEVICE), "epochs_run": len(history), "best_val_mse": best,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "validation_spike_rate": float(spikes.mean().cpu()),
        "learned_threshold_mean": float(threshold.mean().cpu()),
        "learned_leak_mean": float((0.5 + 0.49 * torch.sigmoid(model.encoder.raw_leak)).mean().detach().cpu()),
        "history": history, "loss_weights": weights,
        "loss_ablation": loss_ablation or "full",
    }


def convert_ann_to_snn(tag: str, x_val: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    """Rate-coded LIF conversion of the already supervised DeepNN baseline."""
    from train_baselines import DeepNN

    checkpoint = torch.load(BASELINES / f"deepnn_{tag}.pt", map_location="cpu", weights_only=False)
    input_dim = int(checkpoint["input_dim"])
    ann = DeepNN(input_dim)
    ann.load_state_dict(checkpoint["state_dict"])
    ann.eval()
    layers = [m for m in ann.layers if isinstance(m, nn.Linear)]
    if len(layers) != 4:
        raise ValueError(f"Unexpected DeepNN layer count in {tag}: {len(layers)}")
    # Train-set calibration keeps thresholds independent of validation labels.
    z = np.load(DATA / f"windows_{tag}.npz")
    x_cal = z["X_train"].reshape(len(z["X_train"]), -1).astype(np.float32)
    with torch.no_grad():
        activations = torch.from_numpy(x_cal[: min(8192, len(x_cal))])
        maxima = []
        for layer in layers[:-1]:
            activations = torch.relu(layer(activations))
            maxima.append(activations.amax(dim=0).numpy())
    steps = 24
    # A layer threshold must cover its calibrated activation range; dividing
    # by the number of simulation steps saturates each neuron at one spike per
    # step and destroys the rate representation.
    thresholds = [np.maximum(v, 1e-4).astype(np.float32) for v in maxima]
    x = torch.from_numpy(x_val.reshape(len(x_val), -1).astype(np.float32)).clamp(0, 1)
    with torch.no_grad():
        rates = x
        input_scale = 1.0
        for i, layer in enumerate(layers[:-1]):
            threshold = torch.from_numpy(thresholds[i])
            membrane = torch.zeros((len(x), layer.out_features), dtype=torch.float32)
            emitted_sum = torch.zeros_like(membrane)
            for _ in range(steps):
                membrane = membrane + layer(rates * input_scale)
                emitted = (membrane >= threshold).float()
                membrane = membrane - emitted * threshold
                emitted_sum += emitted
            rates = emitted_sum / steps
            input_scale = threshold
        prediction = layers[-1](rates * input_scale).squeeze(-1).numpy().astype(np.float32)
    return prediction, {"conversion": "rate-coded ReLU-to-LIF using pretrained DeepNN weights", "simulation_steps": steps, "thresholds_per_neuron": [v.tolist() for v in thresholds], "calibration": "first 8192 training windows, no validation labels"}


def write_metrics(path: Path, rows: list[dict[str, object]]) -> None:
    fields = ["subset", "model", "MAE", "MSE", "RMSE", "R2", "CMAPSS_score", "spike_rate"]
    merged: dict[tuple[str, str], dict[str, object]] = {}
    if path.exists():
        with path.open("r", newline="", encoding="utf-8") as handle:
            for old in csv.DictReader(handle):
                merged[(old["subset"], old["model"])] = old
    for row in rows:
        merged[(str(row["subset"]), str(row["model"]))] = row
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(merged.values())


def run_subset(tag: str, stage: str) -> None:
    seed_all()
    data = np.load(DATA / f"windows_{tag}.npz")
    xt = data["X_train"].astype(np.float32)
    yt = data["y_train"].astype(np.float32)
    xv = data["X_val"].astype(np.float32)
    yv = data["y_val"].astype(np.float32)
    rows: list[dict[str, object]] = []

    if stage in {"1", "all"}:
        out1 = OUT / "snn_baselines"
        out1.mkdir(parents=True, exist_ok=True)
        model, pred, info = fit_supervised(DirectSNN(xt.shape[2]), xt, yt, xv, yv, EPOCHS_SNN)
        with torch.no_grad():
            _, validation_spikes = model(torch.from_numpy(xv).to(DEVICE))
        rate = float(validation_spikes.mean().cpu())
        torch.save({"state_dict": model.state_dict(), "input_size": xt.shape[2], "hidden_size": SNN_HIDDEN}, out1 / f"surrogate_snn_{tag}.pt")
        (out1 / f"surrogate_snn_{tag}_history.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
        np.savez_compressed(out1 / f"surrogate_snn_{tag}_validation_predictions.npz", y_true=yv, y_pred=pred)
        rows.append({"subset": tag, "model": "surrogate_snn", **regression_metrics(yv, pred), "spike_rate": rate})

        ann_pred, ann_info = convert_ann_to_snn(tag, xv)
        np.savez_compressed(out1 / f"ann_to_snn_{tag}_validation_predictions.npz", y_true=yv, y_pred=ann_pred)
        (out1 / f"ann_to_snn_{tag}_config.json").write_text(json.dumps(ann_info, indent=2), encoding="utf-8")
        rows.append({"subset": tag, "model": "ann_to_snn", **regression_metrics(yv, ann_pred), "spike_rate": float("nan")})
        write_metrics(out1 / "snn_baseline_validation_metrics.csv", rows)
        print(f"{tag} stage 1 complete: direct SNN RMSE={rows[0]['RMSE']:.2f}; ANN-to-SNN RMSE={rows[1]['RMSE']:.2f}", flush=True)

    if stage == "ann":
        out1 = OUT / "snn_baselines"
        out1.mkdir(parents=True, exist_ok=True)
        ann_pred, ann_info = convert_ann_to_snn(tag, xv)
        np.savez_compressed(out1 / f"ann_to_snn_{tag}_validation_predictions.npz", y_true=yv, y_pred=ann_pred)
        (out1 / f"ann_to_snn_{tag}_config.json").write_text(json.dumps(ann_info, indent=2), encoding="utf-8")
        row = {"subset": tag, "model": "ann_to_snn", **regression_metrics(yv, ann_pred), "spike_rate": float("nan")}
        write_metrics(out1 / "snn_baseline_validation_metrics.csv", [row])
        print(f"{tag} corrected ANN-to-SNN RMSE={row['RMSE']:.2f}", flush=True)

    if stage in {"2", "all"}:
        out2 = OUT / "vanilla_kd"
        out2.mkdir(parents=True, exist_ok=True)
        teacher_train = np.load(TEACHER / f"teacher_signals_{tag}_train.npz")
        teacher_val = np.load(TEACHER / f"teacher_signals_{tag}_val.npz")
        student, pred, info = fit_output_kd(
            SmallLSTM(xt.shape[2]), xt, yt, teacher_train["y_teacher"].astype(np.float32),
            xv, yv, teacher_val["y_teacher"].astype(np.float32),
        )
        torch.save({"state_dict": student.state_dict(), "input_size": xt.shape[2], "hidden_size": 32}, out2 / f"vanilla_kd_lstm_{tag}.pt")
        (out2 / f"vanilla_kd_lstm_{tag}_history.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
        np.savez_compressed(out2 / f"vanilla_kd_lstm_{tag}_validation_predictions.npz", y_true=yv, y_pred=pred)
        row = {"subset": tag, "model": "vanilla_kd_lstm", **regression_metrics(yv, pred), "spike_rate": float("nan")}
        rows.append(row)
        write_metrics(out2 / "vanilla_kd_validation_metrics.csv", rows)
        print(f"{tag} stage 2 complete: output-only KD RMSE={row['RMSE']:.2f}", flush=True)

    if stage in {"3", "all"}:
        out3 = OUT / "spikedistill_rul"
        out3.mkdir(parents=True, exist_ok=True)
        teacher_train = np.load(TEACHER / f"teacher_signals_{tag}_train.npz")
        teacher_val = np.load(TEACHER / f"teacher_signals_{tag}_val.npz")
        # Copy into float32 only per subset while it trains; source remains compact float16.
        student, pred, info = fit_distilled_snn(
            DistilledSNN(xt.shape[2]), xt, yt,
            teacher_train["y_teacher"].astype(np.float32),
            teacher_train["hidden_states"].astype(np.float16),
            teacher_train["attention"].astype(np.float32),
            xv, yv,
            teacher_val["y_teacher"].astype(np.float32),
            teacher_val["hidden_states"].astype(np.float16),
            teacher_val["attention"].astype(np.float32),
        )
        torch.save({"state_dict": student.state_dict(), "input_size": xt.shape[2], "hidden_size": SNN_HIDDEN}, out3 / f"spikedistill_rul_{tag}.pt")
        (out3 / f"spikedistill_rul_{tag}_history.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
        with torch.no_grad():
            _, _, attn_val, spikes_val, _ = student(torch.from_numpy(xv).to(DEVICE))
        np.savez_compressed(
            out3 / f"spikedistill_rul_{tag}_validation_outputs.npz",
            y_true=yv, y_pred=pred, attention=attn_val.cpu().numpy().astype(np.float16),
            spike_raster=spikes_val.cpu().numpy().astype(np.uint8),
        )
        row = {"subset": tag, "model": "spikedistill_rul", **regression_metrics(yv, pred), "spike_rate": info["validation_spike_rate"]}
        rows.append(row)
        write_metrics(out3 / "spikedistill_validation_metrics.csv", rows)
        print(f"{tag} stage 3 complete: SpikeDistill RMSE={row['RMSE']:.2f}, spike rate={row['spike_rate']:.4f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["1", "ann", "2", "3", "all"], default="all")
    parser.add_argument("--subset", choices=[f"FD{i:03d}" for i in range(1, 5)])
    args = parser.parse_args()
    torch.set_num_threads(min(8, torch.get_num_threads()))
    subsets = [args.subset] if args.subset else [f"FD{i:03d}" for i in range(1, 5)]
    for tag in subsets:
        print(f"Starting {tag}, stage {args.stage}", flush=True)
        run_subset(tag, args.stage)


if __name__ == "__main__":
    main()
