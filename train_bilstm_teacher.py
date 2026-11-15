"""Train the attention-BiLSTM teacher for SpikeDistill-RUL.

Uses train/validation windows only for fitting and selection. The official test
split is exported as evaluation targets and is never used for early stopping.
Teacher trajectories (prediction, hidden state, attention) are saved for later
student distillation.

Run: python train_bilstm_teacher.py
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "CMAPSSData" / "processed"
OUT = ROOT / "results" / "bilstm_teacher_tuned"
SEED = 42
HIDDEN_SIZE = 96
ATTENTION_SIZE = 96
BATCH_SIZE = 256
EPOCHS = 60
PATIENCE = 12


def seed_everything() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def cmapss_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    d = y_pred - y_true
    return float(np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0).sum())


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    d = y_pred - y_true
    mse = float(np.mean(d**2))
    total = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "MAE": float(np.mean(np.abs(d))),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "R2": float(1.0 - np.sum(d**2) / total) if total else float("nan"),
        "CMAPSS_score": cmapss_score(y_true, y_pred),
    }


class AttentionBiLSTM(nn.Module):
    def __init__(self, input_size: int) -> None:
        super().__init__()
        self.encoder = nn.LSTM(input_size, HIDDEN_SIZE, batch_first=True, bidirectional=True)
        self.attention_hidden = nn.Linear(HIDDEN_SIZE * 2, ATTENTION_SIZE)
        self.attention_score = nn.Linear(ATTENTION_SIZE, 1, bias=False)
        self.regressor = nn.Sequential(nn.Dropout(0.1), nn.Linear(HIDDEN_SIZE * 2, 1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, _ = self.encoder(x)
        scores = self.attention_score(torch.tanh(self.attention_hidden(hidden))).squeeze(-1)
        attention = torch.softmax(scores, dim=1)
        context = torch.sum(hidden * attention.unsqueeze(-1), dim=1)
        prediction = self.regressor(context).squeeze(-1)
        return prediction, hidden, attention


def infer_outputs(model: nn.Module, x: np.ndarray, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    predictions: list[np.ndarray] = []
    hidden_states: list[np.ndarray] = []
    attention_weights: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x), BATCH_SIZE):
            xb = torch.from_numpy(x[start : start + BATCH_SIZE]).to(device)
            prediction, hidden, attention = model(xb)
            predictions.append(prediction.cpu().numpy().astype(np.float32))
            hidden_states.append(hidden.cpu().numpy().astype(np.float16))
            attention_weights.append(attention.cpu().numpy().astype(np.float16))
    return np.concatenate(predictions), np.concatenate(hidden_states), np.concatenate(attention_weights)


def train_one(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray) -> tuple[AttentionBiLSTM, dict[str, object]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AttentionBiLSTM(x_train.shape[2]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4, min_lr=1e-5
    )
    loss_fn = nn.MSELoss()
    # Capping produces many target=125 windows. Mildly upweighting the lower
    # RUL ranges makes the training target distribution closer to validation.
    sampling_weights = np.where(y_train < 30, 1.4, np.where(y_train < 80, 1.0, 0.85))
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sampling_weights, dtype=torch.double),
        num_samples=len(sampling_weights), replacement=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=BATCH_SIZE,
        sampler=sampler,
    )
    x_val_t = torch.from_numpy(x_val).to(device)
    y_val_t = torch.from_numpy(y_val).to(device)
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        examples = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction, _, _ = model(xb)
            loss = loss_fn(prediction, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.item()) * len(xb)
            examples += len(xb)
        model.eval()
        with torch.no_grad():
            val_prediction, _, _ = model(x_val_t)
            val_loss = float(loss_fn(val_prediction, y_val_t).item())
        scheduler.step(val_loss)
        history.append({"epoch": epoch, "train_mse": total_loss / examples, "val_mse": val_loss})
        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break

    if best_state is None:
        raise RuntimeError("Teacher did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    info = {
        "device": str(device),
        "seed": SEED,
        "epochs_run": len(history),
        "best_val_mse": best_loss,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "final_learning_rate": optimizer.param_groups[0]["lr"],
        "history": history,
    }
    return model, info


def main() -> None:
    seed_everything()
    torch.set_num_threads(min(8, torch.get_num_threads()))
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    config: dict[str, object] = {
        "architecture": "BiLSTM(96 per direction) + additive temporal attention(96) + scalar RUL head",
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "max_epochs": EPOCHS,
        "early_stopping_patience": PATIENCE,
        "optimization": "AdamW with validation ReduceLROnPlateau; weighted sampling (RUL <30: 1.4, 30-79: 1.0, >=80: 0.85)",
        "target": "capped RUL in cycles",
        "official_test_used_for_model_selection": False,
        "subsets": {},
    }

    for fd in range(1, 5):
        tag = f"FD{fd:03d}"
        data = np.load(DATA / f"windows_{tag}.npz")
        x_train = data["X_train"].astype(np.float32)
        y_train = data["y_train"].astype(np.float32)
        x_val = data["X_val"].astype(np.float32)
        y_val = data["y_val"].astype(np.float32)
        model, info = train_one(x_train, y_train, x_val, y_val)
        device = next(model.parameters()).device

        # Export teacher trajectories on train and validation examples for KD.
        for split_name, x, labels, ids, cycles in (
            ("train", x_train, y_train, data["train_unit"], data["train_end_cycle"]),
            ("val", x_val, y_val, data["val_unit"], data["val_end_cycle"]),
        ):
            pred, hidden, attention = infer_outputs(model, x, device)
            np.savez_compressed(
                OUT / f"teacher_signals_{tag}_{split_name}.npz",
                y_true=labels,
                y_teacher=pred,
                hidden_states=hidden,
                attention=attention,
                unit=ids,
                end_cycle=cycles,
                feature_names=data["feature_names"],
            )
        # Keep official test labels for final reporting only; model inference is
        # saved separately and does not feed back into fitting or selection.
        x_test = data["X_test"].astype(np.float32)
        y_test = data["y_test"].astype(np.float32)
        pred_test, _, attention_test = infer_outputs(model, x_test, device)
        np.savez_compressed(
            OUT / f"teacher_signals_{tag}_test_evaluation_only.npz",
            y_true=y_test,
            y_teacher=pred_test,
            attention=attention_test,
            unit=data["test_unit"],
            end_cycle=data["test_end_cycle"],
            feature_names=data["feature_names"],
        )

        torch.save(
            {"state_dict": model.state_dict(), "input_size": int(x_train.shape[2]), "hidden_size": HIDDEN_SIZE, "attention_size": ATTENTION_SIZE},
            OUT / f"attention_bilstm_teacher_{tag}.pt",
        )
        (OUT / f"attention_bilstm_teacher_{tag}_history.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
        row = {"subset": tag, **metrics(y_val, infer_outputs(model, x_val, device)[0])}
        rows.append(row)
        config["subsets"][tag] = {
            "train_shape": list(x_train.shape),
            "validation_shape": list(x_val.shape),
            "test_shape": list(x_test.shape),
            "parameter_count": info["parameter_count"],
            "epochs_run": info["epochs_run"],
            "best_val_mse": info["best_val_mse"],
            "device": info["device"],
            "validation_metrics": row,
        }
        fields = ["subset", "MAE", "MSE", "RMSE", "R2", "CMAPSS_score"]
        with (OUT / "teacher_validation_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        (OUT / "teacher_validation_metrics.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        (OUT / "teacher_run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        print(f"Finished attention-BiLSTM teacher {tag}: RMSE={row['RMSE']:.3f}", flush=True)

    print("All teacher models and distillation signals saved to", OUT, flush=True)


if __name__ == "__main__":
    main()
