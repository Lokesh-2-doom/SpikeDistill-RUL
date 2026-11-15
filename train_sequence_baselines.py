"""Train RNN, LSTM, GRU, BiLSTM and CNN-LSTM C-MAPSS baselines.

Uses the existing leakage-aware 30-cycle windows and validation split.
Run: python train_sequence_baselines.py
Outputs are isolated under results/sequence_baselines/.
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "CMAPSSData" / "processed"
OUT = ROOT / "results" / "sequence_baselines"
SEED = 42
BATCH_SIZE = 256
EPOCHS = 25
PATIENCE = 5
HIDDEN = 64


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cmapss_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    error = y_pred - y_true
    scores = np.where(error < 0, np.exp(-error / 13.0) - 1.0, np.exp(error / 10.0) - 1.0)
    return float(scores.sum())


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    error = y_pred - y_true
    mse = float(np.mean(error**2))
    total = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "MAE": float(np.mean(np.abs(error))),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "R2": float(1.0 - np.sum(error**2) / total) if total else float("nan"),
        "CMAPSS_score": cmapss_score(y_true, y_pred),
    }


class RecurrentRegressor(nn.Module):
    def __init__(self, kind: str, input_size: int, hidden_size: int = HIDDEN) -> None:
        super().__init__()
        if kind == "rnn":
            self.rnn = nn.RNN(input_size, hidden_size, batch_first=True, nonlinearity="tanh")
            output_size = hidden_size
        elif kind == "lstm":
            self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
            output_size = hidden_size
        elif kind == "gru":
            self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
            output_size = hidden_size
        elif kind == "bilstm":
            self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
            output_size = hidden_size * 2
        else:
            raise ValueError(f"Unknown recurrent model: {kind}")
        self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(output_size, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence, _ = self.rnn(x)
        return self.head(sequence[:, -1]).squeeze(-1)


class CNNLSTMRegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = HIDDEN) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_size, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(64, hidden_size, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(hidden_size, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x.transpose(1, 2)).transpose(1, 2)
        sequence, _ = self.lstm(features)
        return self.head(sequence[:, -1]).squeeze(-1)


def build_model(name: str, input_size: int) -> nn.Module:
    if name == "cnn_lstm":
        return CNNLSTMRegressor(input_size)
    return RecurrentRegressor(name, input_size)


def train_one(
    name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
) -> tuple[nn.Module, np.ndarray, dict[str, object]]:
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(name, x_train.shape[2]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    x_val_t = torch.from_numpy(x_val).to(device)
    y_val_t = torch.from_numpy(y_val).to(device)
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        examples = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(xb)
            loss = loss_fn(prediction, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.item()) * len(xb)
            examples += len(xb)

        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(x_val_t), y_val_t).item())
        history.append({"epoch": epoch, "train_mse": total_loss / examples, "val_mse": val_loss})
        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= PATIENCE:
                break

    if best_state is None:
        raise RuntimeError(f"{name} did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        y_pred = model(x_val_t).cpu().numpy().astype(np.float32)
    info = {
        "device": str(device),
        "seed": seed,
        "epochs_run": len(history),
        "best_val_mse": best_loss,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "history": history,
    }
    return model, y_pred, info


def main() -> None:
    torch.set_num_threads(min(8, torch.get_num_threads()))
    OUT.mkdir(parents=True, exist_ok=True)
    model_names = ["rnn", "lstm", "gru", "bilstm", "cnn_lstm"]
    rows: list[dict[str, object]] = []
    config: dict[str, object] = {
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "max_epochs": EPOCHS,
        "early_stopping_patience": PATIENCE,
        "hidden_size": HIDDEN,
        "input": "30-cycle multivariate sensor windows; all models use identical windows and labels",
        "validation": "held-out engines and multiple censored endpoints from preprocess_cmapss.py",
        "models": {},
        "subsets": {},
    }

    for fd in range(1, 5):
        tag = f"FD{fd:03d}"
        arrays = np.load(DATA / f"windows_{tag}.npz")
        x_train = arrays["X_train"].astype(np.float32)
        y_train = arrays["y_train"].astype(np.float32)
        x_val = arrays["X_val"].astype(np.float32)
        y_val = arrays["y_val"].astype(np.float32)
        config["subsets"][tag] = {
            "train_shape": list(x_train.shape),
            "validation_shape": list(x_val.shape),
            "features": arrays["feature_names"].tolist(),
        }

        for name in model_names:
            model, prediction, info = train_one(name, x_train, y_train, x_val, y_val, SEED)
            torch.save(
                {"state_dict": model.state_dict(), "model": name, "input_size": int(x_train.shape[2]), "hidden_size": HIDDEN},
                OUT / f"{name}_{tag}.pt",
            )
            (OUT / f"{name}_{tag}_history.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
            np.savez_compressed(
                OUT / f"{name}_{tag}_validation_predictions.npz",
                y_true=y_val,
                y_pred=prediction,
                unit=arrays["val_unit"],
                end_cycle=arrays["val_end_cycle"],
            )
            rows.append({"subset": tag, "model": name, **calculate_metrics(y_val, prediction)})
            config["models"][f"{name}_{tag}"] = {k: v for k, v in info.items() if k != "history"}

            # Checkpoint progress after every completed model so interrupted
            # runs retain useful results.
            fields = ["subset", "model", "MAE", "MSE", "RMSE", "R2", "CMAPSS_score"]
            with (OUT / "sequence_baseline_validation_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            (OUT / "sequence_baseline_validation_metrics.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
            (OUT / "sequence_baseline_run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
            print(f"Finished {name} {tag}: RMSE={rows[-1]['RMSE']:.3f}", flush=True)

    print("All sequence baselines finished:", OUT, flush=True)


if __name__ == "__main__":
    main()
