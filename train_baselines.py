"""Train initial SpikeDistill-RUL baselines on leakage-safe C-MAPSS windows.

Models: mean predictor, Ridge regression, Random Forest, and DeepNN.
Run after preprocess_cmapss.py: python train_baselines.py
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "CMAPSSData" / "processed"
OUT = ROOT / "results" / "baselines"
SEED = 42
BATCH_SIZE = 256
EPOCHS = 20
PATIENCE = 4


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
    error = y_pred - y_true
    mse = float(np.mean(error ** 2))
    variance = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "MAE": float(np.mean(np.abs(error))),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "R2": float(1 - np.sum(error ** 2) / variance) if variance > 0 else float("nan"),
        "CMAPSS_score": cmapss_score(y_true, y_pred),
    }


class DeepNN(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.10),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.10),
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x).squeeze(1)


def train_deepnn(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray) -> tuple[DeepNN, np.ndarray, dict[str, object]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepNN(x_train.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=BATCH_SIZE, shuffle=True, generator=torch.Generator().manual_seed(SEED),
    )
    x_val_t = torch.from_numpy(x_val).to(device)
    y_val_t = torch.from_numpy(y_val).to(device)
    best_state = None
    best_loss = float("inf")
    no_improvement = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0.0
        count = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * len(xb)
            count += len(xb)
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(x_val_t), y_val_t).item())
        history.append({"epoch": epoch, "train_mse": total / count, "val_mse": val_loss})
        if val_loss < best_loss - 1e-6:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement >= PATIENCE:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        prediction = model(x_val_t).cpu().numpy()
    return model, prediction, {"device": str(device), "epochs_run": len(history), "best_val_mse": best_loss, "history": history}


def save_predictions(tag: str, model_name: str, y: np.ndarray, pred: np.ndarray) -> None:
    np.savez_compressed(OUT / f"{model_name}_{tag}_validation_predictions.npz", y_true=y, y_pred=pred)


def main() -> None:
    seed_everything()
    torch.set_num_threads(min(8, torch.get_num_threads()))
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    run_info: dict[str, object] = {
        "seed": SEED,
        "data": "validation artifacts from leakage-aware preprocess_cmapss.py",
        "models": {
            "mean": "training-target mean",
            "ridge": "Ridge(alpha=10) on flattened 30-cycle windows",
            "random_forest": "75 trees, max_depth=20, min_samples_leaf=2, max_features=0.7 on flattened windows",
            "deepnn": "flattened window MLP 128-64-32 with ReLU/dropout, AdamW, early stopping",
        },
        "subsets": {},
    }

    for index in range(1, 5):
        tag = f"FD{index:03d}"
        data = np.load(DATA / f"windows_{tag}.npz")
        x_train = data["X_train"].reshape(len(data["X_train"]), -1).astype(np.float32)
        y_train = data["y_train"].astype(np.float32)
        x_val = data["X_val"].reshape(len(data["X_val"]), -1).astype(np.float32)
        y_val = data["y_val"].astype(np.float32)
        subset_info: dict[str, object] = {"train_shape": list(x_train.shape), "validation_shape": list(x_val.shape)}

        baseline_mean = np.full_like(y_val, y_train.mean())
        model_results: dict[str, np.ndarray] = {"mean": baseline_mean}

        ridge = Ridge(alpha=10.0)
        ridge.fit(x_train, y_train)
        model_results["ridge"] = ridge.predict(x_val).astype(np.float32)

        forest = RandomForestRegressor(
            n_estimators=75, max_depth=20, min_samples_leaf=2, max_features=0.7,
            random_state=SEED, n_jobs=-1,
        )
        forest.fit(x_train, y_train)
        model_results["random_forest"] = forest.predict(x_val).astype(np.float32)

        model, model_results["deepnn"], deepnn_info = train_deepnn(x_train, y_train, x_val, y_val)
        torch.save({"state_dict": model.state_dict(), "input_dim": x_train.shape[1]}, OUT / f"deepnn_{tag}.pt")
        (OUT / f"deepnn_{tag}_history.json").write_text(json.dumps(deepnn_info, indent=2), encoding="utf-8")
        subset_info["deepnn"] = {k: v for k, v in deepnn_info.items() if k != "history"}

        for model_name, prediction in model_results.items():
            result = {"subset": tag, "model": model_name, **metrics(y_val, prediction)}
            rows.append(result)
            save_predictions(tag, model_name, y_val, prediction)
        run_info["subsets"][tag] = subset_info
        fields = ["subset", "model", "MAE", "MSE", "RMSE", "R2", "CMAPSS_score"]
        with (OUT / "baseline_validation_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        (OUT / "baseline_validation_metrics.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        (OUT / "baseline_run_config.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")
        print(f"Finished {tag}", flush=True)

    print("Saved validation results to", OUT)


if __name__ == "__main__":
    main()
