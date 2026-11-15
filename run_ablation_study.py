"""Validation-only ablations for the full hidden-state-aware SpikeDistill loss.

Compares the complete objective with removal of output, membrane, attention,
or sparsity loss. Uses the same train/validation windows and three seeds.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from train_spike_models import DATA, DEVICE, DistilledSNN, fit_distilled_snn, regression_metrics, seed_all

ROOT = Path(__file__).resolve().parent
TEACHER = ROOT / "results" / "bilstm_teacher_tuned"
OUT = ROOT / "results" / "spike_models" / "ablations"
SEEDS = (42, 43, 44)
VARIANTS = ("full", "no_output", "no_membrane", "no_attention", "no_sparsity")
EPOCHS = 20


def main() -> None:
    torch.set_num_threads(min(8, torch.get_num_threads()))
    OUT.mkdir(parents=True, exist_ok=True)
    metrics_path = OUT / "seed_metrics.csv"
    if metrics_path.exists():
        with metrics_path.open("r", newline="", encoding="utf-8") as f:
            rows: list[dict[str, object]] = list(csv.DictReader(f))
        for row in rows:
            for key in ("seed", "hidden_size", "learning_rate", "sparsity_weight", "MAE", "MSE", "RMSE", "R2", "CMAPSS_score", "spike_rate", "epochs_run"):
                if key in row and row[key] != "":
                    row[key] = int(float(row[key])) if key in {"seed", "hidden_size", "epochs_run"} else float(row[key])
    else:
        rows = []
    summaries: dict[str, object] = {}
    for fd in range(1, 5):
        tag = f"FD{fd:03d}"
        z = np.load(DATA / f"windows_{tag}.npz")
        tr = np.load(TEACHER / f"teacher_signals_{tag}_train.npz")
        va = np.load(TEACHER / f"teacher_signals_{tag}_val.npz")
        xtr, ytr = z["X_train"].astype(np.float32), z["y_train"].astype(np.float32)
        xv, yv = z["X_val"].astype(np.float32), z["y_val"].astype(np.float32)
        teacher_train_y = tr["y_teacher"].astype(np.float32)
        teacher_val_y = va["y_teacher"].astype(np.float32)
        teacher_hidden = tr["hidden_states"].astype(np.float16)
        teacher_val_hidden = va["hidden_states"].astype(np.float16)
        teacher_attention = tr["attention"].astype(np.float32)
        teacher_val_attention = va["attention"].astype(np.float32)
        if tag == "FD001":
            width, lr, leak, sparse = 96, 1.5e-3, .8, .01
        elif tag == "FD002":
            width, lr, leak, sparse = 192, 1.5e-3, .6, .001
        elif tag == "FD003":
            width, lr, leak, sparse = 192, 3e-4, .8, .01
        else:
            width, lr, leak, sparse = 192, 1.5e-3, .8, .01
        subset_dir = OUT / tag
        subset_dir.mkdir(parents=True, exist_ok=True)
        for variant in VARIANTS:
            ablation = None if variant == "full" else variant.removeprefix("no_")
            for seed in SEEDS:
                existing = next((r for r in rows if r["subset"] == tag and r["variant"] == variant and int(r["seed"]) == seed), None)
                if existing is not None:
                    print(f"{tag} {variant} seed={seed}: already complete, reusing saved metrics", flush=True)
                    continue
                seed_all(seed)
                model, pred, info = fit_distilled_snn(
                    DistilledSNN(xtr.shape[2], leak_init=leak, hidden_size=width),
                    xtr, ytr, teacher_train_y, teacher_hidden, teacher_attention,
                    xv, yv, teacher_val_y, teacher_val_hidden, teacher_val_attention,
                    learning_rate=lr, sparsity_weight=sparse, seed=seed,
                    loss_ablation=ablation, epochs_override=EPOCHS,
                )
                metrics = regression_metrics(yv, pred)
                record = {"subset": tag, "variant": variant, "seed": seed,
                          "hidden_size": width, "learning_rate": lr,
                          "sparsity_weight": sparse, **metrics,
                          "spike_rate": info["validation_spike_rate"],
                          "epochs_run": info["epochs_run"],
                          "loss_weights": info["loss_weights"]}
                rows.append(record)
                torch.save({"state_dict": model.state_dict(), "input_size": xtr.shape[2],
                            "hidden_size": width, "seed": seed, "variant": variant,
                            "loss_weights": info["loss_weights"]},
                           subset_dir / f"{variant}_seed{seed}.pt")
                np.savez_compressed(subset_dir / f"{variant}_seed{seed}_validation.npz",
                                    y_true=yv, y_pred=pred)
                (subset_dir / f"{variant}_seed{seed}_history.json").write_text(
                    json.dumps(info["history"], indent=2), encoding="utf-8")
                with metrics_path.open("w", newline="", encoding="utf-8") as f:
                    fields = ["subset", "variant", "seed", "hidden_size", "learning_rate",
                              "sparsity_weight", "MAE", "MSE", "RMSE", "R2",
                              "CMAPSS_score", "spike_rate", "epochs_run"]
                    writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
                    writer.writeheader(); writer.writerows(rows)
                print(f"{tag} {variant} seed={seed}: RMSE={metrics['RMSE']:.2f}, "
                      f"score={metrics['CMAPSS_score']:.1f}, spike={info['validation_spike_rate']:.3f}",
                      flush=True)
        subrows = [r for r in rows if r["subset"] == tag]
        summaries[tag] = {}
        for variant in VARIANTS:
            vr = [r for r in subrows if r["variant"] == variant]
            summaries[tag][variant] = {
                metric: {"mean": float(np.mean([r[metric] for r in vr])),
                         "std": float(np.std([r[metric] for r in vr], ddof=1))}
                for metric in ("RMSE", "MAE", "CMAPSS_score", "spike_rate")
            }
        (subset_dir / "summary.json").write_text(json.dumps(summaries[tag], indent=2), encoding="utf-8")
        (OUT / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
        del z, tr, va, xtr, ytr, xv, yv, teacher_hidden, teacher_val_hidden
    print("Ablation study complete:", OUT, flush=True)


if __name__ == "__main__":
    main()
