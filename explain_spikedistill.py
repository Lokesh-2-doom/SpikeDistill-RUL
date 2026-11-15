"""Validation-only explanations for selected refined SNNs.

Produces Integrated Gradients, grouped temporal permutation-Shapley values,
attention maps, spike-time ablation attributions, and sensor permutation
importance. No official test examples or labels are loaded.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from train_bilstm_teacher import AttentionBiLSTM
from train_spike_models import DATA, DEVICE, DistilledSNN, regression_metrics

ROOT = Path(__file__).resolve().parent
TEACHER = ROOT / "results" / "bilstm_teacher_tuned"
OUT = ROOT / "results" / "spike_models" / "xai"


def student_artifact(tag: str) -> tuple[Path, int]:
    if tag == "FD001":
        return ROOT / "results/spike_models/rmse_refinement/FD001/label_mse_h96_seed42.pt", 96
    if tag == "FD002":
        return ROOT / "results/spike_models/rmse_refinement/FD002/output_mse_h192_seed42.pt", 192
    if tag == "FD003":
        return ROOT / "results/spike_models/fd003_refinement/h192_lr0.0003_kd0.1_seed42.pt", 192
    return ROOT / "results/spike_models/rmse_refinement/FD004/output_mse_h192_seed42.pt", 192


def predict(model: torch.nn.Module, arr: np.ndarray, batch_size: int = 64) -> np.ndarray:
    result = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(arr), batch_size):
            out = model(torch.from_numpy(arr[start:start+batch_size]).to(DEVICE))
            if isinstance(out, tuple): out = out[0]
            result.append(out.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(result)


def integrated_gradients(model, x: np.ndarray, baseline: np.ndarray, steps: int = 32) -> np.ndarray:
    model.eval()
    base = torch.from_numpy(baseline).to(DEVICE)
    sample = torch.from_numpy(x).to(DEVICE)
    total_grad = torch.zeros_like(sample)
    for alpha in torch.linspace(0.0, 1.0, steps, device=DEVICE):
        interp = (base + alpha * (sample - base)).detach().requires_grad_(True)
        pred = model(interp)[0]
        grad = torch.autograd.grad(pred.sum(), interp, retain_graph=False)[0]
        total_grad += grad
    return ((sample - base) * total_grad / steps).detach().cpu().numpy()[0]


def grouped_temporal_shap(model, sample: np.ndarray, baseline: np.ndarray,
                          permutations: int, seed: int) -> np.ndarray:
    """Monte-Carlo permutation Shapley over (time-block, sensor) players."""
    steps, features = sample.shape
    n_blocks = 6
    boundaries = np.linspace(0, steps, n_blocks + 1, dtype=int)
    players = [(b, f) for b in range(n_blocks) for f in range(features)]
    rng = np.random.default_rng(seed)
    shap = np.zeros((n_blocks, features), dtype=np.float64)
    for _ in range(permutations):
        state = baseline.copy()
        path = []
        order = rng.permutation(len(players))
        for player_idx in order:
            block, feature = players[int(player_idx)]
            lo, hi = boundaries[block], boundaries[block + 1]
            state[lo:hi, feature] = sample[lo:hi, feature]
            path.append(state.copy())
        path_pred = predict(model, np.stack(path).astype(np.float32), batch_size=64)
        increments = np.diff(path_pred, prepend=predict(model, baseline[None].astype(np.float32))[0])
        for order_idx, player_idx in enumerate(order):
            block, feature = players[int(player_idx)]
            shap[block, feature] += float(increments[order_idx])
    return (shap / permutations).astype(np.float32)


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows: return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)


def heatmap_image(values: np.ndarray, size: tuple[int, int], symmetric: bool = True) -> Image.Image:
    width, height = size
    a = np.asarray(values, dtype=np.float32)
    if symmetric:
        scale = max(float(np.nanmax(np.abs(a))), 1e-8)
        a = np.clip(a / scale, -1, 1)
        red = np.maximum(a, 0)
        blue = np.maximum(-a, 0)
        rgb = np.stack([255 * red, 255 * (1 - np.abs(a)), 255 * blue], axis=-1)
    else:
        scale = max(float(np.nanmax(a)), 1e-8)
        a = np.clip(a / scale, 0, 1)
        rgb = np.stack([255 * a, 255 * a, 255 * a], axis=-1)
    small = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
    return small.resize((width, height), Image.Resampling.NEAREST)


def make_summary_figure(path: Path, tag: str, ig: np.ndarray, attn_rows: list[dict[str, object]],
                        spike_map: np.ndarray, y: float, pred: float) -> None:
    panel_w, panel_h = 420, 330
    canvas = Image.new("RGB", (panel_w * 3, panel_h), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 8), f"{tag} XAI | true RUL={y:.1f}, predicted={pred:.1f}", fill="black")
    ig_img = heatmap_image(ig.T, (panel_w - 40, panel_h - 65), symmetric=True)
    canvas.paste(ig_img, (25, 45))
    draw.text((20, 28), "Integrated Gradients (sensor x cycle)", fill="black")
    x0, y0 = panel_w + 35, 45
    plot_w, plot_h = panel_w - 60, panel_h - 75
    draw.text((panel_w + 20, 28), "Teacher / student attention", fill="black")
    maxv = max(float(max(max(r["teacher_attention"], r["student_attention"]) for r in attn_rows)), 1e-8)
    def pt(i, val):
        return (x0 + int(i * plot_w / max(1, len(attn_rows)-1)),
                y0 + plot_h - int(float(val) / maxv * plot_h))
    for key, color in (("teacher_attention", (35, 95, 190)), ("student_attention", (210, 65, 45))):
        points = [pt(i, row[key]) for i, row in enumerate(attn_rows)]
        draw.line(points, fill=color, width=3)
    draw.text((x0, y0 + plot_h + 8), "Blue: teacher; red: SNN", fill="black")
    spike_img = heatmap_image(spike_map.T, (panel_w - 40, panel_h - 65), symmetric=False)
    canvas.paste(spike_img, (panel_w * 2 + 25, 45))
    draw.text((panel_w * 2 + 20, 28), "SNN spike raster (neuron x cycle)", fill="black")
    canvas.save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--permutations", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(min(8, torch.get_num_threads()))
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {"data_used": "validation split only", "official_test_loaded": False,
                "integrated_gradients_steps": 32,
                "temporal_shap": {"method": "Monte-Carlo permutation Shapley", "time_blocks": 6,
                                  "players": "each sensor within each temporal block",
                                  "permutations": args.permutations}, "subsets": {}}
    for fd in range(1, 5):
        tag = f"FD{fd:03d}"
        z = np.load(DATA / f"windows_{tag}.npz")
        xtr = z["X_train"].astype(np.float32)
        xv = z["X_val"].astype(np.float32)
        yv = z["y_val"].astype(np.float32)
        names = [str(n) for n in z["feature_names"].tolist()]
        student_path, width = student_artifact(tag)
        student = DistilledSNN(xv.shape[2], hidden_size=width).to(DEVICE)
        student_payload = torch.load(student_path, map_location=DEVICE, weights_only=False)
        student.load_state_dict(student_payload["state_dict"])
        student.eval()
        teacher = AttentionBiLSTM(xv.shape[2]).to(DEVICE)
        teacher_payload = torch.load(TEACHER / f"attention_bilstm_teacher_{tag}.pt",
                                     map_location=DEVICE, weights_only=False)
        teacher.load_state_dict(teacher_payload["state_dict"])
        teacher.eval()
        subset_dir = OUT / tag
        subset_dir.mkdir(parents=True, exist_ok=True)
        baseline = np.mean(xtr, axis=0, dtype=np.float64).astype(np.float32)
        sample_ids = np.unique(np.linspace(0, len(xv)-1, min(args.samples, len(xv)), dtype=int))
        all_ig = []
        shap_rows = []
        for sample_id in sample_ids:
            sample = xv[sample_id]
            ig = integrated_gradients(student, sample[None], baseline[None])
            all_ig.append(ig)
            shap = grouped_temporal_shap(student, sample, baseline, args.permutations,
                                         seed=9000 + fd * 100 + int(sample_id))
            bounds = np.linspace(0, sample.shape[0], shap.shape[0] + 1, dtype=int)
            for block in range(shap.shape[0]):
                for feature, name in enumerate(names):
                    shap_rows.append({"sample_index": int(sample_id), "true_rul": float(yv[sample_id]),
                                      "time_start": int(bounds[block]), "time_end_exclusive": int(bounds[block+1]),
                                      "feature": name, "shap_value": float(shap[block, feature])})
        np.savez_compressed(subset_dir / "integrated_gradients_validation.npz",
                            sample_indices=sample_ids, attributions=np.stack(all_ig),
                            feature_names=np.asarray(names))
        write_rows(subset_dir / "temporal_shap_validation.csv", shap_rows)

        # Attention and spike-raster attribution for a representative endpoint.
        sample_id = int(sample_ids[-1])
        x_one = torch.from_numpy(xv[sample_id:sample_id+1]).to(DEVICE)
        student.eval(); teacher.eval()
        with torch.no_grad():
            pred, _, student_attention, spikes, _ = student(x_one)
            teacher_pred, _, teacher_attention = teacher(x_one)
        attn_rows = [{"cycle_in_window": t, "student_attention": float(student_attention[0,t].cpu()),
                      "teacher_attention": float(teacher_attention[0,t].cpu())}
                     for t in range(xv.shape[1])]
        write_rows(subset_dir / "attention_maps_validation.csv", attn_rows)
        spike_map = spikes[0].detach().cpu().numpy().astype(np.uint8)
        time_effect = []
        full_prediction = float(pred.item())
        for t in range(xv.shape[1]):
            with torch.no_grad():
                ablated = float(student(x_one, ablate_spikes_at=t)[0].item())
            time_effect.append(full_prediction - ablated)
        event_attr = np.zeros_like(spike_map, dtype=np.float32)
        for t, effect in enumerate(time_effect):
            active = spike_map[t].astype(bool)
            count = int(active.sum())
            if count: event_attr[t, active] = effect / count
        np.savez_compressed(subset_dir / "spike_raster_attribution_validation.npz",
                            spikes=spike_map, event_attribution=event_attr,
                            timestep_ablation_effect=np.asarray(time_effect, dtype=np.float32),
                            prediction=np.asarray(full_prediction), true_rul=np.asarray(yv[sample_id]))

        # Sensor permutation importance keeps each sensor's trajectory intact,
        # shuffling it across held-out validation windows.
        base_pred = predict(student, xv)
        base_rmse = regression_metrics(yv, base_pred)["RMSE"]
        rng = np.random.default_rng(4200 + fd)
        importance_rows = []
        for feature, name in enumerate(names):
            perturbed = xv.copy()
            perturbed[:, :, feature] = perturbed[rng.permutation(len(perturbed)), :, feature]
            p = predict(student, perturbed)
            importance_rows.append({"feature": name, "baseline_rmse": base_rmse,
                                    "permuted_rmse": regression_metrics(yv, p)["RMSE"],
                                    "rmse_increase": regression_metrics(yv, p)["RMSE"] - base_rmse})
        importance_rows.sort(key=lambda r: float(r["rmse_increase"]), reverse=True)
        write_rows(subset_dir / "sensor_permutation_importance.csv", importance_rows)

        # Compact visualization bundle for reports/notebooks.
        image = np.mean(np.stack(all_ig), axis=0)
        make_summary_figure(subset_dir / "xai_summary.png", tag, image, attn_rows,
                            spike_map, float(yv[sample_id]), full_prediction)
        manifest["subsets"][tag] = {"checkpoint": str(student_path), "validation_samples": sample_ids.tolist(),
                                    "student_prediction": full_prediction,
                                    "teacher_prediction": float(teacher_pred.item()),
                                    "spike_rate": float(spikes.float().mean().cpu()),
                                    "permutation_importance_baseline_rmse": base_rmse}
        (subset_dir / "xai_manifest.json").write_text(json.dumps(manifest["subsets"][tag], indent=2), encoding="utf-8")
        print(f"{tag} XAI saved; sample SNN prediction={full_prediction:.1f}, IG/SHAP/attention/spikes/permutation done", flush=True)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Validation XAI complete:", OUT, flush=True)


if __name__ == "__main__":
    main()
