"""Create report-ready comparisons from the saved official-test predictions.

Plots are generated with Pillow so the project does not need matplotlib. Model
selection is not performed here; the selected SNN recipe and rankings are read
from existing result artifacts.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
OFFICIAL = RESULTS / "official_test_evaluation"
OUT = RESULTS / "report_figures"
SETS = ("FD001", "FD002", "FD003", "FD004")
SELECTED = "selected_snn_student"
TEACHER = "attention_bilstm_teacher"
MODEL_COLORS = {
    SELECTED: "#0072B2",
    TEACHER: "#D55E00",
    "refined_snn_ensemble": "#009E73",
    "deepnn": "#CC79A7",
    "final_snn_ensemble": "#E69F00",
    "random_forest": "#777777",
}
LABELS = {
    SELECTED: "Selected SNN student ensemble",
    TEACHER: "Attention-BiLSTM teacher",
    "refined_snn_ensemble": "Refined SNN ensemble",
    "deepnn": "DeepNN baseline",
    "final_snn_ensemble": "Full-train SNN ensemble",
    "random_forest": "Random Forest baseline",
    "ridge": "Ridge baseline",
    "mean": "Mean predictor",
    "rnn": "RNN baseline",
    "lstm": "LSTM baseline",
    "gru": "GRU baseline",
    "bilstm": "BiLSTM baseline",
    "cnn_lstm": "CNN-LSTM baseline",
    "vanilla_kd_lstm": "Vanilla KD LSTM",
    "direct_surrogate_snn": "Direct surrogate SNN",
    "ann_to_snn": "ANN-to-SNN",
    "full_loss_spikedistill": "Full-loss SpikeDistill SNN",
    "final_teacher": "Full-train BiLSTM teacher",
    "refined_snn_seed42": "Refined SNN seed 42",
    "refined_snn_seed43": "Refined SNN seed 43",
    "refined_snn_seed44": "Refined SNN seed 44",
    "final_snn_seed42": "Full-train SNN seed 42",
    "final_snn_seed43": "Full-train SNN seed 43",
    "final_snn_seed44": "Full-train SNN seed 44",
}
FONT_PATH = Path(r"C:\Windows\Fonts\arial.ttf")
FONT_BOLD_PATH = Path(r"C:\Windows\Fonts\arialbd.ttf")


def font(size: int, bold: bool = False):
    path = FONT_BOLD_PATH if bold else FONT_PATH
    try:
        return ImageFont.truetype(str(path), size)
    except OSError:
        return ImageFont.load_default()


def load_rankings():
    with (OFFICIAL / "pooled_model_rankings.csv").open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_prediction(tag: str, model: str):
    z = np.load(OFFICIAL / f"{tag}_{model}_test_predictions.npz")
    return z["y_true"].astype(float), z["y_pred"].astype(float)


def draw_title(draw, title: str, subtitle: str, width: int):
    draw.text((70, 42), title, fill="#17202A", font=font(48, True))
    draw.text((72, 105), subtitle, fill="#52616B", font=font(25))
    draw.line((70, 150, width - 70, 150), fill="#CBD5E1", width=2)


def pooled_rmse_ranking(rows):
    width, top, row_h, bottom = 2400, 200, 52, 100
    height = top + row_h * len(rows) + bottom
    image = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(image)
    draw_title(d, "Overall test RMSE ranking", "Pooled across all 707 official-test windows; lower is better", width)
    label_x, chart_x, chart_w = 70, 820, 1450
    max_value = max(float(r["pooled_RMSE"]) for r in rows) * 1.06
    axis_y = top - 25
    d.line((chart_x, axis_y, chart_x + chart_w, axis_y), fill="#6B7280", width=2)
    for tick in range(0, int(max_value) + 1, 5):
        x = chart_x + chart_w * tick / max_value
        d.line((x, axis_y, x, height - bottom + 5), fill="#E8EDF2", width=1)
        d.text((x - 10, axis_y - 32), str(tick), fill="#52616B", font=font(20))
    d.text((chart_x, height - 50), "RMSE", fill="#52616B", font=font(20, True))
    for i, r in enumerate(rows):
        y = top + i * row_h
        model = r["model"]
        label = LABELS.get(model, model)
        d.text((label_x, y + 9), f"{i+1:>2}.  {label}", fill="#17202A", font=font(22, model == SELECTED))
        value = float(r["pooled_RMSE"])
        bar_w = chart_w * value / max_value
        color = MODEL_COLORS.get(model, "#AAB4BE")
        d.rounded_rectangle((chart_x, y + 6, chart_x + bar_w, y + row_h - 6), radius=8, fill=color)
        d.text((chart_x + bar_w + 12, y + 10), f"{value:.2f}", fill="#17202A", font=font(21, model == SELECTED))
    image.save(OUT / "01_overall_rmse_ranking.png", dpi=(300, 300))


def per_subset_comparison():
    models = [SELECTED, TEACHER, "refined_snn_ensemble", "deepnn", "final_snn_ensemble", "random_forest"]
    width, height = 2200, 1450
    image = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(image)
    draw_title(d, "RMSE by C-MAPSS subset", "Selected SNN and leading comparison models; lower is better", width)
    panels = [(100, 220), (1130, 220), (100, 790), (1130, 790)]
    panel_w, panel_h = 970, 500
    for tag, (px, py) in zip(SETS, panels):
        d.rounded_rectangle((px, py, px + panel_w, py + panel_h), radius=18, outline="#D6DEE6", width=2, fill="#FCFDFE")
        d.text((px + 24, py + 18), tag, fill="#17202A", font=font(30, True))
        left, base_y, chart_w, chart_h = px + 80, py + panel_h - 75, panel_w - 120, panel_h - 160
        max_val = 25
        for tick in (0, 5, 10, 15, 20, 25):
            y = base_y - chart_h * tick / max_val
            d.line((left, y, left + chart_w, y), fill="#E4E9EF", width=1)
            d.text((px + 20, y - 12), str(tick), fill="#66727D", font=font(17))
        bar_slot = chart_w / len(models)
        for j, model in enumerate(models):
            r = next(x for x in load_rankings() if x["model"] == model)
            value = float(r[f"{tag}_RMSE"])
            bw = bar_slot * .58
            x0 = left + j * bar_slot + (bar_slot - bw) / 2
            h = chart_h * value / max_val
            d.rounded_rectangle((x0, base_y - h, x0 + bw, base_y), radius=5, fill=MODEL_COLORS.get(model, "#AAB4BE"))
            d.text((x0 - 5, base_y - h - 28), f"{value:.1f}", fill="#17202A", font=font(16, model == SELECTED))
            short = {SELECTED:"SNN*",TEACHER:"Teacher","refined_snn_ensemble":"SNN-ref","deepnn":"DeepNN","final_snn_ensemble":"SNN-full","random_forest":"RF"}[model]
            d.text((x0 - 10, base_y + 10), short, fill="#374151", font=font(16, model == SELECTED))
    # Compact shared legend.
    lx, ly = 100, 1370
    for model in models:
        d.rectangle((lx, ly, lx + 24, ly + 24), fill=MODEL_COLORS.get(model, "#AAB4BE"))
        d.text((lx + 32, ly - 2), LABELS[model], fill="#374151", font=font(18))
        lx += 330 if model != "refined_snn_ensemble" else 380
    image.save(OUT / "02_rmse_by_subset.png", dpi=(300, 300))


def actual_vs_predicted():
    width, height = 2200, 1420
    image = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(image)
    draw_title(d, "Actual vs predicted RUL", "Selected SNN student and attention-BiLSTM teacher; each dot is one test engine", width)
    panels = [(100, 220), (1130, 220), (100, 800), (1130, 800)]
    panel_w, panel_h, axis_size = 970, 520, 405
    for tag, (px, py) in zip(SETS, panels):
        d.rounded_rectangle((px, py, px + panel_w, py + panel_h), radius=18, outline="#D6DEE6", width=2, fill="#FCFDFE")
        d.text((px + 25, py + 18), tag, fill="#17202A", font=font(30, True))
        x0, y0 = px + 105, py + 80 + axis_size
        d.line((x0, y0, x0 + axis_size, y0), fill="#374151", width=2)
        d.line((x0, y0, x0, y0 - axis_size), fill="#374151", width=2)
        for tick in (0, 25, 50, 75, 100, 125):
            xp = x0 + axis_size * tick / 125
            yp = y0 - axis_size * tick / 125
            d.line((xp, y0, xp, y0 - axis_size), fill="#E5EAF0", width=1)
            d.line((x0, yp, x0 + axis_size, yp), fill="#E5EAF0", width=1)
            d.text((xp - 10, y0 + 8), str(tick), fill="#66727D", font=font(14))
            d.text((x0 - 45, yp - 10), str(tick), fill="#66727D", font=font(14))
        d.line((x0, y0, x0 + axis_size, y0 - axis_size), fill="#A0A8B0", width=2)
        for model, color in ((TEACHER, MODEL_COLORS[TEACHER]), (SELECTED, MODEL_COLORS[SELECTED])):
            y, pred = load_prediction(tag, model)
            for yy, pp in zip(y, pred):
                xx = x0 + axis_size * np.clip(yy, 0, 125) / 125
                py2 = y0 - axis_size * np.clip(pp, 0, 125) / 125
                d.ellipse((xx - 4, py2 - 4, xx + 4, py2 + 4), fill=color)
        d.text((px + 340, py + panel_h - 42), "Actual capped RUL", fill="#52616B", font=font(17))
        d.text((px + 12, py + 250), "Predicted", fill="#52616B", font=font(17))
    d.rectangle((100, 1360, 124, 1384), fill=MODEL_COLORS[SELECTED])
    d.text((134, 1358), "Selected SNN", fill="#374151", font=font(20))
    d.rectangle((360, 1360, 384, 1384), fill=MODEL_COLORS[TEACHER])
    d.text((394, 1358), "Attention-BiLSTM teacher", fill="#374151", font=font(20))
    d.line((720, 1372, 770, 1372), fill="#A0A8B0", width=2)
    d.text((782, 1358), "Ideal prediction", fill="#374151", font=font(20))
    image.save(OUT / "03_actual_vs_predicted.png", dpi=(300, 300))


def residual_boxplots():
    width, height = 2200, 1250
    image = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(image)
    draw_title(d, "Test residuals by subset", "Residual = prediction − actual RUL; zero is ideal", width)
    panels = [(100, 220), (1130, 220), (100, 730), (1130, 730)]
    panel_w, panel_h = 970, 450
    for tag, (px, py) in zip(SETS, panels):
        d.rounded_rectangle((px, py, px + panel_w, py + panel_h), radius=18, outline="#D6DEE6", width=2, fill="#FCFDFE")
        d.text((px + 25, py + 18), tag, fill="#17202A", font=font(30, True))
        x0, ymid, chart_h = px + 100, py + 230, 250
        lo, hi = -50, 50
        d.line((x0, ymid - chart_h/2, x0, ymid + chart_h/2), fill="#374151", width=2)
        for tick in (-50, -25, 0, 25, 50):
            yy = ymid - chart_h * tick / (hi - lo)
            d.line((x0, yy, px + panel_w - 35, yy), fill="#E5EAF0" if tick else "#7B8794", width=2 if tick == 0 else 1)
            d.text((px + 25, yy - 10), str(tick), fill="#66727D", font=font(15))
        for i, (model, label, color) in enumerate(((TEACHER, "Teacher", MODEL_COLORS[TEACHER]), (SELECTED, "SNN", MODEL_COLORS[SELECTED]))):
            y, pred = load_prediction(tag, model)
            vals = np.clip(pred - y, lo, hi)
            q1, med, q3 = np.percentile(vals, [25, 50, 75])
            iqr = q3 - q1
            whisk_lo = max(lo, float(np.min(vals[vals >= q1 - 1.5*iqr])))
            whisk_hi = min(hi, float(np.max(vals[vals <= q3 + 1.5*iqr])))
            cx = px + 360 + i * 300
            def sy(v): return ymid - chart_h * v / (hi - lo)
            d.line((cx, sy(whisk_lo), cx, sy(whisk_hi)), fill=color, width=4)
            d.line((cx - 45, sy(whisk_lo), cx + 45, sy(whisk_lo)), fill=color, width=4)
            d.line((cx - 45, sy(whisk_hi), cx + 45, sy(whisk_hi)), fill=color, width=4)
            d.rectangle((cx - 70, sy(q3), cx + 70, sy(q1)), fill=color, outline="#263238", width=2)
            d.line((cx - 70, sy(med), cx + 70, sy(med)), fill="#17202A", width=4)
            d.text((cx - 50, py + panel_h - 45), label, fill="#374151", font=font(20, True))
    image.save(OUT / "04_residual_distributions.png", dpi=(300, 300))


def cmapss_score_ranking(rows):
    rows = sorted(rows, key=lambda r: float(r["combined_CMAPSS_score"]))
    width, top, row_h, bottom = 2400, 200, 52, 100
    height = top + row_h * len(rows) + bottom
    image = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(image)
    draw_title(d, "Combined C-MAPSS score ranking", "Sum of asymmetric scores across all four test subsets; lower is better", width)
    label_x, chart_x, chart_w = 70, 820, 1450
    values = [float(r["combined_CMAPSS_score"]) for r in rows]
    maxv = max(values) * 1.05
    axis_y = top - 25
    for tick in (0, 50000, 100000, 150000, 200000, 250000):
        x = chart_x + chart_w * tick / maxv
        d.line((x, axis_y, x, height - bottom + 5), fill="#E8EDF2", width=1)
        d.text((x - 25, axis_y - 32), f"{tick//1000}k" if tick else "0", fill="#52616B", font=font(18))
    for i, r in enumerate(rows):
        y = top + i * row_h
        model = r["model"]
        d.text((label_x, y + 9), f"{i+1:>2}.  {LABELS.get(model, model)}", fill="#17202A", font=font(22, model == SELECTED))
        value = float(r["combined_CMAPSS_score"])
        bar_w = chart_w * value / maxv
        d.rounded_rectangle((chart_x, y + 6, chart_x + bar_w, y + row_h - 6), radius=8,
                            fill=MODEL_COLORS.get(model, "#AAB4BE"))
        d.text((chart_x + bar_w + 12, y + 10), f"{value:,.0f}", fill="#17202A", font=font(20, model == SELECTED))
    image.save(OUT / "05_cmapss_score_ranking.png", dpi=(300, 300))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_rankings()
    pooled_rmse_ranking(rows)
    per_subset_comparison()
    actual_vs_predicted()
    residual_boxplots()
    cmapss_score_ranking(rows)
    print(f"Saved 5 report figures to {OUT}")


if __name__ == "__main__":
    main()

