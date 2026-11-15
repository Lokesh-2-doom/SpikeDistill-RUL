"""Leakage-aware preprocessing for the C-MAPSS SpikeDistill-RUL project.

Run from the repository root: python preprocess_cmapss.py

Requires pandas and numpy. It preserves source files and writes
processed row tables, train/validation/test windows, fitted preprocessing
parameters, and a reproducibility summary to CMAPSSData/processed/.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent / "CMAPSSData"
OUT = ROOT / "processed"
WINDOW_LENGTH = 30
RUL_CAP = 125
VALIDATION_FRACTION = 0.20
VALIDATION_ENDPOINTS_PER_ENGINE = 5
RANDOM_SEED = 42
COLS = ["unit", "cycle", "setting_1", "setting_2", "setting_3"] + [
    f"sensor_{i}" for i in range(1, 22)
]
SETTING_COLS = ["setting_1", "setting_2", "setting_3"]
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]
MULTI_CONDITION = {"FD002", "FD004"}


def read_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None, names=COLS, engine="python")
    if df.isna().any().any():
        raise ValueError(f"Missing values found in {path}")
    return df


def add_train_rul(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    remaining = out.groupby("unit")["cycle"].transform("max") - out["cycle"]
    out["RUL_raw"] = remaining.astype(np.int32)
    out["RUL"] = np.minimum(remaining, RUL_CAP).astype(np.int32)
    return out


def add_test_rul(df: pd.DataFrame, official_rul: pd.Series) -> pd.DataFrame:
    out = df.copy()
    units = sorted(out["unit"].unique())
    if len(units) != len(official_rul):
        raise ValueError(f"{len(units)} test units but {len(official_rul)} official RUL values")
    rul_by_unit = pd.Series(official_rul.to_numpy(dtype=np.int32), index=units)
    last_cycle = out.groupby("unit")["cycle"].transform("max")
    remaining = out["unit"].map(rul_by_unit) + last_cycle - out["cycle"]
    out["RUL_raw"] = remaining.astype(np.int32)
    out["RUL"] = np.minimum(remaining, RUL_CAP).astype(np.int32)
    return out


def split_engine_ids(df: pd.DataFrame) -> tuple[list[int], list[int]]:
    """Deterministic random split by engine; never split rows/windows of an engine."""
    units = np.asarray(sorted(df["unit"].unique()), dtype=np.int32)
    rng = np.random.default_rng(RANDOM_SEED)
    shuffled = rng.permutation(units)
    n_val = max(1, int(round(len(units) * VALIDATION_FRACTION)))
    val_ids = sorted(shuffled[:n_val].tolist())
    train_ids = sorted(shuffled[n_val:].tolist())
    return train_ids, val_ids


def run_kmeans(x: np.ndarray, n_clusters: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Small deterministic Lloyd K-means implementation for operating settings."""
    rng = np.random.default_rng(seed)
    centers = [x[int(rng.integers(0, len(x)))]]
    # K-means++ initialization improves stability when the six regimes are uneven.
    for _ in range(1, n_clusters):
        dist2 = np.min(((x[:, None, :] - np.asarray(centers)[None, :, :]) ** 2).sum(axis=2), axis=1)
        total = float(dist2.sum())
        if total <= 0:
            candidate = x[int(rng.integers(0, len(x)))]
        else:
            candidate = x[int(rng.choice(len(x), p=dist2 / total))]
        centers.append(candidate)
    centers_arr = np.asarray(centers, dtype=float)
    for _ in range(100):
        dist2 = ((x[:, None, :] - centers_arr[None, :, :]) ** 2).sum(axis=2)
        labels = dist2.argmin(axis=1)
        updated = centers_arr.copy()
        for k in range(n_clusters):
            members = x[labels == k]
            if len(members):
                updated[k] = members.mean(axis=0)
            else:
                updated[k] = x[int(np.argmax(dist2.min(axis=1)))]
        if np.allclose(updated, centers_arr, rtol=0, atol=1e-8):
            centers_arr = updated
            break
        centers_arr = updated
    labels = ((x[:, None, :] - centers_arr[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)
    return labels, centers_arr


def fit_regimes(train: pd.DataFrame, tag: str) -> tuple[pd.DataFrame, dict[str, object]]:
    """Fit operating-condition assignment on training-fold rows only."""
    out = train.copy()
    if tag not in MULTI_CONDITION:
        out["regime"] = 0
        return out, {"method": "single regime", "n_regimes": 1}

    setting_mean = train[SETTING_COLS].mean()
    setting_std = train[SETTING_COLS].std(ddof=0).replace(0, 1)
    scaled_settings = (train[SETTING_COLS] - setting_mean) / setting_std
    labels, centers = run_kmeans(scaled_settings.to_numpy(), n_clusters=6, seed=RANDOM_SEED)
    out["regime"] = labels
    info = {
        "method": "6-cluster KMeans fitted on training-fold operating settings",
        "n_regimes": 6,
        "setting_mean": {k: float(v) for k, v in setting_mean.items()},
        "setting_std": {k: float(v) for k, v in setting_std.items()},
        "scaled_cluster_centers": centers.tolist(),
    }
    return out, info


def assign_regimes(df: pd.DataFrame, tag: str, regime_info: dict[str, object]) -> pd.DataFrame:
    out = df.copy()
    if tag not in MULTI_CONDITION:
        out["regime"] = 0
        return out
    mean = pd.Series(regime_info["setting_mean"])
    std = pd.Series(regime_info["setting_std"])
    scaled = ((out[SETTING_COLS] - mean[SETTING_COLS]) / std[SETTING_COLS]).to_numpy()
    centers = np.asarray(regime_info["scaled_cluster_centers"], dtype=float)
    distance = ((scaled[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    out["regime"] = distance.argmin(axis=1).astype(np.int32)
    return out


def make_validation_censored(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, list[int]]]:
    """Create several chronological pre-failure endpoints per held-out engine."""
    pieces: list[pd.DataFrame] = []
    cutoffs: dict[int, list[int]] = {}
    for unit, group in df.sort_values(["unit", "cycle"]).groupby("unit", sort=True):
        final_cycle = int(group["cycle"].max())
        max_remaining = min(RUL_CAP, final_cycle - 1)
        if max_remaining < 1:
            raise ValueError(f"Validation engine {unit} is too short to censor")
        min_remaining = min(10, max_remaining)
        remaining_values = np.unique(
            np.rint(np.linspace(min_remaining, max_remaining, VALIDATION_ENDPOINTS_PER_ENGINE)).astype(int)
        )
        unit_cutoffs: list[int] = []
        for remaining in remaining_values:
            cutoff = final_cycle - int(remaining)
            trace = group[group["cycle"] <= cutoff].copy()
            trace["validation_case"] = f"{int(unit)}_rul{int(remaining)}"
            pieces.append(trace)
            unit_cutoffs.append(int(cutoff))
        cutoffs[int(unit)] = unit_cutoffs
    return pd.concat(pieces, ignore_index=True), cutoffs


def fit_feature_scaler(train: pd.DataFrame) -> tuple[list[str], dict[str, object]]:
    """Fit sensor constant removal, regime centering and min-max stats on train fold."""
    dropped = [c for c in SENSOR_COLS if train[c].nunique(dropna=False) <= 1]
    features = [c for c in SENSOR_COLS if c not in dropped]
    regime_means = train.groupby("regime")[features].mean()
    global_mean = train[features].mean()
    centered = pd.DataFrame(index=train.index, columns=features, dtype=float)
    for regime, indices in train.groupby("regime").groups.items():
        centered.loc[indices, features] = train.loc[indices, features] - regime_means.loc[regime]
    low = centered.min()
    high = centered.max()
    span = (high - low).replace(0, 1)
    params = {
        "features": features,
        "dropped_constant_sensors": dropped,
        "regime_means": {str(int(k)): {f: float(v) for f, v in row.items()} for k, row in regime_means.to_dict(orient="index").items()},
        "global_mean_fallback": {f: float(v) for f, v in global_mean.items()},
        "centered_min": {f: float(v) for f, v in low.items()},
        "centered_max": {f: float(v) for f, v in high.items()},
        "method": "per-regime training-fold mean centering, then training-fold min-max scaling; no test fitting or clipping",
    }
    return features, params


def transform_features(df: pd.DataFrame, params: dict[str, object]) -> pd.DataFrame:
    out = df.copy()
    features = params["features"]
    means = params["regime_means"]
    global_mean = params["global_mean_fallback"]
    low = pd.Series(params["centered_min"])[features]
    high = pd.Series(params["centered_max"])[features]
    span = (high - low).replace(0, 1)
    centered = pd.DataFrame(index=out.index, columns=features, dtype=float)
    for regime, indices in out.groupby("regime").groups.items():
        mean = pd.Series(means.get(str(int(regime)), global_mean))[features]
        centered.loc[indices, features] = out.loc[indices, features] - mean
    out[features] = (centered[features] - low) / span
    return out


def make_windows(
    df: pd.DataFrame, features: list[str], all_train_windows: bool, group_column: str = "unit"
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[float] = []
    units: list[int] = []
    end_cycles: list[int] = []
    for _, group in df.sort_values(["unit", "cycle"]).groupby(group_column, sort=True):
        unit = int(group["unit"].iloc[0])
        group = group.sort_values("cycle")
        x = group[features].to_numpy(dtype=np.float32)
        y = group["RUL"].to_numpy(dtype=np.float32)
        cycles = group["cycle"].to_numpy(dtype=np.int32)
        if all_train_windows:
            for end in range(WINDOW_LENGTH - 1, len(group)):
                start = end - WINDOW_LENGTH + 1
                xs.append(x[start : end + 1])
                ys.append(float(y[end]))
                units.append(int(unit))
                end_cycles.append(int(cycles[end]))
        else:
            if len(group) < WINDOW_LENGTH:
                pad = np.repeat(x[[0]], WINDOW_LENGTH - len(group), axis=0)
                window = np.concatenate([pad, x], axis=0)
            else:
                window = x[-WINDOW_LENGTH:]
            xs.append(window)
            ys.append(float(y[-1]))
            units.append(int(unit))
            end_cycles.append(int(cycles[-1]))
    if not xs:
        raise ValueError("No windows were created; check window length and input trajectories")
    return (
        np.stack(xs).astype(np.float32),
        np.asarray(ys, dtype=np.float32),
        np.asarray(units, dtype=np.int32),
        np.asarray(end_cycles, dtype=np.int32),
    )


def dataset_profile(df: pd.DataFrame) -> dict[str, object]:
    """Compact EDA values for the report without adding derived model features."""
    counts = df.groupby("unit").size()
    return {
        "rows": int(len(df)),
        "engines": int(df["unit"].nunique()),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_unit_cycle_rows": int(df.duplicated(["unit", "cycle"]).sum()),
        "cycles_per_engine": {"min": int(counts.min()), "median": float(counts.median()), "mean": float(counts.mean()), "max": int(counts.max())},
        "sensor_summary": {
            sensor: {"min": float(df[sensor].min()), "mean": float(df[sensor].mean()), "max": float(df[sensor].max()), "std": float(df[sensor].std(ddof=0))}
            for sensor in SENSOR_COLS
        },
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {
        "description": "Leakage-aware C-MAPSS preparation for SpikeDistill-RUL; raw data are unchanged.",
        "window_length_cycles": WINDOW_LENGTH,
        "rul_target": {"formula": "min(raw_RUL, cap)", "cap_cycles": RUL_CAP, "shape": "piecewise-linear; flat above cap, decreases linearly to zero"},
        "validation": {"method": "hold out whole training engines, then create several deterministic pre-failure endpoints per engine", "fraction": VALIDATION_FRACTION, "endpoints_per_engine": VALIDATION_ENDPOINTS_PER_ENGINE, "random_seed": RANDOM_SEED, "same_split_across_models": True},
        "test_policy": "Official test engines are untouched during fitting/model selection. Their provided RUL labels are included only as evaluation targets.",
        "normalization": "Drop sensors constant in training fold; for FD002/FD004 assign 6 operating regimes using settings and training-fold-fitted KMeans; subtract training-fold regime sensor means; apply training-fold min-max transform. No test fitting and no clipping.",
        "windowing": "All rolling 30-cycle windows from train engines; one latest 30-cycle window for each censored validation endpoint and official test engine. Each window stays within an engine and in chronological cycle order. Short final windows are left-padded by repeating the first observation.",
        "files": {},
    }

    for fd in range(1, 5):
        tag = f"FD{fd:03d}"
        raw_train = add_train_rul(read_table(ROOT / f"train_{tag}.txt"))
        raw_test = add_test_rul(
            read_table(ROOT / f"test_{tag}.txt"),
            pd.read_csv(ROOT / f"RUL_{tag}.txt", header=None).iloc[:, 0],
        )

        train_ids, val_ids = split_engine_ids(raw_train)
        train_fold = raw_train[raw_train["unit"].isin(train_ids)].copy()
        val_fold = raw_train[raw_train["unit"].isin(val_ids)].copy()

        train_fold, regime_info = fit_regimes(train_fold, tag)
        val_fold = assign_regimes(val_fold, tag, regime_info)
        test_fold = assign_regimes(raw_test, tag, regime_info)
        val_fold, val_cutoffs = make_validation_censored(val_fold)
        features, scaler = fit_feature_scaler(train_fold)

        train_fold = transform_features(train_fold, scaler)
        val_fold = transform_features(val_fold, scaler)
        test_fold = transform_features(test_fold, scaler)

        X_train, y_train, train_units, train_ends = make_windows(train_fold, features, True)
        X_val, y_val, val_units, val_ends = make_windows(val_fold, features, False, "validation_case")
        X_test, y_test, test_units, test_ends = make_windows(test_fold, features, False)

        np.savez_compressed(
            OUT / f"windows_{tag}.npz",
            X_train=X_train, y_train=y_train, train_unit=train_units, train_end_cycle=train_ends,
            X_val=X_val, y_val=y_val, val_unit=val_units, val_end_cycle=val_ends,
            X_test=X_test, y_test=y_test, test_unit=test_units, test_end_cycle=test_ends,
            feature_names=np.asarray(features),
        )

        for split_name, frame in (("train", train_fold), ("val", val_fold), ("test", test_fold)):
            keep = (["validation_case"] if split_name == "val" else []) + ["unit", "cycle", "RUL_raw", "RUL"] + features
            frame[keep].sort_values(["unit", "cycle"]).to_csv(
                OUT / f"{split_name}_{tag}_processed.csv", index=False, float_format="%.8g"
            )

        scaler["operating_regimes"] = regime_info
        scaler["fit_train_engine_ids"] = train_ids
        scaler["validation_engine_ids"] = val_ids
        scaler["validation_cutoff_cycles"] = {str(k): v for k, v in val_cutoffs.items()}
        (OUT / f"scaler_{tag}.json").write_text(json.dumps(scaler, indent=2), encoding="utf-8")

        # Final-training artifact: after hyperparameters are selected with the
        # validation artifact, refit preprocessing on every official train
        # engine and train the selected model on all its rolling windows.
        final_train, final_regime_info = fit_regimes(raw_train, tag)
        final_test = assign_regimes(raw_test, tag, final_regime_info)
        final_features, final_scaler = fit_feature_scaler(final_train)
        final_train = transform_features(final_train, final_scaler)
        final_test = transform_features(final_test, final_scaler)
        X_final_train, y_final_train, final_units, final_ends = make_windows(final_train, final_features, True)
        X_final_test, y_final_test, final_test_units, final_test_ends = make_windows(final_test, final_features, False)
        np.savez_compressed(
            OUT / f"final_windows_{tag}.npz",
            X_train=X_final_train, y_train=y_final_train, train_unit=final_units, train_end_cycle=final_ends,
            X_test=X_final_test, y_test=y_final_test, test_unit=final_test_units, test_end_cycle=final_test_ends,
            feature_names=np.asarray(final_features),
        )
        final_scaler["operating_regimes"] = final_regime_info
        final_scaler["fit_train_engine_ids"] = sorted(raw_train["unit"].unique().tolist())
        (OUT / f"final_scaler_{tag}.json").write_text(json.dumps(final_scaler, indent=2), encoding="utf-8")

        summary["files"][tag] = {
            "train_engines": len(train_ids),
            "validation_engines": len(val_ids),
            "official_test_engines": int(raw_test["unit"].nunique()),
            "train_engine_ids": train_ids,
            "validation_engine_ids": val_ids,
            "validation_cutoff_cycles": {str(k): v for k, v in val_cutoffs.items()},
            "feature_count": len(features),
            "dropped_constant_sensors": scaler["dropped_constant_sensors"],
            "window_shapes": {"train": list(X_train.shape), "validation": list(X_val.shape), "test": list(X_test.shape)},
            "final_window_shapes": {"train": list(X_final_train.shape), "test": list(X_final_test.shape)},
            "target_ranges": {"train": [float(y_train.min()), float(y_train.max())], "validation": [float(y_val.min()), float(y_val.max())], "test": [float(y_test.min()), float(y_test.max())]},
            "window_file": f"windows_{tag}.npz",
            "row_files": [f"train_{tag}_processed.csv", f"val_{tag}_processed.csv", f"test_{tag}_processed.csv"],
            "scaler_file": f"scaler_{tag}.json",
            "final_window_file": f"final_windows_{tag}.npz",
            "final_scaler_file": f"final_scaler_{tag}.json",
        }

    (OUT / "preprocessing_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    eda = {
        "purpose": "Raw-data quality and descriptive statistics before preprocessing.",
        "subsets": {
            f"FD{fd:03d}": {
                "official_train": dataset_profile(read_table(ROOT / f"train_FD{fd:03d}.txt")),
                "official_test": dataset_profile(read_table(ROOT / f"test_FD{fd:03d}.txt")),
            }
            for fd in range(1, 5)
        },
    }
    (OUT / "eda_summary.json").write_text(json.dumps(eda, indent=2), encoding="utf-8")
    print(json.dumps(summary["files"], indent=2))
    print(f"\nProcessed outputs: {OUT}")


if __name__ == "__main__":
    main()
