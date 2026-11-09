"""Causal trailing-window feature engineering and leakage-proof scikit-learn pipelines.

Phase 3 of the ES 335 Predictive Maintenance Project:
- Implements `SensorVarianceFilter` to select informative sensors strictly on training folds.
- Implements `TrailingWindowFeatureTransformer` to compute per-engine, strictly causal trailing
  window statistics (`rolling_mean`, `rolling_std`, `slope`) over `[t - k + 1, t]` plus engine age (`cycle`).
- Encapsulates feature screening, trailing window extraction, and standardization inside a
  scikit-learn `Pipeline` so `GroupKFold` never leaks validation fold statistics.
- Provides utilities to quantify how trailing smoothing improves signal correlation with RUL
  and why random row splitting leaks near-identical neighboring cycles across train/test.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.constants import INDEX_COLS, SENSOR_COLS, SENSOR_DESCRIPTIONS

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def _precompute_slope_weights(max_k: int) -> dict[int, np.ndarray]:
    """Precompute variance-stabilized linear OLS slope weight vectors for m = 2 .. max_k.

    For a full window of length k with local cycle indices x = [0, 1, ..., k-1]:
        slope = sum((x - x_mean) * y) / sum((x - x_mean)^2) = w_k @ y
    For incomplete initial boundary windows (2 <= m < k) at the start of an engine's life,
    dividing by the full-window denominator `k * (k^2 - 1) / 12` prevents 2-point random
    sensor noise at cycle t=2,3 from being amplified by up to 67x.
    """
    weights: dict[int, np.ndarray] = {}
    full_denom = float(max_k * (max_k**2 - 1)) / 12.0
    for m in range(2, max_k + 1):
        x = np.arange(m, dtype=np.float64)
        x_centered = x - x.mean()
        weights[m] = x_centered / full_denom
    return weights


def compute_engine_trailing_features(
    sensor_matrix: np.ndarray,
    window_k: int,
    slope_weights: dict[int, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute strictly trailing mean, std, and linear OLS slope for a single engine's trajectory.

    Parameters
    ----------
    sensor_matrix : np.ndarray of shape (T, S)
        Chronologically ordered sensor values for a single engine over T cycles and S sensors.
    window_k : int
        Trailing window length k (covers cycles `[max(1, t - k + 1), t]`).
    slope_weights : dict[int, np.ndarray]
        Precomputed OLS slope weight vectors for lengths `2 .. window_k`.

    Returns
    -------
    roll_mean : np.ndarray of shape (T, S)
    roll_std  : np.ndarray of shape (T, S)
    roll_slope: np.ndarray of shape (T, S)
    """
    n_cycles, n_sensors = sensor_matrix.shape
    roll_mean = np.empty((n_cycles, n_sensors), dtype=np.float64)
    roll_std = np.empty((n_cycles, n_sensors), dtype=np.float64)
    roll_slope = np.empty((n_cycles, n_sensors), dtype=np.float64)

    # First cycle (m = 1): mean is raw reading, std and slope are 0.0 (no prior history)
    roll_mean[0] = sensor_matrix[0]
    roll_std[0] = 0.0
    roll_slope[0] = 0.0

    # Expanding boundary windows for cycles 2 .. min(window_k - 1, n_cycles)
    boundary_end = min(window_k - 1, n_cycles)
    for m in range(2, boundary_end + 1):
        window_slice = sensor_matrix[:m]
        roll_mean[m - 1] = window_slice.mean(axis=0)
        roll_std[m - 1] = window_slice.std(axis=0, ddof=0)
        roll_slope[m - 1] = slope_weights[m] @ window_slice

    # Full windows of length `window_k` for cycles window_k .. n_cycles
    if n_cycles >= window_k:
        # windows shape: (n_cycles - window_k + 1, n_sensors, window_k)
        windows = sliding_window_view(sensor_matrix, window_shape=window_k, axis=0)
        roll_mean[window_k - 1 :] = windows.mean(axis=-1)
        roll_std[window_k - 1 :] = windows.std(axis=-1, ddof=0)
        roll_slope[window_k - 1 :] = np.tensordot(
            windows, slope_weights[window_k], axes=([-1], [0])
        )

    return roll_mean, roll_std, roll_slope


class SensorVarianceFilter(BaseEstimator, TransformerMixin):
    """Fit-on-training-fold sensor filter that drops constant and near-constant channels.

    Parameters
    ----------
    std_threshold : float, default=0.01
        Minimum standard deviation on the training fold required to retain a sensor.
        In FD001, `std_threshold=0.01` drops the 6 zero-variance sensors (`sensor_1, 5, 10, 16, 18, 19`)
        and the near-constant `sensor_6` (`std=0.0014`), retaining the 14 degradation sensors.
    candidate_sensors : list[str] or None, default=None
        Sensor columns to evaluate. Defaults to `SENSOR_COLS` (`sensor_1..sensor_21`).
    """

    def __init__(
        self,
        std_threshold: float = 0.01,
        candidate_sensors: list[str] | None = None,
    ) -> None:
        self.std_threshold = std_threshold
        self.candidate_sensors = candidate_sensors

    def fit(self, X: pd.DataFrame, y: Any = None) -> "SensorVarianceFilter":
        if not isinstance(X, pd.DataFrame):
            raise TypeError("SensorVarianceFilter expects a pandas DataFrame input.")

        candidates = (
            self.candidate_sensors
            if self.candidate_sensors is not None
            else [c for c in SENSOR_COLS if c in X.columns]
        )
        stds = X[candidates].std(axis=0)
        self.sensor_stds_: dict[str, float] = {col: float(stds[col]) for col in candidates}
        self.retained_sensors_: list[str] = [
            col for col in candidates if stds[col] >= self.std_threshold
        ]
        self.dropped_sensors_: list[str] = [
            col for col in candidates if stds[col] < self.std_threshold
        ]
        if not self.retained_sensors_:
            raise ValueError(
                f"No sensors passed std_threshold={self.std_threshold}. Check input DataFrame."
            )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("SensorVarianceFilter expects a pandas DataFrame input.")
        keep_cols = [c for c in INDEX_COLS if c in X.columns] + self.retained_sensors_
        return X[keep_cols].copy()

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.array(INDEX_COLS + self.retained_sensors_, dtype=object)


class TrailingWindowFeatureTransformer(BaseEstimator, TransformerMixin):
    """Per-engine strictly causal trailing-window feature extractor.

    Computes rolling statistics over `[max(1, t - k + 1), t]` grouped by `unit_id` so windows
    never mix engines and never look ahead of cycle `t`.

    Parameters
    ----------
    window_k : int, default=10
        Length of the trailing window in cycles (`k=10` primary, `k=30` sensitivity).
    feature_mode : str, default="trailing_trends"
        Controls which feature blocks are emitted:
        - `"age_only"`: `["cycle"]` (1 feature — Rung 2 baseline)
        - `"instantaneous"`: `["cycle"]` + raw retained sensors at cycle `t` (15 features — Rung 3)
        - `"trailing_trends"`: `["cycle"]` + trailing `mean`, `std`, `slope` for each retained sensor
          (43 features — Rung 4 & Rung 5 primary models)
        - `"combined"`: `["cycle"]` + instantaneous sensors + trailing `mean`, `std`, `slope`
          (57 features)
    include_age : bool, default=True
        Whether to include current operating `cycle` as an age feature.
    return_dataframe : bool, default=True
        If True, returns a pandas DataFrame with named columns and preserved index.
    """

    VALID_MODES = ("age_only", "instantaneous", "trailing_trends", "combined")

    def __init__(
        self,
        window_k: int = 10,
        feature_mode: str = "trailing_trends",
        include_age: bool = True,
        return_dataframe: bool = True,
    ) -> None:
        self.window_k = window_k
        self.feature_mode = feature_mode
        self.include_age = include_age
        self.return_dataframe = return_dataframe

    def fit(self, X: pd.DataFrame, y: Any = None) -> "TrailingWindowFeatureTransformer":
        if self.feature_mode not in self.VALID_MODES:
            raise ValueError(
                f"Invalid feature_mode='{self.feature_mode}'. Must be one of {self.VALID_MODES}."
            )
        if self.window_k < 2:
            raise ValueError(f"window_k must be >= 2, got {self.window_k}.")

        if not isinstance(X, pd.DataFrame):
            raise TypeError("TrailingWindowFeatureTransformer expects a pandas DataFrame input.")

        for req_col in INDEX_COLS:
            if req_col not in X.columns:
                raise ValueError(f"Required column '{req_col}' missing from input DataFrame.")

        self.sensor_cols_: list[str] = [
            c for c in X.columns if c.startswith("sensor_")
        ]
        if self.feature_mode != "age_only" and not self.sensor_cols_:
            raise ValueError("No sensor columns found in X.")

        feature_names: list[str] = []
        if self.include_age or self.feature_mode == "age_only":
            feature_names.append("cycle")

        if self.feature_mode in ("instantaneous", "combined"):
            feature_names.extend(self.sensor_cols_)

        if self.feature_mode in ("trailing_trends", "combined"):
            k = self.window_k
            for s in self.sensor_cols_:
                feature_names.append(f"{s}_mean_k{k}")
            for s in self.sensor_cols_:
                feature_names.append(f"{s}_std_k{k}")
            for s in self.sensor_cols_:
                feature_names.append(f"{s}_slope_k{k}")

        self.feature_names_out_: list[str] = feature_names
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame | np.ndarray:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("TrailingWindowFeatureTransformer expects a pandas DataFrame input.")

        # Case 1: Age-only baseline
        if self.feature_mode == "age_only":
            out_df = X[["cycle"]].astype(np.float64).copy()
            return out_df if self.return_dataframe else out_df.to_numpy()

        # Case 2: Instantaneous sensor values only (no rolling window needed)
        if self.feature_mode == "instantaneous":
            cols = (["cycle"] if self.include_age else []) + self.sensor_cols_
            out_df = X[cols].astype(np.float64).copy()
            return out_df if self.return_dataframe else out_df.to_numpy()

        # Case 3: Trailing window features ("trailing_trends" or "combined")
        # Verify chronological ordering within each unit_id; if unsorted, sort first and restore order
        is_sorted = (
            X["unit_id"].is_monotonic_increasing
            and X.groupby("unit_id", sort=False)["cycle"]
            .apply(lambda s: s.is_monotonic_increasing)
            .all()
        )
        if not is_sorted:
            work_df = X.sort_values(INDEX_COLS)
            restore_order = True
        else:
            work_df = X
            restore_order = False

        n_rows = len(work_df)
        n_sensors = len(self.sensor_cols_)
        sensor_vals = work_df[self.sensor_cols_].to_numpy(dtype=np.float64)
        unit_ids = work_df["unit_id"].to_numpy()

        all_means = np.empty((n_rows, n_sensors), dtype=np.float64)
        all_stds = np.empty((n_rows, n_sensors), dtype=np.float64)
        all_slopes = np.empty((n_rows, n_sensors), dtype=np.float64)

        slope_weights = _precompute_slope_weights(self.window_k)

        # Find contiguous group boundaries for each unit_id
        change_indices = np.flatnonzero(np.diff(unit_ids) != 0) + 1
        starts = np.concatenate(([0], change_indices))
        ends = np.concatenate((change_indices, [n_rows]))

        for start, end in zip(starts, ends):
            eng_matrix = sensor_vals[start:end]
            m_arr, s_arr, sl_arr = compute_engine_trailing_features(
                eng_matrix, self.window_k, slope_weights
            )
            all_means[start:end] = m_arr
            all_stds[start:end] = s_arr
            all_slopes[start:end] = sl_arr

        blocks: list[np.ndarray] = []
        if self.include_age:
            blocks.append(work_df[["cycle"]].to_numpy(dtype=np.float64))
        if self.feature_mode == "combined":
            blocks.append(sensor_vals)
        blocks.extend([all_means, all_stds, all_slopes])

        full_matrix = np.hstack(blocks)
        out_df = pd.DataFrame(full_matrix, index=work_df.index, columns=self.feature_names_out_)

        if restore_order:
            out_df = out_df.loc[X.index]

        return out_df if self.return_dataframe else out_df.to_numpy()

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.array(self.feature_names_out_, dtype=object)


class DataFrameStandardScaler(BaseEstimator, TransformerMixin):
    """Wrapper around `StandardScaler` that preserves pandas DataFrame column names and index."""

    def __init__(self) -> None:
        self.scaler_ = StandardScaler()

    def fit(self, X: pd.DataFrame | np.ndarray, y: Any = None) -> "DataFrameStandardScaler":
        self.scaler_ = StandardScaler()
        self.scaler_.fit(X)
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = list(X.columns)
        else:
            self.feature_names_in_ = [f"f_{i}" for i in range(X.shape[1])]
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        scaled = self.scaler_.transform(X)
        idx = X.index if isinstance(X, pd.DataFrame) else None
        return pd.DataFrame(scaled, index=idx, columns=self.feature_names_in_)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.array(self.feature_names_in_, dtype=object)


def build_preprocessing_pipeline(
    window_k: int = 10,
    feature_mode: str = "trailing_trends",
    std_threshold: float = 0.01,
    include_age: bool = True,
    scale: bool = True,
) -> Pipeline:
    """Construct a leakage-proof preprocessing `Pipeline` for C-MAPSS sensor trajectories.

    When placed inside `GroupKFold` cross-validation:
    1. `sensor_filter` identifies constant/near-constant sensors using ONLY the training fold.
    2. `trailing_features` computes per-engine causal trailing statistics over `[t - k + 1, t]`.
    3. `scaler` standardizes features using ONLY the training fold's mean and standard deviation.
    """
    steps: list[tuple[str, Any]] = [
        ("sensor_filter", SensorVarianceFilter(std_threshold=std_threshold)),
        (
            "trailing_features",
            TrailingWindowFeatureTransformer(
                window_k=window_k,
                feature_mode=feature_mode,
                include_age=include_age,
                return_dataframe=True,
            ),
        ),
    ]
    if scale:
        steps.append(("scaler", DataFrameStandardScaler()))
    return Pipeline(steps)


def compare_feature_correlations_across_windows(
    dev_df: pd.DataFrame,
    window_ks: list[int] = [10, 30],
    std_threshold: float = 0.01,
) -> pd.DataFrame:
    """Compare RUL correlations of raw instantaneous sensors vs trailing window features on dev data."""
    vfilter = SensorVarianceFilter(std_threshold=std_threshold).fit(dev_df)
    retained = vfilter.retained_sensors_

    feat_By_k: dict[int, pd.DataFrame] = {}
    for k in window_ks:
        pipe = build_preprocessing_pipeline(
            window_k=k,
            feature_mode="trailing_trends",
            std_threshold=std_threshold,
            scale=False,
        )
        feat_By_k[k] = pipe.fit_transform(dev_df)

    rows: list[dict[str, Any]] = []
    for s in retained:
        raw_corr_cap = float(dev_df[s].corr(dev_df["rul"]))
        raw_corr_uncap = float(dev_df[s].corr(dev_df["rul_uncapped"]))
        rec: dict[str, Any] = {
            "sensor": s,
            "physical_description": SENSOR_DESCRIPTIONS.get(s, s),
            "corr_raw_instantaneous": round(raw_corr_cap, 4),
            "abs_corr_raw": round(abs(raw_corr_cap), 4),
            "corr_raw_uncapped": round(raw_corr_uncap, 4),
        }
        for k in window_ks:
            f_df = feat_By_k[k]
            c_mean = float(f_df[f"{s}_mean_k{k}"].corr(dev_df["rul"]))
            c_std = float(f_df[f"{s}_std_k{k}"].corr(dev_df["rul"]))
            c_slope = float(f_df[f"{s}_slope_k{k}"].corr(dev_df["rul"]))
            rec[f"corr_mean_k{k}"] = round(c_mean, 4)
            rec[f"abs_corr_mean_k{k}"] = round(abs(c_mean), 4)
            rec[f"corr_std_k{k}"] = round(c_std, 4)
            rec[f"corr_slope_k{k}"] = round(c_slope, 4)
            rec[f"abs_corr_slope_k{k}"] = round(abs(c_slope), 4)
            rec[f"mean_gain_k{k}_vs_raw"] = round(abs(c_mean) - abs(raw_corr_cap), 4)

        rows.append(rec)

    return pd.DataFrame(rows).sort_values("abs_corr_mean_k10", ascending=False).reset_index(drop=True)


def analyze_row_split_leakage_mechanism(
    dev_df: pd.DataFrame,
    window_k: int = 10,
    seed: int = 42,
) -> dict[str, Any]:
    """Quantify why a random row-level split leaks near-identical cycles across train and validation.

    Computes:
    1. Lag-1 within-engine autocorrelation of trailing rolling mean features (shows consecutive
       cycles `t` and `t-1` are ~0.996 correlated).
    2. Under an 80/20 random row split of `dev_df`, the fraction of validation rows whose immediate
       neighbor (`t-1` or `t+1` of the same engine) sits inside the training set, and the cosine /
       Euclidean distance to that leaked neighbor vs. the nearest neighbor in an engine-grouped split.
    """
    pipe = build_preprocessing_pipeline(
        window_k=window_k, feature_mode="trailing_trends", scale=True
    )
    X_scaled = pipe.fit_transform(dev_df)
    mean_cols = [c for c in X_scaled.columns if f"_mean_k{window_k}" in c]

    # 1. Compute lag-1 autocorrelation within each engine for rolling mean features
    lag1_corrs: list[float] = []
    for col in mean_cols:
        s_curr = X_scaled[col]
        s_prev = X_scaled.groupby(dev_df["unit_id"])[col].shift(1)
        valid = s_prev.notna()
        lag1_corrs.append(float(s_curr[valid].corr(s_prev[valid])))

    # 2. Simulate a 80/20 random row split on the 80 development engines
    n_rows = len(dev_df)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_rows)
    n_train_rows = int(0.8 * n_rows)
    train_row_mask = np.zeros(n_rows, dtype=bool)
    train_row_mask[perm[:n_train_rows]] = True
    val_row_indices = np.flatnonzero(~train_row_mask)

    unit_ids = dev_df["unit_id"].to_numpy()
    cycles = dev_df["cycle"].to_numpy()
    train_pairs = set(zip(unit_ids[train_row_mask], cycles[train_row_mask]))

    has_adjacent_neighbor = 0
    has_window_overlap = 0
    for idx in val_row_indices:
        u = int(unit_ids[idx])
        c = int(cycles[idx])
        if (u, c - 1) in train_pairs or (u, c + 1) in train_pairs:
            has_adjacent_neighbor += 1
        if any((u, c + delta) in train_pairs for delta in range(-(window_k - 1), window_k)):
            has_window_overlap += 1

    pct_adjacent_in_train = (has_adjacent_neighbor / len(val_row_indices)) * 100.0
    pct_window_overlap_in_train = (has_window_overlap / len(val_row_indices)) * 100.0

    # Feature L2 distance between cycle t and cycle t-1 of same engine vs different engine
    X_mat = X_scaled[mean_cols].to_numpy()
    same_engine_diffs = []
    for idx in range(1, n_rows):
        if unit_ids[idx] == unit_ids[idx - 1]:
            dist = float(np.linalg.norm(X_mat[idx] - X_mat[idx - 1]))
            same_engine_diffs.append(dist)

    return {
        "window_k": int(window_k),
        "mean_lag1_autocorrelation_rolling_means": round(float(np.mean(lag1_corrs)), 5),
        "min_lag1_autocorrelation_rolling_means": round(float(np.min(lag1_corrs)), 5),
        "max_lag1_autocorrelation_rolling_means": round(float(np.max(lag1_corrs)), 5),
        "random_row_split_val_rows": int(len(val_row_indices)),
        "pct_val_rows_with_adjacent_cycle_in_train": round(float(pct_adjacent_in_train), 2),
        "pct_val_rows_with_overlapping_window_in_train": round(float(pct_window_overlap_in_train), 2),
        "median_feature_l2_dist_adjacent_cycle": round(float(np.median(same_engine_diffs)), 4),
    }


def plot_phase3_feature_diagnostics(
    dev_df: pd.DataFrame,
    corr_df: pd.DataFrame,
    leakage_info: dict[str, Any],
    out_path: Path,
    sample_unit_id: int = 1,
) -> None:
    """Generate a 3-panel Phase 3 diagnostic figure showing trailing features and leakage mechanics."""
    pipe_k10 = build_preprocessing_pipeline(window_k=10, feature_mode="trailing_trends", scale=False)
    pipe_k30 = build_preprocessing_pipeline(window_k=30, feature_mode="trailing_trends", scale=False)

    feat_k10 = pipe_k10.fit_transform(dev_df)
    feat_k30 = pipe_k30.fit_transform(dev_df)

    mask_u = dev_df["unit_id"] == sample_unit_id
    u_raw = dev_df.loc[mask_u].reset_index(drop=True)
    u_k10 = feat_k10.loc[mask_u].reset_index(drop=True)
    u_k30 = feat_k30.loc[mask_u].reset_index(drop=True)

    fig = plt.figure(figsize=(14.5, 9.2), layout="constrained")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1.0])

    # Panel A (Top Left): Raw vs Rolling Mean (k=10, k=30) on Engine 1 for sensor_11
    ax1 = fig.add_subplot(gs[0, 0])
    s_col = "sensor_11"
    ax1.plot(u_raw["cycle"], u_raw[s_col], color="#b0b0b0", alpha=0.65, linewidth=1.0, label="Raw Instantaneous Reading")
    ax1.plot(u_raw["cycle"], u_k10[f"{s_col}_mean_k10"], color="#1f77b4", linewidth=2.0, label="Trailing Rolling Mean (k=10)")
    ax1.plot(u_raw["cycle"], u_k30[f"{s_col}_mean_k30"], color="#d62728", linewidth=2.2, linestyle="--", label="Trailing Rolling Mean (k=30)")
    ax1.axvline(u_raw["max_cycle"].iloc[0] - 125, color="#2ca02c", linestyle=":", linewidth=1.4, label="RUL Cap Onset (-125 cyc)")
    ax1.axvline(u_raw["max_cycle"].iloc[0] - 30, color="#ff7f0e", linestyle="-.", linewidth=1.4, label="Alarm Horizon (w=30 cyc)")
    ax1.set_title(f"A. Trailing Smoothing on Engine #{sample_unit_id} ({s_col}: HPC Static Pressure)")
    ax1.set_xlabel("Operating Cycle ($t$)")
    ax1.set_ylabel("Pressure (psia)")
    ax1.legend(loc="upper left", frameon=True, fontsize=8)

    # Panel B (Top Right): Trailing Linear Drift Slope & Rolling Std on Engine 1
    ax2 = fig.add_subplot(gs[0, 1])
    color_slope = "#8e24aa"
    color_std = "#2ca02c"
    l1 = ax2.plot(
        u_raw["cycle"],
        u_k10[f"{s_col}_slope_k10"],
        color=color_slope,
        linewidth=1.6,
        alpha=0.5,
        label="Trailing Drift Slope (k=10)",
    )
    l2 = ax2.plot(
        u_raw["cycle"],
        u_k30[f"{s_col}_slope_k30"],
        color=color_slope,
        linewidth=2.3,
        label="Trailing Drift Slope (k=30)",
    )
    ax2.axhline(0.0, color="gray", linestyle=":", linewidth=1.0)
    ax2.set_xlabel("Operating Cycle ($t$)")
    ax2.set_ylabel("Linear Drift Rate ($\\Delta$ psia / cycle)", color=color_slope)
    ax2.tick_params(axis="y", labelcolor=color_slope)

    ax2_twin = ax2.twinx()
    ax2_twin.spines["top"].set_visible(False)
    l3 = ax2_twin.plot(
        u_raw["cycle"],
        u_k10[f"{s_col}_std_k10"],
        color=color_std,
        linewidth=1.5,
        linestyle="--",
        alpha=0.8,
        label="Trailing Rolling Std (k=10)",
    )
    ax2_twin.set_ylabel("Rolling Std Dev (psia)", color=color_std)
    ax2_twin.tick_params(axis="y", labelcolor=color_std)

    lines = l1 + l2 + l3
    ax2.legend(lines, [l.get_label() for l in lines], loc="upper left", frameon=True, fontsize=8)
    ax2.set_title(f"B. Trailing Drift Rate (Slope) & Instability (Std) on Engine #{sample_unit_id}")

    # Panel C (Bottom Left): Correlation Gain Across All 14 Sensors (Raw vs Mean k=10 vs Slope k=30)
    ax3 = fig.add_subplot(gs[1, 0])
    y_pos = np.arange(len(corr_df))
    height = 0.26
    ax3.barh(
        y_pos + height,
        corr_df["abs_corr_raw"],
        height=height,
        color="#9ecae1",
        label="Raw Instantaneous (|r|)",
    )
    ax3.barh(
        y_pos,
        corr_df["abs_corr_mean_k10"],
        height=height,
        color="#1f77b4",
        label="Trailing Mean k=10 (|r|)",
    )
    ax3.barh(
        y_pos - height,
        corr_df["abs_corr_slope_k30"],
        height=height,
        color="#8e24aa",
        label="Trailing Slope k=30 (|r|)",
    )
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels([s.replace("sensor_", "S") for s in corr_df["sensor"]], fontsize=8.5)
    ax3.invert_yaxis()
    ax3.set_xlim(0.2, 0.90)
    ax3.set_xlabel("Absolute Pearson Correlation ($|r|$) with Capped RUL")
    ax3.set_title("C. Signal Denoising: Trailing Mean (k=10) & Drift Slope (k=30) vs. Raw")
    ax3.legend(loc="lower right", frameon=True, fontsize=8)

    # Panel D (Bottom Right): Why Random Row Splitting Leaks Trajectory Windows
    ax4 = fig.add_subplot(gs[1, 1])
    rng = np.random.RandomState(42)
    sub_u = u_k10.iloc[120:185].copy()
    sub_cycles = u_raw["cycle"].iloc[120:185].to_numpy()
    sub_vals = sub_u[f"{s_col}_mean_k10"].to_numpy()
    is_val = rng.rand(len(sub_cycles)) < 0.20

    ax4.plot(sub_cycles, sub_vals, color="#d0d0d0", linewidth=1.5, zorder=1)
    ax4.scatter(
        sub_cycles[~is_val],
        sub_vals[~is_val],
        color="#1f77b4",
        s=28,
        label="Random Split: Train Rows (80%)",
        zorder=3,
    )
    ax4.scatter(
        sub_cycles[is_val],
        sub_vals[is_val],
        color="#d62728",
        s=52,
        marker="D",
        label="Random Split: Val Rows (20%)",
        zorder=4,
    )
    ax4.annotate(
        f"Lag-1 Autocorrelation = {leakage_info['mean_lag1_autocorrelation_rolling_means']:.3f}\n"
        f"{leakage_info['pct_val_rows_with_adjacent_cycle_in_train']:.1f}% of Val rows have t-1 or t+1 in Train!\n"
        f"{leakage_info['pct_val_rows_with_overlapping_window_in_train']:.1f}% share overlapping k=10 windows!",
        xy=(sub_cycles[is_val][len(sub_cycles[is_val]) // 2], sub_vals[is_val][len(sub_vals[is_val]) // 2]),
        xytext=(123, sub_vals.max() - 0.18),
        arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.2),
        fontsize=8.2,
        bbox=dict(boxstyle="round,pad=0.35", fc="#fff5f5", ec="#d62728"),
    )
    ax4.set_title(f"D. The Row-Split Leakage Trap (Engine #{sample_unit_id}, Cycles 120–185)")
    ax4.set_xlabel("Operating Cycle ($t$)")
    ax4.set_ylabel(f"{s_col}_mean_k10 (psia)")
    ax4.legend(loc="lower right", frameon=True, fontsize=8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
