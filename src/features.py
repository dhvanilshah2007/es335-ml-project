"""Causal trailing-window feature engineering and leakage-proof scikit-learn pipelines.

Phase 3 of the ES 335 Predictive Maintenance Project:
- `SensorVarianceFilter`: selects informative sensors strictly on training folds.
- `TrailingWindowFeatureTransformer`: computes per-engine, strictly causal trailing window
  statistics (`rolling_mean`, `rolling_std`, `slope`) over `[t - k + 1, t]` plus engine age (`cycle`).
- `build_preprocessing_pipeline`: encapsulates screening, trailing extraction, and scaling inside a
  scikit-learn `Pipeline` so `GroupKFold` never leaks validation fold statistics.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.constants import INDEX_COLS, SENSOR_COLS, SENSOR_DESCRIPTIONS


def _precompute_slope_weights(max_k: int) -> dict[int, np.ndarray]:
    """Precompute variance-stabilized OLS slope weights for window lengths m = 2 .. max_k.

    Dividing incomplete initial boundary windows (`2 <= m < max_k`) by the full-window denominator
    `max_k * (max_k^2 - 1) / 12` prevents early-cycle sensor noise from being amplified.
    """
    weights: dict[int, np.ndarray] = {}
    full_denom = float(max_k * (max_k**2 - 1)) / 12.0
    for m in range(2, max_k + 1):
        x = np.arange(m, dtype=np.float64)
        weights[m] = (x - x.mean()) / full_denom
    return weights


def compute_engine_trailing_features(
    sensor_matrix: np.ndarray,
    window_k: int,
    slope_weights: dict[int, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute strictly trailing mean, std, and linear OLS slope for one engine's trajectory."""
    n_cycles, n_sensors = sensor_matrix.shape
    roll_mean = np.empty((n_cycles, n_sensors), dtype=np.float64)
    roll_std = np.empty((n_cycles, n_sensors), dtype=np.float64)
    roll_slope = np.empty((n_cycles, n_sensors), dtype=np.float64)

    # First cycle (m = 1): mean is raw reading, std and slope are 0.0
    roll_mean[0] = sensor_matrix[0]
    roll_std[0] = 0.0
    roll_slope[0] = 0.0

    # Expanding causal boundary windows for cycles 2 .. min(window_k - 1, n_cycles)
    boundary_end = min(window_k - 1, n_cycles)
    for m in range(2, boundary_end + 1):
        window_slice = sensor_matrix[:m]
        roll_mean[m - 1] = window_slice.mean(axis=0)
        roll_std[m - 1] = window_slice.std(axis=0, ddof=0)
        roll_slope[m - 1] = slope_weights[m] @ window_slice

    # Full windows of length `window_k` for cycles window_k .. n_cycles
    if n_cycles >= window_k:
        windows = sliding_window_view(sensor_matrix, window_shape=window_k, axis=0)
        roll_mean[window_k - 1 :] = windows.mean(axis=-1)
        roll_std[window_k - 1 :] = windows.std(axis=-1, ddof=0)
        roll_slope[window_k - 1 :] = np.tensordot(
            windows, slope_weights[window_k], axes=([-1], [0])
        )

    return roll_mean, roll_std, roll_slope


class SensorVarianceFilter(BaseEstimator, TransformerMixin):
    """Fit-on-training-fold sensor filter that drops constant and near-constant channels."""

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

    Supported `feature_mode` values:
    - `"age_only"`: `["cycle"]` (1 feature)
    - `"instantaneous"`: `["cycle"]` + raw retained sensors at cycle `t` (15 features)
    - `"trailing_trends"`: `["cycle"]` + trailing mean, std, and slope over `[t-k+1, t]` (43 features)
    """

    VALID_MODES = ("age_only", "instantaneous", "trailing_trends")

    def __init__(
        self,
        window_k: int = 30,
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

        self.sensor_cols_: list[str] = [c for c in X.columns if c.startswith("sensor_")]
        if self.feature_mode != "age_only" and not self.sensor_cols_:
            raise ValueError("No sensor columns found in X.")

        feature_names: list[str] = []
        if self.include_age or self.feature_mode == "age_only":
            feature_names.append("cycle")

        if self.feature_mode == "instantaneous":
            feature_names.extend(self.sensor_cols_)
        elif self.feature_mode == "trailing_trends":
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

        if self.feature_mode == "age_only":
            out_df = X[["cycle"]].astype(np.float64).copy()
            return out_df if self.return_dataframe else out_df.to_numpy()

        if self.feature_mode == "instantaneous":
            cols = (["cycle"] if self.include_age else []) + self.sensor_cols_
            out_df = X[cols].astype(np.float64).copy()
            return out_df if self.return_dataframe else out_df.to_numpy()

        # Verify chronological order within each engine; sort temporarily if needed
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

        change_indices = np.flatnonzero(np.diff(unit_ids) != 0) + 1
        starts = np.concatenate(([0], change_indices))
        ends = np.concatenate((change_indices, [n_rows]))

        for start, end in zip(starts, ends):
            m_arr, s_arr, sl_arr = compute_engine_trailing_features(
                sensor_vals[start:end], self.window_k, slope_weights
            )
            all_means[start:end] = m_arr
            all_stds[start:end] = s_arr
            all_slopes[start:end] = sl_arr

        blocks: list[np.ndarray] = []
        if self.include_age:
            blocks.append(work_df[["cycle"]].to_numpy(dtype=np.float64))
        blocks.extend([all_means, all_stds, all_slopes])

        out_df = pd.DataFrame(
            np.hstack(blocks), index=work_df.index, columns=self.feature_names_out_
        )
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
    window_k: int = 30,
    feature_mode: str = "trailing_trends",
    std_threshold: float = 0.01,
    include_age: bool = True,
    scale: bool = True,
) -> Pipeline:
    """Construct a leakage-proof preprocessing `Pipeline` for C-MAPSS sensor trajectories."""
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
    window_k: int = 30,
    std_threshold: float = 0.01,
) -> pd.DataFrame:
    """Compare RUL correlations of raw instantaneous sensors vs trailing window features."""
    vfilter = SensorVarianceFilter(std_threshold=std_threshold).fit(dev_df)
    retained = vfilter.retained_sensors_

    pipe = build_preprocessing_pipeline(
        window_k=window_k,
        feature_mode="trailing_trends",
        std_threshold=std_threshold,
        scale=False,
    )
    f_df = pipe.fit_transform(dev_df)

    rows: list[dict[str, Any]] = []
    for s in retained:
        raw_corr = float(dev_df[s].corr(dev_df["rul"]))
        c_mean = float(f_df[f"{s}_mean_k{window_k}"].corr(dev_df["rul"]))
        c_std = float(f_df[f"{s}_std_k{window_k}"].corr(dev_df["rul"]))
        c_slope = float(f_df[f"{s}_slope_k{window_k}"].corr(dev_df["rul"]))
        rows.append(
            {
                "sensor": s,
                "physical_description": SENSOR_DESCRIPTIONS.get(s, s),
                "corr_raw": round(raw_corr, 4),
                "abs_corr_raw": round(abs(raw_corr), 4),
                "corr_mean": round(c_mean, 4),
                "abs_corr_mean": round(abs(c_mean), 4),
                "corr_std": round(c_std, 4),
                "corr_slope": round(c_slope, 4),
                "abs_corr_slope": round(abs(c_slope), 4),
                "mean_gain_vs_raw": round(abs(c_mean) - abs(raw_corr), 4),
            }
        )

    return pd.DataFrame(rows).sort_values("abs_corr_mean", ascending=False).reset_index(drop=True)
