"""Exploratory Data Analysis (EDA) and Degradation Physics module for C-MAPSS FD001.

Core responsibilities:
- Screens all 3 operating settings and 21 sensors strictly on the 80 development engines.
- Identifies constant/near-constant channels vs. the 14 informative degradation sensors.
- Constructs an unsupervised 1D PCA Composite Health Index on the retained sensors.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.constants import SENSOR_COLS, SENSOR_DESCRIPTIONS, SETTING_COLS


def screen_sensors_and_settings(
    dev_df: pd.DataFrame,
    std_threshold: float = 1e-2,
    min_abs_rul_corr: float = 0.05,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Screen all 3 operating settings and 21 sensors strictly on the 80 development engines.

    Parameters
    ----------
    dev_df : pd.DataFrame
        Development DataFrame (80 engines, 16,340 rows) with `rul` and `rul_uncapped`.
    std_threshold : float, default=1e-2
        Minimum standard deviation required to consider a channel non-constant.
        (Note: `sensor_6` has std=0.00139 in FD001, oscillating between 21.60 and 21.61 psia).
    min_abs_rul_corr : float, default=0.05
        Minimum absolute Pearson correlation with capped RUL on the development split.

    Returns
    -------
    summary_df : pd.DataFrame
        Detailed statistics per channel (mean, std, min, max, range, corr_rul_capped,
        corr_rul_uncapped, retained, status, physical_description).
    retained_sensors : list[str]
        List of informative sensor column names retained for modeling.
    dropped_channels : list[str]
        List of constant/near-constant sensor and setting column names dropped.
    """
    channels = SETTING_COLS + SENSOR_COLS
    records: list[dict[str, Any]] = []

    setting_descriptions = {
        "op_setting_1": "Altitude (Sea Level ~0 ft)",
        "op_setting_2": "Mach Number (~0.0)",
        "op_setting_3": "Throttle Resolver Angle (100%)",
    }

    for col in channels:
        series = dev_df[col]
        val_mean = float(series.mean())
        val_std = float(series.std())
        val_min = float(series.min())
        val_max = float(series.max())
        val_range = val_max - val_min
        unique_cnt = int(series.nunique())

        if val_std < 1e-12:
            corr_capped = 0.0
            corr_uncapped = 0.0
        else:
            corr_capped = float(series.corr(dev_df["rul"]))
            corr_uncapped = float(series.corr(dev_df["rul_uncapped"]))

        is_setting = col in SETTING_COLS
        if is_setting:
            status = "Dropped (Fixed Sea-Level Setting)"
            retain = False
            desc = setting_descriptions[col]
        else:
            desc = SENSOR_DESCRIPTIONS.get(col, col)
            if val_std < 1e-6:
                status = "Dropped (Exactly Constant)"
                retain = False
            elif val_std < std_threshold or abs(corr_capped) < min_abs_rul_corr:
                status = "Dropped (Near-Constant / Uninformative)"
                retain = False
            else:
                trend_dir = "Rising with wear" if corr_capped < 0 else "Falling with wear"
                status = f"Retained ({trend_dir})"
                retain = True

        records.append(
            {
                "channel": col,
                "type": "setting" if is_setting else "sensor",
                "physical_description": desc,
                "mean": round(val_mean, 4),
                "std": round(val_std, 6),
                "min": round(val_min, 4),
                "max": round(val_max, 4),
                "range": round(val_range, 4),
                "unique_values": unique_cnt,
                "corr_rul_capped": round(corr_capped, 4),
                "corr_rul_uncapped": round(corr_uncapped, 4),
                "retained": retain,
                "status": status,
            }
        )

    summary_df = pd.DataFrame(records)
    retained_sensors = summary_df.loc[summary_df["retained"], "channel"].tolist()
    dropped_channels = summary_df.loc[~summary_df["retained"], "channel"].tolist()
    return summary_df, retained_sensors, dropped_channels


def fit_pca_health_index(
    dev_df: pd.DataFrame,
    retained_sensors: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit a StandardScaler + PCA on the 14 retained sensors of the 80 development engines.

    Produces a 1D composite Health Index (PC1 oriented so high values represent healthy early
    operation and low values represent near-failure degradation).
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(dev_df[retained_sensors])

    pca = PCA(n_components=3, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    pc1 = X_pca[:, 0]
    if np.corrcoef(pc1, dev_df["rul"])[0, 1] < 0:
        pc1 = -pc1
        pc1_loadings = -pca.components_[0]
    else:
        pc1_loadings = pca.components_[0]

    out_df = dev_df[["unit_id", "cycle", "max_cycle", "rul_uncapped", "rul", "alarm"]].copy()
    out_df["pca_health_index"] = pc1
    out_df["pc2"] = X_pca[:, 1]

    loadings_dict = {
        sensor: round(float(loading), 4)
        for sensor, loading in zip(retained_sensors, pc1_loadings)
    }

    pca_info = {
        "explained_variance_ratio_pc1": round(float(pca.explained_variance_ratio_[0]), 4),
        "explained_variance_ratio_pc2": round(float(pca.explained_variance_ratio_[1]), 4),
        "explained_variance_ratio_pc3": round(float(pca.explained_variance_ratio_[2]), 4),
        "cumulative_top3_variance": round(float(pca.explained_variance_ratio_[:3].sum()), 4),
        "corr_pc1_with_rul_capped": round(float(np.corrcoef(pc1, dev_df["rul"])[0, 1]), 4),
        "corr_pc1_with_rul_uncapped": round(float(np.corrcoef(pc1, dev_df["rul_uncapped"])[0, 1]), 4),
        "pc1_sensor_loadings": loadings_dict,
    }
    return out_df, pca_info
