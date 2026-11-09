"""Exploratory Data Analysis (EDA) and Degradation Physics module for C-MAPSS FD001.

Phase 2 of the ES 335 Predictive Maintenance Project:
- Computes sensor and operating setting variance, range, and RUL correlation strictly on the
  80 development engines (`load_development_data()`).
- Identifies constant and near-constant channels vs. the 14 informative degradation sensors.
- Quantifies the fixed-interval replacement trade-off (in-service failure rate vs. wasted life).
- Aligns sensor trajectories by forward cycle (`cycle`) vs. reverse cycles-to-failure (`-rul_uncapped`).
- Constructs an unsupervised PCA Health Index on the 14 retained sensors.
- Generates publication-quality figures in `results/figures/` and tabular summaries in `results/metrics/`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.constants import (
    SENSOR_COLS,
    SENSOR_DESCRIPTIONS,
    SETTING_COLS,
)

# Set consistent publication style
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
        corr_rul_uncapped, status, physical_description).
    retained_sensors : list[str]
        List of informative sensor column names retained for modeling.
    dropped_channels : list[str]
        List of constant/near-constant sensor and setting column names dropped.
    """
    channels = SETTING_COLS + SENSOR_COLS
    records: list[dict[str, Any]] = []

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
            desc = {
                "op_setting_1": "Altitude (Sea Level ~0 ft)",
                "op_setting_2": "Mach Number (~0.0)",
                "op_setting_3": "Throttle Resolver Angle (100%)",
            }[col]
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


def analyze_fixed_schedule_tradeoff(
    dev_df: pd.DataFrame,
    candidate_k_values: list[int] | None = None,
) -> pd.DataFrame:
    """Quantify the fundamental dilemma of fixed-interval replacement at cycle K on dev engines.

    For each candidate replacement cycle K:
    - Engines with `max_cycle <= K` fail in service prior to scheduled replacement.
    - Engines with `max_cycle > K` survive to cycle K and are replaced preventively,
      discarding `max_cycle - K` cycles of remaining useful life.
    """
    lifetimes = dev_df.groupby("unit_id")["max_cycle"].first()
    n_engines = len(lifetimes)

    if candidate_k_values is None:
        candidate_k_values = list(range(120, 310, 10)) + [128, 150, 175, 199]
        candidate_k_values = sorted(set(candidate_k_values))

    rows: list[dict[str, Any]] = []
    for k in candidate_k_values:
        failed_mask = lifetimes <= k
        n_failed = int(failed_mask.sum())
        fail_rate_pct = (n_failed / n_engines) * 100.0

        wasted_cycles = np.maximum(0, lifetimes - k)
        mean_wasted_all = float(wasted_cycles.mean())
        mean_wasted_survivors = (
            float(wasted_cycles[~failed_mask].mean()) if (~failed_mask).any() else 0.0
        )
        pct_fleet_life_wasted = float(wasted_cycles.sum() / lifetimes.sum() * 100.0)

        rows.append(
            {
                "replacement_cycle_K": int(k),
                "engines_failed": n_failed,
                "failure_rate_pct": round(fail_rate_pct, 2),
                "mean_wasted_cycles_per_engine": round(mean_wasted_all, 2),
                "mean_wasted_cycles_survivors": round(mean_wasted_survivors, 2),
                "pct_total_fleet_life_wasted": round(pct_fleet_life_wasted, 2),
            }
        )

    return pd.DataFrame(rows)


def fit_pca_health_index(
    dev_df: pd.DataFrame,
    retained_sensors: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit a StandardScaler + PCA on the 14 retained sensors of the 80 development engines.

    Produces a 1-number composite Health Index (PC1 oriented so that high values correspond to
    healthy early life and low values correspond to near-failure degradation).
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(dev_df[retained_sensors])

    pca = PCA(n_components=3, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    pc1 = X_pca[:, 0]
    # Orient PC1 so that it correlates positively with RUL (1 = healthy, declines toward failure)
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


def plot_lifetime_and_fixed_schedule(
    dev_df: pd.DataFrame,
    tradeoff_df: pd.DataFrame,
    out_path: Path,
) -> None:
    """Plot fleet lifetime distribution and the fixed-interval replacement trade-off curve."""
    lifetimes = dev_df.groupby("unit_id")["max_cycle"].first()

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    # Panel A: Lifetime histogram + KDE
    ax = axes[0]
    counts, bins, _ = ax.hist(
        lifetimes, bins=16, color="#1f77b4", edgecolor="white", alpha=0.75
    )
    kde = gaussian_kde(lifetimes)
    x_grid = np.linspace(lifetimes.min() - 10, lifetimes.max() + 10, 200)
    bin_width = bins[1] - bins[0]
    ax.plot(x_grid, kde(x_grid) * len(lifetimes) * bin_width, color="#0d3b66", linewidth=1.8)
    ax.axvline(lifetimes.min(), color="#d62728", linestyle="--", linewidth=1.5, label=f"Min Life = {lifetimes.min()} cyc")
    ax.axvline(lifetimes.median(), color="#2ca02c", linestyle="-", linewidth=1.8, label=f"Median = {lifetimes.median():.0f} cyc")
    ax.axvline(lifetimes.mean(), color="#ff7f0e", linestyle=":", linewidth=1.8, label=f"Mean = {lifetimes.mean():.1f} cyc")
    ax.axvline(lifetimes.max(), color="#9467bd", linestyle="--", linewidth=1.5, label=f"Max Life = {lifetimes.max()} cyc")
    ax.set_title("A. Engine Lifetime Distribution (80 Dev Engines)")
    ax.set_xlabel("Total Operating Cycles Until Failure ($T_{\\mathrm{failure}}$)")
    ax.set_ylabel("Number of Engines")
    ax.legend(loc="upper right", frameon=True)

    # Panel B: Fixed-interval K trade-off
    ax2 = axes[1]
    grid_df = tradeoff_df[tradeoff_df["replacement_cycle_K"].between(120, 300)].sort_values("replacement_cycle_K")
    color_fail = "#d62728"
    color_waste = "#1f77b4"

    l1 = ax2.plot(
        grid_df["replacement_cycle_K"],
        grid_df["failure_rate_pct"],
        color=color_fail,
        linewidth=2.2,
        marker="o",
        markersize=4,
        label="In-Service Failure Rate (%)",
    )
    ax2.set_xlabel("Fixed Scheduled Replacement Cycle ($K$)")
    ax2.set_ylabel("Engines Failing Before Cycle $K$ (%)", color=color_fail)
    ax2.tick_params(axis="y", labelcolor=color_fail)
    ax2.set_ylim(-2, 102)

    ax2_twin = ax2.twinx()
    ax2_twin.spines["top"].set_visible(False)
    l2 = ax2_twin.plot(
        grid_df["replacement_cycle_K"],
        grid_df["mean_wasted_cycles_per_engine"],
        color=color_waste,
        linewidth=2.2,
        linestyle="--",
        marker="s",
        markersize=4,
        label="Mean Wasted Life (cycles/engine)",
    )
    ax2_twin.set_ylabel("Mean Wasted Remaining Life (Cycles)", color=color_waste)
    ax2_twin.tick_params(axis="y", labelcolor=color_waste)
    ax2_twin.set_ylim(-2, 90)

    # Highlight K=128 (shortest engine life in dev set: 1.2% failure, 76.2 wasted cycles)
    ax2.axvline(128, color="gray", linestyle=":", alpha=0.8)
    ax2.annotate(
        "Conservative Schedule (K=128)\n1.2% Failures, 76.2 cyc wasted",
        xy=(128, 5),
        xytext=(142, 18),
        arrowprops=dict(arrowstyle="->", color="gray", lw=1),
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.3", fc="#f8f9fa", ec="#cccccc"),
    )

    lines = l1 + l2
    labels = [line.get_label() for line in lines]
    ax2.legend(lines, labels, loc="center right", frameon=True)
    ax2.set_title("B. The Fixed-Schedule Dilemma: Failures vs. Wasted Life")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_aligned_degradation_trajectories(
    dev_df: pd.DataFrame,
    out_path: Path,
) -> None:
    """Compare unaligned trajectories (vs cycle t) against failure-aligned trajectories (vs -RUL).

    Illustrates why sensor readings aligned by remaining life reveal a coherent degradation signature
    beginning ~125 cycles prior to failure (justifying the piecewise RUL cap at 125).
    """
    key_sensors = [
        ("sensor_11", "Static Pressure at HPC Outlet (psia)"),
        ("sensor_4", "Total Temp at LPT Outlet (°R)"),
        ("sensor_12", "Fuel Flow to Ps30 Ratio (pps/psia)"),
    ]

    sample_units = [1, 4, 11, 18, 25, 34, 49, 63, 76, 96]
    fig, axes = plt.subplots(len(key_sensors), 2, figsize=(13.5, 9.5), sharex="col")

    dev_df = dev_df.copy()
    dev_df["cycles_before_failure"] = -dev_df["rul_uncapped"]

    # Compute mean & interquartile band grouped by cycles_before_failure (for RUL <= 250)
    aligned_sub = dev_df[dev_df["rul_uncapped"] <= 250]

    for row_idx, (col, label) in enumerate(key_sensors):
        ax_left = axes[row_idx, 0]
        ax_right = axes[row_idx, 1]

        # Left column: Forward cycle from start (t)
        for uid in sample_units:
            udf = dev_df[dev_df["unit_id"] == uid]
            ax_left.plot(udf["cycle"], udf[col], alpha=0.45, linewidth=1.0)
            ax_left.scatter(udf["cycle"].iloc[-1], udf[col].iloc[-1], color="#d62728", s=16, zorder=4)

        ax_left.set_ylabel(f"{col}\n({label.split('(')[0].strip()})")
        if row_idx == 0:
            ax_left.set_title("A. Forward Time From Start ($t$)\n[Engines Fail at Different Ages: 128–362 Cycles]")
        if row_idx == len(key_sensors) - 1:
            ax_left.set_xlabel("Operating Cycle From Start ($t$)")

        # Right column: Aligned by cycles before failure (-RUL)
        for uid in sample_units:
            udf = aligned_sub[aligned_sub["unit_id"] == uid]
            ax_right.plot(udf["cycles_before_failure"], udf[col], alpha=0.25, linewidth=0.9, color="#7f7f7f")

        agg = aligned_sub.groupby("cycles_before_failure")[col].agg(
            mean="mean",
            q10=lambda x: np.percentile(x, 10),
            q90=lambda x: np.percentile(x, 90),
        )
        ax_right.plot(agg.index, agg["mean"], color="#1f77b4", linewidth=2.2, label="80-Engine Mean Trend")
        ax_right.fill_between(agg.index, agg["q10"], agg["q90"], color="#1f77b4", alpha=0.2, label="10th–90th Percentile")

        # Mark RUL cap = -125 and Alarm horizon w = -30
        ax_right.axvline(-125, color="#2ca02c", linestyle="--", linewidth=1.5, label="RUL Cap (-125 cyc)")
        ax_right.axvline(-30, color="#d62728", linestyle=":", linewidth=1.6, label="Alarm Window (w = -30 cyc)")

        if row_idx == 0:
            ax_right.set_title("B. Aligned by Cycles Before Failure ($t - T_{\\mathrm{fail}}$)\n[Universal Degradation Onset Below 125 Cycles]")
            ax_right.legend(loc="upper left", frameon=True, fontsize=8.5)
        if row_idx == len(key_sensors) - 1:
            ax_right.set_xlabel("Cycles Relative to Failure ($-\\mathrm{RUL}$, Failure = 0)")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_all_14_retained_sensors(
    dev_df: pd.DataFrame,
    retained_sensors: list[str],
    out_path: Path,
) -> None:
    """Plot failure-aligned mean and 10-90% bands for all 14 retained degradation sensors."""
    fig, axes = plt.subplots(4, 4, figsize=(15, 11), sharex=True)
    axes_flat = axes.flatten()

    sub = dev_df[dev_df["rul_uncapped"] <= 220].copy()
    sub["cycles_before_failure"] = -sub["rul_uncapped"]

    for idx, col in enumerate(retained_sensors):
        ax = axes_flat[idx]
        agg = sub.groupby("cycles_before_failure")[col].agg(
            mean="mean",
            q15=lambda x: np.percentile(x, 15),
            q85=lambda x: np.percentile(x, 85),
        )
        corr_val = float(dev_df[col].corr(dev_df["rul"]))
        line_color = "#d62728" if corr_val < 0 else "#1f77b4"

        ax.plot(agg.index, agg["mean"], color=line_color, linewidth=1.8)
        ax.fill_between(agg.index, agg["q15"], agg["q85"], color=line_color, alpha=0.2)
        ax.axvline(-125, color="#2ca02c", linestyle="--", linewidth=1.0, alpha=0.8)
        ax.axvline(-30, color="#ff7f0e", linestyle=":", linewidth=1.2, alpha=0.9)

        short_desc = SENSOR_DESCRIPTIONS.get(col, col).split("(")[0].strip()
        ax.set_title(f"{col}: {short_desc}\n(r = {corr_val:+.2f} vs RUL)", fontsize=8.5)
        if idx >= 10:
            ax.tick_params(labelbottom=True)
            ax.set_xlabel("Cycles to Failure (-RUL)")

    for extra_idx in range(len(retained_sensors), len(axes_flat)):
        axes_flat[extra_idx].axis("off")

    fig.suptitle(
        "All 14 Retained Degradation Sensors Aligned by Cycles Before Failure (80 Dev Engines)\n"
        "Red = Rising with Degradation | Blue = Falling with Degradation | Dashed Green = RUL Cap (-125) | Dotted Orange = Alarm (-30)",
        fontsize=11.5,
        y=1.01,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_correlation_and_pca_health_index(
    dev_df: pd.DataFrame,
    pca_df: pd.DataFrame,
    retained_sensors: list[str],
    pca_info: dict[str, Any],
    out_path: Path,
) -> None:
    """Plot the 14-sensor correlation heatmap alongside the unsupervised PCA Health Index."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))

    # Panel A: Correlation Matrix among 14 retained sensors + capped RUL
    ax_corr = axes[0]
    corr_cols = retained_sensors + ["rul"]
    corr_mat = dev_df[corr_cols].corr().values
    short_labels = [c.replace("sensor_", "S") if c.startswith("sensor_") else "RUL" for c in corr_cols]

    im = ax_corr.imshow(corr_mat, cmap="coolwarm", vmin=-1.0, vmax=1.0, aspect="auto")
    ax_corr.set_xticks(range(len(short_labels)))
    ax_corr.set_yticks(range(len(short_labels)))
    ax_corr.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=8)
    ax_corr.set_yticklabels(short_labels, fontsize=8)
    for i in range(len(short_labels)):
        for j in range(len(short_labels)):
            val = corr_mat[i, j]
            txt_color = "white" if abs(val) > 0.65 else "black"
            ax_corr.text(j, i, f"{val:.2f}", ha="center", va="center", color=txt_color, fontsize=6.0)
    fig.colorbar(im, ax=ax_corr, fraction=0.046, pad=0.04)
    ax_corr.set_title("A. Correlation Matrix: 14 Retained Sensors & Capped RUL")

    # Panel B: Unsupervised PCA Health Index aligned by cycles to failure
    ax_pca = axes[1]
    sub = pca_df[pca_df["rul_uncapped"] <= 240].copy()
    sub["cycles_before_failure"] = -sub["rul_uncapped"]

    for uid in [1, 4, 11, 18, 25, 34, 49, 63, 76, 96]:
        udf = sub[sub["unit_id"] == uid]
        ax_pca.plot(udf["cycles_before_failure"], udf["pca_health_index"], alpha=0.25, linewidth=0.9, color="gray")

    agg = sub.groupby("cycles_before_failure")["pca_health_index"].agg(
        mean="mean",
        q10=lambda x: np.percentile(x, 10),
        q90=lambda x: np.percentile(x, 90),
    )
    ax_pca.plot(agg.index, agg["mean"], color="#8e24aa", linewidth=2.4, label="Mean PC1 Health Index")
    ax_pca.fill_between(agg.index, agg["q10"], agg["q90"], color="#8e24aa", alpha=0.2, label="10th–90th Percentile Band")
    ax_pca.axvline(-125, color="#2ca02c", linestyle="--", linewidth=1.5, label="RUL Cap (-125 cyc)")
    ax_pca.axvline(-30, color="#d62728", linestyle=":", linewidth=1.6, label="Alarm Window (w = -30 cyc)")

    var_pct = pca_info["explained_variance_ratio_pc1"] * 100.0
    corr_rul = pca_info["corr_pc1_with_rul_capped"]
    ax_pca.set_title(
        f"B. Unsupervised 1D PCA Health Index\n"
        f"[PC1 Explains {var_pct:.1f}% Sensor Variance | r = {corr_rul:+.2f} with Capped RUL]"
    )
    ax_pca.set_xlabel("Cycles Relative to Failure ($-\\mathrm{RUL}$, Failure = 0)")
    ax_pca.set_ylabel("Composite Health Index (PC1 Score)")
    ax_pca.legend(loc="lower left", frameon=True)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
