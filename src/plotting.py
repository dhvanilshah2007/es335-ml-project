"""Centralized, clean matplotlib visualizations for C-MAPSS FD001 predictive maintenance.

Generates 8 focused, 1- to 2-panel figures in `results/figures/` without cluttering core
modeling and simulation modules.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve

from src.constants import SENSOR_DESCRIPTIONS
from src.features import build_preprocessing_pipeline

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def plot_lifetime_distribution(dev_df: pd.DataFrame, out_path: Path) -> None:
    """Plot fleet engine lifetime distribution (80 development engines)."""
    lifetimes = dev_df.groupby("unit_id")["max_cycle"].first()

    fig, ax = plt.subplots(figsize=(7.5, 4.6), layout="constrained")
    _, bins, _ = ax.hist(
        lifetimes, bins=16, color="#1f77b4", edgecolor="white", alpha=0.78
    )
    kde = gaussian_kde(lifetimes)
    x_grid = np.linspace(lifetimes.min() - 10, lifetimes.max() + 10, 200)
    ax.plot(x_grid, kde(x_grid) * len(lifetimes) * (bins[1] - bins[0]), color="#0d3b66", linewidth=1.8)

    ax.axvline(lifetimes.min(), color="#d62728", linestyle="--", linewidth=1.5, label=f"Min = {lifetimes.min()} cyc")
    ax.axvline(lifetimes.median(), color="#2ca02c", linestyle="-", linewidth=1.8, label=f"Median = {lifetimes.median():.0f} cyc")
    ax.axvline(lifetimes.mean(), color="#ff7f0e", linestyle=":", linewidth=1.8, label=f"Mean = {lifetimes.mean():.1f} cyc")
    ax.axvline(lifetimes.max(), color="#9467bd", linestyle="--", linewidth=1.5, label=f"Max = {lifetimes.max()} cyc")

    ax.set_title("Engine Lifetime Distribution (80 Development Engines)")
    ax.set_xlabel("Total Operating Cycles Until Failure ($T_{\\mathrm{failure}}$)")
    ax.set_ylabel("Number of Engines")
    ax.legend(loc="upper right", frameon=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_aligned_degradation_trajectories(dev_df: pd.DataFrame, out_path: Path) -> None:
    """Compare forward-time trajectories vs. failure-aligned trajectories for 3 key sensors."""
    key_sensors = [
        ("sensor_11", "Static Pressure at HPC Outlet (psia)"),
        ("sensor_4", "Total Temp at LPT Outlet (°R)"),
        ("sensor_12", "Fuel Flow to Ps30 Ratio (pps/psia)"),
    ]
    sample_units = [1, 4, 11, 18, 25, 34, 49, 63, 76, 96]

    fig, axes = plt.subplots(len(key_sensors), 2, figsize=(13.0, 9.0), sharex="col", layout="constrained")
    work = dev_df.copy()
    work["cycles_before_failure"] = -work["rul_uncapped"]
    aligned_sub = work[work["rul_uncapped"] <= 250]

    for row_idx, (col, label) in enumerate(key_sensors):
        ax_left, ax_right = axes[row_idx, 0], axes[row_idx, 1]

        for uid in sample_units:
            udf = work[work["unit_id"] == uid]
            ax_left.plot(udf["cycle"], udf[col], alpha=0.45, linewidth=1.0)
            ax_left.scatter(udf["cycle"].iloc[-1], udf[col].iloc[-1], color="#d62728", s=16, zorder=4)

        ax_left.set_ylabel(f"{col}\n({label.split('(')[0].strip()})")
        if row_idx == 0:
            ax_left.set_title("A. Forward Time From Start ($t$)")
        if row_idx == len(key_sensors) - 1:
            ax_left.set_xlabel("Operating Cycle From Start ($t$)")

        for uid in sample_units:
            udf = aligned_sub[aligned_sub["unit_id"] == uid]
            ax_right.plot(udf["cycles_before_failure"], udf[col], alpha=0.25, linewidth=0.9, color="#7f7f7f")

        agg = aligned_sub.groupby("cycles_before_failure")[col].agg(
            mean="mean",
            q10=lambda x: np.percentile(x, 10),
            q90=lambda x: np.percentile(x, 90),
        )
        ax_right.plot(agg.index, agg["mean"], color="#1f77b4", linewidth=2.2, label="80-Engine Mean")
        ax_right.fill_between(agg.index, agg["q10"], agg["q90"], color="#1f77b4", alpha=0.2, label="10th–90th Pct")
        ax_right.axvline(-125, color="#2ca02c", linestyle="--", linewidth=1.5, label="RUL Cap (-125)")
        ax_right.axvline(-30, color="#d62728", linestyle=":", linewidth=1.6, label="Alarm Window (-30)")

        if row_idx == 0:
            ax_right.set_title("B. Aligned by Cycles Before Failure ($t - T_{\\mathrm{fail}}$)")
            ax_right.legend(loc="upper left", frameon=True, fontsize=8.2)
        if row_idx == len(key_sensors) - 1:
            ax_right.set_xlabel("Cycles Relative to Failure ($-\\mathrm{RUL}$, Failure = 0)")

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
    """Plot the 14-sensor correlation heatmap alongside the unsupervised 1D PCA Health Index."""
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4), layout="constrained")

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
            ax_corr.text(
                j, i, f"{val:.2f}", ha="center", va="center",
                color="white" if abs(val) > 0.65 else "black", fontsize=6.0,
            )
    fig.colorbar(im, ax=ax_corr, fraction=0.046, pad=0.04)
    ax_corr.set_title("A. Correlation Matrix: 14 Retained Sensors & Capped RUL")

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
    ax_pca.plot(agg.index, agg["mean"], color="#8e24aa", linewidth=2.3, label="Mean PC1 Health Index")
    ax_pca.fill_between(agg.index, agg["q10"], agg["q90"], color="#8e24aa", alpha=0.2, label="10th–90th Pct")
    ax_pca.axvline(-125, color="#2ca02c", linestyle="--", linewidth=1.5, label="RUL Cap (-125)")
    ax_pca.axvline(-30, color="#d62728", linestyle=":", linewidth=1.6, label="Alarm Window (-30)")

    var_pct = pca_info["explained_variance_ratio_pc1"] * 100.0
    corr_rul = pca_info["corr_pc1_with_rul_capped"]
    ax_pca.set_title(
        f"B. Unsupervised 1D PCA Health Index (PC1 Var={var_pct:.1f}%, r={corr_rul:+.2f})"
    )
    ax_pca.set_xlabel("Cycles Relative to Failure ($-\\mathrm{RUL}$, Failure = 0)")
    ax_pca.set_ylabel("Composite Health Index (PC1 Score)")
    ax_pca.legend(loc="lower left", frameon=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_trailing_feature_diagnostics(
    dev_df: pd.DataFrame,
    corr_df: pd.DataFrame,
    out_path: Path,
    window_k: int = 30,
    sample_unit_id: int = 1,
) -> None:
    """Plot 2-panel feature figure: trailing mean/slope on Engine #1 and RUL correlation gain."""
    pipe = build_preprocessing_pipeline(window_k=window_k, feature_mode="trailing_trends", scale=False)
    feat_df = pipe.fit_transform(dev_df)

    mask_u = dev_df["unit_id"] == sample_unit_id
    u_raw = dev_df.loc[mask_u].reset_index(drop=True)
    u_feat = feat_df.loc[mask_u].reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.9), layout="constrained")

    # Panel A: Engine #1 raw vs trailing mean and slope (k=30)
    ax1 = axes[0]
    s_col = "sensor_11"
    l1 = ax1.plot(u_raw["cycle"], u_raw[s_col], color="#b0b0b0", alpha=0.65, linewidth=1.0, label="Raw Reading")
    l2 = ax1.plot(
        u_raw["cycle"], u_feat[f"{s_col}_mean_k{window_k}"],
        color="#1f77b4", linewidth=2.2, label=f"Trailing Mean (k={window_k})"
    )
    ax1.axvline(u_raw["max_cycle"].iloc[0] - 30, color="#ff7f0e", linestyle=":", linewidth=1.4, label="Alarm Window (-30)")
    ax1.set_title(f"A. Trailing Smoothing & Drift Slope (Engine #{sample_unit_id}, {s_col})")
    ax1.set_xlabel("Operating Cycle ($t$)")
    ax1.set_ylabel("HPC Static Pressure (psia)", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")

    ax1_twin = ax1.twinx()
    ax1_twin.spines["top"].set_visible(False)
    l3 = ax1_twin.plot(
        u_raw["cycle"], u_feat[f"{s_col}_slope_k{window_k}"],
        color="#8e24aa", linewidth=1.9, linestyle="--", label=f"Trailing Drift Slope (k={window_k})"
    )
    ax1_twin.set_ylabel("Drift Slope ($\\Delta$ psia / cycle)", color="#8e24aa")
    ax1_twin.tick_params(axis="y", labelcolor="#8e24aa")

    lines = l1 + l2 + l3
    ax1.legend(lines, [l.get_label() for l in lines], loc="upper left", frameon=True, fontsize=8)

    # Panel B: Raw vs Trailing Mean & Slope correlation |r| across 14 retained sensors
    ax2 = axes[1]
    y_pos = np.arange(len(corr_df))
    h = 0.27
    ax2.barh(y_pos + h, corr_df["abs_corr_raw"], height=h, color="#9ecae1", label="Raw Instantaneous |r|")
    ax2.barh(y_pos, corr_df["abs_corr_mean"], height=h, color="#1f77b4", label=f"Trailing Mean k={window_k} |r|")
    ax2.barh(y_pos - h, corr_df["abs_corr_slope"], height=h, color="#8e24aa", label=f"Trailing Slope k={window_k} |r|")
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([s.replace("sensor_", "S") for s in corr_df["sensor"]], fontsize=8.5)
    ax2.invert_yaxis()
    ax2.set_xlim(0.2, 0.92)
    ax2.set_xlabel("Absolute Pearson Correlation ($|r|$) with Capped RUL")
    ax2.set_title(f"B. Signal Denoising Gain Across 14 Sensors (k={window_k})")
    ax2.legend(loc="lower right", frameon=True, fontsize=8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_cv_baseline_ladder(
    reg_df: pd.DataFrame,
    clf_df: pd.DataFrame,
    out_path: Path,
) -> None:
    """Plot 2-panel CV ladder progression: 5-fold CV RMSE and OOF Policy Cost (c_fail=100)."""
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0), layout="constrained")

    short_reg_names = {
        "rung0_mean_rul": "R0: Mean RUL (Floor)",
        "rung2_age_linear": "R2: Age-Only Linear",
        "rung3_instant_ridge": "R3: Raw Sensors Ridge",
        "rung3_instant_lasso": "R3: Raw Sensors Lasso",
        "rung4_trends_ridge": "R4: Trailing Trends Ridge",
        "rung4_trends_lasso": "R4: Trailing Trends Lasso",
        "rung5_trends_rf": "R5: Trailing Trends RF",
        "rung5_trends_hgb": "R5: Trailing Trends HistGBR",
    }
    reg_colors = ["#b0b0b0", "#fdb863", "#80b1d3", "#80b1d3", "#3182bd", "#3182bd", "#6a51a3", "#2ca02c"]

    ax1 = axes[0]
    y_reg = np.arange(len(reg_df))
    bars1 = ax1.barh(y_reg, reg_df["rmse_all"], color=reg_colors[: len(reg_df)], height=0.65)
    ax1.set_yticks(y_reg)
    ax1.set_yticklabels([short_reg_names.get(m, m) for m in reg_df["model_id"]], fontsize=8.8)
    ax1.invert_yaxis()
    ax1.set_xlabel("5-Fold GroupKFold RMSE (Cycles, Lower is Better)")
    ax1.set_title("A. RUL Regression RMSE Across Baseline Ladder")
    ax1.set_xlim(0, max(reg_df["rmse_all"]) * 1.15)
    for bar, val in zip(bars1, reg_df["rmse_all"]):
        ax1.text(val + 0.6, bar.get_y() + bar.get_height() / 2, f"{val:.2f}", va="center", fontsize=8.5)

    short_clf_names = {
        "rung0_prior_clf": "R0: Never Alarm (Run-to-Fail)",
        "rung1_fixed_schedule": "R1: Fixed Schedule (K*)",
        "rung2_age_logreg": "R2: Age-Only Logistic",
        "rung3_instant_logreg": "R3: Raw Sensors Logistic",
        "rung4_trends_logreg": "R4: Trailing Trends Logistic",
        "rung5_trends_rf_clf": "R5: Trailing Trends RF Clf",
        "rung5_trends_hgb_clf": "R5: Trailing Trends HistGBC",
    }
    clf_colors = ["#b0b0b0", "#d62728", "#fdb863", "#80b1d3", "#3182bd", "#6a51a3", "#2ca02c"]

    ax2 = axes[1]
    y_clf = np.arange(len(clf_df))
    bars2 = ax2.barh(y_clf, clf_df["oof_policy_mean_cost_c100"], color=clf_colors[: len(clf_df)], height=0.65)
    ax2.set_yticks(y_clf)
    ax2.set_yticklabels([short_clf_names.get(m, m) for m in clf_df["model_id"]], fontsize=8.8)
    ax2.invert_yaxis()
    ax2.set_xlabel("Mean Cost / Engine ($c_{\\mathrm{fail}}=100, c_{\\mathrm{waste}}=1$)")
    ax2.set_title("B. Alarm Policy Cost Across Baseline Ladder (5-Fold OOF)")
    ax2.set_xlim(0, 115)
    for bar, val in zip(bars2, clf_df["oof_policy_mean_cost_c100"]):
        ax2.text(val + 1.5, bar.get_y() + bar.get_height() / 2, f"{val:.1f}", va="center", fontsize=8.5)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_cv_pr_and_calibration(
    dev_df: pd.DataFrame,
    oof_probs: dict[str, np.ndarray],
    out_path: Path,
) -> None:
    """Plot Precision-Recall and Probability Calibration curves for Rungs 2–5."""
    y_true = dev_df["alarm"].to_numpy(dtype=int)
    base_rate = float(np.mean(y_true))

    models_to_plot = [
        ("rung2_age_logreg", "R2: Age-Only Logistic", "#ff7f0e", ":"),
        ("rung3_instant_logreg", "R3: Raw Sensors Logistic", "#1f77b4", "--"),
        ("rung4_trends_logreg", "R4: Trailing Trends Logistic", "#8e24aa", "-."),
        ("rung5_trends_rf_clf", "R5: Trailing Trends Random Forest", "#d62728", "--"),
        ("rung5_trends_hgb_clf", "R5: Trailing Trends HistGBC", "#2ca02c", "-"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0), layout="constrained")

    ax_pr = axes[0]
    ax_pr.axhline(base_rate, color="gray", linestyle=":", linewidth=1.2, label=f"Random Baseline (Pos={base_rate:.3f})")
    for mid, label, color, ls in models_to_plot:
        probs = oof_probs[mid]
        prec, rec, _ = precision_recall_curve(y_true, probs)
        ap = float(average_precision_score(y_true, probs))
        ax_pr.plot(rec, prec, color=color, linestyle=ls, linewidth=2.0, label=f"{label} (PR-AUC={ap:.3f})")

    ax_pr.set_xlabel("Recall (True Near-Failure Cycles Detected)")
    ax_pr.set_ylabel("Precision (Alarms Within w=30 Cycles)")
    ax_pr.set_title("A. 5-Fold Out-of-Fold Precision-Recall Curves (w=30)")
    ax_pr.set_xlim(0.0, 1.02)
    ax_pr.set_ylim(0.0, 1.03)
    ax_pr.legend(loc="lower left", frameon=True, fontsize=8.2)

    ax_cal = axes[1]
    ax_cal.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1.2, label="Perfectly Calibrated")
    for mid, label, color, ls in models_to_plot:
        probs = oof_probs[mid]
        brier = float(brier_score_loss(y_true, probs))
        frac_pos, mean_pred = calibration_curve(y_true, probs, n_bins=10, strategy="uniform")
        ax_cal.plot(
            mean_pred, frac_pos, color=color, linestyle=ls, marker="o", markersize=4, linewidth=1.8,
            label=f"{label} (Brier={brier:.4f})",
        )

    ax_cal.set_xlabel("Mean Predicted Probability $P(\\mathrm{RUL} \\leq 30)$")
    ax_cal.set_ylabel("Empirical Fraction of Positives")
    ax_cal.set_title("B. Probability Calibration Curves (Reliability Diagram)")
    ax_cal.set_xlim(-0.02, 1.02)
    ax_cal.set_ylim(-0.02, 1.02)
    ax_cal.legend(loc="upper left", frameon=True, fontsize=8.2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_cost_curve_vs_fixed_schedule(
    sweep_df: pd.DataFrame,
    breakeven_info: dict[str, float],
    boot_samples: dict[str, np.ndarray],
    out_path: Path,
) -> None:
    """Plot 2-panel held-out cost evaluation: Cost Ratio Sweep and Bootstrap Savings CI."""
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0), layout="constrained")

    ax1 = axes[0]
    ratios = sweep_df["cost_ratio"].to_numpy()
    ax1.plot(ratios, sweep_df["run_to_failure_cost"], linestyle=":", color="#6b7280", linewidth=1.6, label="R0: Run-to-Failure")
    ax1.plot(ratios, sweep_df["fixed_test_cost"], marker="s", markersize=5, color="#d97706", linewidth=2.2, label="R1: Locked Fixed Schedule (K*_dev)")
    ax1.plot(ratios, sweep_df["age_logreg_test_cost"], linestyle="--", color="#9333ea", linewidth=1.7, label="R2: Age-Only Logistic")
    ax1.plot(ratios, sweep_df["champ_clf_test_cost"], marker="o", markersize=5, color="#2563eb", linewidth=2.2, label="R4: Trailing Trends Logistic")
    ax1.plot(ratios, sweep_df["tree_clf_test_cost"], marker="^", markersize=5, color="#0891b2", linewidth=1.9, linestyle="-.", label="R5: Trailing Trends HistGBC")
    ax1.plot(ratios, sweep_df["champ_reg_test_cost"], marker="D", markersize=5, color="#16a34a", linewidth=2.3, label="R5: HistGBR RUL Trigger")

    be_clf = breakeven_info.get("champ_clf_breakeven_test_ratio", np.nan)
    if not np.isnan(be_clf):
        ax1.axvline(be_clf, color="#2563eb", linestyle="--", alpha=0.55, linewidth=1.3)
        ax1.text(be_clf + 5, 50.0, f"Logistic break-even\n~{be_clf:.1f}:1", fontsize=8.5, color="#1d4ed8", fontweight="bold")

    ax1.set_title("A. Held-Out Test Cost Across Cost Ratios (20 Unseen Engines)")
    ax1.set_xlabel("Failure-to-Waste Cost Ratio ($c_{\\mathrm{fail}} / c_{\\mathrm{waste}}$)")
    ax1.set_ylabel("Mean Maintenance Cost per Engine")
    ax1.set_xlim(min(ratios), max(ratios))
    ax1.set_ylim(0, 95)
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="upper left", frameon=True, fontsize=8.0)

    ax2 = axes[1]
    sav_clf = boot_samples["paired_savings_champ_clf_vs_fixed"]
    sav_reg = boot_samples["paired_savings_champ_reg_vs_fixed"]
    bins = np.linspace(min(np.min(sav_clf), np.min(sav_reg)) - 2, max(np.max(sav_clf), np.max(sav_reg)) + 2, 26)

    ax2.hist(
        sav_clf, bins=bins, alpha=0.55, color="#2563eb", edgecolor="white",
        label=f"Logistic Classifier Savings\nMean: +{np.mean(sav_clf):.1f} [{np.percentile(sav_clf, 2.5):.1f}, {np.percentile(sav_clf, 97.5):.1f}]",
    )
    ax2.hist(
        sav_reg, bins=bins, alpha=0.55, color="#16a34a", edgecolor="white",
        label=f"HistGBR Trigger Savings\nMean: +{np.mean(sav_reg):.1f} [{np.percentile(sav_reg, 2.5):.1f}, {np.percentile(sav_reg, 97.5):.1f}]",
    )
    ax2.axvline(0.0, color="#dc2626", linestyle="--", linewidth=1.8, label="Zero Savings (Fixed Parity)")
    ax2.set_title("B. Engine-Level Bootstrap Cost Savings at $c_{\\mathrm{fail}}=100$ (B=1,000)")
    ax2.set_xlabel("Paired Cost Reduction per Engine (Fixed minus Model)")
    ax2.set_ylabel("Bootstrap Resamples")
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="upper left", frameon=True, fontsize=8.2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_leakage_comparison(
    leakage_df: pd.DataFrame,
    scatter_payload: dict[str, Any],
    out_path: Path,
) -> None:
    """Plot 2-panel leakage comparison: Proper Engine Split vs. Random Row Split."""
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0), layout="constrained")

    # Panel A: Grouped bar chart of Test RMSE across the 3 actual ladder models
    ax1 = axes[0]
    labels = [lbl.replace(" (", "\n(") for lbl in leakage_df["model_label"]]
    x = np.arange(len(labels))
    width = 0.35

    proper_vals = leakage_df["proper_heldout_engines_test_rmse"].to_numpy()
    leaky_vals = leakage_df["leaky_random_row_split_test_rmse"].to_numpy()

    bars1 = ax1.bar(x - width / 2, proper_vals, width, color="#1e40af", label="Proper Engine Split (20 Unseen Engines)")
    bars2 = ax1.bar(x + width / 2, leaky_vals, width, color="#dc2626", alpha=0.85, label="Flawed Random Row Split (80/20 Rows)")

    for b1, b2 in zip(bars1, bars2):
        ax1.text(b1.get_x() + b1.get_width() / 2, b1.get_height() + 0.25, f"{b1.get_height():.2f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#1e3a8a")
        ax1.text(b2.get_x() + b2.get_width() / 2, b2.get_height() + 0.25, f"{b2.get_height():.2f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#991b1b")

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8.8)
    ax1.set_ylabel("Test RMSE (Cycles, Lower is Better)")
    ax1.set_ylim(0, max(proper_vals) + 3.5)
    ax1.set_title("A. Artificial RMSE Drop Under Random Row Splitting")
    ax1.grid(True, axis="y", alpha=0.25)
    ax1.legend(loc="upper right", frameon=True, fontsize=8.2)

    # Panel B: Engine 21 trajectory showing interleaved train/test cycles under random row split
    ax2 = axes[1]
    traj = scatter_payload["trajectory_example"]
    uid = traj["unit_id"]
    cycles = np.asarray(traj["cycles"])
    true_rul = np.asarray(traj["true_rul"])
    in_tr = np.asarray(traj["in_leaky_train_mask"], dtype=bool)
    pred_proper = np.asarray(traj["pred_proper_unseen_engine"])
    pred_leaky = np.asarray(traj["pred_leaky_row_split"])

    ax2.plot(cycles, true_rul, "k-", linewidth=2.0, label="True Capped RUL")
    ax2.plot(cycles, pred_proper, color="#1e40af", linewidth=1.8, alpha=0.9, label="Proper Unseen-Engine Prediction")
    ax2.plot(cycles[~in_tr], pred_leaky[~in_tr], "o", color="#dc2626", markersize=5, label="Leaky Test Rows (80% Neighbors in Train)")
    ax2.scatter(cycles[in_tr], true_rul[in_tr], color="#9ca3af", s=10, alpha=0.6, label="Same-Engine Cycles Leaked into Train")

    ax2.set_title(f"B. Engine {uid} Trajectory: Why Row Splits Memorize")
    ax2.set_xlabel("Operational Cycle ($t$)")
    ax2.set_ylabel("Capped RUL (Cycles)")
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="lower left", frameon=True, fontsize=8.0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
