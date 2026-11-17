"""Export lightweight, self-consistent JSON data and static fallback PNGs for docs/.

Reuses the repository's exact splits, pipelines, models, and saved evaluation
outputs so every number and chart on the static website traces directly to
the reproducible Python pipeline.

Usage:
    python scripts/export_web_data.py
"""

from __future__ import annotations

import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.cost_simulation import (
    evaluate_locked_classification_ladder_on_test,
    evaluate_locked_regression_ladder_on_test,
    run_row_vs_engine_leakage_experiment,
)
from src.data import (
    load_config,
    load_development_data,
    load_heldout_test_data,
)
from src.eda import fit_pca_health_index, screen_sensors_and_settings
from src.features import build_preprocessing_pipeline
from src.models import (
    run_group_cv_classification_ladder,
    run_group_cv_regression_ladder,
)

PALETTE = {
    "bg": "#F7F8FA",
    "ink": "#1D2733",
    "muted": "#4A5568",
    "dev": "#4F6788",
    "test": "#C98A00",
    "model": "#0B7285",
    "fixed": "#7A828E",
    "fail": "#B93A2B",
    "grid": "#D8DEE9",
}


def _sanitize_json(obj: Any) -> Any:
    """Recursively replace NaN/Inf with None and convert numpy scalars for strict RFC 8259 JSON."""
    if isinstance(obj, dict):
        return {str(k): _sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_json(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        if math.isnan(val) or math.isinf(val):
            return None
        return round(val, 4)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return [_sanitize_json(v) for v in obj.tolist()]
    return obj


def _summarize_threshold_sweep(
    df: pd.DataFrame,
    scores: np.ndarray,
    thresholds: list[float],
    score_is_rul: bool,
    alarm_w: int = 30,
) -> dict[str, Any]:
    """Precompute exact failure counts, wasted cycles, and per-engine first alarms over thresholds."""
    unit_ids = np.sort(df["unit_id"].unique())
    engine_groups = []
    for uid in unit_ids:
        mask = df["unit_id"].to_numpy() == uid
        cycles = df.loc[mask, "cycle"].to_numpy(dtype=int)
        rul_unc = df.loc[mask, "rul_uncapped"].to_numpy(dtype=float)
        sc = scores[mask]
        te = int(df.loc[mask, "max_cycle"].iloc[0])
        engine_groups.append((int(uid), te, cycles, rul_unc, sc))

    failures_list: list[int] = []
    wasted_list: list[float] = []
    premature_list: list[int] = []
    per_engine_alarms: dict[int, list[int | None]] = {uid: [] for uid, *_ in engine_groups}

    for th in thresholds:
        n_fail = 0
        wasted_sum = 0.0
        n_premature = 0
        for uid, te, cycles, rul_unc, sc in engine_groups:
            valid = rul_unc > 0
            trig = (sc <= th) & valid if score_is_rul else (sc >= th) & valid
            if np.any(trig):
                idx = int(np.flatnonzero(trig)[0])
                alm_cyc = int(cycles[idx])
                wasted = float(rul_unc[idx])
                wasted_sum += wasted
                if wasted > alarm_w:
                    n_premature += 1
                per_engine_alarms[uid].append(alm_cyc)
            else:
                n_fail += 1
                per_engine_alarms[uid].append(None)

        failures_list.append(int(n_fail))
        wasted_list.append(round(wasted_sum, 2))
        premature_list.append(int(n_premature))

    return {
        "thresholds": thresholds,
        "failures": failures_list,
        "wasted_sum": wasted_list,
        "premature": premature_list,
        "per_engine_alarms": [
            {
                "unit_id": uid,
                "true_lifetime_Te": te,
                "alarm_cycles": per_engine_alarms[uid],
            }
            for uid, te, *_ in engine_groups
        ],
    }


def _summarize_fixed_k_sweep(df: pd.DataFrame, k_grid: list[int], alarm_w: int = 30) -> dict[str, Any]:
    """Precompute exact failures and wasted cycles across fixed-schedule K values."""
    lifetimes = df.groupby("unit_id")["max_cycle"].first().to_numpy(dtype=float)
    failures_list: list[int] = []
    wasted_list: list[float] = []
    premature_list: list[int] = []
    for k in k_grid:
        failed_mask = lifetimes <= k
        wasted = np.where(failed_mask, 0.0, lifetimes - k)
        failures_list.append(int(np.sum(failed_mask)))
        wasted_list.append(round(float(np.sum(wasted)), 2))
        premature_list.append(int(np.sum((~failed_mask) & (wasted > alarm_w))))
    return {
        "k_values": k_grid,
        "failures": failures_list,
        "wasted_sum": wasted_list,
        "premature": premature_list,
    }


def _build_aligned_sensor_summary(
    dev_df: pd.DataFrame,
    pca_df: pd.DataFrame,
    pca_info: dict[str, Any],
) -> dict[str, Any]:
    """Compute binned failure-aligned sensor and PC1 health-index trajectories on 80 dev engines."""
    work = dev_df[["unit_id", "cycle", "max_cycle", "sensor_11", "sensor_4", "sensor_12", "sensor_7"]].copy()
    work["t_to_fail"] = work["cycle"] - work["max_cycle"]
    work["pc1"] = pca_df["pca_health_index"].to_numpy()
    sub = work[work["t_to_fail"] >= -200].copy()
    sub["bin"] = (sub["t_to_fail"] // 5) * 5

    bins_sorted = sorted(sub["bin"].unique().tolist())
    series: dict[str, dict[str, list[float]]] = {}
    for col in ("sensor_11", "sensor_4", "sensor_12", "sensor_7", "pc1"):
        grp = sub.groupby("bin")[col]
        means = [round(float(grp.mean().loc[b]), 4) for b in bins_sorted]
        stds = [round(float(grp.std(ddof=0).loc[b]), 4) for b in bins_sorted]
        series[col] = {"mean": means, "std": stds}

    return {
        "t_to_fail_bins": [int(b) for b in bins_sorted],
        "series": series,
        "pca_info": pca_info,
    }


def _build_pr_and_calibration(
    dev_df: pd.DataFrame,
    oof_probs: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Extract compact PR curves and reliability calibration curves from dev OOF probabilities."""
    y_true = dev_df["alarm"].to_numpy(dtype=int)
    model_ids = [
        "rung2_age_logreg",
        "rung3_instant_logreg",
        "rung4_trends_logreg",
        "rung5_trends_hgb_clf",
    ]
    out: dict[str, Any] = {"base_rate": round(float(y_true.mean()), 4), "models": {}}

    recall_targets = np.linspace(0.0, 1.0, 41)
    for mid in model_ids:
        probs = oof_probs[mid]
        prec, rec, _ = precision_recall_curve(y_true, probs)
        # Sort by ascending recall for interpolation
        order = np.argsort(rec)
        rec_s = rec[order]
        prec_s = prec[order]
        prec_interp = np.interp(recall_targets, rec_s, prec_s)

        prob_true, prob_pred = calibration_curve(y_true, probs, n_bins=10, strategy="uniform")
        out["models"][mid] = {
            "pr_recall": [round(float(x), 3) for x in recall_targets],
            "pr_precision": [round(float(x), 3) for x in prec_interp],
            "cal_mean_pred": [round(float(x), 3) for x in prob_pred],
            "cal_frac_pos": [round(float(x), 3) for x in prob_true],
        }
    return out


def _downsample_scatter(y_true: np.ndarray, y_pred: np.ndarray, max_pts: int = 320) -> dict[str, list[float]]:
    """Deterministically stride scatter points while covering the full RUL range."""
    order = np.argsort(y_true, kind="mergesort")
    step = max(1, len(order) // max_pts)
    idx = order[::step]
    return {
        "y_true": [round(float(v), 2) for v in y_true[idx]],
        "y_pred": [round(float(v), 2) for v in y_pred[idx]],
    }


def _render_hero_fallback_png(hero: dict[str, Any], out_path: Path) -> None:
    """Render static PNG fallback for the Opening block hero trace (Engine #24)."""
    fig, ax1 = plt.subplots(figsize=(7.6, 3.8), dpi=160)
    fig.patch.set_facecolor(PALETTE["bg"])
    ax1.set_facecolor("#FFFFFF")

    cyc = np.asarray(hero["cycles"], dtype=float)
    s11_z = np.asarray(hero["sensor_11_mean_k30_z"], dtype=float)
    prob = np.asarray(hero["prob_logreg"], dtype=float)

    ax1.plot(cyc, s11_z, color=PALETTE["ink"], lw=2.2, label="Trailing HPC static pressure (Sensor 11, z-score)")
    ax1.axvline(130, color=PALETTE["model"], lw=1.8, ls="-", label="Logistic alarm @130 (17 cycles left)")
    ax1.axvline(140, color=PALETTE["test"], lw=1.6, ls="--", label="HistGBR alarm @140 (7 cycles left)")
    ax1.axvline(147, color=PALETTE["fail"], lw=2.0, ls="-", label="True failure @147 (Fixed K*=179 misses)")

    ax1.set_xlim(1, 155)
    ax1.set_xlabel("Operating cycle t", color=PALETTE["ink"], fontsize=9.5)
    ax1.set_ylabel("Standardized Sensor 11 (k=30)", color=PALETTE["ink"], fontsize=9.5)
    ax1.grid(axis="y", color=PALETTE["grid"], lw=0.6)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    ax2 = ax1.twinx()
    ax2.plot(cyc, prob, color=PALETTE["model"], lw=1.8, alpha=0.85, ls=":", label="Alarm probability P(RUL<=30)")
    ax2.set_ylim(-0.04, 1.06)
    ax2.set_ylabel("Alarm probability", color=PALETTE["model"], fontsize=9.5)
    ax2.spines["top"].set_visible(False)

    ax1.set_title("Held-out test Engine #24 (T_fail = 147 cycles): early warning vs. fixed schedule K*=179",
                  fontsize=10.5, loc="left", color=PALETTE["ink"], pad=10)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def _render_split_architecture_fallback_png(out_path: Path) -> None:
    """Render static PNG fallback for Figure 4 (Engine-level split architecture diagram)."""
    fig, ax = plt.subplots(figsize=(8.6, 3.5), dpi=160)
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 46)
    ax.axis("off")

    def draw_box(x: float, y: float, w: float, h: float, title: str, body: str, edge: str, fill: str = "#FFFFFF") -> None:
        rect = patches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.4,rounding_size=0.8",
            linewidth=1.6, edgecolor=edge, facecolor=fill
        )
        ax.add_patch(rect)
        ax.text(x + 1.5, y + h - 3.2, title, fontsize=9.5, fontweight="bold", color=edge, va="top")
        ax.text(x + 1.5, y + h - 7.5, body, fontsize=8.3, color=PALETTE["ink"], va="top", linespacing=1.35)

    draw_box(
        2, 24, 27, 17,
        "train_FD001.txt",
        "100 run-to-failure engines\n20,631 rows total\nDeterministic split (seed=42)",
        PALETTE["ink"],
    )
    draw_box(
        35, 24, 32, 17,
        "80 Development Engines",
        "16,340 rows | 5-fold GroupKFold by unit_id\nSensor screening, pipeline fit,\nhyperparameters & thresholds (K*, p*, tau*)",
        PALETTE["dev"],
        "#F0F4F8",
    )
    draw_box(
        72, 24, 26, 17,
        "20 Held-Out Test Engines",
        "4,291 rows | Sealed until Phase 5\nEvaluated ONCE with locked rules\n& 1,000 engine bootstrap CIs",
        PALETTE["test"],
        "#FDF8EE",
    )
    draw_box(
        35, 3, 63, 13,
        "Secondary Check: Official test_FD001.txt (100 Truncated Engines, 13,096 rows)",
        "Models trained on all 80 development engines checked once at final cutoff cycle against RUL_FD001.txt.\nNo policy simulation possible because trajectories end prior to failure.",
        PALETTE["fixed"],
        "#FFFFFF",
    )

    ax.annotate("", xy=(34.5, 32.5), xytext=(29.5, 32.5), arrowprops=dict(arrowstyle="->", color=PALETTE["dev"], lw=1.8))
    ax.annotate("", xy=(71.5, 32.5), xytext=(67.5, 32.5), arrowprops=dict(arrowstyle="->", color=PALETTE["test"], lw=1.8, ls="--"))
    ax.annotate("", xy=(66.5, 16.5), xytext=(51.0, 23.5), arrowprops=dict(arrowstyle="->", color=PALETTE["fixed"], lw=1.4, ls=":"))

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> int:
    cfg = load_config()
    rul_cap = int(cfg["framing"]["rul_cap"])
    alarm_w = int(cfg["framing"]["alarm_w"])
    window_k = int(cfg["features"]["window_k"])
    cv_folds = int(cfg["split"]["cv_folds"])
    c_waste = float(cfg["cost_simulation"]["c_waste"])
    default_c_fail = float(cfg["cost_simulation"]["default_c_fail"])
    seed = int(cfg["project"]["random_seed"])

    docs_dir = REPO_ROOT / "docs"
    docs_data_dir = docs_dir / "data"
    docs_fig_dir = docs_dir / "figures"
    docs_data_dir.mkdir(parents=True, exist_ok=True)
    docs_fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. Copy the 9 static pipeline PNG figures as fallbacks
    src_fig_dir = REPO_ROOT / cfg["paths"]["results_figures_dir"]
    for png in sorted(src_fig_dir.glob("*.png")):
        shutil.copy2(png, docs_fig_dir / png.name)

    # 2. Load saved JSON & CSV artifacts
    with open(REPO_ROOT / "data" / "splits" / "split_summary.json", "r", encoding="utf-8") as f:
        split_summary = json.load(f)
    with open(REPO_ROOT / "data" / "processed" / "feature_metadata.json", "r", encoding="utf-8") as f:
        feature_metadata = json.load(f)
    with open(REPO_ROOT / "data" / "processed" / "test_evaluation_summary.json", "r", encoding="utf-8") as f:
        test_summary = json.load(f)

    eda_screening = pd.read_csv(REPO_ROOT / "results" / "metrics" / "eda_sensor_screening.csv").to_dict(orient="records")
    cv_reg_df = pd.read_csv(REPO_ROOT / "results" / "metrics" / "cv_regression_ladder.csv")
    cv_clf_df = pd.read_csv(REPO_ROOT / "results" / "metrics" / "cv_classification_ladder.csv")
    feature_attr = pd.read_csv(REPO_ROOT / "results" / "metrics" / "cv_feature_attribution.csv").to_dict(orient="records")

    # 3. Load dev and held-out test data for curve extraction and threshold grids
    dev_df = load_development_data(rul_cap=rul_cap, alarm_w=alarm_w)
    test_df = load_heldout_test_data(rul_cap=rul_cap, alarm_w=alarm_w)

    dev_lifetimes = sorted([int(x) for x in dev_df.groupby("unit_id")["max_cycle"].first().tolist()])
    test_lifetimes = sorted([int(x) for x in test_df.groupby("unit_id")["max_cycle"].first().tolist()])

    _, retained_sensors, _ = screen_sensors_and_settings(dev_df)
    pca_df, pca_info = fit_pca_health_index(dev_df, retained_sensors)
    aligned_summary = _build_aligned_sensor_summary(dev_df, pca_df, pca_info)

    # Sample trajectory for Figure 3 (first development engine, sensor_11 raw vs trailing mean & slope)
    prep_unscaled = build_preprocessing_pipeline(window_k=window_k, feature_mode="trailing_trends", scale=False)
    X_dev_trends = prep_unscaled.fit_transform(dev_df)
    sample_dev_uid = int(dev_df["unit_id"].iloc[0])
    dev_u_mask = (dev_df["unit_id"] == sample_dev_uid).to_numpy()
    sample_dev_engine = {
        "unit_id": sample_dev_uid,
        "max_cycle": int(dev_df.loc[dev_u_mask, "max_cycle"].iloc[0]),
        "cycles": dev_df.loc[dev_u_mask, "cycle"].astype(int).tolist(),
        "sensor_11_raw": [round(float(x), 4) for x in dev_df.loc[dev_u_mask, "sensor_11"]],
        "sensor_11_mean_k30": [round(float(x), 4) for x in X_dev_trends.loc[dev_u_mask, "sensor_11_mean_k30"]],
        "sensor_11_slope_k30": [round(float(x), 6) for x in X_dev_trends.loc[dev_u_mask, "sensor_11_slope_k30"]],
    }

    # Hero engine #24 (shortest-lived held-out test engine) normalized sensor_11_mean_k30
    prep_scaled = build_preprocessing_pipeline(window_k=window_k, feature_mode="trailing_trends", scale=True)
    prep_scaled.fit(dev_df)
    X_test_scaled = prep_scaled.transform(test_df)
    hero_mask = (test_df["unit_id"] == 24).to_numpy()
    hero_cs = test_summary["error_case_studies"]["case_studies"]["shortest_lived"]
    hero_engine = {
        "unit_id": 24,
        "max_cycle": int(hero_cs["max_cycle"]),
        "cycles": hero_cs["cycles"],
        "true_rul": hero_cs["true_rul"],
        "pred_hgb": hero_cs["pred_hgb"],
        "prob_logreg": hero_cs["prob_logreg"],
        "sensor_11_mean_k30_z": [round(float(x), 3) for x in X_test_scaled.loc[hero_mask, "sensor_11_mean_k30"]],
        "sensor_4_mean_k30_z": [round(float(x), 3) for x in X_test_scaled.loc[hero_mask, "sensor_4_mean_k30"]],
        "fixed_k_dev": int(hero_cs["fixed_k_dev"]),
        "logreg_alarm_cycle": hero_cs["logreg_alarm_cycle"],
        "hgb_alarm_cycle": hero_cs["hgb_alarm_cycle"],
    }
    _render_hero_fallback_png(hero_engine, docs_fig_dir / "hero_engine_24.png")
    _render_split_architecture_fallback_png(docs_fig_dir / "split_architecture.png")

    # 4. Run CV & held-out predictions to export PR/calibration curves and threshold explorer grids
    cv_reg_live, oof_preds = run_group_cv_regression_ladder(
        dev_df, cv_folds=cv_folds, window_k=window_k, rul_cap=rul_cap,
        c_fail=default_c_fail, c_waste=c_waste, alarm_w=alarm_w,
    )
    cv_clf_live, oof_probs = run_group_cv_classification_ladder(
        dev_df, cv_folds=cv_folds, window_k=window_k,
        c_fail=default_c_fail, c_waste=c_waste, alarm_w=alarm_w,
    )
    _, test_preds, _ = evaluate_locked_regression_ladder_on_test(
        dev_df=dev_df, test_df=test_df, cv_reg_df=cv_reg_live,
        window_k=window_k, rul_cap=rul_cap, c_fail=default_c_fail, c_waste=c_waste, alarm_w=alarm_w,
    )
    _, test_probs, _ = evaluate_locked_classification_ladder_on_test(
        dev_df=dev_df, test_df=test_df, cv_clf_df=cv_clf_live,
        window_k=window_k, c_fail=default_c_fail, c_waste=c_waste, alarm_w=alarm_w,
    )

    pr_calibration = _build_pr_and_calibration(dev_df, oof_probs)

    prob_grid = [round(float(x), 2) for x in np.linspace(0.05, 0.95, 91)] + [0.97, 0.98, 0.99, 1.0]
    tau_grid = [round(float(x), 1) for x in np.linspace(8.0, 50.0, 85)]
    k_grid = list(range(120, 365))

    explorer_models = {}
    for mid, label, is_rul, grid in [
        ("rung4_trends_logreg", "Trailing Trends + Logistic Regression (Rung 4)", False, prob_grid),
        ("rung5_trends_hgb_clf", "Trailing Trends + HistGB Classifier (Rung 5)", False, prob_grid),
        ("rung2_age_logreg", "Age-Only Logistic Regression (Rung 2)", False, prob_grid),
        ("rung5_trends_hgb", "Trailing Trends + HistGB Regressor (Rung 5)", True, tau_grid),
    ]:
        dev_scores = oof_preds[mid] if is_rul else oof_probs[mid]
        test_scores = test_preds[mid] if is_rul else test_probs[mid]
        dev_sw = _summarize_threshold_sweep(dev_df, dev_scores, grid, score_is_rul=is_rul, alarm_w=alarm_w)
        test_sw = _summarize_threshold_sweep(test_df, test_scores, grid, score_is_rul=is_rul, alarm_w=alarm_w)
        explorer_models[mid] = {
            "label": label,
            "score_is_rul": is_rul,
            "threshold_symbol": "tau" if is_rul else "p",
            "thresholds": grid,
            "dev_failures": dev_sw["failures"],
            "dev_wasted_sum": dev_sw["wasted_sum"],
            "dev_premature": dev_sw["premature"],
            "test_failures": test_sw["failures"],
            "test_wasted_sum": test_sw["wasted_sum"],
            "test_premature": test_sw["premature"],
            "test_per_engine_alarms": test_sw["per_engine_alarms"],
        }

    fixed_schedule_sweep = {
        "dev": _summarize_fixed_k_sweep(dev_df, k_grid, alarm_w=alarm_w),
        "test": _summarize_fixed_k_sweep(test_df, k_grid, alarm_w=alarm_w),
    }

    # Verify cost=100 consistency with test_evaluation_summary.json
    idx_p99 = prob_grid.index(0.99)
    assert explorer_models["rung4_trends_logreg"]["test_failures"][idx_p99] == 0
    assert round(explorer_models["rung4_trends_logreg"]["test_wasted_sum"][idx_p99] / 20.0, 2) == 18.20

    # 5. Extract leakage scatter & Engine #21 trajectory
    _, scatter_payload = run_row_vs_engine_leakage_experiment(
        dev_df=dev_df, test_df=test_df, window_k=window_k, rul_cap=rul_cap, seed=seed,
    )
    leakage_scatter = {
        "rung5_trends_hgb": {
            "proper": _downsample_scatter(scatter_payload["rung5_trends_hgb"]["y_true_proper"], scatter_payload["rung5_trends_hgb"]["y_pred_proper"]),
            "leaky": _downsample_scatter(scatter_payload["rung5_trends_hgb"]["y_true_leaky"], scatter_payload["rung5_trends_hgb"]["y_pred_leaky"]),
        },
        "rung5_trends_rf": {
            "proper": _downsample_scatter(scatter_payload["rung5_trends_rf"]["y_true_proper"], scatter_payload["rung5_trends_rf"]["y_pred_proper"]),
            "leaky": _downsample_scatter(scatter_payload["rung5_trends_rf"]["y_true_leaky"], scatter_payload["rung5_trends_rf"]["y_pred_leaky"]),
        },
        "trajectory_example": scatter_payload["trajectory_example"],
        "trajectories_by_model": scatter_payload["trajectories_by_model"],
    }

    report_data = {
        "meta": {
            "course": cfg["project"]["course"],
            "dataset": "NASA C-MAPSS FD001",
            "seed": seed,
            "rul_cap": rul_cap,
            "alarm_w": alarm_w,
            "window_k": window_k,
            "dev_engines": 80,
            "dev_rows": int(len(dev_df)),
            "test_engines": 20,
            "test_rows": int(len(test_df)),
            "official_test_engines": 100,
            "official_test_rows": 13096,
            "dev_lifetimes": dev_lifetimes,
            "test_lifetimes": test_lifetimes,
        },
        "hero_engine": hero_engine,
        "sensor_screening": eda_screening,
        "aligned_sensor_summary": aligned_summary,
        "sample_dev_engine": sample_dev_engine,
        "feature_metadata": feature_metadata,
        "feature_attribution": feature_attr,
        "split_summary": split_summary,
        "cv_regression_ladder": cv_reg_df.to_dict(orient="records"),
        "cv_classification_ladder": cv_clf_df.to_dict(orient="records"),
        "pr_and_calibration": pr_calibration,
        "test_regression_ladder": test_summary["test_regression_ladder"],
        "test_classification_ladder": test_summary["test_classification_ladder"],
        "locked_parameters_c100": test_summary["locked_parameters_c100"],
        "breakeven_analysis": test_summary["breakeven_analysis"],
        "bootstrap_ci_summary": test_summary["bootstrap_ci_summary"],
        "cost_ratio_sweep": test_summary["cost_ratio_sweep"],
        "official_fd001_100_engines_benchmark": test_summary["official_fd001_100_engines_benchmark"],
        "sensitivity_summary": test_summary["sensitivity_summary"],
        "limitations": test_summary["limitations"],
        "interactive_explorer": {
            "models": explorer_models,
            "fixed_schedule": fixed_schedule_sweep,
        },
        "leakage_experiment": {
            "summary": test_summary["leakage_experiment"],
            "scatter_and_trajectory": leakage_scatter,
        },
        "error_analysis": {
            "per_engine_error_audit": test_summary["per_engine_error_audit"],
            "error_case_studies": test_summary["error_case_studies"],
        },
    }

    clean_data = _sanitize_json(report_data)
    out_json = docs_data_dir / "report_data.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(clean_data, f, separators=(",", ":"), allow_nan=False)

    size_kb = out_json.stat().st_size / 1024.0
    print(f"Exported {out_json.relative_to(REPO_ROOT)} ({size_kb:.1f} KB) and {len(list(docs_fig_dir.glob('*.png')))} PNG fallbacks in docs/figures/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
