"""Baseline ladder, 5-fold GroupKFold cross-validation, and threshold/policy optimization.

Phase 4 of the ES 335 Predictive Maintenance Project:
- Implements the NASA Asymmetric Scoring function and regression/classification evaluation metrics.
- Evaluates Rungs 0 through 5 of the baseline ladder strictly on the 80 development engines
  using 5-fold `GroupKFold` grouped by `unit_id`.
- Optimizes the fixed replacement schedule `K*` and model alarm thresholds `p*` / `tau*` on
  development folds across failure-to-waste cost ratios (`c_fail : c_waste`).
- Generates diagnostic figures for baseline progression, PR & calibration curves, and
  Lasso / HistGradientBoosting feature importances.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

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


def nasa_asymmetric_score(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    return_mean: bool = True,
) -> float:
    """Compute the NASA C-MAPSS asymmetric scoring penalty.

    Let d_i = y_pred_i - y_true_i:
      - If d_i < 0 (early prediction / under-predicting RUL): exp(-d_i / 13) - 1
      - If d_i >= 0 (late prediction / over-predicting RUL):  exp(+d_i / 10) - 1

    Overestimating remaining life is penalized much more severely because an unexpected
    in-service failure is far more dangerous than a slightly early preventive replacement.
    """
    d = np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
    d_clipped = np.clip(d, -150.0, 150.0)
    penalties = np.where(
        d_clipped < 0.0,
        np.expm1(-d_clipped / 13.0),
        np.expm1(d_clipped / 10.0),
    )
    return float(np.mean(penalties)) if return_mean else float(np.sum(penalties))


def compute_regression_metrics(
    df: pd.DataFrame,
    y_pred_raw: np.ndarray,
    rul_cap: int | None = 125,
) -> dict[str, float]:
    """Compute comprehensive RUL regression metrics across all cycles and near-failure cycles."""
    y_true = df["rul"].to_numpy(dtype=np.float64)
    if rul_cap is not None:
        y_pred = np.clip(y_pred_raw, 0.0, float(rul_cap))
    else:
        y_pred = np.maximum(0.0, y_pred_raw)

    rmse_all = float(root_mean_squared_error(y_true, y_pred))
    mae_all = float(mean_absolute_error(y_true, y_pred))
    r2_all = float(r2_score(y_true, y_pred))
    asym_all = nasa_asymmetric_score(y_true, y_pred, return_mean=True)
    pct_late = float(np.mean(y_pred > y_true) * 100.0)

    # Critical zone evaluation (true uncapped RUL <= 50 cycles)
    crit_mask = df["rul_uncapped"].to_numpy() <= 50
    rmse_crit = float(root_mean_squared_error(y_true[crit_mask], y_pred[crit_mask]))
    asym_crit = nasa_asymmetric_score(y_true[crit_mask], y_pred[crit_mask], return_mean=True)

    return {
        "rmse_all": round(rmse_all, 3),
        "mae_all": round(mae_all, 3),
        "r2_all": round(r2_all, 4),
        "nasa_asym_mean": round(asym_all, 3),
        "rmse_rul_le_50": round(rmse_crit, 3),
        "nasa_asym_rul_le_50": round(asym_crit, 3),
        "pct_late_overpredictions": round(pct_late, 2),
    }


def simulate_fixed_schedule_policy(
    df: pd.DataFrame,
    k_cycle: int,
    c_fail: float = 100.0,
    c_waste: float = 1.0,
) -> dict[str, float]:
    """Simulate a fixed-interval replacement policy at cycle K across all engines in `df`.

    For each engine i with total run-to-failure lifespan `T_i = max_cycle`:
    - If `K < T_i`: engine is replaced preventively at cycle K, throwing away `T_i - K` cycles
      of remaining useful life (`cost = c_waste * (T_i - K)`).
    - If `K >= T_i`: engine fails in service at cycle `T_i` prior to/at replacement (`cost = c_fail`).
    """
    lifetimes = df.groupby("unit_id")["max_cycle"].first().to_numpy(dtype=np.float64)
    n_engines = len(lifetimes)

    failed_mask = lifetimes <= k_cycle
    wasted_cycles = np.where(failed_mask, 0.0, lifetimes - k_cycle)
    costs = np.where(failed_mask, float(c_fail), float(c_waste) * wasted_cycles)

    return {
        "policy_param": float(k_cycle),
        "mean_cost_per_engine": round(float(np.mean(costs)), 2),
        "total_fleet_cost": round(float(np.sum(costs)), 2),
        "in_service_failures": int(np.sum(failed_mask)),
        "failure_rate_pct": round(float(np.mean(failed_mask) * 100.0), 2),
        "mean_wasted_cycles": round(float(np.mean(wasted_cycles)), 2),
        "premature_alarm_rate_pct": round(
            float(np.mean((~failed_mask) & (wasted_cycles > 30)) * 100.0), 2
        ),
    }


def optimize_fixed_schedule_k(
    df: pd.DataFrame,
    c_fail: float = 100.0,
    c_waste: float = 1.0,
    k_grid: range | list[int] = range(100, 365),
) -> tuple[int, dict[str, float]]:
    """Find the fixed replacement cycle K* that minimizes mean maintenance cost per engine on `df`."""
    best_k = 128
    best_stats: dict[str, float] | None = None
    for k in k_grid:
        stats = simulate_fixed_schedule_policy(df, k_cycle=int(k), c_fail=c_fail, c_waste=c_waste)
        if best_stats is None or stats["mean_cost_per_engine"] < best_stats["mean_cost_per_engine"]:
            best_k = int(k)
            best_stats = stats
    assert best_stats is not None
    return best_k, best_stats


def simulate_model_alarm_policy(
    df: pd.DataFrame,
    scores: np.ndarray,
    threshold: float,
    c_fail: float = 100.0,
    c_waste: float = 1.0,
    score_is_rul: bool = False,
    alarm_w: int = 30,
) -> dict[str, float]:
    """Simulate a condition-based replacement policy that replaces an engine at its first alarm.

    For each engine i with failure cycle `T_i = max_cycle`:
    - Evaluate cycles `t = 1 .. T_i - 1` (cycles prior to actual failure at `T_i`).
    - Alarm fires at cycle `t` if `scores[t] >= threshold` (for probability classifiers) or
      `scores[t] <= threshold` (when `score_is_rul=True`, i.e., predicted RUL <= tau).
    - If an alarm fires at some `t_alarm < T_i`:
        Engine is replaced preventively at `t_alarm`.
        Remaining life thrown away = `T_i - t_alarm` (`rul_uncapped` at `t_alarm`).
        Cost = `c_waste * (T_i - t_alarm)`.
    - If no alarm fires prior to `T_i`:
        Engine fails in service at `T_i`.
        Cost = `c_fail`.
    """
    work = df[["unit_id", "cycle", "max_cycle", "rul_uncapped"]].copy()
    if score_is_rul:
        work["triggered"] = (scores <= threshold) & (work["rul_uncapped"] > 0)
    else:
        work["triggered"] = (scores >= threshold) & (work["rul_uncapped"] > 0)

    total_engines = work["unit_id"].nunique()
    triggered_rows = work[work["triggered"]]
    first_alarms = triggered_rows.groupby("unit_id", sort=False).first()

    n_replaced = len(first_alarms)
    n_failed = total_engines - n_replaced

    if n_replaced > 0:
        wasted_replaced = first_alarms["rul_uncapped"].to_numpy(dtype=np.float64)
        n_premature = int(np.sum(wasted_replaced > alarm_w))
        n_timely = int(np.sum((wasted_replaced >= 1) & (wasted_replaced <= alarm_w)))
    else:
        wasted_replaced = np.array([], dtype=np.float64)
        n_premature = 0
        n_timely = 0

    total_wasted_cycles = float(np.sum(wasted_replaced))
    total_cost = float(n_failed * c_fail + c_waste * total_wasted_cycles)
    mean_cost = total_cost / total_engines
    mean_wasted_all = total_wasted_cycles / total_engines

    return {
        "policy_param": round(float(threshold), 4),
        "mean_cost_per_engine": round(mean_cost, 2),
        "total_fleet_cost": round(total_cost, 2),
        "in_service_failures": int(n_failed),
        "failure_rate_pct": round(float(n_failed / total_engines * 100.0), 2),
        "timely_replacement_rate_pct": round(float(n_timely / total_engines * 100.0), 2),
        "premature_alarm_rate_pct": round(float(n_premature / total_engines * 100.0), 2),
        "mean_wasted_cycles": round(mean_wasted_all, 2),
    }


def optimize_alarm_threshold(
    df: pd.DataFrame,
    scores: np.ndarray,
    c_fail: float = 100.0,
    c_waste: float = 1.0,
    score_is_rul: bool = False,
    alarm_w: int = 30,
) -> tuple[float, dict[str, float]]:
    """Find the alarm threshold (`p*` or `tau*`) that minimizes maintenance cost on `df`."""
    if score_is_rul:
        grid = np.linspace(8.0, 50.0, 85)
    else:
        grid = np.linspace(0.05, 0.85, 81)

    best_thresh = float(grid[0])
    best_stats: dict[str, float] | None = None
    for th in grid:
        stats = simulate_model_alarm_policy(
            df,
            scores=scores,
            threshold=float(th),
            c_fail=c_fail,
            c_waste=c_waste,
            score_is_rul=score_is_rul,
            alarm_w=alarm_w,
        )
        if best_stats is None or stats["mean_cost_per_engine"] < best_stats["mean_cost_per_engine"]:
            best_thresh = float(th)
            best_stats = stats

    assert best_stats is not None
    return round(best_thresh, 4), best_stats


def get_regression_ladder_specs() -> list[dict[str, Any]]:
    """Return specifications for Rungs 0 to 5 of the Continuous RUL Regression Ladder."""
    return [
        {
            "model_id": "rung0_mean_rul",
            "rung": "Rung 0",
            "name": "Mean RUL Baseline (Floor)",
            "feature_mode": "age_only",
            "window_k": 10,
            "scale": False,
            "estimator": DummyRegressor(strategy="mean"),
        },
        {
            "model_id": "rung2_age_linear",
            "rung": "Rung 2",
            "name": "Age-Only Linear Regression",
            "feature_mode": "age_only",
            "window_k": 10,
            "scale": True,
            "estimator": Ridge(alpha=1.0),
        },
        {
            "model_id": "rung3_instant_ridge",
            "rung": "Rung 3",
            "name": "Instantaneous Sensors + Ridge (L2)",
            "feature_mode": "instantaneous",
            "window_k": 10,
            "scale": True,
            "estimator": Ridge(alpha=100.0),
        },
        {
            "model_id": "rung3_instant_lasso",
            "rung": "Rung 3",
            "name": "Instantaneous Sensors + Lasso (L1)",
            "feature_mode": "instantaneous",
            "window_k": 10,
            "scale": True,
            "estimator": Lasso(alpha=0.5, max_iter=5000, random_state=42),
        },
        {
            "model_id": "rung4_trends_k10_ridge",
            "rung": "Rung 4",
            "name": "Trailing Trends (k=10) + Ridge (L2)",
            "feature_mode": "trailing_trends",
            "window_k": 10,
            "scale": True,
            "estimator": Ridge(alpha=200.0),
        },
        {
            "model_id": "rung4_trends_k10_lasso",
            "rung": "Rung 4",
            "name": "Trailing Trends (k=10) + Lasso (L1)",
            "feature_mode": "trailing_trends",
            "window_k": 10,
            "scale": True,
            "estimator": Lasso(alpha=0.5, max_iter=5000, random_state=42),
        },
        {
            "model_id": "rung4_trends_k30_ridge",
            "rung": "Rung 4",
            "name": "Trailing Trends (k=30) + Ridge (L2)",
            "feature_mode": "trailing_trends",
            "window_k": 30,
            "scale": True,
            "estimator": Ridge(alpha=200.0),
        },
        {
            "model_id": "rung4_trends_k30_lasso",
            "rung": "Rung 4",
            "name": "Trailing Trends (k=30) + Lasso (L1)",
            "feature_mode": "trailing_trends",
            "window_k": 30,
            "scale": True,
            "estimator": Lasso(alpha=0.5, max_iter=5000, random_state=42),
        },
        {
            "model_id": "rung5_trends_k10_rf",
            "rung": "Rung 5",
            "name": "Trailing Trends (k=10) + Random Forest",
            "feature_mode": "trailing_trends",
            "window_k": 10,
            "scale": False,
            "estimator": RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                min_samples_leaf=12,
                max_features="sqrt",
                random_state=42,
                n_jobs=-1,
            ),
        },
        {
            "model_id": "rung5_trends_k10_hgb",
            "rung": "Rung 5",
            "name": "Trailing Trends (k=10) + HistGradientBoosting",
            "feature_mode": "trailing_trends",
            "window_k": 10,
            "scale": False,
            "estimator": HistGradientBoostingRegressor(
                max_iter=150,
                learning_rate=0.05,
                max_depth=5,
                min_samples_leaf=30,
                l2_regularization=2.0,
                random_state=42,
            ),
        },
        {
            "model_id": "rung5_trends_k30_rf",
            "rung": "Rung 5",
            "name": "Trailing Trends (k=30) + Random Forest",
            "feature_mode": "trailing_trends",
            "window_k": 30,
            "scale": False,
            "estimator": RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                min_samples_leaf=12,
                max_features="sqrt",
                random_state=42,
                n_jobs=-1,
            ),
        },
        {
            "model_id": "rung5_trends_k30_hgb",
            "rung": "Rung 5",
            "name": "Trailing Trends (k=30) + HistGradientBoosting",
            "feature_mode": "trailing_trends",
            "window_k": 30,
            "scale": False,
            "estimator": HistGradientBoostingRegressor(
                max_iter=150,
                learning_rate=0.05,
                max_depth=5,
                min_samples_leaf=30,
                l2_regularization=2.0,
                random_state=42,
            ),
        },
        {
            "model_id": "ablation_combined_k10_hgb",
            "rung": "Ablation",
            "name": "Combined Raw+Trends (k=10) + HistGradientBoosting",
            "feature_mode": "combined",
            "window_k": 10,
            "scale": False,
            "estimator": HistGradientBoostingRegressor(
                max_iter=150,
                learning_rate=0.05,
                max_depth=5,
                min_samples_leaf=30,
                l2_regularization=2.0,
                random_state=42,
            ),
        },
    ]


def get_classification_ladder_specs() -> list[dict[str, Any]]:
    """Return specifications for Rungs 0 to 5 of the Binary Alarm Classification Ladder."""
    return [
        {
            "model_id": "rung0_prior_clf",
            "rung": "Rung 0",
            "name": "Never Alarm / Base-Rate Prior (Floor)",
            "feature_mode": "age_only",
            "window_k": 10,
            "scale": False,
            "estimator": DummyClassifier(strategy="prior"),
        },
        {
            "model_id": "rung2_age_logreg",
            "rung": "Rung 2",
            "name": "Age-Only Logistic Regression",
            "feature_mode": "age_only",
            "window_k": 10,
            "scale": True,
            "estimator": LogisticRegression(C=1.0, max_iter=1000, random_state=42),
        },
        {
            "model_id": "rung3_instant_logreg",
            "rung": "Rung 3",
            "name": "Instantaneous Sensors + Logistic Regression",
            "feature_mode": "instantaneous",
            "window_k": 10,
            "scale": True,
            "estimator": LogisticRegression(C=0.1, max_iter=1000, random_state=42),
        },
        {
            "model_id": "rung4_trends_k10_logreg",
            "rung": "Rung 4",
            "name": "Trailing Trends (k=10) + Logistic Regression (L2)",
            "feature_mode": "trailing_trends",
            "window_k": 10,
            "scale": True,
            "estimator": LogisticRegression(C=0.05, max_iter=1000, random_state=42),
        },
        {
            "model_id": "rung4_trends_k30_logreg",
            "rung": "Rung 4",
            "name": "Trailing Trends (k=30) + Logistic Regression (L2)",
            "feature_mode": "trailing_trends",
            "window_k": 30,
            "scale": True,
            "estimator": LogisticRegression(C=0.05, max_iter=1000, random_state=42),
        },
        {
            "model_id": "rung5_trends_k10_rf_clf",
            "rung": "Rung 5",
            "name": "Trailing Trends (k=10) + Random Forest Classifier",
            "feature_mode": "trailing_trends",
            "window_k": 10,
            "scale": False,
            "estimator": RandomForestClassifier(
                n_estimators=200,
                max_depth=8,
                min_samples_leaf=15,
                max_features="sqrt",
                random_state=42,
                n_jobs=-1,
            ),
        },
        {
            "model_id": "rung5_trends_k10_hgb_clf",
            "rung": "Rung 5",
            "name": "Trailing Trends (k=10) + HistGradientBoosting Classifier",
            "feature_mode": "trailing_trends",
            "window_k": 10,
            "scale": False,
            "estimator": HistGradientBoostingClassifier(
                max_iter=120,
                learning_rate=0.05,
                max_depth=4,
                min_samples_leaf=30,
                l2_regularization=2.0,
                random_state=42,
            ),
        },
        {
            "model_id": "rung5_trends_k30_rf_clf",
            "rung": "Rung 5",
            "name": "Trailing Trends (k=30) + Random Forest Classifier",
            "feature_mode": "trailing_trends",
            "window_k": 30,
            "scale": False,
            "estimator": RandomForestClassifier(
                n_estimators=200,
                max_depth=8,
                min_samples_leaf=15,
                max_features="sqrt",
                random_state=42,
                n_jobs=-1,
            ),
        },
        {
            "model_id": "rung5_trends_k30_hgb_clf",
            "rung": "Rung 5",
            "name": "Trailing Trends (k=30) + HistGradientBoosting Classifier",
            "feature_mode": "trailing_trends",
            "window_k": 30,
            "scale": False,
            "estimator": HistGradientBoostingClassifier(
                max_iter=120,
                learning_rate=0.05,
                max_depth=4,
                min_samples_leaf=30,
                l2_regularization=2.0,
                random_state=42,
            ),
        },
    ]


def run_group_cv_regression_ladder(
    dev_df: pd.DataFrame,
    cv_folds: int = 5,
    rul_cap: int = 125,
    c_fail: float = 100.0,
    c_waste: float = 1.0,
    alarm_w: int = 30,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Evaluate all regression ladder models using 5-fold GroupKFold on the 80 dev engines."""
    gkf = GroupKFold(n_splits=cv_folds)
    specs = get_regression_ladder_specs()

    oof_preds: dict[str, np.ndarray] = {}
    records: list[dict[str, Any]] = []

    for spec in specs:
        mid = spec["model_id"]
        oof_y = np.zeros(len(dev_df), dtype=np.float64)
        fold_rmses: list[float] = []

        for tr_idx, val_idx in gkf.split(dev_df, groups=dev_df["unit_id"]):
            tr_df = dev_df.iloc[tr_idx]
            val_df = dev_df.iloc[val_idx]

            prep = build_preprocessing_pipeline(
                window_k=spec["window_k"],
                feature_mode=spec["feature_mode"],
                scale=spec["scale"],
            )
            full_pipe = Pipeline([("prep", prep), ("model", clone(spec["estimator"]))])
            full_pipe.fit(tr_df, tr_df["rul"])
            preds_val = np.clip(full_pipe.predict(val_df), 0.0, float(rul_cap))
            oof_y[val_idx] = preds_val
            fold_rmses.append(float(root_mean_squared_error(val_df["rul"], preds_val)))

        oof_preds[mid] = oof_y
        reg_metrics = compute_regression_metrics(dev_df, oof_y, rul_cap=rul_cap)

        # Also evaluate policy cost if regression RUL prediction triggers replacement at RUL <= tau*
        best_tau, policy_stats = optimize_alarm_threshold(
            dev_df,
            scores=oof_y,
            c_fail=c_fail,
            c_waste=c_waste,
            score_is_rul=True,
            alarm_w=alarm_w,
        )

        records.append(
            {
                "model_id": mid,
                "rung": spec["rung"],
                "model_name": spec["name"],
                "feature_mode": spec["feature_mode"],
                "window_k": spec["window_k"],
                **reg_metrics,
                "cv_fold_rmse_std": round(float(np.std(fold_rmses)), 3),
                "optimal_rul_trigger_tau": best_tau,
                "policy_mean_cost_c100": policy_stats["mean_cost_per_engine"],
                "policy_failure_rate_pct": policy_stats["failure_rate_pct"],
                "policy_mean_wasted_cycles": policy_stats["mean_wasted_cycles"],
            }
        )

    results_df = pd.DataFrame(records)
    return results_df, oof_preds


def run_group_cv_classification_ladder(
    dev_df: pd.DataFrame,
    cv_folds: int = 5,
    c_fail: float = 100.0,
    c_waste: float = 1.0,
    alarm_w: int = 30,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Evaluate Rung 0–5 alarm classifiers + Fixed Schedule K* using 5-fold GroupKFold on dev data."""
    gkf = GroupKFold(n_splits=cv_folds)
    specs = get_classification_ladder_specs()
    y_true = dev_df["alarm"].to_numpy(dtype=int)

    oof_probs: dict[str, np.ndarray] = {}
    records: list[dict[str, Any]] = []

    # First evaluate Rung 1: Fixed-Interval Replacement Schedule K* (fit K* on train folds, eval on val)
    fixed_val_costs: list[float] = []
    fixed_val_fails: int = 0
    fixed_val_wasted: list[float] = []
    fold_ks: list[int] = []
    for tr_idx, val_idx in gkf.split(dev_df, groups=dev_df["unit_id"]):
        tr_df = dev_df.iloc[tr_idx]
        val_df = dev_df.iloc[val_idx]
        k_star, _ = optimize_fixed_schedule_k(tr_df, c_fail=c_fail, c_waste=c_waste)
        fold_ks.append(k_star)
        val_stats = simulate_fixed_schedule_policy(val_df, k_cycle=k_star, c_fail=c_fail, c_waste=c_waste)
        fixed_val_costs.append(val_stats["mean_cost_per_engine"])
        fixed_val_fails += val_stats["in_service_failures"]
        fixed_val_wasted.append(val_stats["mean_wasted_cycles"])

    k_dev_global, fixed_global_stats = optimize_fixed_schedule_k(dev_df, c_fail=c_fail, c_waste=c_waste)
    records.append(
        {
            "model_id": "rung1_fixed_schedule",
            "rung": "Rung 1",
            "model_name": f"Fixed Replacement Schedule (K*={k_dev_global} cyc)",
            "feature_mode": "cycle_timer",
            "window_k": 0,
            "pr_auc": np.nan,
            "roc_auc": np.nan,
            "brier_score": np.nan,
            "precision_at_opt": np.nan,
            "recall_at_opt": np.nan,
            "f1_at_opt": np.nan,
            "optimal_threshold_p": float(k_dev_global),
            "oof_policy_mean_cost_c100": round(float(np.mean(fixed_val_costs)), 2),
            "failure_rate_pct": round(float(fixed_val_fails / 80.0 * 100.0), 2),
            "premature_alarm_rate_pct": fixed_global_stats["premature_alarm_rate_pct"],
            "mean_wasted_cycles": round(float(np.mean(fixed_val_wasted)), 2),
        }
    )

    for spec in specs:
        mid = spec["model_id"]
        oof_p = np.zeros(len(dev_df), dtype=np.float64)

        for tr_idx, val_idx in gkf.split(dev_df, groups=dev_df["unit_id"]):
            tr_df = dev_df.iloc[tr_idx]
            val_df = dev_df.iloc[val_idx]

            prep = build_preprocessing_pipeline(
                window_k=spec["window_k"],
                feature_mode=spec["feature_mode"],
                scale=spec["scale"],
            )
            full_pipe = Pipeline([("prep", prep), ("model", clone(spec["estimator"]))])
            full_pipe.fit(tr_df, tr_df["alarm"])
            oof_p[val_idx] = full_pipe.predict_proba(val_df)[:, 1]

        oof_probs[mid] = oof_p
        pr_auc = float(average_precision_score(y_true, oof_p))
        roc_auc = float(roc_auc_score(y_true, oof_p)) if len(np.unique(oof_p)) > 1 else 0.5
        brier = float(brier_score_loss(y_true, oof_p))

        if mid == "rung0_prior_clf":
            # Never alarm floor policy: threshold = 1.0 (0 alarms, 100% run to failure)
            best_p = 1.0
            pol_stats = simulate_model_alarm_policy(
                dev_df, oof_p, threshold=1.0, c_fail=c_fail, c_waste=c_waste, alarm_w=alarm_w
            )
            prec, rec, f1 = 0.0, 0.0, 0.0
        else:
            best_p, pol_stats = optimize_alarm_threshold(
                dev_df,
                scores=oof_p,
                c_fail=c_fail,
                c_waste=c_waste,
                score_is_rul=False,
                alarm_w=alarm_w,
            )
            y_hat = (oof_p >= best_p).astype(int)
            prec = float(precision_score(y_true, y_hat, zero_division=0))
            rec = float(recall_score(y_true, y_hat, zero_division=0))
            f1 = float(f1_score(y_true, y_hat, zero_division=0))

        records.append(
            {
                "model_id": mid,
                "rung": spec["rung"],
                "model_name": spec["name"],
                "feature_mode": spec["feature_mode"],
                "window_k": spec["window_k"],
                "pr_auc": round(pr_auc, 4),
                "roc_auc": round(roc_auc, 4),
                "brier_score": round(brier, 4),
                "precision_at_opt": round(prec, 4),
                "recall_at_opt": round(rec, 4),
                "f1_at_opt": round(f1, 4),
                "optimal_threshold_p": best_p,
                "oof_policy_mean_cost_c100": pol_stats["mean_cost_per_engine"],
                "failure_rate_pct": pol_stats["failure_rate_pct"],
                "premature_alarm_rate_pct": pol_stats["premature_alarm_rate_pct"],
                "mean_wasted_cycles": pol_stats["mean_wasted_cycles"],
            }
        )

    results_df = pd.DataFrame(records)
    # Sort by Rung order so Rung 0, Rung 1, Rung 2, Rung 3, Rung 4, Rung 5 appear in sequence
    rung_order = {"Rung 0": 0, "Rung 1": 1, "Rung 2": 2, "Rung 3": 3, "Rung 4": 4, "Rung 5": 5}
    results_df["_order"] = results_df["rung"].map(rung_order)
    results_df = results_df.sort_values(["_order", "oof_policy_mean_cost_c100"]).drop(columns=["_order"]).reset_index(drop=True)
    return results_df, oof_probs


def extract_feature_attribution(
    dev_df: pd.DataFrame,
    window_k: int = 10,
) -> pd.DataFrame:
    """Extract standardized linear/Lasso/Logistic coefficients and HGB permutation importances on dev set."""
    prep = build_preprocessing_pipeline(window_k=window_k, feature_mode="trailing_trends", scale=True)
    X_dev = prep.fit_transform(dev_df)
    y_rul = dev_df["rul"].to_numpy()
    y_alarm = dev_df["alarm"].to_numpy()

    ridge = Ridge(alpha=200.0).fit(X_dev, y_rul)
    lasso = Lasso(alpha=0.5, max_iter=5000, random_state=42).fit(X_dev, y_rul)
    logreg = LogisticRegression(C=0.05, max_iter=1000, random_state=42).fit(X_dev, y_alarm)

    hgb_reg = HistGradientBoostingRegressor(
        max_iter=150, learning_rate=0.05, max_depth=5, min_samples_leaf=30, l2_regularization=2.0, random_state=42
    ).fit(X_dev, y_rul)
    perm_imp = permutation_importance(
        hgb_reg, X_dev, y_rul, n_repeats=5, random_state=42, scoring="neg_root_mean_squared_error"
    )

    df_imp = pd.DataFrame(
        {
            "feature": list(X_dev.columns),
            "ridge_coef": np.round(ridge.coef_, 4),
            "lasso_coef": np.round(lasso.coef_, 4),
            "lasso_selected": np.abs(lasso.coef_) > 1e-5,
            "logreg_alarm_coef": np.round(logreg.coef_[0], 4),
            "hgb_perm_importance_rmse": np.round(perm_imp.importances_mean, 4),
        }
    )
    return df_imp.sort_values("hgb_perm_importance_rmse", ascending=False).reset_index(drop=True)


def plot_cv_baseline_ladder_progression(
    reg_df: pd.DataFrame,
    clf_df: pd.DataFrame,
    out_path: Path,
) -> None:
    """Plot 3-panel progression across the Baseline Ladder (RMSE, NASA Asymmetric Score, Policy Cost)."""
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2), layout="constrained")

    # Select key representative models for clean visual ladder comparison
    reg_sub = reg_df[
        reg_df["model_id"].isin(
            [
                "rung0_mean_rul",
                "rung2_age_linear",
                "rung3_instant_ridge",
                "rung3_instant_lasso",
                "rung4_trends_k10_ridge",
                "rung4_trends_k30_ridge",
                "rung5_trends_k10_rf",
                "rung5_trends_k10_hgb",
                "rung5_trends_k30_hgb",
            ]
        )
    ].copy()

    short_reg_names = {
        "rung0_mean_rul": "R0: Mean RUL (Floor)",
        "rung2_age_linear": "R2: Age-Only Linear",
        "rung3_instant_ridge": "R3: Raw Sensors Ridge",
        "rung3_instant_lasso": "R3: Raw Sensors Lasso",
        "rung4_trends_k10_ridge": "R4: Trends (k=10) Ridge",
        "rung4_trends_k30_ridge": "R4: Trends (k=30) Ridge",
        "rung5_trends_k10_rf": "R5: Trends (k=10) RF",
        "rung5_trends_k10_hgb": "R5: Trends (k=10) HGB",
        "rung5_trends_k30_hgb": "R5: Trends (k=30) HGB",
    }
    reg_sub["label"] = reg_sub["model_id"].map(short_reg_names)
    rung_colors = ["#b0b0b0", "#fdb863", "#80b1d3", "#80b1d3", "#3182bd", "#08519c", "#6a51a3", "#2ca02c", "#1b7837"]

    # Panel A: 5-Fold GroupKFold RMSE
    ax1 = axes[0]
    y_pos = np.arange(len(reg_sub))
    bars1 = ax1.barh(y_pos, reg_sub["rmse_all"], color=rung_colors, height=0.68)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(reg_sub["label"], fontsize=8.5)
    ax1.invert_yaxis()
    ax1.set_xlabel("5-Fold GroupKFold RMSE (Cycles, Lower is Better)")
    ax1.set_title("A. RUL Regression RMSE Across Ladder")
    ax1.set_xlim(0, max(reg_sub["rmse_all"]) * 1.15)
    for bar, val in zip(bars1, reg_sub["rmse_all"]):
        ax1.text(val + 0.6, bar.get_y() + bar.get_height() / 2, f"{val:.2f}", va="center", fontsize=8)

    # Panel B: NASA Asymmetric Score (Critical Near-Failure Zone RUL <= 50)
    ax2 = axes[1]
    bars2 = ax2.barh(y_pos, reg_sub["nasa_asym_rul_le_50"], color=rung_colors, height=0.68)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([""] * len(reg_sub))
    ax2.invert_yaxis()
    ax2.set_xlabel("Mean NASA Asymmetric Penalty (RUL <= 50, Lower is Better)")
    ax2.set_title("B. Asymmetric Penalty in Critical Zone (RUL <= 50)")
    ax2.set_xlim(0, max(reg_sub["nasa_asym_rul_le_50"]) * 1.18)
    for bar, val in zip(bars2, reg_sub["nasa_asym_rul_le_50"]):
        ax2.text(val + max(reg_sub["nasa_asym_rul_le_50"]) * 0.02, bar.get_y() + bar.get_height() / 2, f"{val:.2f}", va="center", fontsize=8)

    # Panel C: Out-of-Fold Maintenance Cost per Engine (c_fail=100, c_waste=1)
    ax3 = axes[2]
    clf_sub = clf_df[
        clf_df["model_id"].isin(
            [
                "rung0_prior_clf",
                "rung1_fixed_schedule",
                "rung2_age_logreg",
                "rung3_instant_logreg",
                "rung4_trends_k10_logreg",
                "rung4_trends_k30_logreg",
                "rung5_trends_k10_rf_clf",
                "rung5_trends_k10_hgb_clf",
                "rung5_trends_k30_hgb_clf",
            ]
        )
    ].copy()
    short_clf_names = {
        "rung0_prior_clf": "R0: Never Alarm (Run-to-Fail)",
        "rung1_fixed_schedule": "R1: Best Fixed Schedule (K*)",
        "rung2_age_logreg": "R2: Age-Only Logistic",
        "rung3_instant_logreg": "R3: Raw Sensors Logistic",
        "rung4_trends_k10_logreg": "R4: Trends (k=10) Logistic",
        "rung4_trends_k30_logreg": "R4: Trends (k=30) Logistic",
        "rung5_trends_k10_rf_clf": "R5: Trends (k=10) RF Clf",
        "rung5_trends_k10_hgb_clf": "R5: Trends (k=10) HGB Clf",
        "rung5_trends_k30_hgb_clf": "R5: Trends (k=30) HGB Clf",
    }
    clf_sub["label"] = clf_sub["model_id"].map(short_clf_names)
    clf_colors = ["#b0b0b0", "#d62728", "#fdb863", "#80b1d3", "#3182bd", "#08519c", "#6a51a3", "#2ca02c", "#1b7837"]
    y_pos_c = np.arange(len(clf_sub))
    bars3 = ax3.barh(y_pos_c, clf_sub["oof_policy_mean_cost_c100"], color=clf_colors, height=0.68)
    ax3.set_yticks(y_pos_c)
    ax3.set_yticklabels(clf_sub["label"], fontsize=8.5)
    ax3.invert_yaxis()
    ax3.set_xlabel("Mean Cost / Engine ($c_{\\mathrm{fail}}=100, c_{\\mathrm{waste}}=1$)")
    ax3.set_title("C. Replacement Policy Cost Across Ladder")
    ax3.set_xlim(0, 115)
    for bar, val in zip(bars3, clf_sub["oof_policy_mean_cost_c100"]):
        ax3.text(val + 1.5, bar.get_y() + bar.get_height() / 2, f"{val:.1f}", va="center", fontsize=8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_cv_pr_and_calibration_curves(
    dev_df: pd.DataFrame,
    oof_probs: dict[str, np.ndarray],
    clf_df: pd.DataFrame,
    out_path: Path,
) -> None:
    """Plot Precision-Recall curves and Reliability (Calibration) curves for the alarm ladder."""
    y_true = dev_df["alarm"].to_numpy(dtype=int)
    base_rate = float(np.mean(y_true))

    models_to_plot = [
        ("rung2_age_logreg", "R2: Age-Only Logistic", "#ff7f0e", ":"),
        ("rung3_instant_logreg", "R3: Raw Sensors Logistic", "#1f77b4", "--"),
        ("rung4_trends_k10_logreg", "R4: Trends (k=10) Logistic", "#8e24aa", "-."),
        ("rung4_trends_k30_logreg", "R4: Trends (k=30) Logistic", "#e377c2", "-."),
        ("rung5_trends_k10_rf_clf", "R5: Trends (k=10) Random Forest", "#d62728", "--"),
        ("rung5_trends_k10_hgb_clf", "R5: Trends (k=10) HistGradientBoosting", "#2ca02c", "-"),
        ("rung5_trends_k30_hgb_clf", "R5: Trends (k=30) HistGradientBoosting", "#0d3b66", "-"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), layout="constrained")

    # Panel A: Precision-Recall Curve
    ax_pr = axes[0]
    ax_pr.axhline(base_rate, color="gray", linestyle=":", linewidth=1.2, label=f"Random Baseline (Pos Rate = {base_rate:.3f})")

    for mid, label, color, ls in models_to_plot:
        probs = oof_probs[mid]
        prec, rec, _ = precision_recall_curve(y_true, probs)
        ap = float(average_precision_score(y_true, probs))
        ax_pr.plot(rec, prec, color=color, linestyle=ls, linewidth=2.0, label=f"{label} (PR-AUC = {ap:.3f})")

    ax_pr.set_xlabel("Recall (Fraction of True Near-Failure Cycles Detected)")
    ax_pr.set_ylabel("Precision (Fraction of Alarms Within w=30 Cycles)")
    ax_pr.set_title("A. 5-Fold Out-of-Fold Precision-Recall Curves (w = 30 Cycles)")
    ax_pr.set_xlim(0.0, 1.02)
    ax_pr.set_ylim(0.0, 1.03)
    ax_pr.legend(loc="lower left", frameon=True, fontsize=8)

    # Panel B: Calibration Curve (Reliability Diagram)
    ax_cal = axes[1]
    ax_cal.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1.2, label="Perfectly Calibrated")

    for mid, label, color, ls in models_to_plot:
        probs = oof_probs[mid]
        brier = float(brier_score_loss(y_true, probs))
        frac_pos, mean_pred = calibration_curve(y_true, probs, n_bins=10, strategy="uniform")
        ax_cal.plot(
            mean_pred,
            frac_pos,
            color=color,
            linestyle=ls,
            marker="o",
            markersize=4,
            linewidth=1.8,
            label=f"{label} (Brier = {brier:.4f})",
        )

    ax_cal.set_xlabel("Mean Predicted Alarm Probability $P(\\mathrm{RUL} \\leq 30)$")
    ax_cal.set_ylabel("Empirical Fraction of Positives ($\\mathrm{RUL} \\leq 30$)")
    ax_cal.set_title("B. Probability Calibration Curves (Reliability Diagram)")
    ax_cal.set_xlim(-0.02, 1.02)
    ax_cal.set_ylim(-0.02, 1.02)
    ax_cal.legend(loc="upper left", frameon=True, fontsize=8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_cv_feature_attribution_and_errors(
    dev_df: pd.DataFrame,
    imp_df: pd.DataFrame,
    oof_preds_reg: dict[str, np.ndarray],
    out_path: Path,
) -> None:
    """Plot Lasso vs HGB top feature importances and Error-vs-True-RUL diagnostic curve."""
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.4), layout="constrained")

    # Panel A: Top 12 features by HGB permutation importance alongside Lasso standardized coefficients
    ax1 = axes[0]
    top_f = imp_df.head(12).copy().iloc[::-1]
    y_pos = np.arange(len(top_f))
    height = 0.36

    ax1.barh(
        y_pos + height / 2,
        top_f["hgb_perm_importance_rmse"],
        height=height,
        color="#2ca02c",
        label="HistGradientBoosting Permutation Importance (RMSE cyc)",
    )
    ax1.barh(
        y_pos - height / 2,
        np.abs(top_f["lasso_coef"]),
        height=height,
        color="#1f77b4",
        label="|Lasso Coefficient| (L1 Standardized Weight)",
    )
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(top_f["feature"], fontsize=8.5)
    ax1.set_xlabel("Feature Attribution Magnitude (Cycles)")
    n_selected = int(imp_df["lasso_selected"].sum())
    ax1.set_title(
        f"A. Top 12 Trailing Features: HGB vs. Lasso\n"
        f"[Lasso Zeroed Out {len(imp_df) - n_selected}/{len(imp_df)} Redundant Features]"
    )
    ax1.legend(loc="lower right", frameon=True, fontsize=8)

    # Panel B: Prediction Error (y_pred - y_true) vs True Uncapped RUL across the Ladder
    ax2 = axes[1]
    sub = dev_df[dev_df["rul_uncapped"] <= 200].copy()
    sub["err_age"] = oof_preds_reg["rung2_age_linear"][sub.index] - sub["rul"]
    sub["err_ridge"] = oof_preds_reg["rung4_trends_k10_ridge"][sub.index] - sub["rul"]
    sub["err_hgb"] = oof_preds_reg["rung5_trends_k30_hgb"][sub.index] - sub["rul"]

    # Bin by 5-cycle intervals of true uncapped RUL
    sub["rul_bin"] = (sub["rul_uncapped"] // 5) * 5
    grp = sub.groupby("rul_bin")

    ax2.axhline(0.0, color="black", linestyle="--", linewidth=1.2, label="Zero Error (Ideal)")
    ax2.plot(grp.mean().index, grp["err_age"].mean(), color="#ff7f0e", linestyle=":", linewidth=2.0, label="R2: Age-Only Linear")
    ax2.plot(grp.mean().index, grp["err_ridge"].mean(), color="#1f77b4", linestyle="--", linewidth=2.0, label="R4: Trailing Trends Ridge")
    ax2.plot(grp.mean().index, grp["err_hgb"].mean(), color="#2ca02c", linewidth=2.4, label="R5: Trailing Trends HGB (Champion)")

    q25 = grp["err_hgb"].quantile(0.25)
    q75 = grp["err_hgb"].quantile(0.75)
    ax2.fill_between(grp.mean().index, q25, q75, color="#2ca02c", alpha=0.18, label="HGB Interquartile Error Band")
    ax2.axvline(30, color="#d62728", linestyle="-.", linewidth=1.3, label="Alarm Window (RUL = 30)")
    ax2.axvline(125, color="gray", linestyle=":", linewidth=1.2, label="RUL Cap (125)")

    ax2.invert_xaxis()
    ax2.set_xlabel("True Uncapped RUL (Cycles Remaining Until Failure)")
    ax2.set_ylabel("Prediction Error $\\widehat{\\mathrm{RUL}} - \\mathrm{RUL}$ (Cycles)")
    ax2.set_title("B. Error vs. True RUL: Over-Prediction Risk Near Failure")
    ax2.legend(loc="lower left", frameon=True, fontsize=8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
