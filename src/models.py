"""Baseline ladder, 5-fold GroupKFold cross-validation, and threshold/policy optimization.

Core responsibilities:
- Implements the NASA Asymmetric Scoring function and regression/classification evaluation metrics.
- Evaluates Rungs 0 through 5 of the streamlined baseline ladder strictly on the 80 development
  engines using 5-fold `GroupKFold` grouped by `unit_id`.
- Optimizes the fixed replacement schedule `K*` and model alarm thresholds `p*` / `tau*` on
  development folds across failure-to-waste cost ratios (`c_fail : c_waste`).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
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
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

from src.features import build_preprocessing_pipeline


def nasa_asymmetric_score(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    return_mean: bool = True,
) -> float:
    """Compute the NASA C-MAPSS asymmetric scoring penalty.

    Let d_i = y_pred_i - y_true_i:
      - If d_i < 0 (early prediction / under-predicting RUL): exp(-d_i / 13) - 1
      - If d_i >= 0 (late prediction / over-predicting RUL):  exp(+d_i / 10) - 1
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
    """Compute RUL regression metrics across all cycles and the near-failure zone (RUL <= 50)."""
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
    """Simulate a fixed-interval replacement policy at cycle K across all engines in `df`."""
    lifetimes = df.groupby("unit_id")["max_cycle"].first().to_numpy(dtype=np.float64)

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
    """Simulate a condition-based policy that replaces each engine at its first alarm cycle."""
    work = df[["unit_id", "cycle", "max_cycle", "rul_uncapped"]].copy()
    if score_is_rul:
        work["triggered"] = (scores <= threshold) & (work["rul_uncapped"] > 0)
    else:
        work["triggered"] = (scores >= threshold) & (work["rul_uncapped"] > 0)

    total_engines = work["unit_id"].nunique()
    first_alarms = work[work["triggered"]].groupby("unit_id", sort=False).first()

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

    return {
        "policy_param": round(float(threshold), 4),
        "mean_cost_per_engine": round(total_cost / total_engines, 2),
        "total_fleet_cost": round(total_cost, 2),
        "in_service_failures": int(n_failed),
        "failure_rate_pct": round(float(n_failed / total_engines * 100.0), 2),
        "timely_replacement_rate_pct": round(float(n_timely / total_engines * 100.0), 2),
        "premature_alarm_rate_pct": round(float(n_premature / total_engines * 100.0), 2),
        "mean_wasted_cycles": round(total_wasted_cycles / total_engines, 2),
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
        grid = np.concatenate([np.linspace(0.05, 0.95, 91), [0.97, 0.98, 0.99, 1.0]])

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


def get_regression_ladder_specs(window_k: int = 30) -> list[dict[str, Any]]:
    """Return specifications for Rungs 0 to 5 of the Continuous RUL Regression Ladder."""
    return [
        {
            "model_id": "rung0_mean_rul",
            "rung": "Rung 0",
            "name": "Mean RUL Baseline (Floor)",
            "feature_mode": "age_only",
            "window_k": window_k,
            "scale": False,
            "estimator": DummyRegressor(strategy="mean"),
        },
        {
            "model_id": "rung2_age_linear",
            "rung": "Rung 2",
            "name": "Age-Only Linear Regression",
            "feature_mode": "age_only",
            "window_k": window_k,
            "scale": True,
            "estimator": Ridge(alpha=1.0),
        },
        {
            "model_id": "rung3_instant_ridge",
            "rung": "Rung 3",
            "name": "Instantaneous Sensors + Ridge (L2)",
            "feature_mode": "instantaneous",
            "window_k": window_k,
            "scale": True,
            "estimator": Ridge(alpha=100.0),
        },
        {
            "model_id": "rung3_instant_lasso",
            "rung": "Rung 3",
            "name": "Instantaneous Sensors + Lasso (L1)",
            "feature_mode": "instantaneous",
            "window_k": window_k,
            "scale": True,
            "estimator": Lasso(alpha=0.5, max_iter=5000, random_state=42),
        },
        {
            "model_id": "rung4_trends_ridge",
            "rung": "Rung 4",
            "name": f"Trailing Trends (k={window_k}) + Ridge (L2)",
            "feature_mode": "trailing_trends",
            "window_k": window_k,
            "scale": True,
            "estimator": Ridge(alpha=200.0),
        },
        {
            "model_id": "rung4_trends_lasso",
            "rung": "Rung 4",
            "name": f"Trailing Trends (k={window_k}) + Lasso (L1)",
            "feature_mode": "trailing_trends",
            "window_k": window_k,
            "scale": True,
            "estimator": Lasso(alpha=0.5, max_iter=5000, random_state=42),
        },
        {
            "model_id": "rung5_trends_rf",
            "rung": "Rung 5",
            "name": f"Trailing Trends (k={window_k}) + Random Forest",
            "feature_mode": "trailing_trends",
            "window_k": window_k,
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
            "model_id": "rung5_trends_hgb",
            "rung": "Rung 5",
            "name": f"Trailing Trends (k={window_k}) + HistGradientBoosting",
            "feature_mode": "trailing_trends",
            "window_k": window_k,
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


def get_classification_ladder_specs(window_k: int = 30) -> list[dict[str, Any]]:
    """Return specifications for Rungs 0 to 5 of the Binary Alarm Classification Ladder."""
    return [
        {
            "model_id": "rung0_prior_clf",
            "rung": "Rung 0",
            "name": "Never Alarm / Base-Rate Prior (Floor)",
            "feature_mode": "age_only",
            "window_k": window_k,
            "scale": False,
            "estimator": DummyClassifier(strategy="prior"),
        },
        {
            "model_id": "rung2_age_logreg",
            "rung": "Rung 2",
            "name": "Age-Only Logistic Regression",
            "feature_mode": "age_only",
            "window_k": window_k,
            "scale": True,
            "estimator": LogisticRegression(C=1.0, max_iter=1000, random_state=42),
        },
        {
            "model_id": "rung3_instant_logreg",
            "rung": "Rung 3",
            "name": "Instantaneous Sensors + Logistic Regression",
            "feature_mode": "instantaneous",
            "window_k": window_k,
            "scale": True,
            "estimator": LogisticRegression(C=0.1, max_iter=1000, random_state=42),
        },
        {
            "model_id": "rung4_trends_logreg",
            "rung": "Rung 4",
            "name": f"Trailing Trends (k={window_k}) + Logistic Regression (L2)",
            "feature_mode": "trailing_trends",
            "window_k": window_k,
            "scale": True,
            "estimator": LogisticRegression(C=0.05, max_iter=1000, random_state=42),
        },
        {
            "model_id": "rung5_trends_rf_clf",
            "rung": "Rung 5",
            "name": f"Trailing Trends (k={window_k}) + Random Forest Classifier",
            "feature_mode": "trailing_trends",
            "window_k": window_k,
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
            "model_id": "rung5_trends_hgb_clf",
            "rung": "Rung 5",
            "name": f"Trailing Trends (k={window_k}) + HistGradientBoosting Classifier",
            "feature_mode": "trailing_trends",
            "window_k": window_k,
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
    window_k: int = 30,
    rul_cap: int = 125,
    c_fail: float = 100.0,
    c_waste: float = 1.0,
    alarm_w: int = 30,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Evaluate all regression ladder models using 5-fold GroupKFold on the 80 dev engines."""
    gkf = GroupKFold(n_splits=cv_folds)
    specs = get_regression_ladder_specs(window_k=window_k)

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

    return pd.DataFrame(records), oof_preds


def run_group_cv_classification_ladder(
    dev_df: pd.DataFrame,
    cv_folds: int = 5,
    window_k: int = 30,
    c_fail: float = 100.0,
    c_waste: float = 1.0,
    alarm_w: int = 30,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Evaluate Rung 0–5 alarm classifiers + Fixed Schedule K* using 5-fold GroupKFold on dev data."""
    gkf = GroupKFold(n_splits=cv_folds)
    specs = get_classification_ladder_specs(window_k=window_k)
    y_true = dev_df["alarm"].to_numpy(dtype=int)

    oof_probs: dict[str, np.ndarray] = {}
    records: list[dict[str, Any]] = []

    # Rung 1: Fixed-Interval Replacement Schedule K* (fit K* on train folds, eval on val)
    fixed_val_costs: list[float] = []
    fixed_val_fails: int = 0
    fixed_val_wasted: list[float] = []
    for tr_idx, val_idx in gkf.split(dev_df, groups=dev_df["unit_id"]):
        tr_df = dev_df.iloc[tr_idx]
        val_df = dev_df.iloc[val_idx]
        k_star, _ = optimize_fixed_schedule_k(tr_df, c_fail=c_fail, c_waste=c_waste)
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
    rung_order = {"Rung 0": 0, "Rung 1": 1, "Rung 2": 2, "Rung 3": 3, "Rung 4": 4, "Rung 5": 5}
    results_df["_order"] = results_df["rung"].map(rung_order)
    results_df = (
        results_df.sort_values(["_order", "oof_policy_mean_cost_c100"])
        .drop(columns=["_order"])
        .reset_index(drop=True)
    )
    return results_df, oof_probs


def extract_feature_attribution(
    dev_df: pd.DataFrame,
    window_k: int = 30,
) -> pd.DataFrame:
    """Extract standardized Ridge/Lasso/Logistic coefficients and HGB permutation importances."""
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
