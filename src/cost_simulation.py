"""Decision-centric maintenance cost simulation, held-out test evaluation, and bootstrap CIs.

Core responsibilities:
- Unseals the 20 held-out test engines once for final evaluation of models trained strictly on the
  80 development engines using locked development thresholds (`K*_dev`, `p*_dev`, `tau*_dev`).
- Sweeps failure-to-waste cost ratios (`c_fail : c_waste`) to find break-even ratios against
  the optimal fixed-interval schedule (`K*_dev`).
- Computes 1,000-iteration engine-level bootstrap 95% confidence intervals for key test metrics
  and paired cost savings.
- Evaluates the secondary sanity-check benchmark on the 100 official `test_FD001.txt` engines.
- Demonstrates and quantifies row-level split leakage vs. engine-grouped splitting on ladder models.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
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
from sklearn.model_selection import GroupKFold, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features import build_preprocessing_pipeline
from src.models import (
    compute_regression_metrics,
    get_classification_ladder_specs,
    get_regression_ladder_specs,
    nasa_asymmetric_score,
    optimize_alarm_threshold,
    optimize_fixed_schedule_k,
    simulate_fixed_schedule_policy,
    simulate_model_alarm_policy,
)


def evaluate_locked_regression_ladder_on_test(
    dev_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cv_reg_df: pd.DataFrame,
    window_k: int = 30,
    rul_cap: int = 125,
    c_fail: float = 100.0,
    c_waste: float = 1.0,
    alarm_w: int = 30,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, Pipeline]]:
    """Fit all regression ladder models on 80 dev engines and evaluate once on 20 test engines."""
    specs = get_regression_ladder_specs(window_k=window_k)
    tau_map = dict(zip(cv_reg_df["model_id"], cv_reg_df["optimal_rul_trigger_tau"]))
    cv_rmse_map = dict(zip(cv_reg_df["model_id"], cv_reg_df["rmse_all"]))
    cv_cost_map = dict(zip(cv_reg_df["model_id"], cv_reg_df["policy_mean_cost_c100"]))

    test_preds: dict[str, np.ndarray] = {}
    fitted_pipes: dict[str, Pipeline] = {}
    records: list[dict[str, Any]] = []

    for spec in specs:
        mid = spec["model_id"]
        prep = build_preprocessing_pipeline(
            window_k=spec["window_k"],
            feature_mode=spec["feature_mode"],
            scale=spec["scale"],
        )
        full_pipe = Pipeline([("prep", prep), ("model", clone(spec["estimator"]))])
        full_pipe.fit(dev_df, dev_df["rul"])
        fitted_pipes[mid] = full_pipe

        preds_test = np.clip(full_pipe.predict(test_df), 0.0, float(rul_cap))
        test_preds[mid] = preds_test

        reg_metrics = compute_regression_metrics(test_df, preds_test, rul_cap=rul_cap)
        locked_tau = float(tau_map[mid])
        pol_stats = simulate_model_alarm_policy(
            test_df,
            scores=preds_test,
            threshold=locked_tau,
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
                "dev_cv_rmse": round(float(cv_rmse_map[mid]), 3),
                **reg_metrics,
                "locked_dev_tau": locked_tau,
                "dev_cv_policy_cost_c100": round(float(cv_cost_map[mid]), 2),
                "test_policy_mean_cost_c100": pol_stats["mean_cost_per_engine"],
                "test_failure_rate_pct": pol_stats["failure_rate_pct"],
                "test_in_service_failures": pol_stats["in_service_failures"],
                "test_mean_wasted_cycles": pol_stats["mean_wasted_cycles"],
                "test_premature_alarm_rate_pct": pol_stats["premature_alarm_rate_pct"],
            }
        )

    return pd.DataFrame(records), test_preds, fitted_pipes


def evaluate_locked_classification_ladder_on_test(
    dev_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cv_clf_df: pd.DataFrame,
    window_k: int = 30,
    c_fail: float = 100.0,
    c_waste: float = 1.0,
    alarm_w: int = 30,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, Pipeline]]:
    """Fit all classification ladder models on 80 dev engines and evaluate once on 20 test engines."""
    specs = get_classification_ladder_specs(window_k=window_k)
    p_map = dict(zip(cv_clf_df["model_id"], cv_clf_df["optimal_threshold_p"]))
    cv_prauc_map = dict(zip(cv_clf_df["model_id"], cv_clf_df["pr_auc"]))
    cv_cost_map = dict(zip(cv_clf_df["model_id"], cv_clf_df["oof_policy_mean_cost_c100"]))

    y_true = test_df["alarm"].to_numpy(dtype=int)
    test_probs: dict[str, np.ndarray] = {}
    fitted_pipes: dict[str, Pipeline] = {}
    records: list[dict[str, Any]] = []

    k_dev_locked = int(p_map["rung1_fixed_schedule"])
    fixed_test_stats = simulate_fixed_schedule_policy(
        test_df, k_cycle=k_dev_locked, c_fail=c_fail, c_waste=c_waste
    )
    records.append(
        {
            "model_id": "rung1_fixed_schedule",
            "rung": "Rung 1",
            "model_name": f"Fixed Replacement Schedule (Locked K*_dev={k_dev_locked} cyc)",
            "feature_mode": "cycle_timer",
            "window_k": 0,
            "dev_cv_pr_auc": np.nan,
            "test_pr_auc": np.nan,
            "test_roc_auc": np.nan,
            "test_brier_score": np.nan,
            "test_precision_at_locked_p": np.nan,
            "test_recall_at_locked_p": np.nan,
            "test_f1_at_locked_p": np.nan,
            "locked_dev_threshold_p": float(k_dev_locked),
            "dev_cv_policy_cost_c100": round(float(cv_cost_map["rung1_fixed_schedule"]), 2),
            "test_policy_mean_cost_c100": fixed_test_stats["mean_cost_per_engine"],
            "test_failure_rate_pct": fixed_test_stats["failure_rate_pct"],
            "test_in_service_failures": fixed_test_stats["in_service_failures"],
            "test_premature_alarm_rate_pct": fixed_test_stats["premature_alarm_rate_pct"],
            "test_mean_wasted_cycles": fixed_test_stats["mean_wasted_cycles"],
        }
    )

    for spec in specs:
        mid = spec["model_id"]
        prep = build_preprocessing_pipeline(
            window_k=spec["window_k"],
            feature_mode=spec["feature_mode"],
            scale=spec["scale"],
        )
        full_pipe = Pipeline([("prep", prep), ("model", clone(spec["estimator"]))])
        full_pipe.fit(dev_df, dev_df["alarm"])
        fitted_pipes[mid] = full_pipe

        probs_test = full_pipe.predict_proba(test_df)[:, 1]
        test_probs[mid] = probs_test

        pr_auc = float(average_precision_score(y_true, probs_test))
        roc_auc = float(roc_auc_score(y_true, probs_test)) if len(np.unique(probs_test)) > 1 else 0.5
        brier = float(brier_score_loss(y_true, probs_test))

        locked_p = float(p_map[mid])
        pol_stats = simulate_model_alarm_policy(
            test_df,
            scores=probs_test,
            threshold=locked_p,
            c_fail=c_fail,
            c_waste=c_waste,
            score_is_rul=False,
            alarm_w=alarm_w,
        )

        if mid == "rung0_prior_clf":
            prec, rec, f1 = 0.0, 0.0, 0.0
        else:
            y_hat = (probs_test >= locked_p).astype(int)
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
                "dev_cv_pr_auc": round(float(cv_prauc_map[mid]), 4),
                "test_pr_auc": round(pr_auc, 4),
                "test_roc_auc": round(roc_auc, 4),
                "test_brier_score": round(brier, 4),
                "test_precision_at_locked_p": round(prec, 4),
                "test_recall_at_locked_p": round(rec, 4),
                "test_f1_at_locked_p": round(f1, 4),
                "locked_dev_threshold_p": locked_p,
                "dev_cv_policy_cost_c100": round(float(cv_cost_map[mid]), 2),
                "test_policy_mean_cost_c100": pol_stats["mean_cost_per_engine"],
                "test_failure_rate_pct": pol_stats["failure_rate_pct"],
                "test_in_service_failures": pol_stats["in_service_failures"],
                "test_premature_alarm_rate_pct": pol_stats["premature_alarm_rate_pct"],
                "test_mean_wasted_cycles": pol_stats["mean_wasted_cycles"],
            }
        )

    results_df = pd.DataFrame(records)
    rung_order = {"Rung 0": 0, "Rung 1": 1, "Rung 2": 2, "Rung 3": 3, "Rung 4": 4, "Rung 5": 5}
    results_df["_order"] = results_df["rung"].map(rung_order)
    results_df = (
        results_df.sort_values(["_order", "test_policy_mean_cost_c100"])
        .drop(columns=["_order"])
        .reset_index(drop=True)
    )
    return results_df, test_probs, fitted_pipes


def extract_per_engine_policy_outcomes(
    df: pd.DataFrame,
    scores: np.ndarray | None,
    threshold: float,
    c_fail: float = 100.0,
    c_waste: float = 1.0,
    policy_type: str = "prob",
    alarm_w: int = 30,
) -> pd.DataFrame:
    """Compute per-engine replacement cycle, wasted life, failure status, and cost on `df`."""
    records: list[dict[str, Any]] = []
    work = df[["unit_id", "cycle", "max_cycle", "rul_uncapped"]].copy()
    if scores is not None:
        work["score"] = scores

    for unit_id, grp in work.groupby("unit_id", sort=True):
        t_e = int(grp["max_cycle"].iloc[0])

        if policy_type == "never":
            alarm_cycle, failed, wasted, cost = None, True, 0.0, float(c_fail)
        elif policy_type == "fixed":
            k_cyc = int(round(threshold))
            if k_cyc < t_e:
                alarm_cycle = k_cyc
                failed = False
                wasted = float(t_e - k_cyc)
                cost = float(c_waste * wasted)
            else:
                alarm_cycle, failed, wasted, cost = None, True, 0.0, float(c_fail)
        elif policy_type == "rul":
            trig = grp[(grp["score"] <= threshold) & (grp["rul_uncapped"] > 0)]
            if len(trig) > 0:
                first_row = trig.iloc[0]
                alarm_cycle = int(first_row["cycle"])
                failed = False
                wasted = float(first_row["rul_uncapped"])
                cost = float(c_waste * wasted)
            else:
                alarm_cycle, failed, wasted, cost = None, True, 0.0, float(c_fail)
        elif policy_type == "prob":
            trig = grp[(grp["score"] >= threshold) & (grp["rul_uncapped"] > 0)]
            if len(trig) > 0:
                first_row = trig.iloc[0]
                alarm_cycle = int(first_row["cycle"])
                failed = False
                wasted = float(first_row["rul_uncapped"])
                cost = float(c_waste * wasted)
            else:
                alarm_cycle, failed, wasted, cost = None, True, 0.0, float(c_fail)
        else:
            raise ValueError(f"Unknown policy_type: {policy_type}")

        records.append(
            {
                "unit_id": int(unit_id),
                "true_lifetime_Te": t_e,
                "alarm_cycle": alarm_cycle,
                "failed_in_service": bool(failed),
                "wasted_cycles": round(wasted, 2),
                "premature_alarm": bool((not failed) and (wasted > alarm_w)),
                "timely_alarm": bool((not failed) and (1 <= wasted <= alarm_w)),
                "engine_cost": round(cost, 2),
            }
        )

    return pd.DataFrame(records)


def run_cost_ratio_sweep(
    dev_df: pd.DataFrame,
    test_df: pd.DataFrame,
    oof_preds: dict[str, np.ndarray],
    oof_probs: dict[str, np.ndarray],
    test_preds: dict[str, np.ndarray],
    test_probs: dict[str, np.ndarray],
    cost_ratios: list[int | float] | None = None,
    c_waste: float = 1.0,
    alarm_w: int = 30,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Sweep cost ratios (`c_fail : c_waste`) using dev-locked thresholds on held-out test engines."""
    if cost_ratios is None:
        cost_ratios = [10, 20, 30, 50, 75, 100, 150, 200, 300]

    rows: list[dict[str, Any]] = []
    for ratio in cost_ratios:
        cf = float(ratio) * c_waste

        k_dev, fixed_dev = optimize_fixed_schedule_k(dev_df, c_fail=cf, c_waste=c_waste)
        fixed_test = simulate_fixed_schedule_policy(test_df, k_cycle=k_dev, c_fail=cf, c_waste=c_waste)

        p_age, _ = optimize_alarm_threshold(
            dev_df, oof_probs["rung2_age_logreg"], c_fail=cf, c_waste=c_waste, score_is_rul=False, alarm_w=alarm_w
        )
        age_test = simulate_model_alarm_policy(
            test_df, test_probs["rung2_age_logreg"], threshold=p_age, c_fail=cf, c_waste=c_waste, score_is_rul=False, alarm_w=alarm_w
        )

        p_champ_clf, champ_clf_dev = optimize_alarm_threshold(
            dev_df, oof_probs["rung4_trends_logreg"], c_fail=cf, c_waste=c_waste, score_is_rul=False, alarm_w=alarm_w
        )
        champ_clf_test = simulate_model_alarm_policy(
            test_df, test_probs["rung4_trends_logreg"], threshold=p_champ_clf, c_fail=cf, c_waste=c_waste, score_is_rul=False, alarm_w=alarm_w
        )

        p_tree_clf, tree_clf_dev = optimize_alarm_threshold(
            dev_df, oof_probs["rung5_trends_hgb_clf"], c_fail=cf, c_waste=c_waste, score_is_rul=False, alarm_w=alarm_w
        )
        tree_clf_test = simulate_model_alarm_policy(
            test_df, test_probs["rung5_trends_hgb_clf"], threshold=p_tree_clf, c_fail=cf, c_waste=c_waste, score_is_rul=False, alarm_w=alarm_w
        )

        tau_champ_reg, champ_reg_dev = optimize_alarm_threshold(
            dev_df, oof_preds["rung5_trends_hgb"], c_fail=cf, c_waste=c_waste, score_is_rul=True, alarm_w=alarm_w
        )
        champ_reg_test = simulate_model_alarm_policy(
            test_df, test_preds["rung5_trends_hgb"], threshold=tau_champ_reg, c_fail=cf, c_waste=c_waste, score_is_rul=True, alarm_w=alarm_w
        )

        fixed_test_cost = fixed_test["mean_cost_per_engine"]
        clf_savings_pct = (
            (fixed_test_cost - champ_clf_test["mean_cost_per_engine"]) / fixed_test_cost * 100.0
            if fixed_test_cost > 0
            else 0.0
        )
        tree_savings_pct = (
            (fixed_test_cost - tree_clf_test["mean_cost_per_engine"]) / fixed_test_cost * 100.0
            if fixed_test_cost > 0
            else 0.0
        )
        reg_savings_pct = (
            (fixed_test_cost - champ_reg_test["mean_cost_per_engine"]) / fixed_test_cost * 100.0
            if fixed_test_cost > 0
            else 0.0
        )

        rows.append(
            {
                "cost_ratio": int(ratio),
                "c_fail": cf,
                "c_waste": c_waste,
                "run_to_failure_cost": cf,
                "fixed_K_dev": k_dev,
                "fixed_dev_cost": fixed_dev["mean_cost_per_engine"],
                "fixed_test_cost": fixed_test_cost,
                "fixed_test_fail_pct": fixed_test["failure_rate_pct"],
                "fixed_test_wasted_cycles": fixed_test["mean_wasted_cycles"],
                "age_logreg_p_dev": p_age,
                "age_logreg_test_cost": age_test["mean_cost_per_engine"],
                "champ_clf_p_dev": p_champ_clf,
                "champ_clf_dev_cost": champ_clf_dev["mean_cost_per_engine"],
                "champ_clf_test_cost": champ_clf_test["mean_cost_per_engine"],
                "champ_clf_test_fail_pct": champ_clf_test["failure_rate_pct"],
                "champ_clf_test_wasted_cycles": champ_clf_test["mean_wasted_cycles"],
                "champ_clf_savings_vs_fixed_pct": round(clf_savings_pct, 2),
                "tree_clf_p_dev": p_tree_clf,
                "tree_clf_dev_cost": tree_clf_dev["mean_cost_per_engine"],
                "tree_clf_test_cost": tree_clf_test["mean_cost_per_engine"],
                "tree_clf_test_fail_pct": tree_clf_test["failure_rate_pct"],
                "tree_clf_test_wasted_cycles": tree_clf_test["mean_wasted_cycles"],
                "tree_clf_savings_vs_fixed_pct": round(tree_savings_pct, 2),
                "champ_reg_tau_dev": tau_champ_reg,
                "champ_reg_dev_cost": champ_reg_dev["mean_cost_per_engine"],
                "champ_reg_test_cost": champ_reg_test["mean_cost_per_engine"],
                "champ_reg_test_fail_pct": champ_reg_test["failure_rate_pct"],
                "champ_reg_test_wasted_cycles": champ_reg_test["mean_wasted_cycles"],
                "champ_reg_savings_vs_fixed_pct": round(reg_savings_pct, 2),
            }
        )

    sweep_df = pd.DataFrame(rows)

    def _find_breakeven(ratios_arr: np.ndarray, fixed_arr: np.ndarray, model_arr: np.ndarray) -> float:
        diff = fixed_arr - model_arr
        if np.all(diff >= 0):
            return float(ratios_arr[0])
        for idx in range(1, len(diff)):
            if diff[idx - 1] < 0 and diff[idx] >= 0:
                r0, r1 = float(ratios_arr[idx - 1]), float(ratios_arr[idx])
                d0, d1 = float(diff[idx - 1]), float(diff[idx])
                return round(r0 + (0.0 - d0) * (r1 - r0) / (d1 - d0), 2)
        return float("nan")

    r_arr = sweep_df["cost_ratio"].to_numpy(dtype=float)
    f_test = sweep_df["fixed_test_cost"].to_numpy(dtype=float)
    f_dev = sweep_df["fixed_dev_cost"].to_numpy(dtype=float)

    breakeven_info = {
        "champ_clf_breakeven_test_ratio": _find_breakeven(
            r_arr, f_test, sweep_df["champ_clf_test_cost"].to_numpy(dtype=float)
        ),
        "champ_clf_breakeven_dev_ratio": _find_breakeven(
            r_arr, f_dev, sweep_df["champ_clf_dev_cost"].to_numpy(dtype=float)
        ),
        "tree_clf_breakeven_test_ratio": _find_breakeven(
            r_arr, f_test, sweep_df["tree_clf_test_cost"].to_numpy(dtype=float)
        ),
        "champ_reg_breakeven_test_ratio": _find_breakeven(
            r_arr, f_test, sweep_df["champ_reg_test_cost"].to_numpy(dtype=float)
        ),
        "champ_reg_breakeven_dev_ratio": _find_breakeven(
            r_arr, f_dev, sweep_df["champ_reg_dev_cost"].to_numpy(dtype=float)
        ),
    }
    return sweep_df, breakeven_info


def run_engine_bootstrap_ci(
    test_df: pd.DataFrame,
    test_preds: dict[str, np.ndarray],
    test_probs: dict[str, np.ndarray],
    locked_params: dict[str, float],
    n_iterations: int = 1000,
    c_fail: float = 100.0,
    c_waste: float = 1.0,
    alarm_w: int = 30,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Compute 1,000-iteration engine-level bootstrap 95% CIs over the 20 held-out test engines."""
    rng = np.random.RandomState(seed)
    engine_ids = np.sort(test_df["unit_id"].unique())
    n_engines = len(engine_ids)

    engine_row_indices = {
        int(uid): np.flatnonzero(test_df["unit_id"].to_numpy() == uid) for uid in engine_ids
    }

    k_fixed = int(round(locked_params["fixed_k_dev"]))
    fixed_outcomes = extract_per_engine_policy_outcomes(
        test_df, None, threshold=k_fixed, c_fail=c_fail, c_waste=c_waste, policy_type="fixed", alarm_w=alarm_w
    ).set_index("unit_id")

    champ_clf_outcomes = extract_per_engine_policy_outcomes(
        test_df,
        test_probs["rung4_trends_logreg"],
        threshold=locked_params["champ_clf_p_dev"],
        c_fail=c_fail,
        c_waste=c_waste,
        policy_type="prob",
        alarm_w=alarm_w,
    ).set_index("unit_id")

    tree_clf_outcomes = extract_per_engine_policy_outcomes(
        test_df,
        test_probs["rung5_trends_hgb_clf"],
        threshold=locked_params["tree_clf_p_dev"],
        c_fail=c_fail,
        c_waste=c_waste,
        policy_type="prob",
        alarm_w=alarm_w,
    ).set_index("unit_id")

    champ_reg_outcomes = extract_per_engine_policy_outcomes(
        test_df,
        test_preds["rung5_trends_hgb"],
        threshold=locked_params["champ_reg_tau_dev"],
        c_fail=c_fail,
        c_waste=c_waste,
        policy_type="rul",
        alarm_w=alarm_w,
    ).set_index("unit_id")

    y_rul_all = test_df["rul"].to_numpy(dtype=np.float64)
    y_alarm_all = test_df["alarm"].to_numpy(dtype=int)

    metric_keys = [
        "champ_reg_rmse",
        "champ_reg_mae",
        "champ_reg_nasa_asym",
        "ridge_reg_rmse",
        "champ_clf_pr_auc",
        "champ_clf_brier",
        "tree_clf_pr_auc",
        "tree_clf_brier",
        "fixed_cost_c100",
        "champ_clf_cost_c100",
        "tree_clf_cost_c100",
        "champ_reg_cost_c100",
        "paired_savings_champ_clf_vs_fixed",
        "paired_savings_tree_clf_vs_fixed",
        "paired_savings_champ_reg_vs_fixed",
    ]
    boot_samples = {k: np.zeros(n_iterations) for k in metric_keys}

    for b in range(n_iterations):
        sampled_uids = rng.choice(engine_ids, size=n_engines, replace=True)
        row_idx = np.concatenate([engine_row_indices[int(u)] for u in sampled_uids])

        y_rul_b = y_rul_all[row_idx]
        y_alarm_b = y_alarm_all[row_idx]

        pred_hgb_b = test_preds["rung5_trends_hgb"][row_idx]
        pred_ridge_b = test_preds["rung4_trends_ridge"][row_idx]
        prob_logreg_b = test_probs["rung4_trends_logreg"][row_idx]
        prob_hgb_clf_b = test_probs["rung5_trends_hgb_clf"][row_idx]

        boot_samples["champ_reg_rmse"][b] = root_mean_squared_error(y_rul_b, pred_hgb_b)
        boot_samples["champ_reg_mae"][b] = mean_absolute_error(y_rul_b, pred_hgb_b)
        boot_samples["champ_reg_nasa_asym"][b] = nasa_asymmetric_score(y_rul_b, pred_hgb_b, return_mean=True)
        boot_samples["ridge_reg_rmse"][b] = root_mean_squared_error(y_rul_b, pred_ridge_b)

        boot_samples["champ_clf_pr_auc"][b] = average_precision_score(y_alarm_b, prob_logreg_b)
        boot_samples["champ_clf_brier"][b] = brier_score_loss(y_alarm_b, prob_logreg_b)
        boot_samples["tree_clf_pr_auc"][b] = average_precision_score(y_alarm_b, prob_hgb_clf_b)
        boot_samples["tree_clf_brier"][b] = brier_score_loss(y_alarm_b, prob_hgb_clf_b)

        c_fix = float(fixed_outcomes.loc[sampled_uids, "engine_cost"].mean())
        c_clf = float(champ_clf_outcomes.loc[sampled_uids, "engine_cost"].mean())
        c_tree = float(tree_clf_outcomes.loc[sampled_uids, "engine_cost"].mean())
        c_reg = float(champ_reg_outcomes.loc[sampled_uids, "engine_cost"].mean())

        boot_samples["fixed_cost_c100"][b] = c_fix
        boot_samples["champ_clf_cost_c100"][b] = c_clf
        boot_samples["tree_clf_cost_c100"][b] = c_tree
        boot_samples["champ_reg_cost_c100"][b] = c_reg

        boot_samples["paired_savings_champ_clf_vs_fixed"][b] = c_fix - c_clf
        boot_samples["paired_savings_tree_clf_vs_fixed"][b] = c_fix - c_tree
        boot_samples["paired_savings_champ_reg_vs_fixed"][b] = c_fix - c_reg

    point_estimates = {
        "champ_reg_rmse": float(root_mean_squared_error(y_rul_all, test_preds["rung5_trends_hgb"])),
        "champ_reg_mae": float(mean_absolute_error(y_rul_all, test_preds["rung5_trends_hgb"])),
        "champ_reg_nasa_asym": float(nasa_asymmetric_score(y_rul_all, test_preds["rung5_trends_hgb"], return_mean=True)),
        "ridge_reg_rmse": float(root_mean_squared_error(y_rul_all, test_preds["rung4_trends_ridge"])),
        "champ_clf_pr_auc": float(average_precision_score(y_alarm_all, test_probs["rung4_trends_logreg"])),
        "champ_clf_brier": float(brier_score_loss(y_alarm_all, test_probs["rung4_trends_logreg"])),
        "tree_clf_pr_auc": float(average_precision_score(y_alarm_all, test_probs["rung5_trends_hgb_clf"])),
        "tree_clf_brier": float(brier_score_loss(y_alarm_all, test_probs["rung5_trends_hgb_clf"])),
        "fixed_cost_c100": float(fixed_outcomes["engine_cost"].mean()),
        "champ_clf_cost_c100": float(champ_clf_outcomes["engine_cost"].mean()),
        "tree_clf_cost_c100": float(tree_clf_outcomes["engine_cost"].mean()),
        "champ_reg_cost_c100": float(champ_reg_outcomes["engine_cost"].mean()),
        "paired_savings_champ_clf_vs_fixed": float(
            fixed_outcomes["engine_cost"].mean() - champ_clf_outcomes["engine_cost"].mean()
        ),
        "paired_savings_tree_clf_vs_fixed": float(
            fixed_outcomes["engine_cost"].mean() - tree_clf_outcomes["engine_cost"].mean()
        ),
        "paired_savings_champ_reg_vs_fixed": float(
            fixed_outcomes["engine_cost"].mean() - champ_reg_outcomes["engine_cost"].mean()
        ),
    }

    descriptions = {
        "champ_reg_rmse": ("Champion Regressor (HistGBR)", "Test RMSE (cycles)"),
        "champ_reg_mae": ("Champion Regressor (HistGBR)", "Test MAE (cycles)"),
        "champ_reg_nasa_asym": ("Champion Regressor (HistGBR)", "Test NASA Asymmetric Score (Mean)"),
        "ridge_reg_rmse": ("Linear Regressor (Ridge)", "Test RMSE (cycles)"),
        "champ_clf_pr_auc": ("Champion Classifier (Logistic L2)", "Test PR-AUC"),
        "champ_clf_brier": ("Champion Classifier (Logistic L2)", "Test Brier Score"),
        "tree_clf_pr_auc": ("Tree Classifier (HistGBC)", "Test PR-AUC"),
        "tree_clf_brier": ("Tree Classifier (HistGBC)", "Test Brier Score"),
        "fixed_cost_c100": (f"Locked Fixed Schedule (K*_dev={k_fixed})", "Test Policy Cost per Engine (c=100)"),
        "champ_clf_cost_c100": ("Champion Classifier Policy (Logistic L2)", "Test Policy Cost per Engine (c=100)"),
        "tree_clf_cost_c100": ("Tree Classifier Policy (HistGBC)", "Test Policy Cost per Engine (c=100)"),
        "champ_reg_cost_c100": ("Champion Regressor Policy (HistGBR)", "Test Policy Cost per Engine (c=100)"),
        "paired_savings_champ_clf_vs_fixed": (
            "Paired Savings: Fixed minus Logistic Classifier",
            "Cost Reduction per Engine (c=100)",
        ),
        "paired_savings_tree_clf_vs_fixed": (
            "Paired Savings: Fixed minus HistGBC Classifier",
            "Cost Reduction per Engine (c=100)",
        ),
        "paired_savings_champ_reg_vs_fixed": (
            "Paired Savings: Fixed minus HistGBR Regressor",
            "Cost Reduction per Engine (c=100)",
        ),
    }

    summary_rows: list[dict[str, Any]] = []
    for key, arr in boot_samples.items():
        model_label, metric_label = descriptions[key]
        ci_low, ci_high = np.percentile(arr, [2.5, 97.5])
        summary_rows.append(
            {
                "metric_key": key,
                "model_or_comparison": model_label,
                "metric_name": metric_label,
                "test_point_estimate": round(point_estimates[key], 4),
                "bootstrap_mean": round(float(np.mean(arr)), 4),
                "bootstrap_std": round(float(np.std(arr)), 4),
                "ci95_lower": round(float(ci_low), 4),
                "ci95_upper": round(float(ci_high), 4),
                "prob_positive_savings": (
                    round(float(np.mean(arr > 0.0) * 100.0), 2)
                    if key.startswith("paired_savings_")
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(summary_rows), boot_samples


def evaluate_official_fd001_benchmark(
    fitted_reg_pipes: dict[str, Pipeline],
    official_test_df: pd.DataFrame,
    window_k: int = 30,
    rul_cap: int = 125,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate locked regression models on the 100 official right-censored `test_FD001.txt` engines."""
    specs = get_regression_ladder_specs(window_k=window_k)
    last_cycle_mask = (official_test_df["cycle"] == official_test_df["max_cycle"]).to_numpy()
    last_df = official_test_df.loc[last_cycle_mask].copy().reset_index(drop=True)
    y_true_last = last_df["rul"].to_numpy(dtype=np.float64)
    y_true_all = official_test_df["rul"].to_numpy(dtype=np.float64)

    records: list[dict[str, Any]] = []
    last_cycle_preds_df = last_df[["unit_id", "max_cycle", "rul_uncapped", "rul"]].copy()

    for spec in specs:
        mid = spec["model_id"]
        pipe = fitted_reg_pipes[mid]
        preds_all = np.clip(pipe.predict(official_test_df), 0.0, float(rul_cap))
        preds_last = preds_all[last_cycle_mask]
        last_cycle_preds_df[mid] = np.round(preds_last, 3)

        records.append(
            {
                "model_id": mid,
                "rung": spec["rung"],
                "model_name": spec["name"],
                "last_cycle_rmse_100_engines": round(float(root_mean_squared_error(y_true_last, preds_last)), 3),
                "last_cycle_mae_100_engines": round(float(mean_absolute_error(y_true_last, preds_last)), 3),
                "last_cycle_r2_100_engines": round(float(r2_score(y_true_last, preds_last)), 4),
                "last_cycle_nasa_sum_100_engines": round(nasa_asymmetric_score(y_true_last, preds_last, return_mean=False), 2),
                "last_cycle_nasa_mean_100_engines": round(nasa_asymmetric_score(y_true_last, preds_last, return_mean=True), 3),
                "all_13096_cycles_rmse": round(float(root_mean_squared_error(y_true_all, preds_all)), 3),
                "all_13096_cycles_mae": round(float(mean_absolute_error(y_true_all, preds_all)), 3),
            }
        )

    return pd.DataFrame(records), last_cycle_preds_df


def run_row_vs_engine_leakage_experiment(
    dev_df: pd.DataFrame,
    test_df: pd.DataFrame,
    window_k: int = 30,
    rul_cap: int = 125,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Quantify optimistic score inflation when splitting randomly by row instead of by `unit_id`.

    Compares the three actual trailing-trend ladder models (`rung4_trends_ridge`,
    `rung5_trends_hgb`, `rung5_trends_rf`) under:
    1. Proper Engine Split (5-fold `GroupKFold` on dev; 80 train engines vs 20 unseen test engines).
    2. Flawed Random Row Split (5-fold `KFold(shuffle=True)` on dev; 80/20 random row split).
    """
    full_df = pd.concat([dev_df, test_df], axis=0).sort_values(["unit_id", "cycle"]).reset_index(drop=True)

    prep_unscaled = build_preprocessing_pipeline(
        window_k=window_k, feature_mode="trailing_trends", scale=False
    )
    X_dev_proper = prep_unscaled.fit_transform(dev_df)
    X_test_proper = prep_unscaled.transform(test_df)
    y_dev_proper = dev_df["rul"].to_numpy(dtype=np.float64)
    y_test_proper = test_df["rul"].to_numpy(dtype=np.float64)

    X_full = prep_unscaled.fit_transform(full_df)
    y_full = full_df["rul"].to_numpy(dtype=np.float64)
    units_full = full_df["unit_id"].to_numpy(dtype=int)
    cycles_full = full_df["cycle"].to_numpy(dtype=int)

    idx_all = np.arange(len(full_df))
    tr_row_idx, te_row_idx = train_test_split(idx_all, test_size=0.20, random_state=seed, shuffle=True)

    X_tr_leaky, y_tr_leaky = X_full.iloc[tr_row_idx], y_full[tr_row_idx]
    X_te_leaky, y_te_leaky = X_full.iloc[te_row_idx], y_full[te_row_idx]

    spec_by_id = {s["model_id"]: s for s in get_regression_ladder_specs(window_k=window_k)}
    models_to_compare = [
        (
            f"Ridge Linear (k={window_k})",
            "rung4_trends_ridge",
            Pipeline([("scaler", StandardScaler()), ("reg", clone(spec_by_id["rung4_trends_ridge"]["estimator"]))]),
        ),
        (
            "HistGradientBoosting (depth=5)",
            "rung5_trends_hgb",
            clone(spec_by_id["rung5_trends_hgb"]["estimator"]),
        ),
        (
            "Random Forest (depth=10)",
            "rung5_trends_rf",
            clone(spec_by_id["rung5_trends_rf"]["estimator"]),
        ),
    ]

    gkf = GroupKFold(n_splits=5)
    kf_leaky = KFold(n_splits=5, shuffle=True, random_state=seed)

    records: list[dict[str, Any]] = []
    scatter_payload: dict[str, Any] = {}
    fitted_proper_models: dict[str, Any] = {}
    fitted_leaky_models: dict[str, Any] = {}

    for label, mid, est in models_to_compare:
        gkf_preds = np.zeros(len(dev_df))
        for tr_i, val_i in gkf.split(X_dev_proper, y_dev_proper, groups=dev_df["unit_id"]):
            m = clone(est).fit(X_dev_proper.iloc[tr_i], y_dev_proper[tr_i])
            gkf_preds[val_i] = np.clip(m.predict(X_dev_proper.iloc[val_i]), 0.0, float(rul_cap))
        gkf_rmse = float(root_mean_squared_error(y_dev_proper, gkf_preds))

        kf_preds = np.zeros(len(dev_df))
        for tr_i, val_i in kf_leaky.split(X_dev_proper, y_dev_proper):
            m = clone(est).fit(X_dev_proper.iloc[tr_i], y_dev_proper[tr_i])
            kf_preds[val_i] = np.clip(m.predict(X_dev_proper.iloc[val_i]), 0.0, float(rul_cap))
        kf_rmse = float(root_mean_squared_error(y_dev_proper, kf_preds))

        m_proper = clone(est).fit(X_dev_proper, y_dev_proper)
        fitted_proper_models[mid] = m_proper
        pred_test_proper = np.clip(m_proper.predict(X_test_proper), 0.0, float(rul_cap))
        test_proper_rmse = float(root_mean_squared_error(y_test_proper, pred_test_proper))
        test_proper_r2 = float(r2_score(y_test_proper, pred_test_proper))
        test_proper_asym = nasa_asymmetric_score(y_test_proper, pred_test_proper, return_mean=True)

        m_leaky = clone(est).fit(X_tr_leaky, y_tr_leaky)
        fitted_leaky_models[mid] = m_leaky
        pred_test_leaky = np.clip(m_leaky.predict(X_te_leaky), 0.0, float(rul_cap))
        test_leaky_rmse = float(root_mean_squared_error(y_te_leaky, pred_test_leaky))
        test_leaky_r2 = float(r2_score(y_te_leaky, pred_test_leaky))
        test_leaky_asym = nasa_asymmetric_score(y_te_leaky, pred_test_leaky, return_mean=True)

        inflation_pct = (test_proper_rmse - test_leaky_rmse) / test_proper_rmse * 100.0
        records.append(
            {
                "model_id": mid,
                "model_label": label,
                "proper_groupkfold_cv_rmse": round(gkf_rmse, 3),
                "leaky_random_kfold_cv_rmse": round(kf_rmse, 3),
                "proper_heldout_engines_test_rmse": round(test_proper_rmse, 3),
                "leaky_random_row_split_test_rmse": round(test_leaky_rmse, 3),
                "artificial_rmse_drop_cycles": round(test_proper_rmse - test_leaky_rmse, 3),
                "artificial_error_reduction_pct": round(inflation_pct, 2),
                "proper_test_r2": round(test_proper_r2, 4),
                "leaky_test_r2": round(test_leaky_r2, 4),
                "proper_test_nasa_asym": round(test_proper_asym, 3),
                "leaky_test_nasa_asym": round(test_leaky_asym, 3),
            }
        )

        if mid in ("rung5_trends_hgb", "rung5_trends_rf"):
            scatter_payload[mid] = {
                "y_true_proper": y_test_proper,
                "y_pred_proper": pred_test_proper,
                "y_true_leaky": y_te_leaky,
                "y_pred_leaky": pred_test_leaky,
            }

    # Capture held-out Engine 21 trajectory under Random Forest (shows interleaved train/test rows)
    sample_uid = int(test_df["unit_id"].unique()[3])
    uid_mask_full = units_full == sample_uid
    in_leaky_train = np.isin(np.flatnonzero(uid_mask_full), tr_row_idx)
    X_uid = X_full.loc[uid_mask_full]

    scatter_payload["trajectory_example"] = {
        "unit_id": sample_uid,
        "model_label": "Random Forest (depth=10)",
        "cycles": cycles_full[uid_mask_full].tolist(),
        "true_rul": y_full[uid_mask_full].tolist(),
        "in_leaky_train_mask": in_leaky_train.tolist(),
        "pred_proper_unseen_engine": np.clip(
            fitted_proper_models["rung5_trends_rf"].predict(X_uid), 0.0, float(rul_cap)
        ).tolist(),
        "pred_leaky_row_split": np.clip(
            fitted_leaky_models["rung5_trends_rf"].predict(X_uid), 0.0, float(rul_cap)
        ).tolist(),
    }
    return pd.DataFrame(records), scatter_payload
