"""CLI script for Phase 4: Baseline Ladder & 5-Fold GroupKFold Cross-Validation on 80 Dev Engines.

Usage:
    python scripts/run_phase4_cv.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))

from src.data import load_config, load_development_data
from src.models import (
    extract_feature_attribution,
    optimize_alarm_threshold,
    optimize_fixed_schedule_k,
    plot_cv_baseline_ladder_progression,
    plot_cv_feature_attribution_and_errors,
    plot_cv_pr_and_calibration_curves,
    run_group_cv_classification_ladder,
    run_group_cv_regression_ladder,
)


def main() -> int:
    cfg = load_config()
    rul_cap = int(cfg["framing"]["rul_cap"])
    alarm_w = int(cfg["framing"]["alarm_w"])
    cv_folds = int(cfg["split"]["cv_folds"])
    c_waste = float(cfg["cost_simulation"]["c_waste"])
    default_c_fail = float(cfg["cost_simulation"]["default_c_fail"])
    cost_ratios = [float(r) for r in cfg["cost_simulation"]["cost_ratios"]]

    metrics_dir = REPO_ROOT / cfg["paths"]["results_metrics_dir"]
    figures_dir = REPO_ROOT / cfg["paths"]["results_figures_dir"]
    processed_dir = REPO_ROOT / cfg["paths"]["processed_dir"]

    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 82)
    print("Phase 4: Baseline Ladder & 5-Fold GroupKFold Cross-Validation (80 Dev Engines)")
    print("=" * 82)

    t0 = time.perf_counter()
    dev_df = load_development_data(rul_cap=rul_cap, alarm_w=alarm_w)
    print(f"\n1. Loaded Development Split: {dev_df['unit_id'].nunique()} engines, {len(dev_df):,} cycles.")

    # 2. Run Continuous RUL Regression Ladder (5-Fold GroupKFold)
    reg_df, oof_preds_reg = run_group_cv_regression_ladder(
        dev_df=dev_df,
        cv_folds=cv_folds,
        rul_cap=rul_cap,
        c_fail=default_c_fail,
        c_waste=c_waste,
        alarm_w=alarm_w,
    )
    reg_csv_path = metrics_dir / "cv_regression_ladder.csv"
    reg_df.to_csv(reg_csv_path, index=False)

    print(f"\n2. Continuous RUL Regression Ladder (5-Fold GroupKFold on 80 Dev Engines):")
    print(
        f"   {'Rung':<8} | {'Model Name':<44} | {'RMSE':>6} | {'MAE':>6} | {'R2':>6} | "
        f"{'Asym(All)':>9} | {'Asym(<=50)':>10} | {'Cost(c=100)':>11}"
    )
    print("   " + "-" * 116)
    for _, r in reg_df.iterrows():
        print(
            f"   {r['rung']:<8} | {r['model_name']:<44} | {r['rmse_all']:6.2f} | {r['mae_all']:6.2f} | "
            f"{r['r2_all']:6.3f} | {r['nasa_asym_mean']:9.2f} | {r['nasa_asym_rul_le_50']:10.2f} | "
            f"{r['policy_mean_cost_c100']:11.2f}"
        )

    # 3. Run Binary Alarm Classification & Policy Ladder (5-Fold GroupKFold)
    clf_df, oof_probs_clf = run_group_cv_classification_ladder(
        dev_df=dev_df,
        cv_folds=cv_folds,
        c_fail=default_c_fail,
        c_waste=c_waste,
        alarm_w=alarm_w,
    )
    clf_csv_path = metrics_dir / "cv_classification_ladder.csv"
    clf_df.to_csv(clf_csv_path, index=False)

    print(f"\n3. Binary Alarm Classification & Policy Ladder (w = {alarm_w} cycles, 5-Fold GroupKFold):")
    print(
        f"   {'Rung':<8} | {'Model Name':<50} | {'PR-AUC':>6} | {'Brier':>6} | "
        f"{'Opt p*':>6} | {'Fail%':>6} | {'Wasted':>6} | {'Cost(c=100)':>11}"
    )
    print("   " + "-" * 116)
    for _, r in clf_df.iterrows():
        pr_str = f"{r['pr_auc']:6.4f}" if not np.isnan(r["pr_auc"]) else "   N/A"
        br_str = f"{r['brier_score']:6.4f}" if not np.isnan(r["brier_score"]) else "   N/A"
        print(
            f"   {r['rung']:<8} | {r['model_name']:<50} | {pr_str} | {br_str} | "
            f"{r['optimal_threshold_p']:6.2f} | {r['failure_rate_pct']:5.1f}% | "
            f"{r['mean_wasted_cycles']:6.1f} | {r['oof_policy_mean_cost_c100']:11.2f}"
        )

    # 4. Extract Feature Attribution (Ridge, Lasso, Logistic, HistGradientBoosting)
    imp_df = extract_feature_attribution(dev_df, window_k=10)
    imp_csv_path = metrics_dir / "cv_feature_attribution.csv"
    imp_df.to_csv(imp_csv_path, index=False)

    n_lasso_kept = int(imp_df["lasso_selected"].sum())
    print(
        f"\n4. Feature Attribution & Regularization Sparsity (k=10, 43 features):"
        f"\n   - Lasso (L1, alpha=0.5) retained {n_lasso_kept}/43 features and zeroed out {43 - n_lasso_kept} redundant features."
        f"\n   - Top 5 Features by HGB Permutation Importance:"
    )
    for _, r in imp_df.head(5).iterrows():
        print(
            f"     * {r['feature']:<22} | HGB Perm RMSE={r['hgb_perm_importance_rmse']:6.3f} "
            f"| Lasso Coef={r['lasso_coef']:+7.3f} | LogReg Coef={r['logreg_alarm_coef']:+6.3f}"
        )

    # 5. Lock Development-Tuned Thresholds Across Cost Ratios for Phase 5
    # Identify champion regression model and champion classification model on dev CV
    non_ablation_reg = reg_df[reg_df["rung"] != "Ablation"].sort_values("rmse_all")
    champ_reg_row = non_ablation_reg.iloc[0]
    champ_reg_id = str(champ_reg_row["model_id"])

    clf_models_only = clf_df[clf_df["model_id"] != "rung1_fixed_schedule"].sort_values("pr_auc", ascending=False)
    champ_clf_row = clf_models_only.iloc[0]
    champ_clf_id = str(champ_clf_row["model_id"])

    tree_clf_id = "rung5_trends_k30_hgb_clf"
    tree_clf_row = clf_df[clf_df["model_id"] == tree_clf_id].iloc[0]
    linear_clf_id = "rung4_trends_k30_logreg"
    linear_reg_id = "rung4_trends_k30_ridge"

    cost_ratio_locks: dict[str, Any] = {}
    for ratio in cost_ratios:
        c_f = ratio * c_waste
        k_star, fixed_st = optimize_fixed_schedule_k(dev_df, c_fail=c_f, c_waste=c_waste)
        p_champ, champ_clf_st = optimize_alarm_threshold(
            dev_df, oof_probs_clf[champ_clf_id], c_fail=c_f, c_waste=c_waste, score_is_rul=False, alarm_w=alarm_w
        )
        p_tree, tree_clf_st = optimize_alarm_threshold(
            dev_df, oof_probs_clf[tree_clf_id], c_fail=c_f, c_waste=c_waste, score_is_rul=False, alarm_w=alarm_w
        )
        p_lin, lin_clf_st = optimize_alarm_threshold(
            dev_df, oof_probs_clf[linear_clf_id], c_fail=c_f, c_waste=c_waste, score_is_rul=False, alarm_w=alarm_w
        )
        tau_champ, champ_reg_st = optimize_alarm_threshold(
            dev_df, oof_preds_reg[champ_reg_id], c_fail=c_f, c_waste=c_waste, score_is_rul=True, alarm_w=alarm_w
        )
        cost_ratio_locks[str(int(ratio))] = {
            "c_fail": c_f,
            "c_waste": c_waste,
            "fixed_schedule_K_dev": int(k_star),
            "fixed_dev_mean_cost": fixed_st["mean_cost_per_engine"],
            "champion_clf_p_dev": p_champ,
            "champion_clf_dev_mean_cost": champ_clf_st["mean_cost_per_engine"],
            "tree_clf_p_dev": p_tree,
            "tree_clf_dev_mean_cost": tree_clf_st["mean_cost_per_engine"],
            "linear_clf_p_dev": p_lin,
            "linear_clf_dev_mean_cost": lin_clf_st["mean_cost_per_engine"],
            "champion_reg_tau_dev": tau_champ,
            "champion_reg_dev_mean_cost": champ_reg_st["mean_cost_per_engine"],
        }

    champion_payload = {
        "phase": "Phase 4: Baseline Ladder & 5-Fold GroupKFold Cross-Validation",
        "dev_engines_count": 80,
        "cv_folds": cv_folds,
        "default_cost_ratio": int(default_c_fail / c_waste),
        "champion_regression_model": {
            "model_id": champ_reg_id,
            "model_name": str(champ_reg_row["model_name"]),
            "feature_mode": str(champ_reg_row["feature_mode"]),
            "window_k": int(champ_reg_row["window_k"]),
            "cv_rmse": float(champ_reg_row["rmse_all"]),
            "cv_mae": float(champ_reg_row["mae_all"]),
            "cv_r2": float(champ_reg_row["r2_all"]),
            "cv_nasa_asym_mean": float(champ_reg_row["nasa_asym_mean"]),
        },
        "champion_classification_model": {
            "model_id": champ_clf_id,
            "model_name": str(champ_clf_row["model_name"]),
            "feature_mode": str(champ_clf_row["feature_mode"]),
            "window_k": int(champ_clf_row["window_k"]),
            "cv_pr_auc": float(champ_clf_row["pr_auc"]),
            "cv_brier_score": float(champ_clf_row["brier_score"]),
            "cv_policy_mean_cost_c100": float(champ_clf_row["oof_policy_mean_cost_c100"]),
        },
        "tree_classification_model": {
            "model_id": tree_clf_id,
            "model_name": str(tree_clf_row["model_name"]),
            "feature_mode": str(tree_clf_row["feature_mode"]),
            "window_k": int(tree_clf_row["window_k"]),
            "cv_pr_auc": float(tree_clf_row["pr_auc"]),
            "cv_brier_score": float(tree_clf_row["brier_score"]),
            "cv_policy_mean_cost_c100": float(tree_clf_row["oof_policy_mean_cost_c100"]),
        },
        "linear_benchmark_models": {
            "regression_id": linear_reg_id,
            "classification_id": linear_clf_id,
        },
        "locked_dev_thresholds_by_cost_ratio": cost_ratio_locks,
    }

    champ_json_path = processed_dir / "champion_cv_config.json"
    with open(champ_json_path, "w", encoding="utf-8") as f:
        json.dump(champion_payload, f, indent=2)

    print(
        f"\n5. Champion Models Selected on 80 Development Engines (Locked for Phase 5):"
        f"\n   - Champion Regressor  : {champ_reg_row['model_name']} "
        f"(RMSE = {champ_reg_row['rmse_all']:.2f} cyc, Asym = {champ_reg_row['nasa_asym_mean']:.2f})"
        f"\n   - Champion Classifier : {champ_clf_row['model_name']} "
        f"(PR-AUC = {champ_clf_row['pr_auc']:.4f}, Cost@100 = {champ_clf_row['oof_policy_mean_cost_c100']:.2f} vs Fixed K* Cost = {clf_df.loc[clf_df['model_id']=='rung1_fixed_schedule', 'oof_policy_mean_cost_c100'].iloc[0]:.2f})"
    )

    # 6. Generate Phase 4 Diagnostic Plots
    fig1_path = figures_dir / "cv_baseline_ladder_progression.png"
    fig2_path = figures_dir / "cv_pr_and_calibration_curves.png"
    fig3_path = figures_dir / "cv_feature_importance_and_lasso.png"

    plot_cv_baseline_ladder_progression(reg_df, clf_df, fig1_path)
    plot_cv_pr_and_calibration_curves(dev_df, oof_probs_clf, clf_df, fig2_path)
    plot_cv_feature_attribution_and_errors(dev_df, imp_df, oof_preds_reg, fig3_path)

    elapsed = time.perf_counter() - t0
    print("\n6. Saved Phase 4 Tables, Configs & Figures:")
    print(f"   - Regression ladder CSV     : {reg_csv_path.relative_to(REPO_ROOT)}")
    print(f"   - Classification ladder CSV : {clf_csv_path.relative_to(REPO_ROOT)}")
    print(f"   - Feature attribution CSV   : {imp_csv_path.relative_to(REPO_ROOT)}")
    print(f"   - Locked CV champion JSON   : {champ_json_path.relative_to(REPO_ROOT)}")
    print(f"   - Figure 1 (Ladder Prog.)   : {fig1_path.relative_to(REPO_ROOT)}")
    print(f"   - Figure 2 (PR & Calib.)    : {fig2_path.relative_to(REPO_ROOT)}")
    print(f"   - Figure 3 (Import. & Err.) : {fig3_path.relative_to(REPO_ROOT)}")
    print("=" * 82)
    print(f"[SUCCESS] Phase 4 completed in {elapsed:.2f}s strictly on 80 development engines.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
