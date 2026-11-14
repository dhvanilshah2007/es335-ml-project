"""Decision-Centric Cost Policy Simulation & Single Held-Out Test Evaluation.

Usage:
    python scripts/run_test_evaluation.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.cost_simulation import (
    evaluate_locked_classification_ladder_on_test,
    evaluate_locked_regression_ladder_on_test,
    evaluate_official_fd001_benchmark,
    extract_per_engine_policy_outcomes,
    run_cost_ratio_sweep,
    run_engine_bootstrap_ci,
    run_row_vs_engine_leakage_experiment,
)
from src.data import (
    load_config,
    load_development_data,
    load_heldout_test_data,
    load_official_benchmark_test_data,
)
from src.models import (
    run_group_cv_classification_ladder,
    run_group_cv_regression_ladder,
)
from src.plotting import (
    plot_cost_curve_vs_fixed_schedule,
    plot_leakage_comparison,
)


def main() -> int:
    cfg = load_config()
    rul_cap = int(cfg["framing"]["rul_cap"])
    alarm_w = int(cfg["framing"]["alarm_w"])
    window_k = int(cfg["features"]["window_k"])
    cv_folds = int(cfg["split"]["cv_folds"])
    c_waste = float(cfg["cost_simulation"]["c_waste"])
    default_c_fail = float(cfg["cost_simulation"]["default_c_fail"])
    cost_ratios = [float(r) for r in cfg["cost_simulation"]["cost_ratios"]]
    n_boot = int(cfg["evaluation"]["bootstrap_iterations"])
    seed = int(cfg["project"]["random_seed"])

    print("=" * 84)
    print(f"COST POLICY SIMULATION & HELD-OUT TEST EVALUATION (k={window_k})")
    print("=" * 84)

    dev_df = load_development_data(rul_cap=rul_cap, alarm_w=alarm_w)
    test_df = load_heldout_test_data(rul_cap=rul_cap, alarm_w=alarm_w)
    official_test_df = load_official_benchmark_test_data(rul_cap=rul_cap, alarm_w=alarm_w)
    print(
        f"[1/6] Loaded 80 Dev ({len(dev_df):,} rows), 20 Held-Out Test ({len(test_df):,} rows), "
        f"and 100 Official Truncated Test engines ({len(official_test_df):,} rows)."
    )

    cv_reg_df, oof_preds = run_group_cv_regression_ladder(
        dev_df, cv_folds=cv_folds, window_k=window_k, rul_cap=rul_cap,
        c_fail=default_c_fail, c_waste=c_waste, alarm_w=alarm_w,
    )
    cv_clf_df, oof_probs = run_group_cv_classification_ladder(
        dev_df, cv_folds=cv_folds, window_k=window_k,
        c_fail=default_c_fail, c_waste=c_waste, alarm_w=alarm_w,
    )

    print("\n[2/6] Evaluating locked Regression & Classification Ladders on 20 Held-Out Test engines...")
    test_reg_df, test_preds, fitted_reg_pipes = evaluate_locked_regression_ladder_on_test(
        dev_df=dev_df, test_df=test_df, cv_reg_df=cv_reg_df,
        window_k=window_k, rul_cap=rul_cap, c_fail=default_c_fail, c_waste=c_waste, alarm_w=alarm_w,
    )
    test_clf_df, test_probs, _ = evaluate_locked_classification_ladder_on_test(
        dev_df=dev_df, test_df=test_df, cv_clf_df=cv_clf_df,
        window_k=window_k, c_fail=default_c_fail, c_waste=c_waste, alarm_w=alarm_w,
    )

    print("\n--- Held-Out Test Set: Regression Ladder (20 Unseen Engines) ---")
    print(
        test_reg_df[
            ["rung", "model_name", "dev_cv_rmse", "rmse_all", "mae_all", "r2_all", "nasa_asym_mean", "test_policy_mean_cost_c100"]
        ].to_string(index=False)
    )

    print("\n--- Held-Out Test Set: Classification & Policy Ladder (w=30) ---")
    print(
        test_clf_df[
            ["rung", "model_name", "dev_cv_pr_auc", "test_pr_auc", "test_brier_score", "locked_dev_threshold_p", "test_policy_mean_cost_c100", "test_failure_rate_pct", "test_mean_wasted_cycles"]
        ].to_string(index=False)
    )

    print("\n[3/6] Running Cost Ratio Sweep & Break-Even Analysis...")
    sweep_df, breakeven_info = run_cost_ratio_sweep(
        dev_df=dev_df, test_df=test_df, oof_preds=oof_preds, oof_probs=oof_probs,
        test_preds=test_preds, test_probs=test_probs, cost_ratios=cost_ratios,
        c_waste=c_waste, alarm_w=alarm_w,
    )
    print(
        sweep_df[
            ["cost_ratio", "fixed_K_dev", "fixed_test_cost", "champ_clf_p_dev", "champ_clf_test_cost", "champ_clf_savings_vs_fixed_pct", "tree_clf_test_cost", "champ_reg_test_cost"]
        ].to_string(index=False)
    )
    print("\nBreak-Even Cost Ratios:")
    for k, v in breakeven_info.items():
        print(f"  - {k}: {v}:1")

    print(f"\n[4/6] Running {n_boot:,} Engine-Level Bootstrap Resamples...")
    locked_params = {
        "fixed_k_dev": float(test_clf_df.loc[test_clf_df["model_id"] == "rung1_fixed_schedule", "locked_dev_threshold_p"].iloc[0]),
        "champ_clf_p_dev": float(test_clf_df.loc[test_clf_df["model_id"] == "rung4_trends_logreg", "locked_dev_threshold_p"].iloc[0]),
        "tree_clf_p_dev": float(test_clf_df.loc[test_clf_df["model_id"] == "rung5_trends_hgb_clf", "locked_dev_threshold_p"].iloc[0]),
        "champ_reg_tau_dev": float(test_reg_df.loc[test_reg_df["model_id"] == "rung5_trends_hgb", "locked_dev_tau"].iloc[0]),
    }
    boot_df, boot_samples = run_engine_bootstrap_ci(
        test_df=test_df, test_preds=test_preds, test_probs=test_probs, locked_params=locked_params,
        n_iterations=n_boot, c_fail=default_c_fail, c_waste=c_waste, alarm_w=alarm_w, seed=seed,
    )
    print(
        boot_df[["model_or_comparison", "metric_name", "test_point_estimate", "ci95_lower", "ci95_upper", "prob_positive_savings"]].to_string(index=False)
    )

    print("\n[5/6] Evaluating Secondary Sanity Check on 100 Official `test_FD001.txt` Engines...")
    official_bench_df, _ = evaluate_official_fd001_benchmark(
        fitted_reg_pipes=fitted_reg_pipes, official_test_df=official_test_df, window_k=window_k, rul_cap=rul_cap,
    )
    print(
        official_bench_df[
            ["rung", "model_name", "last_cycle_rmse_100_engines", "last_cycle_mae_100_engines", "last_cycle_r2_100_engines", "last_cycle_nasa_sum_100_engines"]
        ].to_string(index=False)
    )

    print("\n[6/6] Running Row-Split vs. Engine-Split Leakage Experiment...")
    leakage_df, scatter_payload = run_row_vs_engine_leakage_experiment(
        dev_df=dev_df, test_df=test_df, window_k=window_k, rul_cap=rul_cap, seed=seed,
    )
    print(
        leakage_df[
            ["model_label", "proper_groupkfold_cv_rmse", "leaky_random_kfold_cv_rmse", "proper_heldout_engines_test_rmse", "leaky_random_row_split_test_rmse", "artificial_error_reduction_pct"]
        ].to_string(index=False)
    )

    metrics_dir = REPO_ROOT / cfg["paths"]["results_metrics_dir"]
    processed_dir = REPO_ROOT / cfg["paths"]["processed_dir"]
    fig_dir = REPO_ROOT / cfg["paths"]["results_figures_dir"]
    metrics_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    boot_df.to_csv(metrics_dir / "test_final_metrics.csv", index=False)
    sweep_df.to_csv(metrics_dir / "cost_policy_comparison.csv", index=False)
    leakage_df.to_csv(metrics_dir / "leakage_experiment_results.csv", index=False)

    per_engine_fixed = extract_per_engine_policy_outcomes(
        test_df, None, threshold=locked_params["fixed_k_dev"], c_fail=default_c_fail, c_waste=c_waste, policy_type="fixed"
    )
    per_engine_clf = extract_per_engine_policy_outcomes(
        test_df, test_probs["rung4_trends_logreg"], threshold=locked_params["champ_clf_p_dev"],
        c_fail=default_c_fail, c_waste=c_waste, policy_type="prob",
    )
    per_engine_tree = extract_per_engine_policy_outcomes(
        test_df, test_probs["rung5_trends_hgb_clf"], threshold=locked_params["tree_clf_p_dev"],
        c_fail=default_c_fail, c_waste=c_waste, policy_type="prob",
    )
    per_engine_reg = extract_per_engine_policy_outcomes(
        test_df, test_preds["rung5_trends_hgb"], threshold=locked_params["champ_reg_tau_dev"],
        c_fail=default_c_fail, c_waste=c_waste, policy_type="rul",
    )

    test_summary = {
        "module": "Decision-Centric Cost Policy Simulation & Held-Out Test Evaluation",
        "primary_window_k": window_k,
        "heldout_test_engines_count": 20,
        "heldout_test_rows": int(len(test_df)),
        "locked_parameters_c100": locked_params,
        "test_regression_ladder": test_reg_df.to_dict(orient="records"),
        "test_classification_ladder": test_clf_df.to_dict(orient="records"),
        "breakeven_analysis": breakeven_info,
        "bootstrap_ci_summary": boot_df.to_dict(orient="records"),
        "cost_ratio_sweep": sweep_df.to_dict(orient="records"),
        "leakage_experiment": leakage_df.to_dict(orient="records"),
        "official_fd001_100_engines_benchmark": official_bench_df.to_dict(orient="records"),
        "per_engine_test_outcomes_c100": {
            "fixed_schedule": per_engine_fixed.to_dict(orient="records"),
            "champion_classifier_logreg": per_engine_clf.to_dict(orient="records"),
            "tree_classifier_hgb": per_engine_tree.to_dict(orient="records"),
            "champion_regressor_hgb": per_engine_reg.to_dict(orient="records"),
        },
    }
    with open(processed_dir / "test_evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(test_summary, f, indent=2)

    plot_cost_curve_vs_fixed_schedule(
        sweep_df=sweep_df, breakeven_info=breakeven_info, boot_samples=boot_samples,
        out_path=fig_dir / "cost_curve_vs_fixed_schedule.png",
    )
    plot_leakage_comparison(
        leakage_df=leakage_df, scatter_payload=scatter_payload,
        out_path=fig_dir / "leakage_comparison_scatter.png",
    )

    print("\nSaved Test Evaluation Outputs:")
    print(f"  - {(metrics_dir / 'test_final_metrics.csv').relative_to(REPO_ROOT)}")
    print(f"  - {(metrics_dir / 'cost_policy_comparison.csv').relative_to(REPO_ROOT)}")
    print(f"  - {(metrics_dir / 'leakage_experiment_results.csv').relative_to(REPO_ROOT)}")
    print(f"  - {(processed_dir / 'test_evaluation_summary.json').relative_to(REPO_ROOT)}")
    print(f"  - {(fig_dir / 'cost_curve_vs_fixed_schedule.png').relative_to(REPO_ROOT)}")
    print(f"  - {(fig_dir / 'leakage_comparison_scatter.png').relative_to(REPO_ROOT)}")
    print("=" * 84)
    return 0


if __name__ == "__main__":
    sys.exit(main())
