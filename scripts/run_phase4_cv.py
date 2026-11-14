"""CLI script for Phase 4: Baseline Ladder & 5-Fold GroupKFold Cross-Validation on 80 Dev Engines.

Usage:
    python scripts/run_phase4_cv.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))

from src.data import load_config, load_development_data
from src.models import (
    extract_feature_attribution,
    run_group_cv_classification_ladder,
    run_group_cv_regression_ladder,
)
from src.plotting import plot_cv_baseline_ladder, plot_cv_pr_and_calibration


def main() -> int:
    cfg = load_config()
    rul_cap = int(cfg["framing"]["rul_cap"])
    alarm_w = int(cfg["framing"]["alarm_w"])
    window_k = int(cfg["features"]["window_k"])
    cv_folds = int(cfg["split"]["cv_folds"])
    c_waste = float(cfg["cost_simulation"]["c_waste"])
    default_c_fail = float(cfg["cost_simulation"]["default_c_fail"])

    metrics_dir = REPO_ROOT / cfg["paths"]["results_metrics_dir"]
    figures_dir = REPO_ROOT / cfg["paths"]["results_figures_dir"]
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"Phase 4: Baseline Ladder & 5-Fold GroupKFold CV (k={window_k}, 80 Dev Engines)")
    print("=" * 80)

    t0 = time.perf_counter()
    dev_df = load_development_data(rul_cap=rul_cap, alarm_w=alarm_w)
    print(f"\n1. Loaded Development Split: {dev_df['unit_id'].nunique()} engines, {len(dev_df):,} cycles.")

    reg_df, _ = run_group_cv_regression_ladder(
        dev_df=dev_df,
        cv_folds=cv_folds,
        window_k=window_k,
        rul_cap=rul_cap,
        c_fail=default_c_fail,
        c_waste=c_waste,
        alarm_w=alarm_w,
    )
    reg_csv_path = metrics_dir / "cv_regression_ladder.csv"
    reg_df.to_csv(reg_csv_path, index=False)

    print("\n2. Continuous RUL Regression Ladder (5-Fold GroupKFold):")
    print(f"   {'Rung':<8} | {'Model Name':<44} | {'RMSE':>6} | {'MAE':>6} | {'R2':>6} | {'Asym':>7}")
    print("   " + "-" * 92)
    for _, r in reg_df.iterrows():
        print(
            f"   {r['rung']:<8} | {r['model_name']:<44} | {r['rmse_all']:6.2f} | "
            f"{r['mae_all']:6.2f} | {r['r2_all']:6.3f} | {r['nasa_asym_mean']:7.2f}"
        )

    clf_df, oof_probs_clf = run_group_cv_classification_ladder(
        dev_df=dev_df,
        cv_folds=cv_folds,
        window_k=window_k,
        c_fail=default_c_fail,
        c_waste=c_waste,
        alarm_w=alarm_w,
    )
    clf_csv_path = metrics_dir / "cv_classification_ladder.csv"
    clf_df.to_csv(clf_csv_path, index=False)

    print(f"\n3. Binary Alarm Classification & Policy Ladder (w={alarm_w}, 5-Fold GroupKFold):")
    print(
        f"   {'Rung':<8} | {'Model Name':<50} | {'PR-AUC':>6} | {'Brier':>6} | "
        f"{'Opt p*':>6} | {'Fail%':>6} | {'Cost(c=100)':>11}"
    )
    print("   " + "-" * 108)
    for _, r in clf_df.iterrows():
        pr_str = f"{r['pr_auc']:6.4f}" if not np.isnan(r["pr_auc"]) else "   N/A"
        br_str = f"{r['brier_score']:6.4f}" if not np.isnan(r["brier_score"]) else "   N/A"
        print(
            f"   {r['rung']:<8} | {r['model_name']:<50} | {pr_str} | {br_str} | "
            f"{r['optimal_threshold_p']:6.2f} | {r['failure_rate_pct']:5.1f}% | "
            f"{r['oof_policy_mean_cost_c100']:11.2f}"
        )

    imp_df = extract_feature_attribution(dev_df, window_k=window_k)
    imp_csv_path = metrics_dir / "cv_feature_attribution.csv"
    imp_df.to_csv(imp_csv_path, index=False)
    n_lasso_kept = int(imp_df["lasso_selected"].sum())
    print(f"\n4. Feature Attribution (Lasso kept {n_lasso_kept}/{len(imp_df)} features; top 3 by HGB perm):")
    for _, r in imp_df.head(3).iterrows():
        print(f"   * {r['feature']:<22} | HGB Perm RMSE={r['hgb_perm_importance_rmse']:6.3f} | Lasso={r['lasso_coef']:+7.3f}")

    fig1_path = figures_dir / "cv_baseline_ladder_progression.png"
    fig2_path = figures_dir / "cv_pr_and_calibration_curves.png"
    plot_cv_baseline_ladder(reg_df, clf_df, fig1_path)
    plot_cv_pr_and_calibration(dev_df, oof_probs_clf, fig2_path)

    elapsed = time.perf_counter() - t0
    print("\n5. Saved Phase 4 Outputs:")
    print(f"   - {reg_csv_path.relative_to(REPO_ROOT)}")
    print(f"   - {clf_csv_path.relative_to(REPO_ROOT)}")
    print(f"   - {imp_csv_path.relative_to(REPO_ROOT)}")
    print(f"   - {fig1_path.relative_to(REPO_ROOT)}")
    print(f"   - {fig2_path.relative_to(REPO_ROOT)}")
    print("=" * 80)
    print(f"[SUCCESS] Phase 4 completed in {elapsed:.2f}s strictly on 80 development engines.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
