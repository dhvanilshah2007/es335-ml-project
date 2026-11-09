"""CLI script for Phase 3: Trailing-Window Feature Engineering & Leakage-Proof Pipelines.

Verifies:
1. Exact feature dimensionality across all baseline ladder modes (`age_only`, `instantaneous`,
   `trailing_trends` for k=10 & k=30, `combined`).
2. Strict causality test (modifying future cycles `> t` has zero effect on features at `<= t`).
3. Zero cross-engine window bleed (transforming one engine alone equals transforming all 80).
4. Zero validation-fold leakage inside 5-fold `GroupKFold` on the 80 development engines.
5. Signal denoising gain (`|r|` with capped RUL for raw vs k=10 vs k=30 trailing features).
6. Quantitative row-split leakage audit (lag-1 autocorrelation & overlapping window rate).

Usage:
    python scripts/run_features.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))

from src.data import load_config, load_development_data
from src.features import (
    analyze_row_split_leakage_mechanism,
    build_preprocessing_pipeline,
    compare_feature_correlations_across_windows,
    plot_phase3_feature_diagnostics,
)


def verify_causality_and_isolation(dev_df) -> dict[str, bool]:
    """Run mathematical unit tests proving strict causality and zero cross-engine bleed."""
    pipe = build_preprocessing_pipeline(window_k=10, feature_mode="trailing_trends", scale=False)
    full_feats = pipe.fit_transform(dev_df)

    # Test 1: Strict Causality (perturbing cycles > t on Engine 1 cannot change features at <= t)
    u1_mask = dev_df["unit_id"] == 1
    u1_df = dev_df.loc[u1_mask].copy().reset_index(drop=True)
    u1_orig = pipe.transform(u1_df)

    cutoff_t = 100
    u1_perturbed = u1_df.copy()
    sensor_cols = [c for c in u1_perturbed.columns if c.startswith("sensor_")]
    u1_perturbed.loc[u1_perturbed["cycle"] > cutoff_t, sensor_cols] += 999.9
    u1_pert_feats = pipe.transform(u1_perturbed)

    max_diff_past = float(
        np.max(np.abs(u1_orig.iloc[:cutoff_t].to_numpy() - u1_pert_feats.iloc[:cutoff_t].to_numpy()))
    )
    causality_passed = max_diff_past == 0.0

    # Test 2: Zero Cross-Engine Bleed (transforming Engine 1 alone == slicing Engine 1 from 80-engine batch)
    u1_from_batch = full_feats.loc[u1_mask].reset_index(drop=True)
    max_diff_isolation = float(np.max(np.abs(u1_orig.to_numpy() - u1_from_batch.to_numpy())))
    isolation_passed = max_diff_isolation < 1e-12

    # Test 3: Order Invariance (shuffling input rows and transforming restores exact same features)
    shuffled_df = dev_df.sample(frac=1.0, random_state=42)
    shuffled_feats = pipe.transform(shuffled_df).loc[dev_df.index]
    max_diff_order = float(np.max(np.abs(full_feats.to_numpy() - shuffled_feats.to_numpy())))
    order_invariance_passed = max_diff_order < 1e-12

    assert causality_passed, f"Causality test failed! max_diff_past={max_diff_past}"
    assert isolation_passed, f"Engine isolation test failed! max_diff={max_diff_isolation}"
    assert order_invariance_passed, f"Order invariance failed! max_diff={max_diff_order}"

    return {
        "strict_causality_verified": causality_passed,
        "zero_cross_engine_bleed_verified": isolation_passed,
        "row_order_invariance_verified": order_invariance_passed,
    }


def verify_group_kfold_pipeline(dev_df, cv_folds: int = 5, window_k: int = 10) -> dict[str, int | float]:
    """Verify that the preprocessing pipeline fits cleanly inside 5-fold GroupKFold with zero NaNs."""
    gkf = GroupKFold(n_splits=cv_folds)
    fold_retained_counts: list[int] = []
    val_means_abs: list[float] = []

    for fold_idx, (tr_idx, val_idx) in enumerate(gkf.split(dev_df, groups=dev_df["unit_id"])):
        tr_df = dev_df.iloc[tr_idx]
        val_df = dev_df.iloc[val_idx]

        pipe = build_preprocessing_pipeline(
            window_k=window_k, feature_mode="trailing_trends", std_threshold=0.01, scale=True
        )
        X_tr = pipe.fit_transform(tr_df)
        X_val = pipe.transform(val_df)

        assert not X_tr.isna().any().any(), f"NaNs found in train fold {fold_idx + 1}"
        assert not X_val.isna().any().any(), f"NaNs found in val fold {fold_idx + 1}"
        assert X_tr.shape[1] == 43 and X_val.shape[1] == 43

        # Verify training fold is standardized to zero mean, unit std
        assert np.max(np.abs(X_tr.mean(axis=0))) < 1e-9
        fold_retained_counts.append(len(pipe.named_steps["sensor_filter"].retained_sensors_))
        val_means_abs.append(float(np.mean(np.abs(X_val.mean(axis=0)))))

    return {
        "cv_folds_verified": cv_folds,
        "features_per_fold": 43,
        "retained_sensors_each_fold": fold_retained_counts,
        "mean_abs_val_fold_feature_offset": round(float(np.mean(val_means_abs)), 4),
    }


def main() -> int:
    cfg = load_config()
    rul_cap = int(cfg["framing"]["rul_cap"])
    alarm_w = int(cfg["framing"]["alarm_w"])

    metrics_dir = REPO_ROOT / cfg["paths"]["results_metrics_dir"]
    figures_dir = REPO_ROOT / cfg["paths"]["results_figures_dir"]
    processed_dir = REPO_ROOT / cfg["paths"]["processed_dir"]

    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("Phase 3: Trailing-Window Feature Engineering & Pipelines (80 Dev Engines)")
    print("=" * 78)

    t0 = time.perf_counter()
    dev_df = load_development_data(rul_cap=rul_cap, alarm_w=alarm_w)

    # 1. Verify Feature Modes & Dimensionality across the Baseline Ladder
    print(f"\n1. Feature Matrix Modes on 80 Development Engines ({len(dev_df):,} rows):")
    mode_specs = [
        ("age_only", 10, "Rung 2: Age-Only Baseline"),
        ("instantaneous", 10, "Rung 3: Current Sensor Values + Age"),
        ("trailing_trends", 10, "Rung 4/5: Trailing Window Trends (k=10) + Age [PRIMARY]"),
        ("trailing_trends", 30, "Sensitivity: Trailing Window Trends (k=30) + Age"),
        ("combined", 10, "Optional: Instantaneous + Trailing Trends (k=10) + Age"),
    ]
    mode_summary = {}
    for mode_name, k_val, desc in mode_specs:
        t_start = time.perf_counter()
        pipe = build_preprocessing_pipeline(window_k=k_val, feature_mode=mode_name, scale=True)
        X_out = pipe.fit_transform(dev_df)
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        key = f"{mode_name}_k{k_val}" if mode_name != "age_only" and mode_name != "instantaneous" else mode_name
        mode_summary[key] = {
            "description": desc,
            "num_features": int(X_out.shape[1]),
            "feature_names": list(X_out.columns),
            "compute_time_ms": round(elapsed_ms, 2),
        }
        print(
            f"   - {key:<22}: {X_out.shape[0]:,d} x {X_out.shape[1]:2d} features "
            f"({elapsed_ms:5.1f} ms) | {desc}"
        )

    # 2. Causality, Isolation & GroupKFold Pipeline Verification
    causality_checks = verify_causality_and_isolation(dev_df)
    cv_checks = verify_group_kfold_pipeline(dev_df, cv_folds=5, window_k=10)

    print("\n2. Mathematical Integrity & Leakage Prevention Tests:")
    print(f"   - Strict Causality (no look-ahead > t)      : PASSED ({causality_checks['strict_causality_verified']})")
    print(f"   - Zero Cross-Engine Window Bleed            : PASSED ({causality_checks['zero_cross_engine_bleed_verified']})")
    print(f"   - Row-Order Invariance                      : PASSED ({causality_checks['row_order_invariance_verified']})")
    print(
        f"   - 5-Fold GroupKFold Pipeline Containment    : PASSED "
        f"(Retained sensors per fold = {cv_checks['retained_sensors_each_fold']})"
    )

    # 3. Compare RUL Correlations: Raw vs Trailing Window Features (k=10, k=30)
    corr_df = compare_feature_correlations_across_windows(dev_df, window_ks=[10, 30])
    corr_csv_path = metrics_dir / "feature_correlation_comparison.csv"
    corr_df.to_csv(corr_csv_path, index=False)

    print("\n3. Signal Denoising Gain (|r| with Capped RUL: Raw vs Trailing Mean & Slope):")
    for _, r in corr_df.head(6).iterrows():
        print(
            f"   * {r['sensor']:<10} | Raw |r|={r['abs_corr_raw']:.4f} "
            f"-> Mean k=10 |r|={r['abs_corr_mean_k10']:.4f} ({r['mean_gain_k10_vs_raw']:+.4f}) "
            f"-> Mean k=30 |r|={r['abs_corr_mean_k30']:.4f} ({r['mean_gain_k30_vs_raw']:+.4f}) "
            f"| Slope k=30 |r|={r['abs_corr_slope_k30']:.4f}"
        )
    avg_raw = corr_df["abs_corr_raw"].mean()
    avg_k10 = corr_df["abs_corr_mean_k10"].mean()
    avg_k30 = corr_df["abs_corr_mean_k30"].mean()
    print(
        f"   [14-Sensor Average |r|]: Raw = {avg_raw:.4f} -> Trailing Mean (k=10) = {avg_k10:.4f} "
        f"-> Trailing Mean (k=30) = {avg_k30:.4f}"
    )

    # 4. Quantitative Audit of the Row-Split Leakage Trap
    leakage_info = analyze_row_split_leakage_mechanism(dev_df, window_k=10, seed=42)
    print("\n4. Quantitative Audit of Why Random Row Splitting Leaks Data:")
    print(
        f"   - Within-engine Lag-1 Autocorrelation (k=10 means): "
        f"{leakage_info['mean_lag1_autocorrelation_rolling_means']:.5f} "
        f"(range {leakage_info['min_lag1_autocorrelation_rolling_means']:.5f}–"
        f"{leakage_info['max_lag1_autocorrelation_rolling_means']:.5f})"
    )
    print(
        f"   - Under an 80/20 random row split, {leakage_info['pct_val_rows_with_adjacent_cycle_in_train']:.1f}% "
        f"of validation rows have cycle t-1 or t+1 of the SAME engine in the training set!"
    )
    print(
        f"   - {leakage_info['pct_val_rows_with_overlapping_window_in_train']:.2f}% of validation rows share "
        f"overlapping k=10 window cycles with training rows of the same engine!"
    )

    # 5. Save Feature Metadata JSON and Diagnostic Plot
    fig_path = figures_dir / "trailing_features_and_leakage_mechanism.png"
    plot_phase3_feature_diagnostics(dev_df, corr_df, leakage_info, fig_path, sample_unit_id=1)

    metadata_payload = {
        "phase": "Phase 3: Trailing-Window Feature Engineering & Pipelines",
        "dev_engines_count": int(dev_df["unit_id"].nunique()),
        "dev_rows_count": int(len(dev_df)),
        "early_cycle_boundary_rule": (
            "Expanding causal window for cycles t < k (at t=1: mean=raw, std=0.0, slope=0.0; "
            "at 2 <= t < k: computed over cycles 1..t). Preserves all early cycles without NaNs."
        ),
        "feature_modes": mode_summary,
        "integrity_verification": {
            **causality_checks,
            **cv_checks,
        },
        "signal_denoising_summary": {
            "mean_abs_corr_raw_14_sensors": round(float(avg_raw), 4),
            "mean_abs_corr_trailing_mean_k10": round(float(avg_k10), 4),
            "mean_abs_corr_trailing_mean_k30": round(float(avg_k30), 4),
        },
        "row_split_leakage_audit": leakage_info,
    }
    meta_path = processed_dir / "feature_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata_payload, f, indent=2)

    total_sec = time.perf_counter() - t0
    print("\n5. Saved Phase 3 Artifacts:")
    print(f"   - Feature correlation CSV : {corr_csv_path.relative_to(REPO_ROOT)}")
    print(f"   - Feature metadata JSON   : {meta_path.relative_to(REPO_ROOT)}")
    print(f"   - Phase 3 diagnostic plot : {fig_path.relative_to(REPO_ROOT)}")
    print("=" * 78)
    print(f"[SUCCESS] Phase 3 completed in {total_sec:.2f}s with zero data leakage.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
