"""CLI script for Phase 3: Trailing-Window Feature Engineering & Leakage-Proof Pipelines.

Verifies:
1. Feature dimensionality across ladder modes (`age_only`, `instantaneous`, `trailing_trends`).
2. Strict causality (modifying cycles > t has zero effect on features at <= t).
3. Zero cross-engine window bleed (transforming one engine alone equals slicing from batch).
4. Zero validation-fold leakage inside 5-fold `GroupKFold` on the 80 development engines.

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
    SensorVarianceFilter,
    build_preprocessing_pipeline,
    compare_feature_correlations_across_windows,
)
from src.plotting import plot_trailing_feature_diagnostics


def verify_causality_and_isolation(dev_df, window_k: int = 30) -> dict[str, bool]:
    """Run unit tests proving strict causality, zero cross-engine bleed, and order invariance."""
    pipe = build_preprocessing_pipeline(window_k=window_k, feature_mode="trailing_trends", scale=False)
    full_feats = pipe.fit_transform(dev_df)

    u1_mask = dev_df["unit_id"] == 1
    u1_df = dev_df.loc[u1_mask].copy().reset_index(drop=True)
    u1_orig = pipe.transform(u1_df)

    # Test 1: Strict causality
    cutoff_t = 100
    u1_perturbed = u1_df.copy()
    sensor_cols = [c for c in u1_perturbed.columns if c.startswith("sensor_")]
    u1_perturbed.loc[u1_perturbed["cycle"] > cutoff_t, sensor_cols] += 999.9
    u1_pert_feats = pipe.transform(u1_perturbed)
    max_diff_past = float(
        np.max(np.abs(u1_orig.iloc[:cutoff_t].to_numpy() - u1_pert_feats.iloc[:cutoff_t].to_numpy()))
    )
    causality_passed = max_diff_past == 0.0

    # Test 2: Zero cross-engine bleed
    u1_from_batch = full_feats.loc[u1_mask].reset_index(drop=True)
    max_diff_isolation = float(np.max(np.abs(u1_orig.to_numpy() - u1_from_batch.to_numpy())))
    isolation_passed = max_diff_isolation < 1e-12

    # Test 3: Row-order invariance
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


def verify_group_kfold_pipeline(dev_df, cv_folds: int = 5, window_k: int = 30) -> dict[str, int | list[int]]:
    """Verify that the preprocessing pipeline fits cleanly inside 5-fold GroupKFold with zero NaNs."""
    gkf = GroupKFold(n_splits=cv_folds)
    fold_retained_counts: list[int] = []

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
        assert np.max(np.abs(X_tr.mean(axis=0))) < 1e-9
        fold_retained_counts.append(len(pipe.named_steps["sensor_filter"].retained_sensors_))

    return {
        "cv_folds_verified": cv_folds,
        "features_per_fold": 43,
        "retained_sensors_each_fold": fold_retained_counts,
    }


def main() -> int:
    cfg = load_config()
    rul_cap = int(cfg["framing"]["rul_cap"])
    alarm_w = int(cfg["framing"]["alarm_w"])
    window_k = int(cfg["features"]["window_k"])
    cv_folds = int(cfg["split"]["cv_folds"])

    figures_dir = REPO_ROOT / cfg["paths"]["results_figures_dir"]
    processed_dir = REPO_ROOT / cfg["paths"]["processed_dir"]
    figures_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"Phase 3: Trailing-Window Feature Engineering (k={window_k}, 80 Dev Engines)")
    print("=" * 72)

    t0 = time.perf_counter()
    dev_df = load_development_data(rul_cap=rul_cap, alarm_w=alarm_w)

    vfilter = SensorVarianceFilter(std_threshold=0.01).fit(dev_df)
    retained_sensors = vfilter.retained_sensors_
    dropped_sensors = vfilter.dropped_sensors_

    mode_specs = [
        ("age_only", "Rung 2: Age-Only Baseline"),
        ("instantaneous", "Rung 3: Current Sensor Values + Age"),
        ("trailing_trends", f"Rung 4/5: Trailing Window Trends (k={window_k}) + Age"),
    ]
    mode_summary = {}
    print(f"\n1. Feature Modes on 80 Dev Engines ({len(dev_df):,} rows):")
    for mode_name, desc in mode_specs:
        pipe = build_preprocessing_pipeline(window_k=window_k, feature_mode=mode_name, scale=True)
        X_out = pipe.fit_transform(dev_df)
        mode_summary[mode_name] = {
            "description": desc,
            "num_features": int(X_out.shape[1]),
            "feature_names": list(X_out.columns),
        }
        print(f"   - {mode_name:<18}: {X_out.shape[0]:,d} x {X_out.shape[1]:2d} features | {desc}")

    causality_checks = verify_causality_and_isolation(dev_df, window_k=window_k)
    cv_checks = verify_group_kfold_pipeline(dev_df, cv_folds=cv_folds, window_k=window_k)
    print("\n2. Pipeline Integrity Checks:")
    print(f"   - Strict Causality       : {causality_checks['strict_causality_verified']}")
    print(f"   - Zero Cross-Engine Bleed: {causality_checks['zero_cross_engine_bleed_verified']}")
    print(f"   - Row-Order Invariance   : {causality_checks['row_order_invariance_verified']}")
    print(f"   - 5-Fold GroupKFold      : {cv_checks['retained_sensors_each_fold']} sensors retained/fold")

    corr_df = compare_feature_correlations_across_windows(dev_df, window_k=window_k)
    avg_raw = float(corr_df["abs_corr_raw"].mean())
    avg_mean = float(corr_df["abs_corr_mean"].mean())
    print(f"\n3. Signal Denoising Gain (14-Sensor Mean |r| with Capped RUL):")
    print(f"   - Raw Instantaneous = {avg_raw:.4f} -> Trailing Mean (k={window_k}) = {avg_mean:.4f}")

    fig_path = figures_dir / "trailing_window_features.png"
    plot_trailing_feature_diagnostics(dev_df, corr_df, fig_path, window_k=window_k, sample_unit_id=1)

    metadata_payload = {
        "phase": "Phase 3: Trailing-Window Feature Engineering & Pipelines",
        "primary_window_k": window_k,
        "dev_engines_count": int(dev_df["unit_id"].nunique()),
        "dev_rows_count": int(len(dev_df)),
        "retained_sensors": retained_sensors,
        "dropped_sensors": dropped_sensors,
        "early_cycle_boundary_rule": (
            "Expanding causal window for cycles t < k (at t=1: mean=raw, std=0.0, slope=0.0; "
            "at 2 <= t < k: computed over cycles 1..t)."
        ),
        "feature_modes": mode_summary,
        "integrity_verification": {**causality_checks, **cv_checks},
        "signal_denoising_summary": {
            "mean_abs_corr_raw_14_sensors": round(avg_raw, 4),
            f"mean_abs_corr_trailing_mean_k{window_k}": round(avg_mean, 4),
        },
    }
    meta_path = processed_dir / "feature_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata_payload, f, indent=2)

    elapsed = time.perf_counter() - t0
    print("\n4. Saved Phase 3 Outputs:")
    print(f"   - {meta_path.relative_to(REPO_ROOT)}")
    print(f"   - {fig_path.relative_to(REPO_ROOT)}")
    print("=" * 72)
    print(f"[SUCCESS] Phase 3 completed in {elapsed:.2f}s.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
