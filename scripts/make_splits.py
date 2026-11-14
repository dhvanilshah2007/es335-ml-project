"""Generate and persist leakage-proof engine splits and verify targets.

Usage:
    python scripts/make_splits.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add repository root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))

from src.data import (
    add_targets,
    generate_engine_splits,
    load_config,
    load_development_data,
    load_heldout_test_data,
    load_official_benchmark_test_data,
    load_raw_fd001,
    save_engine_splits,
)


def main() -> int:
    cfg = load_config()
    seed = int(cfg["project"]["random_seed"])
    dev_count = int(cfg["split"]["dev_engines_count"])
    test_count = int(cfg["split"]["heldout_test_engines_count"])
    cv_folds = int(cfg["split"]["cv_folds"])
    rul_cap = cfg["framing"]["rul_cap"]
    alarm_w = int(cfg["framing"]["alarm_w"])

    print("=" * 72)
    print("Leakage-Proof Data Foundation & Engine Split Protocol")
    print("=" * 72)

    train_df, test_df, rul_series = load_raw_fd001(REPO_ROOT / cfg["paths"]["raw_data_dir"])
    split_payload = generate_engine_splits(
        train_df=train_df,
        dev_count=dev_count,
        test_count=test_count,
        cv_folds=cv_folds,
        seed=seed,
    )

    saved_paths = save_engine_splits(split_payload, REPO_ROOT / cfg["paths"]["splits_dir"])

    dev_stats = split_payload["lifetime_summary"]["dev"]
    test_stats = split_payload["lifetime_summary"]["heldout_test"]

    print(f"\n1. Engine-Level Partition (Fixed Seed = {seed}):")
    print(
        f"   - Development Set   : {dev_stats['count']} engines, {dev_stats['total_rows']:,} cycles "
        f"| lifetime mean={dev_stats['mean']} (std={dev_stats['std']}), "
        f"min={dev_stats['min']}, median={dev_stats['median']}, max={dev_stats['max']}"
    )
    print(
        f"   - Held-Out Test Set : {test_stats['count']} engines, {test_stats['total_rows']:,} cycles "
        f"| lifetime mean={test_stats['mean']} (std={test_stats['std']}), "
        f"min={test_stats['min']}, median={test_stats['median']}, max={test_stats['max']}"
    )
    print(f"   - Held-Out Test IDs : {split_payload['test_engine_ids']}")

    print(f"\n2. 5-Fold GroupKFold Cross-Validation over 80 Development Engines:")
    for fold in split_payload["cv_folds"]:
        print(
            f"   - Fold {fold['fold']}: {fold['val_engine_count']} val engines "
            f"({fold['val_row_count']:,} rows, mean life={fold['val_lifetime_mean']}) | "
            f"IDs={fold['val_engine_ids']}"
        )

    # Verify target creation on development data
    dev_df = load_development_data(rul_cap=rul_cap, alarm_w=alarm_w)
    heldout_df = load_heldout_test_data(rul_cap=rul_cap, alarm_w=alarm_w)
    official_test_df = load_official_benchmark_test_data(rul_cap=rul_cap, alarm_w=alarm_w)

    assert len(dev_df) + len(heldout_df) == len(train_df), "Row count mismatch after splitting!"
    assert set(dev_df["unit_id"]).isdisjoint(set(heldout_df["unit_id"])), "Leakage detected!"

    print(f"\n3. Target Formulation Verification on 80 Development Engines ({len(dev_df):,} rows):")
    for cap_val in cfg["framing"]["rul_cap_sensitivity"]:
        tmp = add_targets(dev_df, rul_cap=cap_val, alarm_w=alarm_w)
        cap_label = f"Cap={cap_val}" if cap_val is not None else "Uncapped"
        at_cap_pct = (tmp["rul"] == cap_val).mean() * 100.0 if cap_val is not None else 0.0
        print(
            f"   - RUL ({cap_label:<8}): mean={tmp['rul'].mean():6.2f}, "
            f"std={tmp['rul'].std():5.2f}, min={tmp['rul'].min():3d}, "
            f"max={tmp['rul'].max():3d} ({at_cap_pct:4.1f}% of cycles at cap)"
        )

    print("\n4. Binary Alarm Class Balance on 80 Development Engines (Alarm = 1 iff RUL <= w):")
    for w_val in cfg["framing"]["alarm_w_sensitivity"]:
        tmp = add_targets(dev_df, rul_cap=rul_cap, alarm_w=w_val)
        pos_count = int(tmp["alarm"].sum())
        pos_pct = tmp["alarm"].mean() * 100.0
        default_tag = " [DEFAULT]" if w_val == alarm_w else ""
        print(
            f"   - Horizon w={w_val:2d} cycles: {pos_count:4,d} / {len(tmp):,d} positive rows "
            f"({pos_pct:5.2f}% positive class){default_tag}"
        )

    # Verify official benchmark test set at final cutoff cycle per engine
    cutoff_rows = official_test_df.groupby("unit_id").tail(1)
    assert len(cutoff_rows) == 100
    assert (cutoff_rows["rul_uncapped"].values == rul_series.values).all()
    print(
        f"\n5. Official Benchmark Test Set (`test_FD001.txt` + `RUL_FD001.txt`):"
        f"\n   - 100 truncated engines ({len(official_test_df):,} total rows); "
        f"cutoff RUL min={rul_series.min()}, median={rul_series.median():.1f}, max={rul_series.max()}"
    )

    print("\n6. Saved Split Artifacts:")
    for key, path in saved_paths.items():
        print(f"   - {key:<15}: {path.relative_to(REPO_ROOT)}")

    print("=" * 72)
    print("[SUCCESS] Engine splits generated and verified with zero data leakage.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
