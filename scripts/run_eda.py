"""CLI script for Phase 2: Exploratory Data Analysis & Degradation Physics on 80 Dev Engines.

Usage:
    python scripts/run_eda.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add repository root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))

from src.data import load_config, load_development_data
from src.eda import (
    analyze_fixed_schedule_tradeoff,
    fit_pca_health_index,
    plot_aligned_degradation_trajectories,
    plot_all_14_retained_sensors,
    plot_correlation_and_pca_health_index,
    plot_lifetime_and_fixed_schedule,
    screen_sensors_and_settings,
)


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

    print("=" * 76)
    print("Phase 2: Exploratory Data Analysis & Degradation Physics (80 Dev Engines)")
    print("=" * 76)

    # 1. Load strictly the 80 development engines (16,340 rows)
    dev_df = load_development_data(rul_cap=rul_cap, alarm_w=alarm_w)
    print(f"\n1. Loaded Development Split: {dev_df['unit_id'].nunique()} engines, {len(dev_df):,} rows.")

    # 2. Sensor & Operating Setting Variance Screening
    sensor_summary_df, retained_sensors, dropped_channels = screen_sensors_and_settings(dev_df)
    sensor_csv_path = metrics_dir / "eda_sensor_screening.csv"
    sensor_summary_df.to_csv(sensor_csv_path, index=False)

    print(f"\n2. Sensor & Setting Screening (strictly fit on 80 dev engines):")
    print(f"   - Total raw channels : {len(sensor_summary_df)} (3 settings + 21 sensors)")
    print(f"   - Dropped channels   : {len(dropped_channels)} -> {dropped_channels}")
    print(f"   - Retained sensors   : {len(retained_sensors)} -> {retained_sensors}")

    print("\n   Top 6 Sensors by Absolute Correlation with Capped RUL:")
    top_corr = (
        sensor_summary_df[sensor_summary_df["retained"]]
        .assign(abs_corr=lambda d: d["corr_rul_capped"].abs())
        .sort_values("abs_corr", ascending=False)
        .head(6)
    )
    for _, row in top_corr.iterrows():
        print(
            f"     * {row['channel']:<10} | r_capped={row['corr_rul_capped']:+.4f} "
            f"| r_uncapped={row['corr_rul_uncapped']:+.4f} | std={row['std']:8.4f} | {row['physical_description']}"
        )

    # 3. Fixed-Interval Replacement Trade-Off Analysis
    tradeoff_df = analyze_fixed_schedule_tradeoff(dev_df)
    tradeoff_csv_path = metrics_dir / "eda_fixed_schedule_tradeoff.csv"
    tradeoff_df.to_csv(tradeoff_csv_path, index=False)

    print("\n3. Fixed-Schedule Replacement Trade-Off (Why Fixed K Cannot Win):")
    highlight_ks = [128, 150, 175, 199, 230]
    sub_trade = tradeoff_df[tradeoff_df["replacement_cycle_K"].isin(highlight_ks)]
    for _, r in sub_trade.iterrows():
        print(
            f"     * Fixed K={int(r['replacement_cycle_K']):3d} cycles: "
            f"Failures={int(r['engines_failed']):2d}/80 ({r['failure_rate_pct']:5.1f}%) | "
            f"Mean Wasted Life={r['mean_wasted_cycles_per_engine']:5.1f} cycles/engine "
            f"({r['pct_total_fleet_life_wasted']:4.1f}% of fleet life)"
        )

    # 4. Unsupervised PCA Health Index (Syllabus Topic 8)
    pca_df, pca_info = fit_pca_health_index(dev_df, retained_sensors)
    print("\n4. Unsupervised 1D PCA Composite Health Index (Syllabus Topic 8):")
    print(
        f"   - Variance explained: PC1 = {pca_info['explained_variance_ratio_pc1']*100:.2f}%, "
        f"PC2 = {pca_info['explained_variance_ratio_pc2']*100:.2f}%, "
        f"PC3 = {pca_info['explained_variance_ratio_pc3']*100:.2f}% "
        f"(Top-3 Cumulative = {pca_info['cumulative_top3_variance']*100:.2f}%)"
    )
    print(
        f"   - Correlation of PC1 Health Index with RUL: "
        f"r_capped = {pca_info['corr_pc1_with_rul_capped']:+.4f}, "
        f"r_uncapped = {pca_info['corr_pc1_with_rul_uncapped']:+.4f}"
    )

    # 5. Save Retained Sensors & EDA Metadata JSON
    eda_manifest = {
        "dev_engines_count": int(dev_df["unit_id"].nunique()),
        "dev_rows_count": int(len(dev_df)),
        "retained_sensors": retained_sensors,
        "dropped_channels": dropped_channels,
        "pca_health_index": pca_info,
    }
    manifest_path = processed_dir / "retained_sensors.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(eda_manifest, f, indent=2)

    # 6. Generate Publication-Quality Figures
    fig1_path = figures_dir / "engine_lifetime_distribution.png"
    fig2_path = figures_dir / "sensor_trajectories_aligned.png"
    fig3_path = figures_dir / "all_14_retained_sensors_trends.png"
    fig4_path = figures_dir / "sensor_correlation_and_pca_health_index.png"

    plot_lifetime_and_fixed_schedule(dev_df, tradeoff_df, fig1_path)
    plot_aligned_degradation_trajectories(dev_df, fig2_path)
    plot_all_14_retained_sensors(dev_df, retained_sensors, fig3_path)
    plot_correlation_and_pca_health_index(dev_df, pca_df, retained_sensors, pca_info, fig4_path)

    print("\n5. Saved Phase 2 Tables & Figures:")
    print(f"   - Sensor screening CSV : {sensor_csv_path.relative_to(REPO_ROOT)}")
    print(f"   - Fixed-K tradeoff CSV : {tradeoff_csv_path.relative_to(REPO_ROOT)}")
    print(f"   - Sensor manifest JSON : {manifest_path.relative_to(REPO_ROOT)}")
    print(f"   - Figure 1 (Lifetimes) : {fig1_path.relative_to(REPO_ROOT)}")
    print(f"   - Figure 2 (Alignment) : {fig2_path.relative_to(REPO_ROOT)}")
    print(f"   - Figure 3 (14 Sensors): {fig3_path.relative_to(REPO_ROOT)}")
    print(f"   - Figure 4 (Corr & PCA): {fig4_path.relative_to(REPO_ROOT)}")

    print("=" * 76)
    print("[SUCCESS] Phase 2 EDA & Degradation Physics completed strictly on 80 dev engines.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
