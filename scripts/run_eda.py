"""CLI script for Phase 2: Exploratory Data Analysis & Degradation Physics on 80 Dev Engines.

Usage:
    python scripts/run_eda.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))

from src.data import load_config, load_development_data
from src.eda import fit_pca_health_index, screen_sensors_and_settings
from src.plotting import (
    plot_aligned_degradation_trajectories,
    plot_correlation_and_pca_health_index,
    plot_lifetime_distribution,
)


def main() -> int:
    cfg = load_config()
    rul_cap = int(cfg["framing"]["rul_cap"])
    alarm_w = int(cfg["framing"]["alarm_w"])

    metrics_dir = REPO_ROOT / cfg["paths"]["results_metrics_dir"]
    figures_dir = REPO_ROOT / cfg["paths"]["results_figures_dir"]
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Phase 2: Exploratory Data Analysis & Degradation Physics (80 Dev Engines)")
    print("=" * 72)

    dev_df = load_development_data(rul_cap=rul_cap, alarm_w=alarm_w)
    print(f"\n1. Loaded Development Split: {dev_df['unit_id'].nunique()} engines, {len(dev_df):,} rows.")

    # Sensor & setting screening strictly on 80 dev engines
    sensor_summary_df, retained_sensors, dropped_channels = screen_sensors_and_settings(dev_df)
    sensor_csv_path = metrics_dir / "eda_sensor_screening.csv"
    sensor_summary_df.to_csv(sensor_csv_path, index=False)

    print(f"\n2. Sensor & Setting Screening (80 dev engines):")
    print(f"   - Dropped channels ({len(dropped_channels)}): {dropped_channels}")
    print(f"   - Retained sensors ({len(retained_sensors)}): {retained_sensors}")

    # Unsupervised 1D PCA Health Index
    pca_df, pca_info = fit_pca_health_index(dev_df, retained_sensors)
    print("\n3. Unsupervised 1D PCA Composite Health Index:")
    print(
        f"   - PC1 variance explained = {pca_info['explained_variance_ratio_pc1']*100:.2f}% "
        f"| Top-3 cumulative = {pca_info['cumulative_top3_variance']*100:.2f}%"
    )
    print(f"   - Correlation with capped RUL: r = {pca_info['corr_pc1_with_rul_capped']:+.4f}")

    # Generate 3 clean Phase 2 figures
    fig1_path = figures_dir / "engine_lifetime_distribution.png"
    fig2_path = figures_dir / "sensor_trajectories_aligned.png"
    fig3_path = figures_dir / "sensor_correlation_and_pca_health_index.png"

    plot_lifetime_distribution(dev_df, fig1_path)
    plot_aligned_degradation_trajectories(dev_df, fig2_path)
    plot_correlation_and_pca_health_index(dev_df, pca_df, retained_sensors, pca_info, fig3_path)

    print("\n4. Saved Phase 2 Outputs:")
    print(f"   - {sensor_csv_path.relative_to(REPO_ROOT)}")
    print(f"   - {fig1_path.relative_to(REPO_ROOT)}")
    print(f"   - {fig2_path.relative_to(REPO_ROOT)}")
    print(f"   - {fig3_path.relative_to(REPO_ROOT)}")
    print("=" * 72)
    print("[SUCCESS] Phase 2 completed strictly on 80 development engines.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
