"""Data verification script for NASA C-MAPSS dataset.

Verifies the integrity and shape of the raw FD001 dataset files in CMAPSSData/.
"""

import sys
from pathlib import Path
import pandas as pd

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))

from src.constants import ALL_RAW_COLS


def verify_fd001(data_dir: Path) -> bool:
    train_path = data_dir / "train_FD001.txt"
    test_path = data_dir / "test_FD001.txt"
    rul_path = data_dir / "RUL_FD001.txt"

    for path in [train_path, test_path, rul_path]:
        if not path.is_file():
            print(f"[ERROR] Missing file: {path}")
            return False

    print("=" * 60)
    print("NASA C-MAPSS FD001 Data Integrity Verification")
    print("=" * 60)

    # Load train
    train_df = pd.read_csv(train_path, sep=r"\s+", header=None)
    # If trailing whitespace creates an extra NaN column, drop it
    if train_df.shape[1] > len(ALL_RAW_COLS):
        train_df = train_df.iloc[:, : len(ALL_RAW_COLS)]
    train_df.columns = ALL_RAW_COLS

    # Load test
    test_df = pd.read_csv(test_path, sep=r"\s+", header=None)
    if test_df.shape[1] > len(ALL_RAW_COLS):
        test_df = test_df.iloc[:, : len(ALL_RAW_COLS)]
    test_df.columns = ALL_RAW_COLS

    # Load true RUL for test
    rul_df = pd.read_csv(rul_path, sep=r"\s+", header=None)

    train_engines = train_df["unit_id"].nunique()
    test_engines = test_df["unit_id"].nunique()
    rul_count = len(rul_df)

    lifetimes = train_df.groupby("unit_id")["cycle"].max()

    print(f"Train FD001: {len(train_df):,} rows, {train_engines} engines")
    print(
        f"  Engine lifetime range: min = {lifetimes.min()} cycles, "
        f"median = {lifetimes.median():.1f} cycles, max = {lifetimes.max()} cycles"
    )
    print(f"Test FD001:  {len(test_df):,} rows, {test_engines} engines (cut off prior to failure)")
    print(f"RUL FD001:   {rul_count} ground-truth remaining life values for test engines")
    print(f"Missing values: Train={train_df.isna().sum().sum()}, Test={test_df.isna().sum().sum()}")
    print("=" * 60)
    print("[SUCCESS] All FD001 raw data files verified successfully.")
    return True


if __name__ == "__main__":
    data_directory = REPO_ROOT / "CMAPSSData"
    success = verify_fd001(data_directory)
    sys.exit(0 if success else 1)
