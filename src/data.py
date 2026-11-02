"""Data ingestion, target formulation, and leakage-proof split management for C-MAPSS FD001.

Phase 1 of the ES 335 Predictive Maintenance Project:
- Loads whitespace-delimited NASA C-MAPSS FD001 files (`train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt`).
- Computes piecewise linear RUL targets (capped at `rul_cap`) and binary lead-time alarm targets (`RUL <= w`).
- Partitions the 100 training engines into 80 development engines and 20 held-out test engines
  using a fixed random seed (`seed=42`).
- Generates and persists 5-fold `GroupKFold` cross-validation assignments across the 80 development engines.
- Enforces an explicit API boundary between development data (Phases 2-4) and held-out test data (Phase 5).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import GroupKFold

from src.constants import ALL_RAW_COLS, INDEX_COLS

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"


def load_config(config_path: Path | str | None = None) -> dict[str, Any]:
    """Load the project YAML configuration file."""
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _read_cmapss_table(file_path: Path) -> pd.DataFrame:
    """Read a whitespace-delimited C-MAPSS trajectory file and assign standard column names."""
    if not file_path.is_file():
        raise FileNotFoundError(f"C-MAPSS data file not found: {file_path}")

    df = pd.read_csv(file_path, sep=r"\s+", header=None)
    if df.shape[1] > len(ALL_RAW_COLS):
        df = df.iloc[:, : len(ALL_RAW_COLS)]
    if df.shape[1] != len(ALL_RAW_COLS):
        raise ValueError(
            f"Expected {len(ALL_RAW_COLS)} columns in {file_path}, got {df.shape[1]}"
        )

    df.columns = ALL_RAW_COLS
    df["unit_id"] = df["unit_id"].astype(int)
    df["cycle"] = df["cycle"].astype(int)
    df = df.sort_values(INDEX_COLS).reset_index(drop=True)
    return df


def load_raw_fd001(
    data_dir: Path | str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Load raw FD001 training trajectories, test trajectories, and ground-truth test RULs.

    Returns
    -------
    train_df : pd.DataFrame
        20,631 rows x 26 columns for 100 run-to-failure training engines.
    test_df : pd.DataFrame
        13,096 rows x 26 columns for 100 right-censored/truncated test engines.
    rul_series : pd.Series
        True remaining useful life at the final recorded cycle for each of the 100 test engines,
        indexed by `unit_id` (1..100).
    """
    base_dir = Path(data_dir) if data_dir is not None else (REPO_ROOT / "CMAPSSData")
    train_path = base_dir / "train_FD001.txt"
    test_path = base_dir / "test_FD001.txt"
    rul_path = base_dir / "RUL_FD001.txt"

    train_df = _read_cmapss_table(train_path)
    test_df = _read_cmapss_table(test_path)

    if not rul_path.is_file():
        raise FileNotFoundError(f"C-MAPSS RUL file not found: {rul_path}")
    rul_df = pd.read_csv(rul_path, sep=r"\s+", header=None)
    rul_series = rul_df.iloc[:, 0].astype(int)
    rul_series.index = pd.Index(range(1, len(rul_series) + 1), name="unit_id")
    rul_series.name = "true_cutoff_rul"

    return train_df, test_df, rul_series


def add_targets(
    df: pd.DataFrame,
    rul_cap: int | None = 125,
    alarm_w: int = 30,
    cutoff_rul: pd.Series | dict[int, int] | None = None,
) -> pd.DataFrame:
    """Compute uncapped RUL, piecewise-capped RUL, and binary alarm targets per engine cycle.

    Parameters
    ----------
    df : pd.DataFrame
        Trajectory DataFrame containing `unit_id` and `cycle`.
    rul_cap : int or None, default=125
        Maximum RUL value for piecewise linear target formulation. If `None`, RUL is uncapped.
    alarm_w : int, default=30
        Lead-time horizon in cycles for the binary alarm (`alarm = 1` iff `rul_uncapped <= alarm_w`).
    cutoff_rul : pd.Series or dict or None, default=None
        If `None`, assumes trajectories run to failure (`rul_uncapped = max_cycle - cycle`).
        If provided (e.g. for official `test_FD001.txt`), maps `unit_id` to remaining life at
        `max_cycle`, so `rul_uncapped = (max_cycle - cycle) + cutoff_rul[unit_id]`.

    Returns
    -------
    pd.DataFrame
        Copy of `df` with added target columns:
        - `max_cycle`: last observed cycle of the engine in `df`
        - `rul_uncapped`: true remaining cycles until failure
        - `rul`: RUL target capped at `rul_cap` (or uncapped if `rul_cap is None`)
        - `alarm`: binary indicator `1` if `rul_uncapped <= alarm_w` else `0`
    """
    out = df.copy()
    max_cycles = out.groupby("unit_id")["cycle"].transform("max")
    out["max_cycle"] = max_cycles

    if cutoff_rul is None:
        out["rul_uncapped"] = out["max_cycle"] - out["cycle"]
    else:
        extra_rul = out["unit_id"].map(cutoff_rul)
        if extra_rul.isna().any():
            missing_ids = out.loc[extra_rul.isna(), "unit_id"].unique().tolist()
            raise ValueError(f"Missing cutoff_rul values for unit_ids: {missing_ids}")
        out["rul_uncapped"] = (out["max_cycle"] - out["cycle"]) + extra_rul.astype(int)

    if rul_cap is not None:
        out["rul"] = out["rul_uncapped"].clip(upper=int(rul_cap))
    else:
        out["rul"] = out["rul_uncapped"]

    out["alarm"] = (out["rul_uncapped"] <= int(alarm_w)).astype(int)
    return out


def generate_engine_splits(
    train_df: pd.DataFrame,
    dev_count: int = 80,
    test_count: int = 20,
    cv_folds: int = 5,
    seed: int = 42,
) -> dict[str, Any]:
    """Deterministically partition training engines into dev and held-out test sets and 5 CV folds.

    Parameters
    ----------
    train_df : pd.DataFrame
        Raw FD001 training DataFrame containing 100 run-to-failure engines.
    dev_count : int, default=80
        Number of development engines used for EDA, feature selection, and cross-validation.
    test_count : int, default=20
        Number of held-out test engines sealed until final evaluation and policy simulation.
    cv_folds : int, default=5
        Number of `GroupKFold` splits over the development engines.
    seed : int, default=42
        Fixed random seed for reproducible engine permutation.

    Returns
    -------
    dict[str, Any]
        Dictionary containing `dev_engine_ids`, `test_engine_ids`, `cv_folds`, and lifetime statistics.
    """
    engine_ids = np.sort(train_df["unit_id"].unique())
    total_engines = len(engine_ids)
    if dev_count + test_count != total_engines:
        raise ValueError(
            f"dev_count ({dev_count}) + test_count ({test_count}) must equal total engines ({total_engines})"
        )

    rng = np.random.RandomState(seed)
    permuted = rng.permutation(engine_ids)
    dev_ids = [int(x) for x in np.sort(permuted[:dev_count])]
    test_ids = [int(x) for x in np.sort(permuted[dev_count:])]

    # Verify strict disjointness
    assert len(set(dev_ids).intersection(set(test_ids))) == 0, "Leakage: Dev and Test engine sets overlap!"
    assert len(dev_ids) == dev_count and len(test_ids) == test_count

    # Generate 5-fold GroupKFold assignments on the 80 development engines
    dev_df = train_df[train_df["unit_id"].isin(dev_ids)].sort_values(INDEX_COLS).reset_index(drop=True)
    gkf = GroupKFold(n_splits=cv_folds)
    folds_info: list[dict[str, Any]] = []

    lifetimes = train_df.groupby("unit_id")["cycle"].max()

    for fold_idx, (tr_idx, val_idx) in enumerate(gkf.split(dev_df, groups=dev_df["unit_id"])):
        tr_units = [int(u) for u in np.sort(dev_df.iloc[tr_idx]["unit_id"].unique())]
        val_units = [int(u) for u in np.sort(dev_df.iloc[val_idx]["unit_id"].unique())]
        assert len(set(tr_units).intersection(set(val_units))) == 0
        val_lifetimes = lifetimes.loc[val_units]
        folds_info.append(
            {
                "fold": int(fold_idx + 1),
                "train_engine_count": len(tr_units),
                "val_engine_count": len(val_units),
                "train_row_count": int(len(tr_idx)),
                "val_row_count": int(len(val_idx)),
                "val_lifetime_mean": round(float(val_lifetimes.mean()), 2),
                "val_lifetime_min": int(val_lifetimes.min()),
                "val_lifetime_max": int(val_lifetimes.max()),
                "val_engine_ids": val_units,
                "train_engine_ids": tr_units,
            }
        )

    dev_lifetimes = lifetimes.loc[dev_ids]
    test_lifetimes = lifetimes.loc[test_ids]

    return {
        "dataset": "NASA C-MAPSS FD001",
        "random_seed": int(seed),
        "total_engines": int(total_engines),
        "dev_engine_count": int(dev_count),
        "test_engine_count": int(test_count),
        "cv_folds_count": int(cv_folds),
        "dev_engine_ids": dev_ids,
        "test_engine_ids": test_ids,
        "lifetime_summary": {
            "dev": {
                "count": int(dev_count),
                "total_rows": int(len(dev_df)),
                "mean": round(float(dev_lifetimes.mean()), 2),
                "std": round(float(dev_lifetimes.std()), 2),
                "min": int(dev_lifetimes.min()),
                "q25": round(float(dev_lifetimes.quantile(0.25)), 2),
                "median": round(float(dev_lifetimes.median()), 2),
                "q75": round(float(dev_lifetimes.quantile(0.75)), 2),
                "max": int(dev_lifetimes.max()),
            },
            "heldout_test": {
                "count": int(test_count),
                "total_rows": int(len(train_df[train_df["unit_id"].isin(test_ids)])),
                "mean": round(float(test_lifetimes.mean()), 2),
                "std": round(float(test_lifetimes.std()), 2),
                "min": int(test_lifetimes.min()),
                "q25": round(float(test_lifetimes.quantile(0.25)), 2),
                "median": round(float(test_lifetimes.median()), 2),
                "q75": round(float(test_lifetimes.quantile(0.75)), 2),
                "max": int(test_lifetimes.max()),
            },
        },
        "cv_folds": folds_info,
    }


def save_engine_splits(split_payload: dict[str, Any], splits_dir: Path | str | None = None) -> dict[str, Path]:
    """Persist development engine IDs, held-out test engine IDs, and CV fold definitions to JSON files."""
    out_dir = Path(splits_dir) if splits_dir is not None else (REPO_ROOT / "data" / "splits")
    out_dir.mkdir(parents=True, exist_ok=True)

    dev_path = out_dir / "dev_engines.json"
    test_path = out_dir / "test_engines.json"
    summary_path = out_dir / "split_summary.json"

    dev_payload = {
        "dataset": split_payload["dataset"],
        "role": "development_80_engines",
        "usage": "Used for EDA, sensor filtering, feature engineering, hyperparameter tuning, and 5-fold GroupKFold CV.",
        "random_seed": split_payload["random_seed"],
        "engine_count": split_payload["dev_engine_count"],
        "engine_ids": split_payload["dev_engine_ids"],
        "lifetime_stats": split_payload["lifetime_summary"]["dev"],
    }

    test_payload = {
        "dataset": split_payload["dataset"],
        "role": "heldout_test_20_engines",
        "usage": "SEALED until Phase 5. Used strictly ONCE for final model evaluation and maintenance policy cost simulation.",
        "random_seed": split_payload["random_seed"],
        "engine_count": split_payload["test_engine_count"],
        "engine_ids": split_payload["test_engine_ids"],
        "lifetime_stats": split_payload["lifetime_summary"]["heldout_test"],
    }

    with open(dev_path, "w", encoding="utf-8") as f:
        json.dump(dev_payload, f, indent=2)

    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(test_payload, f, indent=2)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(split_payload, f, indent=2)

    return {
        "dev_engines": dev_path,
        "test_engines": test_path,
        "split_summary": summary_path,
    }


def load_engine_splits(splits_dir: Path | str | None = None) -> tuple[list[int], list[int], dict[str, Any]]:
    """Load saved development engine IDs, held-out test engine IDs, and full split summary."""
    in_dir = Path(splits_dir) if splits_dir is not None else (REPO_ROOT / "data" / "splits")
    dev_path = in_dir / "dev_engines.json"
    test_path = in_dir / "test_engines.json"
    summary_path = in_dir / "split_summary.json"

    if not (dev_path.is_file() and test_path.is_file() and summary_path.is_file()):
        raise FileNotFoundError(
            f"Split files not found in {in_dir}. Run `python scripts/make_splits.py` first."
        )

    with open(dev_path, "r", encoding="utf-8") as f:
        dev_ids = json.load(f)["engine_ids"]
    with open(test_path, "r", encoding="utf-8") as f:
        test_ids = json.load(f)["engine_ids"]
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    assert len(set(dev_ids).intersection(set(test_ids))) == 0, "Corrupted split files: overlap detected!"
    return dev_ids, test_ids, summary


def load_development_data(
    rul_cap: int | None = 125,
    alarm_w: int = 30,
    data_dir: Path | str | None = None,
    splits_dir: Path | str | None = None,
) -> pd.DataFrame:
    """Load strictly the 80 development engines with RUL and alarm targets attached.

    Safe to call in Phases 2, 3, and 4 without touching the 20 held-out test engines.
    """
    train_df, _, _ = load_raw_fd001(data_dir=data_dir)
    dev_ids, _, _ = load_engine_splits(splits_dir=splits_dir)
    dev_df = train_df[train_df["unit_id"].isin(dev_ids)].sort_values(INDEX_COLS).reset_index(drop=True)
    return add_targets(dev_df, rul_cap=rul_cap, alarm_w=alarm_w)


def load_heldout_test_data(
    rul_cap: int | None = 125,
    alarm_w: int = 30,
    data_dir: Path | str | None = None,
    splits_dir: Path | str | None = None,
) -> pd.DataFrame:
    """Load the 20 held-out test engines (run-to-failure) for Phase 5 final evaluation ONLY."""
    train_df, _, _ = load_raw_fd001(data_dir=data_dir)
    _, test_ids, _ = load_engine_splits(splits_dir=splits_dir)
    test_df = train_df[train_df["unit_id"].isin(test_ids)].sort_values(INDEX_COLS).reset_index(drop=True)
    return add_targets(test_df, rul_cap=rul_cap, alarm_w=alarm_w)


def load_official_benchmark_test_data(
    rul_cap: int | None = 125,
    alarm_w: int = 30,
    data_dir: Path | str | None = None,
) -> pd.DataFrame:
    """Load the official truncated 100-engine FD001 test set (`test_FD001.txt` + `RUL_FD001.txt`)."""
    _, test_df, rul_series = load_raw_fd001(data_dir=data_dir)
    return add_targets(test_df, rul_cap=rul_cap, alarm_w=alarm_w, cutoff_rul=rul_series)
