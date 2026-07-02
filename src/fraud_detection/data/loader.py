# =============================================================================
# data/loader.py
#
# Load and validate a raw transaction CSV. The CSV must contain a column named
# config.LABEL_COL (default "Class") with values config.FRAUD_LABEL (1) and
# config.NORMAL_LABEL (0).
# =============================================================================

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from fraud_detection import config

# Repo-root data directory (…/data). Datasets are gitignored; fetch with
# scripts/fetch_data.py. Resolved relative to this file so it works regardless
# of the current working directory.
DATA_DIR = Path(__file__).resolve().parents[3] / "data"
DEFAULT_DATA_FILE = "creditcard.csv"


def resolve_data_path(name_or_path: str | os.PathLike[str]) -> Path:
    """Resolve a dataset argument to an absolute path.

    Accepts either a bare filename (looked up inside the repo ``data/`` dir) or
    an explicit path (used as-is).
    """
    p = Path(name_or_path)
    if p.is_absolute() or p.parent != Path("."):
        return p
    return DATA_DIR / p


def load_data(name_or_path: str | os.PathLike[str] = DEFAULT_DATA_FILE) -> pd.DataFrame:
    """Load a raw CSV dataset from disk and validate the label column."""
    data_path = resolve_data_path(name_or_path)

    if not data_path.is_file():
        raise FileNotFoundError(
            f"Data file not found: {data_path}\n"
            f"Place the CSV in {DATA_DIR} or run: python scripts/fetch_data.py"
        )

    print(f"[loader] Loading data from: {data_path}")
    df = pd.read_csv(data_path)

    if config.LABEL_COL not in df.columns:
        raise ValueError(
            f"Label column '{config.LABEL_COL}' not found in the dataset.\n"
            f"Available columns: {list(df.columns)}\n"
            f"Update config.LABEL_COL (config/default.yaml) to match your dataset."
        )

    n_normal = int((df[config.LABEL_COL] == config.NORMAL_LABEL).sum())
    n_fraud = int((df[config.LABEL_COL] == config.FRAUD_LABEL).sum())
    print(f"[loader] Loaded {len(df)} rows x {df.shape[1]} columns")
    print(
        f"[loader] Label distribution: "
        f"{config.NORMAL_LABEL} (normal): {n_normal}, "
        f"{config.FRAUD_LABEL} (fraud): {n_fraud}"
    )

    return df
