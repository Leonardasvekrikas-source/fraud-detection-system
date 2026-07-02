"""Central configuration, loaded from ``config/default.yaml``.

The YAML file is the source of truth. This module exposes the values as
module-level constants (``config.LGBM_PARAMS``, ``config.WINDOW_SIZE``, ...) so
the pipeline code can read them with plain attribute access.

Resolution order for the YAML file:

1. ``$FRAUD_DETECTION_CONFIG`` if set (lets an experiment point at its own copy);
2. the first ``config/default.yaml`` found walking up from this file;
3. built-in defaults below — so the package always imports even if the YAML is
   missing (e.g. when installed as a wheel without the repo checkout).

All values are from Yousefimehr & Ghatee (2025), ESWA 262, 125661.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

# Built-in fallback defaults — kept in sync with config/default.yaml. These
# guarantee the package imports even without the YAML on disk.
_DEFAULTS: dict[str, Any] = {
    "LABEL_COL": "Class",
    "FRAUD_LABEL": 1,
    "NORMAL_LABEL": 0,
    "N_SPLITS": 5,
    "RANDOM_STATE": 42,
    "SS1_TEST_SIZE": 0.3,
    "SEQ_Q_PERCENT": 0.30,
    "F2VOTE_MIN_VOTES": 2,
    "F2VOTE_K_PER_METHOD": None,
    "OCSVM_KERNEL": "rbf",
    "OCSVM_NU": 0.1,
    "TARGET_FRAUD_RATIO": 0.05,
    "LGBM_PARAMS": {
        "colsample_bytree": 0.7,
        "learning_rate": 0.01,
        "max_bin": 100,
        "max_depth": 16,
        "min_child_samples": 100,
        "min_child_weight": 0.001,
        "n_estimators": 5000,
        "num_leaves": 1000,
        "objective": "binary",
        "metric": "binary_logloss",
        "verbose": -1,
        "random_state": 42,
    },
    "LSTM_UNITS": 50,
    "LSTM_OPTIMIZER": "adam",
    "LSTM_LOSS": "binary_crossentropy",
    "LSTM_EPOCHS": 10,
    "LSTM_BATCH_SIZE": 512,
    "LSTM_RANDOM_STATE": 42,
    "WINDOW_SIZE": 3,
    "AUTO_TUNE_WINDOW": False,
    "WINDOW_SIZE_MAX": 10,
    "INFERENCE_THETA": 0.5,
}


def find_config_file() -> Path | None:
    """Locate the active ``default.yaml`` (env var, then upward search)."""
    env = os.environ.get("FRAUD_DETECTION_CONFIG")
    if env:
        return Path(env)
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "config" / "default.yaml"
        if candidate.is_file():
            return candidate
    return None


def load_values(path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Return the config dict: built-in defaults overlaid with the YAML file."""
    values = {**_DEFAULTS}
    yaml_path = Path(path) if path is not None else find_config_file()
    if yaml_path is not None and yaml_path.is_file():
        with open(yaml_path, encoding="utf-8") as fh:
            overlay = yaml.safe_load(fh) or {}
        values.update(overlay)
    return values


# ---------------------------------------------------------------------------
# Populate module-level constants at import time.
# ---------------------------------------------------------------------------
_VALUES = load_values()

CONFIG_FILE = find_config_file()

LABEL_COL: str = _VALUES["LABEL_COL"]
FRAUD_LABEL: int = _VALUES["FRAUD_LABEL"]
NORMAL_LABEL: int = _VALUES["NORMAL_LABEL"]

N_SPLITS: int = _VALUES["N_SPLITS"]
RANDOM_STATE: int = _VALUES["RANDOM_STATE"]
SS1_TEST_SIZE: float = _VALUES["SS1_TEST_SIZE"]
SEQ_Q_PERCENT: float = _VALUES["SEQ_Q_PERCENT"]

F2VOTE_MIN_VOTES: int = _VALUES["F2VOTE_MIN_VOTES"]
F2VOTE_K_PER_METHOD = _VALUES["F2VOTE_K_PER_METHOD"]

OCSVM_KERNEL: str = _VALUES["OCSVM_KERNEL"]
OCSVM_NU: float = _VALUES["OCSVM_NU"]

TARGET_FRAUD_RATIO: float = _VALUES["TARGET_FRAUD_RATIO"]

LGBM_PARAMS: dict[str, Any] = _VALUES["LGBM_PARAMS"]

LSTM_UNITS: int = _VALUES["LSTM_UNITS"]
LSTM_OPTIMIZER: str = _VALUES["LSTM_OPTIMIZER"]
LSTM_LOSS: str = _VALUES["LSTM_LOSS"]
LSTM_EPOCHS: int = _VALUES["LSTM_EPOCHS"]
LSTM_BATCH_SIZE: int = _VALUES["LSTM_BATCH_SIZE"]
LSTM_RANDOM_STATE: int = _VALUES["LSTM_RANDOM_STATE"]

WINDOW_SIZE: int = _VALUES["WINDOW_SIZE"]
AUTO_TUNE_WINDOW: bool = _VALUES["AUTO_TUNE_WINDOW"]
WINDOW_SIZE_MAX: int = _VALUES["WINDOW_SIZE_MAX"]

INFERENCE_THETA: float = _VALUES["INFERENCE_THETA"]


def reload(path: str | os.PathLike[str] | None = None) -> None:
    """Re-load config values and rebind the module constants in place.

    Used by the CLI's ``--config`` flag. Because pipeline code reads
    ``config.X`` at call time (attribute access on this module), rebinding the
    module globals here is picked up by everything downstream.
    """
    global _VALUES, CONFIG_FILE
    if path is not None:
        os.environ["FRAUD_DETECTION_CONFIG"] = str(path)
    else:
        # Fall back to the default upward search (not any previously set env var).
        os.environ.pop("FRAUD_DETECTION_CONFIG", None)
    _VALUES = load_values(path)
    CONFIG_FILE = Path(path) if path is not None else find_config_file()
    globals().update(_VALUES)
