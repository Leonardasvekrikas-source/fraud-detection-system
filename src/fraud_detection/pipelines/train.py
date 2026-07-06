# =============================================================================
# pipelines/train.py
#
# Train ONCE on the full dataset and persist a servable decision-level fusion
# model (the winning production config). Unlike the cross-validation pipelines
# in subsystem1/subsystem2 (which measure generalisation), this trains a single
# deployable model on all available labelled data and saves it.
#
# Steps:
#   1. dedup
#   2. StandardScaler.fit on the full dataset
#   3. F2Vote.fit on the scaled dataset
#   4. Subsystem 1: HybridOS -> LightGBM.fit                      -> P1
#   5. Subsystem 2: HybridUS -> create_windows(W) -> LSTM.fit     -> P2  (optional)
#   6. Persist a ModelBundle (scaler + F2Vote + LightGBM + LSTM + metadata)
# =============================================================================

from __future__ import annotations

from pathlib import Path

import pandas as pd

from fraud_detection import __version__, config
from fraud_detection.artifacts.store import ModelBundle
from fraud_detection.models.lightgbm_model import LightGBMModel
from fraud_detection.models.lstm_model import LSTMModel
from fraud_detection.preprocessing.base_preprocessor import BasePreprocessor
from fraud_detection.preprocessing.feature_selection import F2VoteSelector
from fraud_detection.preprocessing.hybrid_os import HybridOS
from fraud_detection.preprocessing.hybrid_us import HybridUS
from fraud_detection.preprocessing.windowing import create_windows

# Contiguous tail used to train the LSTM subsystem (see note in train_bundle).
# Keeps HybridUS's OneClassSVM tractable while preserving transaction sequences.
LSTM_TRAIN_TAIL = 120_000


def train_bundle(df: pd.DataFrame, include_lstm: bool = True) -> ModelBundle:
    """Train both subsystems on the full dataset and return a ModelBundle."""
    print("\n" + "=" * 70)
    print("  TRAIN-ONCE  ->  servable decision-level fusion model")
    print("=" * 70)

    feature_cols = [c for c in df.columns if c != config.LABEL_COL]

    df_clean = BasePreprocessor.remove_duplicates(df)
    X_all = df_clean[feature_cols].values
    y_all = df_clean[config.LABEL_COL].values

    # Scaler fitted on the full dataset (this is the deployed model).
    pre = BasePreprocessor()
    X_scaled = pre.fit_transform(X_all)

    # F2Vote fitted on the scaled dataset.
    f2vote = F2VoteSelector()
    f2vote.fit(X_scaled, y_all)
    X_f2 = f2vote.transform(X_scaled)

    selected_features = [feature_cols[i] for i in f2vote.selected_indices_]

    # Background sample of scaled + F2Vote-selected features, for LIME's
    # perturbation statistics when explaining the LSTM (only used with an LSTM).
    background = None
    if include_lstm:
        import numpy as np

        rng = np.random.default_rng(config.RANDOM_STATE)
        idx = rng.choice(len(X_f2), size=min(2000, len(X_f2)), replace=False)
        background = X_f2[idx].astype("float32")

    # -- Subsystem 1: HybridOS -> LightGBM ------------------------------------
    print("\n[train] Subsystem 1: HybridOS + LightGBM ...")
    X_os, y_os = HybridOS().fit_resample(X_f2, y_all)
    lgbm = LightGBMModel().fit(X_os, y_os)

    # -- Subsystem 2: HybridUS -> windows -> LSTM -----------------------------
    # HybridUS fits a OneClassSVM on the normal class, which is ~O(n^2): on the
    # full dataset that call is intractable. We train the LSTM on a CONTIGUOUS
    # recent tail instead - a random subsample would shred the transaction
    # sequences the LSTM depends on, but the tail keeps them intact. Same
    # precedent as experiments/paysim_generalization.py. LightGBM (Subsystem 1)
    # above still trains on the FULL dataset.
    lstm_keras = None
    if include_lstm:
        print("\n[train] Subsystem 2: HybridUS + windows + LSTM ...")
        tail = min(LSTM_TRAIN_TAIL, len(X_f2))
        X_tail, y_tail = X_f2[-tail:], y_all[-tail:]
        print(
            f"[train] LSTM on contiguous tail: {tail} rows "
            f"({int((y_tail == config.FRAUD_LABEL).sum())} fraud) of {len(X_f2)} total"
        )
        X_us, y_us = HybridUS().fit_resample(X_tail, y_tail)
        X_win, y_win = create_windows(X_us, y_us, config.WINDOW_SIZE)
        lstm = LSTMModel()
        lstm.build(input_shape=(config.WINDOW_SIZE, X_win.shape[2]))
        lstm.fit(X_win, y_win, verbose=0)
        lstm_keras = lstm.keras_model
    else:
        print("\n[train] Skipping LSTM (include_lstm=False).")

    metadata = {
        "package_version": __version__,
        "config_file": str(config.CONFIG_FILE) if config.CONFIG_FILE else None,
        "label_col": config.LABEL_COL,
        "theta": config.INFERENCE_THETA,
        "window_size": config.WINDOW_SIZE,
        "raw_feature_columns": feature_cols,
        "selected_features": selected_features,
        "selected_indices": f2vote.selected_indices_.tolist(),
        "n_train_rows": int(len(df_clean)),
        "lgbm_params": config.LGBM_PARAMS,
    }

    return ModelBundle(
        scaler=pre.scaler,
        f2vote=f2vote,
        lightgbm=lgbm.booster,
        lstm=lstm_keras,
        background=background,
        metadata=metadata,
    )


def train_and_save(
    df: pd.DataFrame, output_dir: str | Path, include_lstm: bool = True
) -> Path:
    """Train a bundle and persist it to ``output_dir``. Returns the path."""
    bundle = train_bundle(df, include_lstm=include_lstm)
    return bundle.save(output_dir)
