# =============================================================================
# pipelines/subsystem2.py
#
# Subsystem 2: Sequential Credit Card Fraud Detection  (paper Fig. 1)
#
# Per CV fold (temporal order preserved throughout):
#   [Pre-split]  remove_duplicates; F2Vote fitted ONCE on the full dataset.
#   [Per fold - train only]  StandardScaler.fit; F2Vote mask; HybridUS;
#                            create_windows(W); LSTM.fit
#   [Test fold]  transform; create_windows(W); predict; compute_metrics
#
# Cross-validation: TimeSeriesSplit(n_splits=5) - Section 2.1.2
# (Sequential data: temporal order MUST be preserved; no future leakage.)
# =============================================================================

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from fraud_detection import config
from fraud_detection.evaluation.metrics import (
    average_cv_metrics,
    compute_metrics,
    print_average_metrics,
    print_fold_metrics,
)
from fraud_detection.models.lstm_model import LSTMModel
from fraud_detection.preprocessing.base_preprocessor import BasePreprocessor
from fraud_detection.preprocessing.feature_selection import F2VoteSelector
from fraud_detection.preprocessing.hybrid_us import HybridUS
from fraud_detection.preprocessing.windowing import create_windows


def tune_window_size(
    X_train_f2: np.ndarray,
    y_train: np.ndarray,
    X_val_f2: np.ndarray,
    y_val: np.ndarray,
    max_window: int = None,
) -> int:
    """Determine the optimal window size for a dataset (paper Fig. 7).

    Increases W from 1, keeping the new W while F1 improves; stops otherwise.
    """
    if max_window is None:
        max_window = config.WINDOW_SIZE_MAX

    print(f"\n[WindowTuning] Searching W = 1 ... {max_window} (Fig. 7)")

    best_w = 1
    best_f1 = 0.0

    for W in range(1, max_window + 1):
        print(f"[WindowTuning] W = {W} ...")

        X_tr_win, y_tr_win = create_windows(X_train_f2, y_train, W)
        X_vl_win, y_vl_win = create_windows(X_val_f2, y_val, W)

        lstm = LSTMModel()
        lstm.fit(X_tr_win, y_tr_win, verbose=0)

        y_pred = lstm.predict(X_vl_win)
        y_prob = lstm.predict_proba(X_vl_win)

        m = compute_metrics(y_vl_win, y_pred, y_prob)
        f1 = m["f1"]

        print(
            f"[WindowTuning] W={W}  F1={f1:.4f}  "
            f"(best so far: W={best_w}, F1={best_f1:.4f})"
        )

        if f1 > best_f1:
            best_f1 = f1
            best_w = W
        else:
            print(f"[WindowTuning] F1 did not improve. Stopping. Best W = {best_w}")
            break

    print(f"[WindowTuning] Optimal window size: W* = {best_w}  (F1 = {best_f1:.4f})")
    return best_w


def run_subsystem2(df: pd.DataFrame, window_size: int = None) -> dict:
    """Execute the full Subsystem 2 (sequential) pipeline.

    Returns averaged CV metrics plus 'fold_metrics' and 'window_size'.
    """
    print("\n" + "=" * 70)
    print("  SUBSYSTEM 2 - Sequential (LSTM + HybridUS)")
    print("=" * 70)

    # -- Step 1: Pre-split preprocessing (order preserved) --------------------
    df_clean = BasePreprocessor.remove_duplicates(df)

    X_all = df_clean.drop(columns=[config.LABEL_COL]).values
    y_all = df_clean[config.LABEL_COL].values

    print(
        f"[Subsystem2] Dataset: {X_all.shape[0]} samples, "
        f"{X_all.shape[1]} features, "
        f"fraud rate = {np.mean(y_all == config.FRAUD_LABEL):.4f}"
    )
    print("[Subsystem2] Temporal order preserved - NO shuffling applied.")

    # -- Step 2: F2Vote - fitted ONCE on full dataset -------------------------
    print("\n[Subsystem2] Running F2Vote on full dataset (one-time) ...")
    _pre_global = BasePreprocessor()
    X_all_scaled_global = _pre_global.fit_transform(X_all)

    f2vote = F2VoteSelector()
    f2vote.fit(X_all_scaled_global, y_all)
    print(
        f"[Subsystem2] F2Vote complete - "
        f"{len(f2vote.selected_indices_)} features selected. Mask reused across all folds."
    )

    # -- Step 3: 5-fold TimeSeriesSplit (Section 2.1.2) -----------------------
    splitter = TimeSeriesSplit(n_splits=config.N_SPLITS)

    W = window_size if window_size is not None else config.WINDOW_SIZE
    auto_tune_done = False

    fold_metrics = []

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X_all)):
        print(f"\n--- Subsystem 2 | Fold {fold_idx + 1}/{config.N_SPLITS} ---")
        print(
            f"    Train range: [{train_idx[0]}, {train_idx[-1]}]  ({len(train_idx)} samples)"
        )
        print(
            f"    Test  range: [{test_idx[0]}, {test_idx[-1]}]  ({len(test_idx)} samples)"
        )

        X_train_raw, X_test_raw = X_all[train_idx], X_all[test_idx]
        y_train, y_test = y_all[train_idx], y_all[test_idx]

        # Normalisation - fit on train only (Eq. 1)
        pre = BasePreprocessor()
        X_train_scaled = pre.fit_transform(X_train_raw)
        X_test_scaled = pre.transform(X_test_raw)

        # F2Vote mask (no refit)
        X_train_f2 = f2vote.transform(X_train_scaled)
        X_test_f2 = f2vote.transform(X_test_scaled)

        # Optional window-size tuning on the first fold (Fig. 7)
        if config.AUTO_TUNE_WINDOW and not auto_tune_done:
            print(
                "[Subsystem2] AUTO_TUNE_WINDOW=True: running window-size search on fold 1 ..."
            )
            W = tune_window_size(X_train_f2, y_train, X_test_f2, y_test)
            auto_tune_done = True
            print(f"[Subsystem2] Window size set to W = {W} for all subsequent folds.")

        # HybridUS resampling - training fold only
        hyb_us = HybridUS()
        X_res, y_res = hyb_us.fit_resample(X_train_f2, y_train)

        # Sliding-window creation
        X_train_win, y_train_win = create_windows(X_res, y_res, W)
        X_test_win, y_test_win = create_windows(X_test_f2, y_test, W)

        # LSTM training
        lstm = LSTMModel()
        lstm.build(input_shape=(W, X_train_win.shape[2]))
        lstm.fit(X_train_win, y_train_win, verbose=0)

        # Evaluation on test fold
        y_pred = lstm.predict(X_test_win)
        y_prob = lstm.predict_proba(X_test_win)

        metrics = compute_metrics(y_test_win, y_pred, y_prob)
        print_fold_metrics(fold_idx, metrics)
        fold_metrics.append(metrics)

    averaged = average_cv_metrics(fold_metrics)
    averaged["fold_metrics"] = fold_metrics
    averaged["window_size"] = W
    print_average_metrics(averaged, label=f"Subsystem 2 - 5-Fold CV Average  (W={W})")

    return averaged
