# =============================================================================
# pipelines/subsystem1.py
#
# Subsystem 1: Non-Sequential Credit Card Fraud Detection  (paper Fig. 1)
#
# Per CV fold:
#   [Pre-split]  remove_duplicates; F2Vote fitted ONCE on the full dataset
#                (paper Section 4.7.1: "applied only once during training").
#   [Per fold - train only]  StandardScaler.fit; F2Vote mask; HybridOS; LightGBM.fit
#   [Test fold - no fitting]  transform; predict; compute_metrics
#
# Cross-validation: ShuffleSplit(n_splits=5, test_size=0.3) - Section 2.1.2
# (Non-sequential data: temporal order not required, shuffle is fine.)
# =============================================================================

import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit

from fraud_detection import config
from fraud_detection.evaluation.metrics import (
    average_cv_metrics,
    compute_metrics,
    print_average_metrics,
    print_fold_metrics,
)
from fraud_detection.models.lightgbm_model import LightGBMModel
from fraud_detection.preprocessing.base_preprocessor import BasePreprocessor
from fraud_detection.preprocessing.feature_selection import F2VoteSelector
from fraud_detection.preprocessing.hybrid_os import HybridOS


def run_subsystem1(df: pd.DataFrame) -> dict:
    """Execute the full Subsystem 1 (non-sequential) pipeline.

    Returns averaged CV metrics plus a 'fold_metrics' list.
    """
    print("\n" + "=" * 70)
    print("  SUBSYSTEM 1 - Non-Sequential (LightGBM + HybridOS)")
    print("=" * 70)

    # -- Step 1: Pre-split preprocessing (Section 2.1.1) ----------------------
    df_clean = BasePreprocessor.remove_duplicates(df)

    X_all = df_clean.drop(columns=[config.LABEL_COL]).values
    y_all = df_clean[config.LABEL_COL].values

    print(
        f"[Subsystem1] Dataset: {X_all.shape[0]} samples, "
        f"{X_all.shape[1]} features, "
        f"fraud rate = {np.mean(y_all == config.FRAUD_LABEL):.4f}"
    )

    # -- Step 2: F2Vote - fitted ONCE on full dataset (Section 4.7.1) ---------
    print("\n[Subsystem1] Running F2Vote on full dataset (one-time) ...")
    _pre_global = BasePreprocessor()
    X_all_scaled_global = _pre_global.fit_transform(X_all)

    f2vote = F2VoteSelector()
    f2vote.fit(X_all_scaled_global, y_all)
    print(
        f"[Subsystem1] F2Vote complete - "
        f"{len(f2vote.selected_indices_)} features selected. Mask reused across all folds."
    )

    # -- Step 3: 5-fold ShuffleSplit (Section 2.1.2) --------------------------
    splitter = ShuffleSplit(
        n_splits=config.N_SPLITS,
        test_size=config.SS1_TEST_SIZE,
        random_state=config.RANDOM_STATE,
    )

    fold_metrics = []

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X_all)):
        print(f"\n--- Subsystem 1 | Fold {fold_idx + 1}/{config.N_SPLITS} ---")

        X_train_raw, X_test_raw = X_all[train_idx], X_all[test_idx]
        y_train, y_test = y_all[train_idx], y_all[test_idx]

        # Normalisation - fit on train only (Eq. 1)
        pre = BasePreprocessor()
        X_train_scaled = pre.fit_transform(X_train_raw)
        X_test_scaled = pre.transform(X_test_raw)

        # F2Vote mask (no refit)
        X_train_f2 = f2vote.transform(X_train_scaled)
        X_test_f2 = f2vote.transform(X_test_scaled)

        # HybridOS resampling - training fold only
        hyb_os = HybridOS()
        X_res, y_res = hyb_os.fit_resample(X_train_f2, y_train)

        # LightGBM training
        model = LightGBMModel()
        model.fit(X_res, y_res)

        # Evaluation on test fold
        y_pred = model.predict(X_test_f2)
        y_prob = model.predict_proba(X_test_f2)

        metrics = compute_metrics(y_test, y_pred, y_prob)
        print_fold_metrics(fold_idx, metrics)
        fold_metrics.append(metrics)

    averaged = average_cv_metrics(fold_metrics)
    averaged["fold_metrics"] = fold_metrics
    print_average_metrics(averaged, label="Subsystem 1 - 5-Fold CV Average")

    return averaged
