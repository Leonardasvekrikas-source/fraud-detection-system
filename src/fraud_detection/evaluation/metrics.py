# =============================================================================
# evaluation/metrics.py
#
# All evaluation metrics reported in the paper (Appendix B), computed per fold
# and averaged across folds:
#   Accuracy, Precision, Recall, F1, AUC, MCC, Balanced Accuracy.
# ROC curves are recorded per fold so mean +/- std can be plotted.
# =============================================================================

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from fraud_detection import config


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    """Compute all seven metrics from Appendix B for a single fold.

    Returns a dict with keys: accuracy, precision, recall, f1, auc, mcc,
    balanced_accuracy, and the ROC arrays fpr, tpr.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob, pos_label=config.FRAUD_LABEL)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, pos_label=config.FRAUD_LABEL, zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, pos_label=config.FRAUD_LABEL, zero_division=0
        ),
        "f1": f1_score(y_true, y_pred, pos_label=config.FRAUD_LABEL, zero_division=0),
        "auc": roc_auc_score(y_true, y_prob),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "fpr": fpr,
        "tpr": tpr,
    }


def average_cv_metrics(fold_metrics: list) -> dict:
    """Average scalar metrics over all CV folds (mean +/- std).

    Returns a dict with <metric>_mean and <metric>_std keys, plus fpr_list and
    tpr_list for ROC plotting.
    """
    scalar_keys = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "auc",
        "mcc",
        "balanced_accuracy",
    ]

    averaged = {}
    for key in scalar_keys:
        values = [m[key] for m in fold_metrics]
        averaged[f"{key}_mean"] = float(np.mean(values))
        averaged[f"{key}_std"] = float(np.std(values))

    averaged["fpr_list"] = [m["fpr"] for m in fold_metrics]
    averaged["tpr_list"] = [m["tpr"] for m in fold_metrics]

    return averaged


def print_fold_metrics(fold: int, metrics: dict) -> None:
    """Print metrics for a single fold in a readable format."""
    print(
        f"  Fold {fold + 1} | "
        f"Acc={metrics['accuracy']:.6f}  "
        f"Prec={metrics['precision']:.4f}  "
        f"Rec={metrics['recall']:.4f}  "
        f"F1={metrics['f1']:.4f}  "
        f"AUC={metrics['auc']:.4f}  "
        f"MCC={metrics['mcc']:.4f}  "
        f"BalAcc={metrics['balanced_accuracy']:.4f}"
    )


def print_average_metrics(averaged: dict, label: str = "5-Fold CV Average") -> None:
    """Print averaged metrics in the same column order as the paper's tables."""
    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"{'=' * 70}")
    row = (
        f"  Accuracy : {averaged['accuracy_mean']:.6f} +/- {averaged['accuracy_std']:.6f}\n"
        f"  Precision: {averaged['precision_mean']:.4f} +/- {averaged['precision_std']:.4f}\n"
        f"  Recall   : {averaged['recall_mean']:.4f} +/- {averaged['recall_std']:.4f}\n"
        f"  F1       : {averaged['f1_mean']:.4f} +/- {averaged['f1_std']:.4f}\n"
        f"  AUC      : {averaged['auc_mean']:.4f} +/- {averaged['auc_std']:.4f}\n"
        f"  MCC      : {averaged['mcc_mean']:.4f} +/- {averaged['mcc_std']:.4f}\n"
        f"  Bal.Acc  : {averaged['balanced_accuracy_mean']:.4f} "
        f"+/- {averaged['balanced_accuracy_std']:.4f}"
    )
    print(row)
    print(f"{'=' * 70}\n")
