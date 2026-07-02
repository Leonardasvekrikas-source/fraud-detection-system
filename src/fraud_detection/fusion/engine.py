# =============================================================================
# fusion/engine.py
#
# Decision-level fusion / inference engine  -  paper Section 2.6 / Algorithm 1.
#
# Combines P1 (LightGBM) and P2 (LSTM) into a three-class decision:
#
#     P_sum = P1 + P2
#     if   P_sum <  theta       -> "Normal"
#     elif P_sum >= 1 + theta   -> "Fraud"
#     else                      -> "Expert-Checking"   (route to a human,
#                                                        with SHAP/LIME support)
#
# This is the decision-level fusion at the heart of the project's argument:
# the two models stay separable, so the score is attributable to each and a
# fast SHAP TreeExplainer can explain the gradient-boosted component.
# =============================================================================

import numpy as np

from fraud_detection import config

LABEL_NORMAL = "Normal"
LABEL_FRAUD = "Fraud"
LABEL_EXPERT_CHECKING = "Expert-Checking"


def classify_single(p1: float, p2: float, theta: float = None) -> str:
    """Run Algorithm 1 for a single transaction.

    Returns one of "Normal", "Fraud", or "Expert-Checking".
    """
    if theta is None:
        theta = config.INFERENCE_THETA

    p_sum = p1 + p2
    if p_sum < theta:
        return LABEL_NORMAL
    elif p_sum >= 1.0 + theta:
        return LABEL_FRAUD
    return LABEL_EXPERT_CHECKING


def classify_batch(
    p1_array: np.ndarray,
    p2_array: np.ndarray,
    theta: float = None,
    verbose: bool = True,
) -> np.ndarray:
    """Run Algorithm 1 for a batch of transactions.

    Returns an array of str labels, one per transaction.
    """
    if theta is None:
        theta = config.INFERENCE_THETA

    p1 = np.asarray(p1_array, dtype=float)
    p2 = np.asarray(p2_array, dtype=float)
    p_sum = p1 + p2

    labels = np.where(
        p_sum < theta,
        LABEL_NORMAL,
        np.where(p_sum >= 1.0 + theta, LABEL_FRAUD, LABEL_EXPERT_CHECKING),
    )

    if verbose:
        n_total = len(labels)
        n_normal = int(np.sum(labels == LABEL_NORMAL))
        n_fraud = int(np.sum(labels == LABEL_FRAUD))
        n_expert = int(np.sum(labels == LABEL_EXPERT_CHECKING))
        print(
            f"[InferenceEngine] theta={theta}  ->  "
            f"Normal: {n_normal} ({100 * n_normal / n_total:.3f}%)  "
            f"Fraud: {n_fraud} ({100 * n_fraud / n_total:.3f}%)  "
            f"Expert-Checking: {n_expert} ({100 * n_expert / n_total:.3f}%)"
        )

    return labels
