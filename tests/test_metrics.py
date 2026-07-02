import numpy as np

from fraud_detection.evaluation.metrics import average_cv_metrics, compute_metrics


def test_perfect_predictions():
    y_true = np.array([0, 0, 1, 1])
    y_pred = y_true.copy()
    y_prob = np.array([0.01, 0.02, 0.99, 0.98])
    m = compute_metrics(y_true, y_pred, y_prob)
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0
    assert m["auc"] == 1.0


def test_known_confusion():
    # TP=2, FP=1, FN=0 -> precision 2/3, recall 1.0, f1 0.8
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    y_prob = np.array([0.1, 0.4, 0.9, 0.8])
    m = compute_metrics(y_true, y_pred, y_prob)
    assert abs(m["precision"] - 2 / 3) < 1e-9
    assert m["recall"] == 1.0
    assert abs(m["f1"] - 0.8) < 1e-9


def test_average_cv_metrics_mean_std():
    folds = [
        compute_metrics(np.array([0, 1]), np.array([0, 1]), np.array([0.1, 0.9])),
        compute_metrics(np.array([0, 1]), np.array([0, 1]), np.array([0.2, 0.7])),
    ]
    avg = average_cv_metrics(folds)
    assert avg["f1_mean"] == 1.0
    assert avg["f1_std"] == 0.0
    assert len(avg["fpr_list"]) == 2
