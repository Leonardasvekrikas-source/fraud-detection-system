"""Test the Expected Calibration Error helper (pure logic, no training)."""

import numpy as np

from fraud_detection.experiments.calibration import expected_calibration_error


def test_ece_is_zero_when_perfectly_calibrated():
    # predicted probability equals the outcome -> no calibration gap
    y = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    p = y.astype(float)
    assert expected_calibration_error(y, p, n_bins=5) == 0.0


def test_ece_flags_overconfidence():
    # 100 samples, 10% positive, but the model says 0.9 for all -> huge gap
    y = np.array([0] * 90 + [1] * 10)
    p = np.full(100, 0.9)
    ece = expected_calibration_error(y, p, n_bins=10)
    assert ece > 0.5  # confidence 0.9 vs accuracy 0.1


def test_ece_in_unit_range():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=200)
    p = rng.random(200)
    assert 0.0 <= expected_calibration_error(y, p) <= 1.0
