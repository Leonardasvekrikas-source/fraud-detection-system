import numpy as np

from fraud_detection.fusion import engine


def test_single_decisions_default_theta():
    # theta = 0.5
    assert engine.classify_single(0.1, 0.1) == engine.LABEL_NORMAL  # 0.2 < 0.5
    assert engine.classify_single(0.8, 0.8) == engine.LABEL_FRAUD  # 1.6 >= 1.5
    assert engine.classify_single(0.5, 0.4) == engine.LABEL_EXPERT_CHECKING  # 0.9


def test_boundary_conditions():
    # p_sum == theta -> not Normal (strict <), so Expert-Checking
    assert engine.classify_single(0.25, 0.25) == engine.LABEL_EXPERT_CHECKING  # 0.5
    # p_sum == 1 + theta -> Fraud (>=)
    assert engine.classify_single(0.75, 0.75) == engine.LABEL_FRAUD  # 1.5


def test_custom_theta():
    assert engine.classify_single(0.3, 0.3, theta=0.7) == engine.LABEL_NORMAL  # 0.6 < 0.7


def test_batch_matches_single():
    p1 = np.array([0.1, 0.8, 0.5])
    p2 = np.array([0.1, 0.8, 0.4])
    batch = engine.classify_batch(p1, p2, verbose=False)
    singles = [engine.classify_single(a, b) for a, b in zip(p1, p2, strict=True)]
    assert list(batch) == singles
