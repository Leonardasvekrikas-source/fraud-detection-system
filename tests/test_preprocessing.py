import numpy as np
import pandas as pd
import pytest

from fraud_detection.preprocessing.base_preprocessor import BasePreprocessor


def test_remove_duplicates():
    df = pd.DataFrame({"a": [1, 1, 2], "b": [3, 3, 4], "Class": [0, 0, 1]})
    clean = BasePreprocessor.remove_duplicates(df)
    assert len(clean) == 2
    # index is reset
    assert list(clean.index) == [0, 1]


def test_scaler_fit_on_train_only():
    rng = np.random.default_rng(0)
    X_train = rng.normal(loc=5.0, scale=2.0, size=(100, 3))
    pre = BasePreprocessor()
    X_train_scaled = pre.fit_transform(X_train)
    # training data is standardised to ~0 mean, ~1 std
    assert np.allclose(X_train_scaled.mean(axis=0), 0.0, atol=1e-6)
    assert np.allclose(X_train_scaled.std(axis=0), 1.0, atol=1e-6)

    # test data uses TRAIN statistics -> its mean is generally not 0
    X_test = rng.normal(loc=9.0, scale=2.0, size=(50, 3))
    X_test_scaled = pre.transform(X_test)
    assert not np.allclose(X_test_scaled.mean(axis=0), 0.0, atol=0.1)


def test_transform_before_fit_raises():
    with pytest.raises(RuntimeError):
        BasePreprocessor().transform(np.zeros((2, 2)))
