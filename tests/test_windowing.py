import numpy as np

from fraud_detection.preprocessing.windowing import create_windows


def test_shape_and_labels():
    X = np.arange(10).reshape(5, 2).astype(float)  # 5 rows, 2 features
    y = np.array([0, 1, 0, 1, 1])
    X_win, y_win = create_windows(X, y, window_size=3)

    assert X_win.shape == (5, 3, 2)
    # label of each window is the label of the last (current) transaction
    np.testing.assert_array_equal(y_win, y)


def test_left_zero_padding_for_early_rows():
    X = np.arange(10).reshape(5, 2).astype(float)
    y = np.zeros(5)
    X_win, _ = create_windows(X, y, window_size=3)

    # row 0: only current transaction available -> two zero rows then X[0]
    np.testing.assert_array_equal(X_win[0, 0], [0, 0])
    np.testing.assert_array_equal(X_win[0, 1], [0, 0])
    np.testing.assert_array_equal(X_win[0, 2], X[0])

    # row 1: one pad row then X[0], X[1]
    np.testing.assert_array_equal(X_win[1, 0], [0, 0])
    np.testing.assert_array_equal(X_win[1, 1], X[0])
    np.testing.assert_array_equal(X_win[1, 2], X[1])


def test_no_future_leakage():
    # A full window ending at i must contain only rows <= i.
    X = np.arange(12).reshape(6, 2).astype(float)
    y = np.zeros(6)
    X_win, _ = create_windows(X, y, window_size=3)
    # window at i=4 should be rows [2, 3, 4]
    np.testing.assert_array_equal(X_win[4], X[2:5])
