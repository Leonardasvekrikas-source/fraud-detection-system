# =============================================================================
# preprocessing/windowing.py
#
# Sliding-window sequence creation for Subsystem 2  (paper Section 2.4).
#
# The LSTM model requires 3-D input of shape (n_samples, window_size, n_features).
# Each sample i becomes a window of the W most recent transactions
# [i-W+1, ..., i] and the label is the label of the last transaction.
#
# Temporal order must be preserved throughout - do NOT shuffle the output
# before feeding it to the LSTM.
#
# For the first (W-1) samples, where a full window cannot be formed, the window
# is zero-padded on the LEFT. This is the standard approach for streaming /
# real-time inference and does not introduce data leakage because only zeros
# (not future or external data) are used.
# =============================================================================

import numpy as np


def create_windows(X: np.ndarray, y: np.ndarray, window_size: int):
    """Create overlapping sliding windows over a time-ordered transaction sequence.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Time-ordered, normalised, feature-selected transaction data.
    y : np.ndarray, shape (n_samples,)
        Time-ordered binary labels (0 = normal, 1 = fraud).
    window_size : int
        Number of consecutive transactions per window (W in the paper).

    Returns
    -------
    X_win : np.ndarray, shape (n_samples, window_size, n_features)
    y_win : np.ndarray, shape (n_samples,)
        Label of the LAST transaction in each window.
    """
    n_samples, n_features = X.shape

    X_win = np.zeros((n_samples, window_size, n_features), dtype=X.dtype)
    y_win = np.asarray(y, dtype=y.dtype if hasattr(y, "dtype") else float)

    for i in range(n_samples):
        start = i - window_size + 1
        if start >= 0:
            # Full window available
            X_win[i] = X[start : i + 1]
        else:
            # Partial window: zero-pad on the left
            available = i + 1
            pad_len = window_size - available
            X_win[i, pad_len:, :] = X[0:available]
            # X_win[i, :pad_len, :] stays zero (already initialised)

    print(
        f"[windowing] create_windows: "
        f"{n_samples} samples x W={window_size} x {n_features} features "
        f"-> X_win shape {X_win.shape}"
    )

    return X_win, y_win
