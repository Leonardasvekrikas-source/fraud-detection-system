# =============================================================================
# preprocessing/base_preprocessor.py
#
# Shared preprocessing steps executed identically by both subsystems.
# Implements the two categories described in the paper Section 2.1:
#
#   1. Preprocessing INDEPENDENT of statistical summary  (pre-split)
#      -> Duplicate removal  (Section 2.1.1)
#
#   2. Preprocessing DEPENDENT on statistical summary  (post-split, train only)
#      -> Standard normalisation fitted on training data  (Eq. 1)
#
# No fitting information from the test fold ever touches train statistics.
# =============================================================================

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class BasePreprocessor:
    """Statistical-independent and statistical-dependent preprocessing (Section 2.1).

    Usage
    -----
    # --- Pre-split step (call once on the full raw DataFrame) ---
    clean_df = BasePreprocessor.remove_duplicates(raw_df)

    # --- Post-split steps (one instance per CV fold) ---
    pre = BasePreprocessor()
    X_train_scaled = pre.fit_transform(X_train)   # fits scaler on train
    X_test_scaled  = pre.transform(X_test)         # applies same scaler to test
    """

    def __init__(self):
        self._scaler = StandardScaler()
        self._fitted = False

    # -------------------------------------------------------------------------
    # Pre-split  -  statistical-independent  (Section 2.1.1)
    # -------------------------------------------------------------------------

    @staticmethod
    def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
        """Remove exact duplicate rows from the raw dataset.

        Must be called BEFORE any train/test split to avoid data leakage caused
        by identical samples appearing in both splits. Index is reset.
        """
        before = len(df)
        df_clean = df.drop_duplicates().reset_index(drop=True)
        removed = before - len(df_clean)
        print(
            f"[BasePreprocessor] remove_duplicates: "
            f"{before} rows -> {len(df_clean)} rows  ({removed} duplicates removed)"
        )
        return df_clean

    # -------------------------------------------------------------------------
    # Post-split  -  statistical-dependent  (Section 2.1.3 / Eq. 1)
    # -------------------------------------------------------------------------

    def fit(self, X_train: np.ndarray) -> "BasePreprocessor":
        """Compute mean and standard deviation from the training fold only."""
        self._scaler.fit(X_train)
        self._fitted = True
        print(
            f"[BasePreprocessor] fit: scaler fitted on {X_train.shape[0]} "
            f"samples, {X_train.shape[1]} features"
        )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply the normalisation  x' = (x - mu) / sigma  (Eq. 1)."""
        if not self._fitted:
            raise RuntimeError("BasePreprocessor.fit() must be called before transform().")
        return self._scaler.transform(X)

    def fit_transform(self, X_train: np.ndarray) -> np.ndarray:
        """Fit on X_train and return its normalised version."""
        self.fit(X_train)
        return self.transform(X_train)

    @property
    def scaler(self) -> StandardScaler:
        """The fitted StandardScaler (needed for persistence / serving)."""
        return self._scaler
