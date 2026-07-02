# =============================================================================
# preprocessing/feature_selection.py
#
# F2Vote  -  paper Section 2.2
#
# Six independent feature selection methods are run on the training data.
# A feature is retained if it is selected by at least F2VOTE_MIN_VOTES (2)
# of the six methods.
#
#   1. Information Value (WOE-based)
#   2. Recursive Feature Elimination  (RFE with Logistic Regression)
#   3. Variable importance - Random Forest
#   4. Variable importance - Extra Trees
#   5. Chi-Square best variables  (SelectKBest)
#   6. L1-based feature selection  (Lasso / L1 Logistic Regression)
#
# Each method selects the top-k features (k = n_features // 2 by default).
#
# IMPORTANT: fit() uses only training data. transform() is applied to both
# train and test folds using the mask learned from training.
# =============================================================================

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import RFE, SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

from fraud_detection import config


class F2VoteSelector:
    """Feature selector that combines six methods via majority voting.

    Parameters
    ----------
    k : int or None
        Number of features each method selects. None -> n_features // 2.
    min_votes : int
        Minimum number of methods that must select a feature for it to be
        retained. Default from config.F2VOTE_MIN_VOTES (= 2).
    random_state : int
        Seed for reproducibility.
    """

    def __init__(self, k=None, min_votes=None, random_state=None):
        self.k = k
        self.min_votes = min_votes if min_votes is not None else config.F2VOTE_MIN_VOTES
        self.random_state = random_state if random_state is not None else config.RANDOM_STATE

        self.selected_indices_ = None  # int array of retained column indices
        self.selected_mask_ = None  # bool array length n_features
        self.vote_counts_ = None  # int array: votes each feature received
        self._feature_names = None  # column names if DataFrame was passed

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def fit(self, X_train, y_train):
        """Run all six methods on the training fold and determine the voting mask."""
        if isinstance(X_train, pd.DataFrame):
            self._feature_names = list(X_train.columns)
            X = X_train.values
        else:
            X = np.asarray(X_train)

        y = np.asarray(y_train)
        n_features = X.shape[1]
        k = self.k if self.k is not None else n_features // 2
        k = min(k, n_features)  # guard: k cannot exceed total features

        print(
            f"[F2Vote] fit: {n_features} input features, "
            f"selecting top {k} per method, min_votes={self.min_votes}"
        )

        masks = []
        labels = []

        for method in (
            self._method_iv,
            self._method_rfe,
            self._method_random_forest,
            self._method_extra_trees,
            self._method_chi2,
            self._method_l1,
        ):
            mask, name = method(X, y, k)
            masks.append(mask)
            labels.append(name)

        # Tally votes
        vote_counts = np.sum(np.stack(masks, axis=0), axis=0)  # shape (n_features,)
        selected_mask = vote_counts >= self.min_votes

        self.vote_counts_ = vote_counts
        self.selected_mask_ = selected_mask
        self.selected_indices_ = np.where(selected_mask)[0]

        print("[F2Vote] Per-method selection counts:")
        for name, mask in zip(labels, masks, strict=True):
            print(f"         {name:<30s}: {int(mask.sum())} features selected")

        feature_ids = (
            [self._feature_names[i] for i in self.selected_indices_]
            if self._feature_names
            else self.selected_indices_.tolist()
        )
        print(
            f"[F2Vote] Final selection ({len(self.selected_indices_)} features "
            f"with >= {self.min_votes} votes): {feature_ids}"
        )

        return self

    def transform(self, X):
        """Return only the columns selected by F2Vote."""
        if self.selected_mask_ is None:
            raise RuntimeError("F2VoteSelector.fit() must be called before transform().")

        if isinstance(X, pd.DataFrame):
            return X.values[:, self.selected_indices_]
        return np.asarray(X)[:, self.selected_indices_]

    def fit_transform(self, X_train, y_train):
        """Fit on X_train and immediately return its reduced version."""
        self.fit(X_train, y_train)
        return self.transform(X_train)

    # -------------------------------------------------------------------------
    # Private: individual feature selection methods
    # Each returns (bool_mask_shape_n_features, method_name_string)
    # -------------------------------------------------------------------------

    def _method_iv(self, X, y, k):
        """Method 1: Information Value using Weight-of-Evidence (Section 2.2 #1)."""
        n_features = X.shape[1]
        iv_scores = np.zeros(n_features)

        n_events = max(np.sum(y == 1), 1)
        n_non_events = max(np.sum(y == 0), 1)

        for j in range(n_features):
            col = X[:, j]
            try:
                bins = pd.qcut(col, q=10, duplicates="drop", labels=False)
            except Exception:
                bins = pd.cut(col, bins=10, labels=False)

            iv = 0.0
            for b in np.unique(bins[~np.isnan(bins)]):
                mask_b = bins == b
                events_b = np.sum(y[mask_b] == 1)
                nonevt_b = np.sum(y[mask_b] == 0)

                pct_e = events_b / n_events
                pct_ne = nonevt_b / n_non_events

                if pct_e < 1e-9 or pct_ne < 1e-9:
                    continue

                woe = np.log(pct_e / pct_ne)
                iv += (pct_e - pct_ne) * woe

            iv_scores[j] = iv

        top_k = np.argsort(iv_scores)[::-1][:k]
        mask = np.zeros(n_features, dtype=bool)
        mask[top_k] = True
        return mask, "1_Information_Value_(WOE)"

    def _method_rfe(self, X, y, k):
        """Method 2: Recursive Feature Elimination (Section 2.2 #2)."""
        estimator = LogisticRegression(
            max_iter=1000, random_state=self.random_state, n_jobs=-1
        )
        rfe = RFE(estimator=estimator, n_features_to_select=k, step=1)
        rfe.fit(X, y)
        return rfe.support_, "2_RFE_(LogisticRegression)"

    def _method_random_forest(self, X, y, k):
        """Method 3: Variable importance using Random Forest (Section 2.2 #3)."""
        rf = RandomForestClassifier(
            n_estimators=100, random_state=self.random_state, n_jobs=-1
        )
        rf.fit(X, y)
        importances = rf.feature_importances_
        top_k = np.argsort(importances)[::-1][:k]
        mask = np.zeros(X.shape[1], dtype=bool)
        mask[top_k] = True
        return mask, "3_Random_Forest_Importance"

    def _method_extra_trees(self, X, y, k):
        """Method 4: Variable importance using Extra Trees (Section 2.2 #4)."""
        et = ExtraTreesClassifier(
            n_estimators=100, random_state=self.random_state, n_jobs=-1
        )
        et.fit(X, y)
        importances = et.feature_importances_
        top_k = np.argsort(importances)[::-1][:k]
        mask = np.zeros(X.shape[1], dtype=bool)
        mask[top_k] = True
        return mask, "4_Extra_Trees_Importance"

    def _method_chi2(self, X, y, k):
        """Method 5: Chi-Square best variables (Section 2.2 #5).

        Chi2 requires non-negative input -> features are scaled to [0,1] with
        MinMaxScaler (fit on this training fold only).
        """
        scaler_mm = MinMaxScaler()
        X_nonneg = scaler_mm.fit_transform(X)

        selector = SelectKBest(chi2, k=k)
        selector.fit(X_nonneg, y)
        return selector.get_support(), "5_Chi_Square"

    def _method_l1(self, X, y, k):
        """Method 6: L1-based feature selection (Section 2.2 #6)."""
        coef = np.zeros(X.shape[1])
        for C in [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]:
            lr = LogisticRegression(
                penalty="l1",
                solver="liblinear",
                C=C,
                max_iter=1000,
                random_state=self.random_state,
            )
            lr.fit(X, y)
            coef = np.abs(lr.coef_[0])
            n_nonzero = np.sum(coef > 0)
            if n_nonzero >= k:
                break

        top_k = np.argsort(coef)[::-1][:k]
        mask = np.zeros(X.shape[1], dtype=bool)
        mask[top_k] = True
        return mask, "6_L1_(Lasso)"
