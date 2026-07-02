import numpy as np

from fraud_detection.preprocessing.feature_selection import F2VoteSelector


def _make_data(n=300, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    # feature 0 is strongly predictive (label + small noise); the rest are noise
    informative = y + rng.normal(0, 0.1, size=n)
    noise = rng.normal(0, 1.0, size=(n, 5))
    X = np.column_stack([informative, noise])
    return X, y


def test_selects_informative_feature_and_shapes():
    X, y = _make_data()
    sel = F2VoteSelector()
    sel.fit(X, y)

    # mask covers all features; votes are within [0, 6]
    assert sel.selected_mask_.shape == (6,)
    assert sel.vote_counts_.min() >= 0
    assert sel.vote_counts_.max() <= 6

    # the strongly-informative feature (index 0) should be retained
    assert 0 in sel.selected_indices_

    # transform reduces to exactly the selected columns
    X_reduced = sel.transform(X)
    assert X_reduced.shape[1] == len(sel.selected_indices_)


def test_min_votes_threshold_effect():
    X, y = _make_data()
    strict = F2VoteSelector(min_votes=6).fit(X, y)
    loose = F2VoteSelector(min_votes=1).fit(X, y)
    # a stricter vote threshold never selects more features than a loose one
    assert len(strict.selected_indices_) <= len(loose.selected_indices_)
