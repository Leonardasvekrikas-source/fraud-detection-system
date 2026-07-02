import numpy as np
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler

from fraud_detection.artifacts.store import ModelBundle
from fraud_detection.preprocessing.feature_selection import F2VoteSelector


def _fit_components(seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=300)
    informative = y + rng.normal(0, 0.1, size=300)
    noise = rng.normal(0, 1.0, size=(300, 5))
    X = np.column_stack([informative, noise])

    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    f2vote = F2VoteSelector().fit(X_scaled, y)
    X_f2 = f2vote.transform(X_scaled)
    clf = LGBMClassifier(n_estimators=10, verbose=-1).fit(X_f2, y)
    return X, scaler, f2vote, clf


def test_bundle_roundtrip_lightgbm_only(tmp_path):
    X, scaler, f2vote, clf = _fit_components()
    bundle = ModelBundle(scaler=scaler, f2vote=f2vote, lightgbm=clf, metadata={"theta": 0.5})

    before = bundle.predict_p1(X)
    bundle.save(tmp_path)

    loaded = ModelBundle.load(tmp_path, with_lstm=False)
    after = loaded.predict_p1(X)

    np.testing.assert_allclose(before, after)
    assert loaded.metadata["theta"] == 0.5
    assert loaded.lstm is None
