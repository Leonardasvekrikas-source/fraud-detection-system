"""Smoke tests for the Phase 1 serving API (LightGBM-only fixture bundle).

These skip automatically if the serving extra (fastapi/shap/httpx) is not
installed, so the core test suite still runs in a minimal environment.
"""

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

pytest.importorskip("fastapi")
pytest.importorskip("shap")
from fastapi.testclient import TestClient  # noqa: E402
from lightgbm import LGBMClassifier  # noqa: E402

from fraud_detection.artifacts.store import ModelBundle  # noqa: E402
from fraud_detection.preprocessing.feature_selection import F2VoteSelector  # noqa: E402
from fraud_detection.serving.app import create_app  # noqa: E402

RAW_COLS = [f"f{i}" for i in range(6)]


def _fixture_bundle() -> ModelBundle:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=300)
    informative = y + rng.normal(0, 0.1, size=300)
    noise = rng.normal(0, 1.0, size=(300, 5))
    X = np.column_stack([informative, noise])

    scaler = StandardScaler().fit(X)
    f2 = F2VoteSelector().fit(scaler.transform(X), y)
    clf = LGBMClassifier(n_estimators=15, verbose=-1).fit(f2.transform(scaler.transform(X)), y)
    selected = [RAW_COLS[i] for i in f2.selected_indices_]

    meta = {
        "raw_feature_columns": RAW_COLS,
        "selected_features": selected,
        "theta": 0.5,
        "window_size": 3,
    }
    return ModelBundle(scaler=scaler, f2vote=f2, lightgbm=clf, metadata=meta)


def test_health_schema_and_score():
    client = TestClient(create_app(bundle=_fixture_bundle()))

    health = client.get("/health").json()
    assert health["model_loaded"] is True
    assert health["mode"] == "lightgbm_only"

    schema = client.get("/schema").json()
    assert schema["raw_feature_columns"] == RAW_COLS

    # dict payload
    resp = client.post("/score", json={"features": {c: 0.0 for c in RAW_COLS}, "top_k": 5})
    assert resp.status_code == 200
    body = resp.json()
    assert body["decision"] in ("Normal", "Fraud", "Expert-Checking")
    assert body["mode"] == "lightgbm_only"
    assert body["p2"] is None
    assert 0.0 <= body["p1"] <= 1.0
    assert 1 <= len(body["explanation"]["top_features"]) <= 5

    # ordered-list payload
    resp2 = client.post("/score", json={"features": [0.0] * len(RAW_COLS)})
    assert resp2.status_code == 200

    # home page renders
    assert client.get("/").status_code == 200


def test_missing_features_returns_422():
    client = TestClient(create_app(bundle=_fixture_bundle()))
    resp = client.post("/score", json={"features": {RAW_COLS[0]: 1.0}})
    assert resp.status_code == 422


def test_no_model_returns_503():
    client = TestClient(create_app(model_dir="does/not/exist"))
    assert client.get("/health").json()["model_loaded"] is False
    resp = client.post("/score", json={"features": [0.0] * len(RAW_COLS)})
    assert resp.status_code == 503
