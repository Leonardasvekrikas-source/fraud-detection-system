"""Tests for the champion/challenger promotion logic (no Airflow needed)."""

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler

from fraud_detection.artifacts.store import ModelBundle
from fraud_detection.pipelines.promotion import (
    evaluate_on_holdout,
    is_better,
    load_metrics,
    promote,
    save_metrics,
)
from fraud_detection.preprocessing.feature_selection import F2VoteSelector

RAW = [f"f{i}" for i in range(6)]


def test_is_better_rules():
    assert is_better({"f1": 0.5}, None) is True  # no champion -> promote first model
    assert is_better({"f1": 0.9}, {"f1": 0.8}) is True  # strictly better
    assert is_better({"f1": 0.7}, {"f1": 0.8}) is False  # worse -> keep champion
    assert is_better({"f1": 0.8}, {"f1": 0.8}) is False  # ties do not promote


def test_metrics_roundtrip(tmp_path):
    p = tmp_path / "m.json"
    save_metrics({"f1": 0.42}, p)
    assert load_metrics(p) == {"f1": 0.42}
    assert load_metrics(tmp_path / "missing.json") is None


def _tiny_bundle():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=300)
    X = np.column_stack([y + rng.normal(0, 0.1, 300), rng.normal(0, 1, (300, 5))])
    scaler = StandardScaler().fit(X)
    f2 = F2VoteSelector().fit(scaler.transform(X), y)
    clf = LGBMClassifier(n_estimators=10, verbose=-1).fit(f2.transform(scaler.transform(X)), y)
    meta = {"raw_feature_columns": RAW, "selected_features": [RAW[i] for i in f2.selected_indices_]}
    df = pd.DataFrame(X, columns=RAW)
    df["Class"] = y
    return ModelBundle(scaler=scaler, f2vote=f2, lightgbm=clf, metadata=meta), df


def test_evaluate_on_holdout_returns_all_metrics():
    bundle, df = _tiny_bundle()
    metrics = evaluate_on_holdout(bundle, df)
    assert set(metrics) == {"accuracy", "precision", "recall", "f1", "auc", "mcc", "balanced_accuracy"}
    assert all(isinstance(v, float) for v in metrics.values())


def test_promote_replaces_production_dir(tmp_path):
    bundle, _ = _tiny_bundle()
    cand = tmp_path / "candidate"
    prod = tmp_path / "production"
    bundle.save(cand)
    promote(cand, prod)
    assert (prod / "lightgbm.joblib").is_file()
    # promoting again over an existing production dir works (atomic replace)
    promote(cand, prod)
    assert (prod / "metadata.json").is_file()
