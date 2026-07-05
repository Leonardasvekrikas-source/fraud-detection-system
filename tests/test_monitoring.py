"""Tests for the drift-monitoring logic that does not need Evidently
(scenario construction + parsing Evidently's report dict)."""

import json

import numpy as np
import pandas as pd

from fraud_detection.monitoring.drift import _summarise, make_scenario


def _toy_df(n=400, seed=0):
    rng = np.random.default_rng(seed)
    cols = {f"V{i}": rng.normal(0, 1, n) for i in range(1, 15)}
    cols["Amount"] = rng.exponential(50, n)
    cols["Class"] = (rng.random(n) < 0.1).astype(int)
    return pd.DataFrame(cols)


def test_stable_scenario_leaves_data_unchanged():
    df = _toy_df()
    ref, cur, changes = make_scenario(df, "stable")
    assert changes == {}
    assert len(ref) == len(cur)  # both clean samples of equal size


def test_drift_scenario_shifts_features_and_inflates_fraud():
    df = _toy_df()
    ref, cur, changes = make_scenario(df, "drift")
    # documented perturbations are recorded
    assert "V14" in changes and "Amount" in changes and "Class" in changes
    # fraud upsampling makes current larger and raises fraud prevalence
    assert len(cur) > len(ref)
    assert cur["Class"].mean() > ref["Class"].mean()


def test_summarise_applies_correct_drift_direction():
    report_dict = {
        "metrics": [
            {"metric_name": "DriftedColumnsCount(drift_share=0.5)", "value": {"count": 2.0, "share": 0.5}},
            # distance methods: drifted when score > threshold
            {"metric_name": "ValueDrift(column=V14,method=Wasserstein distance (normed),threshold=0.1)", "value": 2.0},
            {"metric_name": "ValueDrift(column=V1,method=Wasserstein distance (normed),threshold=0.1)", "value": 0.05},
            # p-value methods: drifted when score < threshold
            {"metric_name": "ValueDrift(column=Xp,method=K-S p_value,threshold=0.05)", "value": 0.01},
            {"metric_name": "ValueDrift(column=Yp,method=K-S p_value,threshold=0.05)", "value": 0.5},
        ]
    }
    summary = _summarise(report_dict, "drift")
    drifted = {c["column"]: c["drifted"] for c in summary["columns"]}
    assert drifted == {"V14": True, "V1": False, "Xp": True, "Yp": False}
    assert summary["drifted_columns_count"] == 2.0
    # summary must be JSON-serialisable (no numpy scalars)
    json.dumps(summary)
