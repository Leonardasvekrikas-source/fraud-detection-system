# =============================================================================
# pipelines/promotion.py  -  champion/challenger model promotion logic
#
# The decision logic behind the Airflow retraining DAG, kept HERE (in the
# tested package) rather than in the DAG, so it is unit-tested and reusable:
#
#   - evaluate a bundle on a holdout set,
#   - decide whether a freshly-trained CANDIDATE beats the live CHAMPION,
#   - promote (atomically replace) the production model only if it is better.
#
# This is the standard, honest answer to "how do you retrain safely?": never
# blindly ship a new model - evaluate the challenger against the champion on the
# same holdout and promote only on a real improvement.
# =============================================================================

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from fraud_detection import config
from fraud_detection.artifacts.store import ModelBundle
from fraud_detection.evaluation.metrics import compute_metrics

SCALARS = ["accuracy", "precision", "recall", "f1", "auc", "mcc", "balanced_accuracy"]


def evaluate_on_holdout(bundle: ModelBundle, df_holdout: pd.DataFrame) -> dict:
    """Score a bundle's LightGBM head on a labelled holdout; return metrics."""
    features = bundle.metadata["raw_feature_columns"]
    X = df_holdout[features].values
    y = df_holdout[config.LABEL_COL].values
    p1 = bundle.predict_p1(X)
    m = compute_metrics(y, (p1 >= 0.5).astype(int), p1)
    return {k: float(m[k]) for k in SCALARS}


def is_better(candidate: dict, champion: dict | None, key: str = "f1") -> bool:
    """True if the candidate should be promoted (no champion yet, or it wins)."""
    if champion is None:
        return True
    return float(candidate.get(key, 0.0)) > float(champion.get(key, 0.0))


def load_metrics(path: str | Path) -> dict | None:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def save_metrics(metrics: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def promote(candidate_dir: str | Path, production_dir: str | Path) -> None:
    """Atomically replace the production model directory with the candidate."""
    src, dst = Path(candidate_dir), Path(production_dir)
    tmp = dst.with_name(dst.name + ".new")
    if tmp.exists():
        shutil.rmtree(tmp)
    shutil.copytree(src, tmp)
    if dst.exists():
        shutil.rmtree(dst)
    tmp.rename(dst)
