# =============================================================================
# airflow/dags/retrain_promote_dag.py
#
# Champion/challenger retraining DAG (Phase 3):
#
#   ingest -> train_candidate -> evaluate_candidate -> promote_if_better
#
# It trains a fresh CANDIDATE model, evaluates it on a holdout, then evaluates
# the current CHAMPION (production) model on the SAME holdout and promotes the
# candidate ONLY if it is genuinely better. This is the safe, honest retraining
# loop - never blindly ship a new model.
#
# The decision logic lives in the tested package (fraud_detection.pipelines.
# promotion); this DAG only orchestrates. Runs LightGBM-only with a reduced
# config (airflow/config/dag_config.yaml) so a full run finishes quickly - a
# documented demo speed-up, not the full training config.
#
# Paths are read from env (set in docker-compose):
#   FRAUD_DATA   -> the dataset CSV        (default /opt/airflow/data/creditcard.csv)
#   MODELS_DIR   -> model store            (default /opt/airflow/models)
# =============================================================================

from __future__ import annotations

import os
from datetime import datetime

import pandas as pd
from airflow.decorators import dag, task

DATA_FILE = os.environ.get("FRAUD_DATA", "/opt/airflow/data/creditcard.csv")
MODELS_DIR = os.environ.get("MODELS_DIR", "/opt/airflow/models")
CANDIDATE_DIR = f"{MODELS_DIR}/candidate"
PRODUCTION_DIR = f"{MODELS_DIR}/production"
CANDIDATE_METRICS = f"{MODELS_DIR}/candidate_metrics.json"
PRODUCTION_METRICS = f"{MODELS_DIR}/production_metrics.json"
TRAIN_FRACTION = 0.8  # chronological: first 80% train, last 20% holdout
# Demo speed-up: subsample the training slice (all fraud + random normals) so a
# full DAG run finishes in ~1-2 min. The holdout is left intact for honest eval.
TRAIN_SAMPLE = int(os.environ.get("TRAIN_SAMPLE", "40000"))


def _load_split():
    """Load the dataset; return (subsampled train, full holdout), time-ordered."""
    from fraud_detection import config
    from fraud_detection.data.loader import load_data

    df = load_data(DATA_FILE)
    cut = int(len(df) * TRAIN_FRACTION)
    train, holdout = df.iloc[:cut], df.iloc[cut:].reset_index(drop=True)

    if TRAIN_SAMPLE and len(train) > TRAIN_SAMPLE:
        fraud = train[train[config.LABEL_COL] == config.FRAUD_LABEL]
        normal = train[train[config.LABEL_COL] == config.NORMAL_LABEL].sample(
            n=max(TRAIN_SAMPLE - len(fraud), 0), random_state=config.RANDOM_STATE
        )
        train = pd.concat([fraud, normal]).sort_index()

    return train.reset_index(drop=True), holdout


@dag(
    dag_id="fraud_retrain_promote",
    description="Retrain a candidate model and promote it only if it beats the live champion.",
    schedule=None,  # trigger manually (or set a cron); this is a local demo, not a 24/7 server
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["fraud", "mlops", "retraining"],
)
def retrain_promote():
    @task
    def ingest() -> dict:
        """Validate the incoming data and report the split."""
        from fraud_detection import config

        train, holdout = _load_split()
        info = {
            "n_train": int(len(train)),
            "n_holdout": int(len(holdout)),
            "train_fraud": float(train[config.LABEL_COL].mean()),
            "holdout_fraud": float(holdout[config.LABEL_COL].mean()),
        }
        print(f"[ingest] {info}")
        return info

    @task
    def train_candidate() -> str:
        """Train a fresh candidate model on the train split; save it."""
        from fraud_detection.pipelines.train import train_bundle

        train, _ = _load_split()
        bundle = train_bundle(train, include_lstm=False)
        bundle.save(CANDIDATE_DIR)
        print(f"[train_candidate] saved candidate -> {CANDIDATE_DIR}")
        return CANDIDATE_DIR

    @task
    def evaluate_candidate(candidate_dir: str) -> dict:
        """Score the candidate on the holdout; persist its metrics."""
        from fraud_detection.artifacts.store import ModelBundle
        from fraud_detection.pipelines.promotion import evaluate_on_holdout, save_metrics

        _, holdout = _load_split()
        bundle = ModelBundle.load(candidate_dir, with_lstm=False)
        metrics = evaluate_on_holdout(bundle, holdout)
        save_metrics(metrics, CANDIDATE_METRICS)
        print(f"[evaluate_candidate] holdout metrics: {metrics}")
        return metrics

    @task
    def promote_if_better(candidate_metrics: dict) -> str:
        """Evaluate the champion on the same holdout; promote candidate if better."""
        from pathlib import Path

        from fraud_detection.artifacts.store import ModelBundle
        from fraud_detection.pipelines.promotion import (
            evaluate_on_holdout,
            is_better,
            promote,
            save_metrics,
        )

        champion_metrics = None
        if Path(PRODUCTION_DIR).exists():
            _, holdout = _load_split()
            champ = ModelBundle.load(PRODUCTION_DIR, with_lstm=False)
            champion_metrics = evaluate_on_holdout(champ, holdout)
            print(f"[promote] champion holdout F1={champion_metrics['f1']:.4f}")
        else:
            print("[promote] no champion yet — first model will be promoted")

        if is_better(candidate_metrics, champion_metrics, key="f1"):
            promote(CANDIDATE_DIR, PRODUCTION_DIR)
            save_metrics(candidate_metrics, PRODUCTION_METRICS)
            decision = (
                f"PROMOTED candidate (F1={candidate_metrics['f1']:.4f}) "
                + (f"over champion (F1={champion_metrics['f1']:.4f})" if champion_metrics else "as first model")
            )
        else:
            decision = (
                f"KEPT champion (F1={champion_metrics['f1']:.4f}) — "
                f"candidate (F1={candidate_metrics['f1']:.4f}) did not improve"
            )
        print(f"[promote] {decision}")
        return decision

    # --- task graph -------------------------------------------------------
    ingest()
    candidate_dir = train_candidate()
    metrics = evaluate_candidate(candidate_dir)
    promote_if_better(metrics)


retrain_promote()
