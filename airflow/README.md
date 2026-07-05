# airflow/ — Phase 3: retraining DAG (champion/challenger)

The drift monitor detects staleness; **this** is what acts on it. A local Airflow
(orchestrator) runs a DAG that retrains a model and promotes it **only if it beats the live one**:

```
ingest → train_candidate → evaluate_candidate → promote_if_better
                                                    │
             (candidate beats the champion on a holdout? swap it in : keep champion)
```

This is the honest answer to *"how do you retrain safely?"* — never blindly ship a new model.
The **decision logic** lives in the tested package
([`fraud_detection.pipelines.promotion`](../src/fraud_detection/pipelines/promotion.py), unit-tested
in [`tests/test_promotion.py`](../tests/test_promotion.py)); the DAG
([`dags/retrain_promote_dag.py`](dags/retrain_promote_dag.py)) only orchestrates.

> **Scope, honestly:** this is a **local docker-compose** you bring up to *run* the DAG, not a
> 24/7 server. The DAG trains **LightGBM-only** on a **40k subsample** with a **reduced config**
> ([`config/dag_config.yaml`](config/dag_config.yaml)) so a full run finishes in ~1–2 min — a
> documented demo speed-up, not the real training path (`fraud-detect train`).

## Run it

Requires Docker and the dataset at `../data/creditcard.csv`
(`python scripts/fetch_data.py --dataset european`, or copy it in).

```bash
cd airflow
docker compose build                 # build the Airflow image (installs fraud_detection)
docker compose up airflow-init       # one-time: init DB + create admin (airflow / airflow)
docker compose up -d                 # start webserver + scheduler
# open http://localhost:8080 → enable/trigger the 'fraud_retrain_promote' DAG → watch it run
docker compose down                  # stop  (add -v to also remove the volumes)
```

## What each task does

| Task | What it does |
|------|--------------|
| `ingest` | Load the data, split chronologically (80% train / 20% holdout), log the split. |
| `train_candidate` | Train a fresh **candidate** LightGBM bundle on the train slice; save it. |
| `evaluate_candidate` | Score the candidate on the **holdout**; persist its metrics. |
| `promote_if_better` | Score the current **champion** on the same holdout; **promote** the candidate only if its F1 is higher (else keep the champion). |

On the **first** run there's no champion, so the candidate is promoted and becomes the champion.
Re-running trains a new candidate and compares it against that champion — the core MLOps
retrain-and-promote loop.

## Services (minimal LocalExecutor)

`postgres` (metadata DB) · `airflow-init` (one-shot DB migrate + admin user) · `airflow-webserver`
(UI at :8080) · `airflow-scheduler`. No Celery/Redis/workers — unnecessary for a local demo.
