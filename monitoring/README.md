# monitoring/ — Phase 3: drift & quality monitoring (Evidently)

A trained fraud model degrades **silently** as production data drifts away from what it was
trained on (new fraud tactics, changing spend). This module compares a **reference** distribution
(training-time) against a **current** batch and reports which features / the target have drifted,
using [Evidently](https://www.evidentlyai.com/)'s statistical tests (Wasserstein / Jensen-Shannon
distance for large samples, K-S / chi-square otherwise).

> The drift is **simulated and clearly labelled** — this is offline public data, not a live feed.

```bash
pip install -e ".[monitoring]"
python -m fraud_detection.monitoring.drift --scenario stable   # baseline: should NOT alarm
python -m fraud_detection.monitoring.drift --scenario drift    # should flag the shifted features
```

Outputs go to `monitoring/reports/`:
- `drift_report_<scenario>.html` — the full interactive Evidently report (gitignored; open locally).
- `drift_summary_<scenario>.json` — a compact, committed summary (the drift decision per column).

## Result — the monitor discriminates

The **drift** scenario perturbs the strongest fraud-signal features (V14, V10, V12, V4) and scales
`Amount` up, and inflates the fraud rate (a simulated fraud wave). The monitor:

| Scenario | Drifted columns | Top flagged (distance > threshold 0.1) |
|----------|-----------------|----------------------------------------|
| **stable** | **0 / 35** | *(none — no false alarm)* |
| **drift** | **11 / 35 (35%)** | V14 (2.06), V12 (1.50), V10 (1.36), V4 (1.00), Amount (0.56) … |

It flags **exactly the features that were shifted**, ranked by magnitude (plus a few correlated
PCA components) — and stays quiet on a stable batch. In a real deployment this signal is what
would **trigger retraining** — which is what the Airflow DAG in [`../airflow/`](../airflow/) does.
