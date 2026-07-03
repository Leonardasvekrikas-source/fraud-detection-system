# experiments/ — Phase 2: the study, made legible

Reproducible, **MLflow-tracked** experiments that turn the thesis's arguments into numbers
anyone can regenerate. Tracking uses a local SQLite backend (MLflow 3 retired the file store):

```bash
pip install -e ".[lstm,experiments]"
python -m fraud_detection.experiments.fusion_comparison --data creditcard.csv
python -m fraud_detection.experiments.threshold_cost   --data creditcard.csv
mlflow ui --backend-store-uri sqlite:///mlflow.db        # browse runs at :5000
```

Result tables/plots are written to `experiments/results/` and committed as evidence.

## 1. Fusion comparison — feature-level vs decision-level ✅

`fusion_comparison.py` trains both base models on one aligned chronological 60/20/20 split,
extracts LightGBM leaf embeddings (PCA→50) and LSTM hidden states, trains the feature-level
MLP meta-classifier, and evaluates all four configs on the same test set.

| config | precision | recall | F1 | AUC |
|---|---|---|---|---|
| LightGBM (Subsystem 1) | 0.963 | 0.703 | 0.813 | 0.964 |
| LSTM (Subsystem 2) | 1.000 | 0.676 | 0.806 | 0.972 |
| **decision-level fusion** (Algorithm 1) | 1.000 | 0.608 | 0.756 | 0.972 |
| **feature-level fusion** (leaf ⊕ hidden → MLP) | 0.032 | 0.905 | **0.062** | 0.968 |

**Finding.** Feature-level fusion keeps competitive *ranking* (AUC 0.968) but its precision and
F1 collapse — its learned decision boundary is miscalibrated by the extreme class imbalance in
the meta-classifier's training data (the leaf embeddings, compressed 5000→50 by PCA, retain
only ~59% variance). Decision-level fusion stays clean with far less machinery. This reproduces
the thesis result (LSTM F1 and feature-level recall match to the digit) and is the empirical
backbone of the project's argument: **decision-level fusion is the better production choice.**

## 2. Threshold as a cost decision ✅

`threshold_cost.py` sweeps the LightGBM threshold and, for a range of cost ratios
`R = cost(false negative) / cost(false positive)`, finds the expected-cost-minimising operating
point — reframing the PR curve as a business decision (fraud loss vs false-alarm friction).
See `experiments/results/threshold_cost.{csv,png}`.

## 3. PaySim generalization — second-dataset evidence 🚧

Planned: run the pipeline on the structurally different PaySim dataset (config override for its
`isFraud` label + interpretable features) to test whether the architecture generalises.
