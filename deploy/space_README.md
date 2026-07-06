---
title: Explainable Fraud Detection
emoji: 🕵️
colorFrom: indigo
colorTo: red
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Real-time fraud scoring: LightGBM+LSTM fusion, SHAP+LIME
---

# Explainable Fraud Detection — live demo

Score a credit-card transaction in real time and see **why**. This FastAPI service runs the
full two-model system and returns a fraud probability from **each** subsystem, the fused verdict
(Normal / Fraud / **Expert-Checking**), and **two** explanations side by side — so the score is
never a black box.

**Try it:** paste a transaction (or click *Load example*) → per-model probabilities + verdict +
the features driving each model.

## Under the hood — decision-level fusion + dual XAI
- **LightGBM** (Subsystem 1) → P1, explained by an exact, fast **SHAP TreeExplainer**.
- **LSTM** (Subsystem 2) → P2 over a sliding window, explained by model-agnostic **LIME**
  (a local surrogate — weights are directional drivers, not exact attributions).
- **Algorithm-1 fusion:** `P_sum = P1 + P2` at threshold θ=0.5 gives a three-way verdict, with an
  **Expert-Checking** tier when the two models disagree — a built-in human-in-the-loop.
- Trained on the public [European Credit Card Fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud);
  reproduces the source method (Yousefimehr & Ghatee, 2025).

## Full project (code, experiments, write-up)
👉 **https://github.com/Leonardasvekrikas-source/fraud-detection-system**

Reproducible training/eval, an MLflow-tracked fusion vs. cost-vs-threshold vs. generalization
study, probability calibration, latency profiling, a drift-monitored retraining loop, tests, and CI.
