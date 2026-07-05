---
title: Explainable Fraud Detection
emoji: 🕵️
colorFrom: indigo
colorTo: red
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Real-time credit-card fraud scoring with SHAP explanations
---

# Explainable Fraud Detection — live demo

Score a credit-card transaction in real time and see **why**. This FastAPI service returns a
fraud probability, a verdict (Normal / Fraud / **Expert-Checking**), and a **SHAP** breakdown of
the features driving the decision — so the score is never a black box.

**Try it:** paste a transaction (or click *Load example*) → probability + verdict + the top
contributing features.

## Under the hood
- **LightGBM** gradient-boosting on tabular transaction features, with an exact, fast **SHAP
  TreeExplainer** for per-request explanations.
- Part of a larger engineering study arguing that **decision-level fusion** (keeping LightGBM
  and an LSTM separable) is the better production choice than feature-level fusion — because it
  stays simpler, lower-latency, and explainable. This demo runs the lean LightGBM + SHAP path.
- Trained on the public [European Credit Card Fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud);
  reproduces the source method (Yousefimehr & Ghatee, 2025).

## Full project (code, experiments, write-up)
👉 **https://github.com/Leonardasvekrikas-source/fraud-detection-system**

Reproducible training/eval, an MLflow-tracked fusion vs. cost-vs-threshold vs. generalization
study, tests, and CI.
