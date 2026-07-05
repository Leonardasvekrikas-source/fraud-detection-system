# 2-minute walkthrough — recording script

A tight, recruiter-friendly video that shows the system working, then the engineering behind it.
Record with [Loom](https://www.loom.com/) (easiest, gives a shareable link) or OBS. Keep it **under
2 minutes** — talk to the *value*, not every detail. Then paste the link into the README badge slot.

**Setup before recording:** open two browser tabs — (1) the
[live demo](https://leonardasvekrikas-source-fraud-detection-demo.hf.space), (2) the
[GitHub repo](https://github.com/Leonardasvekrikas-source/fraud-detection-system). Have the
fraud transaction from `docs/sample_transactions.json` copied to your clipboard.

---

### 0:00–0:15 — Hook (demo tab)
> "This is a real-time, explainable credit-card fraud detector — deployed and live. Let me show
> you what it does, then how it's built."

Have the demo page already loaded.

### 0:15–0:50 — Score a fraud (the money shot)
Paste the fraud transaction → click **Score transaction**.
> "I paste a transaction, and it returns a fraud probability, a verdict — **Fraud** — and, crucially,
> **why**: a SHAP explanation showing V14, V12, V10 drove the decision. It's not a black box. In
> fraud, that explanation is what an analyst — and a regulator — needs."

Point at the SHAP bars as you say the feature names.

### 0:50–1:10 — Contrast with a normal transaction
Click **Load example** (all-zeros) or paste the normal example → Score.
> "A normal transaction comes back **Normal**, low probability, different drivers. The model
> discriminates, and it explains itself either way."

### 1:10–1:40 — The engineering (repo tab)
Scroll the README: the architecture diagram, the results table, the green CI badge.
> "Behind it: an independent reproduction of a 2025 research paper — reproduced bit-for-bit — turned
> into a tested, CI-checked package. The architecture is a decision-level fusion of LightGBM and an
> LSTM, chosen deliberately: it stays simple, low-latency, and explainable. I backed that choice with
> tracked experiments — including an honest negative result where the more complex fusion actually
> failed."

### 1:40–2:00 — MLOps + close
Scroll to the monitoring / airflow sections.
> "And it doesn't stop at deployment: there's drift monitoring with Evidently and a champion/challenger
> retraining pipeline in Airflow that promotes a new model only if it beats the live one. Reproducible,
> deployed, explainable, monitored. Code and live demo are linked below — thanks for watching."

---

**Tips**
- Rehearse once; record in one take (Loom lets you re-record instantly).
- Speak to *why it matters*, not the code line-by-line.
- Face-cam bubble on (Loom default) makes it personable — recruiters like seeing you.
- Trim dead air at the start/end.
