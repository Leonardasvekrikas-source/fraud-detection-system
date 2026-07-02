# serving/ — Phase 1: real-time explainable API

A FastAPI service that scores a transaction and returns the fraud probability, the
decision-level fusion verdict (Normal / Fraud / Expert-Checking), and a **SHAP explanation**
of the top contributing features — the explainability payoff of keeping the two models
separable (exact, fast `TreeExplainer` on the LightGBM component).

Code lives in the package: [`src/fraud_detection/serving/`](../src/fraud_detection/serving/)
(`app.py`, `explain.py`, `ui.py`). The container image is defined by the root
[`Dockerfile`](../Dockerfile); step-by-step deployment is in [`DEPLOY.md`](../DEPLOY.md).

## Run locally

```bash
pip install -e ".[serve]"
fraud-detect train --save artifacts/model          # or --no-lstm for a lean build
MODEL_DIR=artifacts/model uvicorn fraud_detection.serving.app:app --port 7860
# open http://localhost:7860  (paste-a-transaction demo UI)
```

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/` | demo UI |
| GET | `/health` | liveness + whether a model is loaded and in which mode |
| GET | `/schema` | expected raw feature columns, selected features, θ, window size |
| POST | `/score` | score one transaction → probability + verdict + SHAP explanation |

`POST /score` body — features as a name→value map **or** an ordered list:

```json
{ "features": { "Time": 0, "V1": -1.36, "V2": -0.07, "...": 0, "Amount": 149.62 }, "top_k": 8 }
```

## Deploy (Hugging Face Spaces, Docker SDK)

```bash
docker build -t fraud-demo .        # from the repo root
docker run -p 7860:7860 fraud-demo
```

The image installs the lean serving extra (no TensorFlow) and serves on port 7860. A trained
bundle must exist at `artifacts/model` at build time. If the bundle contains an LSTM but TF is
absent, the service degrades to LightGBM-only (P2 disabled) rather than failing — add
`tensorflow~=2.17` to the image to enable full fusion.

> **Honesty note.** With a LightGBM-only build the service returns a single-model verdict at
> the 0.5 threshold and labels the response `"mode": "lightgbm_only"`. Full Algorithm-1 fusion
> (`"mode": "fusion"`) requires the LSTM (P2).
