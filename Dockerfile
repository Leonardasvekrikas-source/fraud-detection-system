# Explainable fraud detection API — portable image (Hugging Face Spaces, Render,
# Fly.io, or local). HF Docker Spaces expect this Dockerfile at the repo root and
# serve on port 7860.
#
# Build from the repo root:
#   docker build -t fraud-demo .
#   docker run -p 7860:7860 fraud-demo   # -> http://localhost:7860
#
# A trained model must exist at artifacts/model at build time:
#   fraud-detect train --save artifacts/model            (full fusion, needs TF)
#   fraud-detect train --save artifacts/model --no-lstm  (lean, LightGBM-only)
#
# Installs the lean serving extra (no TensorFlow). If the bundle contains an
# LSTM it loads LightGBM-only (P2 disabled); to enable full fusion in-container,
# add `tensorflow~=2.17` to the pip install below.

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_DIR=/app/artifacts/model \
    PORT=7860

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY config ./config
RUN pip install --upgrade pip && pip install ".[serve]"

# Trained model bundle (scaler + F2Vote + LightGBM [+ LSTM]).
COPY artifacts ./artifacts

EXPOSE 7860
CMD ["sh", "-c", "uvicorn fraud_detection.serving.app:app --host 0.0.0.0 --port ${PORT}"]
