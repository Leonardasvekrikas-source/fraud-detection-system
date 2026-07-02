# Explainable fraud detection API — portable image (runs on Hugging Face Spaces,
# Render, Fly.io, or locally). HF Spaces serves on port 7860 by default.
#
# Build from the REPO ROOT (context = repo root):
#   docker build -f serving/Dockerfile -t fraud-demo .
#   docker run -p 7860:7860 fraud-demo
#
# A trained model must exist at artifacts/model before building:
#   fraud-detect train --save artifacts/model            (full fusion, needs TF)
#   fraud-detect train --save artifacts/model --no-lstm  (lean, LightGBM-only)
#
# This image installs the lean serving extra (no TensorFlow). If the bundle
# contains an LSTM it loads LightGBM-only (P2 disabled); to enable full fusion
# in the container, add `tensorflow~=2.17` to the pip install below.

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
