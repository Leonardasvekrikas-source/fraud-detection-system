# Custom Airflow image with the fraud_detection package installed, so the DAG
# can `from fraud_detection...`. Build context = repo root (see docker-compose).
#
# LightGBM-only (no TensorFlow) to keep the image lean; libgomp1 is LightGBM's
# OpenMP runtime.
FROM apache/airflow:2.10.5-python3.12

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
USER airflow

# Install the package (without the heavy [lstm]/[serve]/[monitoring] extras).
COPY --chown=airflow:airflow pyproject.toml README.md /opt/fraud/
COPY --chown=airflow:airflow src /opt/fraud/src
COPY --chown=airflow:airflow config /opt/fraud/config
RUN pip install --no-cache-dir /opt/fraud
