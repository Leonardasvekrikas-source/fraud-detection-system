# =============================================================================
# serving/app.py  -  Phase 1 real-time explainable scoring API.
#
# A FastAPI service that loads a persisted ModelBundle and, for a transaction,
# returns: the fraud probability, the decision-level fusion verdict
# (Normal / Fraud / Expert-Checking), and a SHAP explanation of the top
# contributing features.
#
#   uvicorn fraud_detection.serving.app:app --port 7860
#
# The model directory is taken from $MODEL_DIR (default: artifacts/model). The
# app starts even if no model is present yet (endpoints return 503 until one
# is), so the container is deployable before training completes.
# =============================================================================

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from fraud_detection import config
from fraud_detection.artifacts.store import ModelBundle
from fraud_detection.fusion import engine
from fraud_detection.serving.explain import LightGBMExplainer
from fraud_detection.serving.ui import DEMO_HTML

DEFAULT_MODEL_DIR = "artifacts/model"
_ASSET_DIR = Path(__file__).parent


def _load_asset(name: str) -> dict:
    """Load a static demo asset (curated examples / feature metadata) shipped in
    the serving package. Missing or unreadable -> empty (the UI degrades)."""
    p = _ASSET_DIR / name
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}
    except Exception:  # noqa: BLE001 - a bad asset must never break serving
        return {}


class ScoreRequest(BaseModel):
    # Either a name->value mapping or an ordered list matching raw_feature_columns.
    features: dict[str, float] | list[float]
    top_k: int = 8


def _build_raw_row(features, raw_cols: list[str]) -> np.ndarray:
    """Turn the request payload into a (1, n_raw_features) array in column order."""
    if isinstance(features, dict):
        missing = [c for c in raw_cols if c not in features]
        if missing:
            raise HTTPException(
                status_code=422,
                detail=f"Missing features (need all {len(raw_cols)}): {missing[:12]}"
                + (" ..." if len(missing) > 12 else ""),
            )
        return np.array([[float(features[c]) for c in raw_cols]], dtype=float)

    # list form
    if len(features) != len(raw_cols):
        raise HTTPException(
            status_code=422,
            detail=f"Expected {len(raw_cols)} feature values, got {len(features)}.",
        )
    return np.array([features], dtype=float)


def create_app(bundle: ModelBundle | None = None, model_dir: str | None = None) -> FastAPI:
    """Build the FastAPI app. Inject a bundle for tests, or load from disk."""
    app = FastAPI(
        title="Explainable Fraud Detection",
        version="0.1.0",
        description="LightGBM-LSTM decision-level fusion with SHAP explanations.",
    )

    load_error: str | None = None
    if bundle is None:
        target = model_dir or os.environ.get("MODEL_DIR", DEFAULT_MODEL_DIR)
        try:
            bundle = ModelBundle.load(target, with_lstm=True)
            print(f"[serving] loaded model bundle from {target}")
        except Exception as exc:  # noqa: BLE001 - report any load failure to /health
            load_error = f"{type(exc).__name__}: {exc}"
            print(f"[serving] no model loaded ({load_error}); /score will return 503.")

    explainer = None
    lstm_explainer = None
    if bundle is not None:
        feature_names = bundle.metadata.get("selected_features") or [
            f"f{i}" for i in range(bundle.lightgbm.n_features_in_)
        ]
        explainer = LightGBMExplainer(bundle.lightgbm, feature_names)
        if bundle.lstm is not None and bundle.background is not None:
            from fraud_detection.serving.explain_lstm import LSTMLimeExplainer

            w = int(bundle.metadata.get("window_size", config.WINDOW_SIZE))
            lstm_explainer = LSTMLimeExplainer(bundle.lstm, bundle.background, feature_names, w)

    app.state.bundle = bundle
    app.state.explainer = explainer
    app.state.lstm_explainer = lstm_explainer
    app.state.feature_meta = _load_asset("feature_meta.json")
    app.state.demo_examples = _load_asset("demo_examples.json")
    app.state.load_error = load_error
    app.state.theta = (
        float(bundle.metadata.get("theta", config.INFERENCE_THETA))
        if bundle is not None
        else config.INFERENCE_THETA
    )

    def _require_model() -> ModelBundle:
        if app.state.bundle is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "No model loaded. Train and persist one with "
                    "`fraud-detect train --save artifacts/model`, then set "
                    f"$MODEL_DIR. (load error: {app.state.load_error})"
                ),
            )
        return app.state.bundle

    @app.get("/health")
    def health() -> dict:
        b = app.state.bundle
        return {
            "status": "ok",
            "model_loaded": b is not None,
            "has_lstm": (b is not None and b.lstm is not None),
            "mode": "fusion" if (b is not None and b.lstm is not None) else "lightgbm_only",
            "load_error": app.state.load_error,
        }

    @app.get("/schema")
    def schema() -> dict:
        b = _require_model()
        return {
            "raw_feature_columns": b.metadata.get("raw_feature_columns", []),
            "selected_features": b.metadata.get("selected_features", []),
            "theta": app.state.theta,
            "window_size": b.metadata.get("window_size", config.WINDOW_SIZE),
            "feature_meta": app.state.feature_meta,
        }

    @app.get("/examples")
    def examples() -> dict:
        """Curated real transactions (10 each Normal / Fraud / Expert-Checking),
        each verified to produce that verdict, for the demo's quick-try buttons."""
        return app.state.demo_examples

    @app.post("/score")
    def score(req: ScoreRequest) -> dict:
        b = _require_model()
        raw_cols = b.metadata.get("raw_feature_columns", [])
        if not raw_cols:
            raise HTTPException(500, "Model metadata has no raw_feature_columns.")

        X_raw = _build_raw_row(req.features, raw_cols)

        p1 = float(b.predict_p1(X_raw)[0])
        x_selected = b.transform_features(X_raw)[0]
        explanation_lightgbm = app.state.explainer.explain_row(x_selected, top_k=req.top_k)

        theta = app.state.theta
        explanation_lstm = None
        if b.lstm is not None:
            p2 = float(b.predict_p2(X_raw)[0])
            decision = engine.classify_single(p1, p2, theta=theta)
            p_sum = p1 + p2
            mode = "fusion"
            if app.state.lstm_explainer is not None:
                explanation_lstm = app.state.lstm_explainer.explain_row(x_selected, top_k=req.top_k)
        else:
            # LightGBM-only build: full Algorithm 1 needs P2. Report a single-model
            # verdict at the standard 0.5 threshold and label the mode honestly.
            p2 = None
            decision = engine.LABEL_FRAUD if p1 >= 0.5 else engine.LABEL_NORMAL
            p_sum = p1
            mode = "lightgbm_only"

        return {
            "decision": decision,
            "mode": mode,
            "p1": round(p1, 6),
            "p2": (round(p2, 6) if p2 is not None else None),
            "p_sum": round(p_sum, 6),
            "theta": theta,
            "explanation_lightgbm": explanation_lightgbm,  # SHAP (P1)
            "explanation_lstm": explanation_lstm,  # LIME (P2), when fusion
        }

    @app.get("/", response_class=HTMLResponse)
    def home() -> str:
        return DEMO_HTML

    return app


# Module-level app for `uvicorn fraud_detection.serving.app:app`.
app = create_app()
