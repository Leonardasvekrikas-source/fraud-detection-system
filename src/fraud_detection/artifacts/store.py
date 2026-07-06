# =============================================================================
# artifacts/store.py
#
# Persist and load the trained, *servable* decision-level fusion model.
#
# A production model is the winning decision-level config: the two subsystems
# stay separable, so the bundle stores each component independently:
#   - scaler.joblib     the StandardScaler fitted on training data
#   - f2vote.joblib     the fitted F2VoteSelector (feature mask)
#   - lightgbm.joblib   the fitted LGBMClassifier (Subsystem 1)  -> P1
#   - lstm.keras        the fitted Keras LSTM     (Subsystem 2)  -> P2  (optional)
#   - metadata.json     config snapshot, feature names, theta, W, versions
#
# Keeping the models separate is exactly what makes the score attributable to
# each subsystem and lets a fast SHAP TreeExplainer explain the LightGBM part.
# =============================================================================

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from fraud_detection import config
from fraud_detection.fusion import engine
from fraud_detection.preprocessing.windowing import create_windows


class ModelBundle:
    """A saveable/loadable bundle of the fitted serving components."""

    SCALER_FILE = "scaler.joblib"
    F2VOTE_FILE = "f2vote.joblib"
    LGBM_FILE = "lightgbm.joblib"
    LSTM_FILE = "lstm.keras"
    BACKGROUND_FILE = "background.npy"
    META_FILE = "metadata.json"

    def __init__(
        self, scaler, f2vote, lightgbm, lstm=None, background=None,
        metadata: dict[str, Any] | None = None,
    ):
        self.scaler = scaler
        self.f2vote = f2vote
        self.lightgbm = lightgbm  # a fitted LGBMClassifier
        self.lstm = lstm  # a fitted Keras model, or None
        # a sample of scaled + F2Vote-selected training features, for LIME's
        # perturbation statistics (only needed for the LSTM/LIME path)
        self.background = background
        self.metadata = metadata or {}

    # -- persistence ----------------------------------------------------------

    def save(self, directory: str | Path) -> Path:
        """Write all components to ``directory`` (created if needed)."""
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.scaler, d / self.SCALER_FILE)
        joblib.dump(self.f2vote, d / self.F2VOTE_FILE)
        joblib.dump(self.lightgbm, d / self.LGBM_FILE)
        if self.lstm is not None:
            self.lstm.save(d / self.LSTM_FILE)
        if self.background is not None:
            np.save(d / self.BACKGROUND_FILE, np.asarray(self.background, dtype=np.float32))

        meta = {**self.metadata, "has_lstm": self.lstm is not None}
        (d / self.META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")

        print(f"[ModelBundle] saved to {d.resolve()}")
        return d

    @classmethod
    def load(cls, directory: str | Path, with_lstm: bool = True) -> ModelBundle:
        """Load a bundle. ``with_lstm=False`` skips the (heavy) Keras model."""
        d = Path(directory)
        scaler = joblib.load(d / cls.SCALER_FILE)
        f2vote = joblib.load(d / cls.F2VOTE_FILE)
        lightgbm = joblib.load(d / cls.LGBM_FILE)

        metadata = {}
        meta_path = d / cls.META_FILE
        if meta_path.is_file():
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))

        bg_path = d / cls.BACKGROUND_FILE
        background = np.load(bg_path) if bg_path.is_file() else None

        lstm = None
        lstm_path = d / cls.LSTM_FILE
        if with_lstm and lstm_path.exists() and metadata.get("has_lstm", True):
            try:
                import tensorflow as tf  # lazy: serving LightGBM-only needs no TF

                lstm = tf.keras.models.load_model(lstm_path)
            except ImportError:
                # TensorFlow not installed (e.g. a lean serving image). Degrade
                # gracefully to a LightGBM-only bundle instead of failing.
                print(
                    "[ModelBundle] TensorFlow not available; loading LightGBM-only "
                    "(P2 / full fusion disabled)."
                )

        return cls(scaler, f2vote, lightgbm, lstm=lstm, background=background, metadata=metadata)

    # -- inference ------------------------------------------------------------

    def transform_features(self, X_raw: np.ndarray) -> np.ndarray:
        """Scale then apply the F2Vote mask, exactly as during training."""
        X_scaled = self.scaler.transform(np.asarray(X_raw, dtype=float))
        return self.f2vote.transform(X_scaled)

    def predict_p1(self, X_raw: np.ndarray) -> np.ndarray:
        """Subsystem 1 fraud probability P1 (LightGBM)."""
        X_f2 = self.transform_features(X_raw)
        proba = self.lightgbm.predict_proba(X_f2)
        fraud_col = list(self.lightgbm.classes_).index(config.FRAUD_LABEL)
        return proba[:, fraud_col]

    def predict_p2(self, X_raw: np.ndarray, window_size: int | None = None) -> np.ndarray:
        """Subsystem 2 fraud probability P2 (LSTM over sliding windows).

        Input rows are treated as a time-ordered sequence; the first W-1 rows
        get left-zero-padded windows (as in training / streaming inference).
        """
        if self.lstm is None:
            raise RuntimeError("This bundle has no LSTM; P2 is unavailable.")
        W = window_size or int(self.metadata.get("window_size", config.WINDOW_SIZE))
        X_f2 = self.transform_features(X_raw)
        dummy_y = np.zeros(len(X_f2))
        X_win, _ = create_windows(X_f2, dummy_y, W)
        return self.lstm.predict(X_win, verbose=0).ravel()

    def classify(self, X_raw: np.ndarray, theta: float | None = None) -> np.ndarray:
        """Full decision-level fusion: returns Normal / Fraud / Expert-Checking."""
        if theta is None:
            theta = float(self.metadata.get("theta", config.INFERENCE_THETA))
        p1 = self.predict_p1(X_raw)
        p2 = self.predict_p2(X_raw)
        return engine.classify_batch(p1, p2, theta=theta, verbose=False)
