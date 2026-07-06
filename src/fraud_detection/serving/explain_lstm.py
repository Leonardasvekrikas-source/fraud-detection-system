# =============================================================================
# serving/explain_lstm.py  -  LIME explanation for the LSTM subsystem
#
# The LSTM is a black box, so (unlike LightGBM + SHAP) we explain it with LIME:
# a model-agnostic method that fits a local linear surrogate around one
# prediction. This mirrors the thesis's dual-XAI design (SHAP for the tree,
# LIME for the network).
#
# The catch LIME needs solving: LIME works on 2-D tabular data, but the LSTM
# consumes 3-D windows (n, W, features). The wrapper below places each perturbed
# 2-D sample in the LAST timestep of a zero-padded window and calls the LSTM -
# isolating the current transaction's contribution to P2 (the same streaming,
# single-transaction view the API scores).
#
# Honest caveat (from the thesis): LIME's local R^2 on this LSTM is low, so the
# weights are DIRECTIONAL indicators, not precise attributions.
#
# lime is imported lazily so importing this module doesn't require it.
# =============================================================================

from __future__ import annotations

import numpy as np


class LSTMLimeExplainer:
    """Explains the LSTM's fraud probability (P2) for one transaction via LIME."""

    def __init__(self, lstm_model, background: np.ndarray, feature_names: list[str], window_size: int):
        import lime.lime_tabular

        self.lstm = lstm_model
        self.window_size = int(window_size)
        self.feature_names = list(feature_names)
        self.n_features = len(feature_names)
        self._explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=np.asarray(background, dtype=float),
            feature_names=self.feature_names,
            class_names=["normal", "fraud"],
            mode="classification",
            discretize_continuous=True,
            random_state=42,
        )

    def _predict_proba(self, X_2d: np.ndarray) -> np.ndarray:
        """LIME-compatible: 2-D (n, features) -> zero-padded windows -> (n, 2)."""
        X_2d = np.asarray(X_2d, dtype=np.float32)
        n = X_2d.shape[0]
        X_win = np.zeros((n, self.window_size, self.n_features), dtype=np.float32)
        X_win[:, -1, :] = X_2d  # current transaction in the last timestep
        p_fraud = self.lstm.predict(X_win, verbose=0).ravel()
        return np.column_stack([1.0 - p_fraud, p_fraud])

    def explain_row(self, x_selected: np.ndarray, top_k: int = 8, num_samples: int = 800) -> dict:
        """Explain the LSTM's fraud prediction for one (scaled, selected) row."""
        x = np.asarray(x_selected, dtype=float).ravel()
        exp = self._explainer.explain_instance(
            x, self._predict_proba, num_features=top_k, num_samples=num_samples, labels=(1,)
        )
        contributions = [
            {"feature": cond, "weight": round(float(w), 6)} for cond, w in exp.as_list(label=1)
        ]
        return {
            "top_features": contributions,
            "local_r2": round(float(exp.score), 4),
            "note": (
                "LIME fits a local linear surrogate (weights are directional, not exact); "
                "the LSTM sees a zero-padded single-transaction window, so sequential context "
                "is limited."
            ),
        }
