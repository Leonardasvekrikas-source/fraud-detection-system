# =============================================================================
# serving/explain.py
#
# SHAP explanation for the LightGBM subsystem.
#
# This is the explainability payoff of decision-level fusion: because the two
# models stay separable, we can run a fast, EXACT SHAP TreeExplainer on the
# gradient-boosted component (polynomial time, no sampling) and attribute the
# score to individual features - something a merged feature-level model could
# not offer as cleanly.
#
# SHAP is imported lazily so that importing this module does not pull in SHAP
# (and its numba/llvmlite stack) unless an explanation is actually requested.
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FeatureContribution:
    feature: str
    value: float  # the (scaled, F2Vote-selected) feature value fed to the model
    shap_value: float  # signed contribution toward the fraud margin

    def as_dict(self) -> dict:
        return {
            "feature": self.feature,
            "value": round(float(self.value), 6),
            "shap_value": round(float(self.shap_value), 6),
        }


class LightGBMExplainer:
    """Wraps a SHAP TreeExplainer over a fitted LightGBM classifier."""

    def __init__(self, booster, feature_names: list[str]):
        import shap  # lazy

        self._explainer = shap.TreeExplainer(booster)
        self.feature_names = list(feature_names)

    def _positive_class(self, arr):
        """Pick the positive-class slice from SHAP output shapes across versions."""
        # shap may return: (n, f) array, list[per-class (n, f)], or (n, f, n_classes)
        if isinstance(arr, list):
            return np.asarray(arr[-1])  # last = positive class
        arr = np.asarray(arr)
        if arr.ndim == 3:
            return arr[:, :, -1]
        return arr

    def explain_row(self, x_selected: np.ndarray, top_k: int = 8) -> dict:
        """Explain a single transaction (already scaled + F2Vote-selected).

        Returns a dict with the base value and the top-k feature contributions
        ranked by absolute SHAP value.
        """
        x = np.asarray(x_selected, dtype=float).reshape(1, -1)

        shap_values = self._positive_class(self._explainer.shap_values(x))[0]

        base = self._explainer.expected_value
        if isinstance(base, (list, np.ndarray)):
            base = np.asarray(base).ravel()
            base = float(base[-1])
        else:
            base = float(base)

        order = np.argsort(np.abs(shap_values))[::-1][:top_k]
        contributions = [
            FeatureContribution(
                feature=self.feature_names[i],
                value=x[0, i],
                shap_value=shap_values[i],
            ).as_dict()
            for i in order
        ]

        return {
            "base_value": round(base, 6),
            "top_features": contributions,
            "note": (
                "SHAP values are contributions to the LightGBM fraud margin "
                "(log-odds), computed exactly via TreeExplainer."
            ),
        }
