# =============================================================================
# models/lightgbm_model.py
#
# Subsystem 1 model  -  paper Section 2.4
#
# LightGBM: a gradient-boosting framework that grows trees leaf-wise. Chosen for
# efficient handling of large, high-dimensional tabular data and strong tabular
# fraud-detection performance. Hyperparameters: Table A.14 of the paper.
# =============================================================================

import lightgbm as lgb
import numpy as np

from fraud_detection import config


class LightGBMModel:
    """Wrapper around LGBMClassifier with the hyperparameters from Table A.14.

    Parameters
    ----------
    params : dict or None
        LightGBM parameters. None -> use config.LGBM_PARAMS.
    """

    def __init__(self, params: dict = None):
        self.params = params if params is not None else dict(config.LGBM_PARAMS)
        self._model = lgb.LGBMClassifier(**self.params)
        self._fitted = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "LightGBMModel":
        """Train LightGBM on the (resampled) training fold."""
        print(
            f"[LightGBMModel] fit: {X_train.shape[0]} samples, "
            f"{X_train.shape[1]} features, "
            f"fraud rate = {np.mean(y_train == config.FRAUD_LABEL):.4f}"
        )
        self._model.fit(X_train, y_train)
        self._fitted = True
        print("[LightGBMModel] training complete.")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return hard binary predictions (0 or 1)."""
        self._check_fitted()
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return fraud probability scores P(class=fraud | x)."""
        self._check_fitted()
        proba = self._model.predict_proba(X)
        fraud_col = list(self._model.classes_).index(config.FRAUD_LABEL)
        return proba[:, fraud_col]

    @property
    def booster(self):
        """The underlying fitted LGBMClassifier (for SHAP TreeExplainer / persistence)."""
        self._check_fitted()
        return self._model

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("LightGBMModel.fit() must be called before predict().")
