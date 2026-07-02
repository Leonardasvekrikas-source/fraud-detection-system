# =============================================================================
# models/lstm_model.py
#
# Subsystem 2 model  -  paper Section 2.4
#
# Single-layer LSTM for sequence-wise fraud detection.
#   Input -> LSTM(50 units) -> Dense(1, sigmoid)      (Table A.15)
#   Optimizer Adam, loss binary_crossentropy, 10 epochs, batch 512.
#   Input shape: (batch, window_size, n_features).
#
# NOTE: TensorFlow is imported lazily inside build() so that importing this
# module (and the rest of the package) does not require TensorFlow to be
# installed - useful for the serving/explainability paths and for unit tests
# that never touch the LSTM.
# =============================================================================

import os

import numpy as np

from fraud_detection import config

# Quieten TensorFlow's info/warning logs before it is imported.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


class LSTMModel:
    """Single-layer LSTM classifier for sequence-wise fraud detection."""

    def __init__(
        self,
        lstm_units=None,
        optimizer=None,
        loss=None,
        epochs=None,
        batch_size=None,
        random_state=None,
    ):
        self.lstm_units = lstm_units if lstm_units is not None else config.LSTM_UNITS
        self.optimizer = optimizer if optimizer is not None else config.LSTM_OPTIMIZER
        self.loss = loss if loss is not None else config.LSTM_LOSS
        self.epochs = epochs if epochs is not None else config.LSTM_EPOCHS
        self.batch_size = batch_size if batch_size is not None else config.LSTM_BATCH_SIZE
        self.random_state = (
            random_state if random_state is not None else config.LSTM_RANDOM_STATE
        )

        self._model = None
        self._fitted = False

    # -------------------------------------------------------------------------

    def build(self, input_shape: tuple) -> "LSTMModel":
        """Construct the Keras model. ``input_shape`` = (window_size, n_features)."""
        import tensorflow as tf
        from tensorflow.keras.layers import LSTM, Dense
        from tensorflow.keras.models import Sequential

        tf.random.set_seed(self.random_state)

        self._model = Sequential(name="LightGBM_LSTM_Subsystem2")
        self._model.add(
            LSTM(
                units=self.lstm_units,
                input_shape=input_shape,
                return_sequences=False,
                name="lstm_layer",
            )
        )
        self._model.add(Dense(units=1, activation="sigmoid", name="output"))

        self._model.compile(
            optimizer=self.optimizer,
            loss=self.loss,
            metrics=["precision", "recall"],
        )

        print(
            f"[LSTMModel] build: input_shape={input_shape}, LSTM units={self.lstm_units}"
        )
        self._model.summary(print_fn=lambda s: print(f"  {s}"))
        return self

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, verbose: int = 0) -> "LSTMModel":
        """Train the LSTM on windowed training data (n_samples, W, n_features)."""
        if self._model is None:
            self.build(input_shape=X_train.shape[1:])

        print(
            f"[LSTMModel] fit: {X_train.shape[0]} windows, "
            f"epochs={self.epochs}, batch={self.batch_size}, "
            f"fraud rate={np.mean(y_train == config.FRAUD_LABEL):.4f}"
        )
        self._model.fit(
            X_train,
            y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=verbose,
        )
        self._fitted = True
        print("[LSTMModel] training complete.")
        return self

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return hard binary predictions."""
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return fraud probability scores P(fraud | window)."""
        self._check_fitted()
        return self._model.predict(X, verbose=0).ravel()

    @property
    def keras_model(self):
        """The underlying fitted Keras model (for persistence / serving)."""
        self._check_fitted()
        return self._model

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("LSTMModel.fit() must be called before predict().")
