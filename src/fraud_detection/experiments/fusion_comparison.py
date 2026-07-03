# =============================================================================
# experiments/fusion_comparison.py  -  Phase 2, experiment #1
#
# Empirically compares FOUR configurations on a single aligned split so the
# central claim of the project - decision-level fusion is the better production
# choice than feature-level fusion - rests on reproducible numbers, not just the
# thesis write-up:
#
#   1. LightGBM alone                (Subsystem 1)
#   2. LSTM alone                    (Subsystem 2)
#   3. Decision-level fusion         (Algorithm 1: P1 + P2 vs theta)
#   4. Feature-level fusion          (LightGBM leaf embeddings [PCA-50] ++
#                                     LSTM hidden state [50] -> MLP meta-classifier)
#
# Design (following the thesis fusion experiment):
#   - one chronological 60/20/20 split (train / val / test); order preserved
#   - base models trained on TRAIN (HybridOS for LightGBM, HybridUS for LSTM)
#   - internal representations extracted on VAL; PCA + the MLP meta-classifier
#     are fit on VAL; everything is evaluated on TEST
#
# All four configs are logged to MLflow (local ./mlruns) when mlflow is
# installed; results are always printed and written to a CSV.
#
# Run:  python -m fraud_detection.experiments.fusion_comparison --data creditcard.csv
# =============================================================================

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from fraud_detection import config
from fraud_detection.data.loader import load_data
from fraud_detection.evaluation.metrics import compute_metrics
from fraud_detection.fusion import engine
from fraud_detection.models.lightgbm_model import LightGBMModel
from fraud_detection.models.lstm_model import LSTMModel
from fraud_detection.preprocessing.base_preprocessor import BasePreprocessor
from fraud_detection.preprocessing.feature_selection import F2VoteSelector
from fraud_detection.preprocessing.hybrid_os import HybridOS
from fraud_detection.preprocessing.hybrid_us import HybridUS
from fraud_detection.preprocessing.windowing import create_windows

SCALAR_METRICS = ["accuracy", "precision", "recall", "f1", "auc", "mcc", "balanced_accuracy"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def chronological_split(X, y, train_frac=0.6, val_frac=0.2):
    """Split time-ordered arrays into (train, val, test) without shuffling."""
    n = len(X)
    n_train = int(n * train_frac)
    n_val = int(n * (train_frac + val_frac))
    return (
        (X[:n_train], y[:n_train]),
        (X[n_train:n_val], y[n_train:n_val]),
        (X[n_val:], y[n_val:]),
    )


def _scalars(metrics: dict) -> dict:
    return {k: float(metrics[k]) for k in SCALAR_METRICS}


def _lstm_hidden_extractor(lstm: LSTMModel):
    """A Keras sub-model that outputs the LSTM layer's hidden state (50-dim)."""
    import tensorflow as tf

    model = lstm.keras_model
    return tf.keras.Model(inputs=model.inputs, outputs=model.get_layer("lstm_layer").output)


# ---------------------------------------------------------------------------
# main experiment
# ---------------------------------------------------------------------------

def run(data_file: str = "creditcard.csv", out_dir: str = "experiments/results") -> list[dict]:
    from sklearn.decomposition import PCA

    df = load_data(data_file)
    df = BasePreprocessor.remove_duplicates(df)

    feature_cols = [c for c in df.columns if c != config.LABEL_COL]
    X_all = df[feature_cols].values
    y_all = df[config.LABEL_COL].values

    (X_tr, y_tr), (X_val, y_val), (X_te, y_te) = chronological_split(X_all, y_all)
    print(
        f"[fusion] split — train {len(y_tr)} ({y_tr.sum()} fraud), "
        f"val {len(y_val)} ({y_val.sum()} fraud), test {len(y_te)} ({y_te.sum()} fraud)"
    )

    # -- shared preprocessing: fit on TRAIN only --------------------------------
    pre = BasePreprocessor()
    Xtr_s = pre.fit_transform(X_tr)
    Xval_s = pre.transform(X_val)
    Xte_s = pre.transform(X_te)

    f2 = F2VoteSelector().fit(Xtr_s, y_tr)
    Xtr_f = f2.transform(Xtr_s)
    Xval_f = f2.transform(Xval_s)
    Xte_f = f2.transform(Xte_s)

    # -- Subsystem 1: LightGBM (P1 + leaf embeddings) ---------------------------
    print("\n[fusion] training LightGBM (Subsystem 1) ...")
    X_os, y_os = HybridOS().fit_resample(Xtr_f, y_tr)
    lgbm = LightGBMModel().fit(X_os, y_os)
    p1_te = lgbm.predict_proba(Xte_f)

    leaves_val = lgbm.booster.predict(Xval_f, pred_leaf=True).astype(np.float32)
    leaves_te = lgbm.booster.predict(Xte_f, pred_leaf=True).astype(np.float32)
    n_comp = min(50, leaves_val.shape[1], leaves_val.shape[0])  # 50 for the full 5000-tree model
    pca = PCA(n_components=n_comp, random_state=config.RANDOM_STATE).fit(leaves_val)
    leaf_val = pca.transform(leaves_val)
    leaf_te = pca.transform(leaves_te)
    print(
        f"[fusion] PCA(leaf {leaves_val.shape[1]}->{n_comp}) "
        f"retained variance: {pca.explained_variance_ratio_.sum():.4f}"
    )

    # -- Subsystem 2: LSTM (P2 + hidden states) ---------------------------------
    print("\n[fusion] training LSTM (Subsystem 2) ...")
    X_us, y_us = HybridUS().fit_resample(Xtr_f, y_tr)
    W = config.WINDOW_SIZE
    Xtr_win, ytr_win = create_windows(X_us, y_us, W)
    Xval_win, _ = create_windows(Xval_f, y_val, W)
    Xte_win, _ = create_windows(Xte_f, y_te, W)

    lstm = LSTMModel()
    lstm.build(input_shape=(W, Xtr_win.shape[2]))
    lstm.fit(Xtr_win, ytr_win, verbose=0)
    p2_te = lstm.predict_proba(Xte_win)

    hidden = _lstm_hidden_extractor(lstm)
    h_val = hidden.predict(Xval_win, verbose=0)
    h_te = hidden.predict(Xte_win, verbose=0)

    # -- Feature-level fusion: concat -> MLP meta-classifier --------------------
    print("\n[fusion] training feature-level meta-classifier (MLP) ...")
    z_val = np.hstack([leaf_val, h_val])  # (n_val, 100)
    z_te = np.hstack([leaf_te, h_te])
    p_feat_te = _train_meta_classifier(z_val, y_val, z_te)

    # -- Evaluate all four configs on TEST --------------------------------------
    results = []

    m_lgbm = compute_metrics(y_te, (p1_te >= 0.5).astype(int), p1_te)
    results.append({"config": "lightgbm", **_scalars(m_lgbm)})

    m_lstm = compute_metrics(y_te, (p2_te >= 0.5).astype(int), p2_te)
    results.append({"config": "lstm", **_scalars(m_lstm)})

    # Decision-level fusion (Algorithm 1): Expert-Checking counts as "not
    # auto-confirmed fraud" -> negative in the binary confusion; AUC uses the
    # mean probability as the ranking score.
    labels = engine.classify_batch(p1_te, p2_te, verbose=False)
    y_dec = (labels == engine.LABEL_FRAUD).astype(int)
    m_dec = compute_metrics(y_te, y_dec, (p1_te + p2_te) / 2.0)
    results.append({"config": "decision_level_fusion", **_scalars(m_dec)})

    m_feat = compute_metrics(y_te, (p_feat_te >= 0.5).astype(int), p_feat_te)
    results.append({"config": "feature_level_fusion", **_scalars(m_feat)})

    _print_table(results)
    _save_csv(results, out_dir)
    _log_mlflow(results, extra={"pca_variance": float(pca.explained_variance_ratio_.sum())})
    return results


def _train_meta_classifier(z_train, y_train, z_test):
    """MLP on concatenated embeddings; returns fraud probabilities on z_test."""
    import tensorflow as tf
    from tensorflow.keras.layers import Dense, Dropout, Input
    from tensorflow.keras.models import Model

    tf.random.set_seed(config.RANDOM_STATE)
    inp = Input(shape=(z_train.shape[1],))
    x = Dense(64, activation="relu")(inp)
    x = Dropout(0.3)(x)
    x = Dense(32, activation="relu")(x)
    out = Dense(1, activation="sigmoid")(x)
    mlp = Model(inp, out)
    mlp.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="binary_crossentropy")

    n = len(y_train)
    n_fraud = max(int(y_train.sum()), 1)
    class_weight = {0: 0.5, 1: n / (2.0 * n_fraud)}
    print(f"[fusion] meta-classifier class_weight[fraud]={class_weight[1]:.1f} "
          f"(val fraud={n_fraud}/{n})")

    mlp.fit(z_train, y_train, epochs=50, batch_size=256, class_weight=class_weight, verbose=0)
    return mlp.predict(z_test, verbose=0).ravel()


def _print_table(results: list[dict]) -> None:
    print("\n" + "=" * 92)
    print("  FUSION COMPARISON — European test split (20%)")
    print("=" * 92)
    header = f"  {'config':<24}" + "".join(f"{m[:9]:>10}" for m in SCALAR_METRICS)
    print(header)
    print("  " + "-" * 90)
    for r in results:
        row = f"  {r['config']:<24}" + "".join(f"{r[m]:>10.4f}" for m in SCALAR_METRICS)
        print(row)
    print("=" * 92 + "\n")


def _save_csv(results: list[dict], out_dir: str) -> Path:
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    path = d / "fusion_comparison.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["config", *SCALAR_METRICS])
        writer.writeheader()
        writer.writerows(results)
    print(f"[fusion] wrote {path}")
    return path


def _log_mlflow(results: list[dict], extra: dict) -> None:
    import os

    try:
        import mlflow
    except ImportError:
        print("[fusion] mlflow not installed; skipping tracking (pip install '.[experiments]').")
        return

    # MLflow >= 3 blocks the legacy ./mlruns file store; default to a local
    # SQLite backend (view with: mlflow ui --backend-store-uri sqlite:///mlflow.db).
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri("sqlite:///mlflow.db")

    mlflow.set_experiment("fusion_comparison")
    for r in results:
        with mlflow.start_run(run_name=r["config"]):
            mlflow.log_params({"config": r["config"], "dataset": "european", **extra})
            mlflow.log_metrics({m: r[m] for m in SCALAR_METRICS})
    print(
        f"[fusion] logged {len(results)} runs to MLflow experiment 'fusion_comparison' "
        f"({mlflow.get_tracking_uri()})."
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Feature- vs decision-level fusion comparison.")
    parser.add_argument("--data", default="creditcard.csv")
    parser.add_argument("--out", default="experiments/results")
    args = parser.parse_args(argv)
    run(args.data, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
