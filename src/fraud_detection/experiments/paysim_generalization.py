# =============================================================================
# experiments/paysim_generalization.py  -  Phase 2, experiment #3
#
# Second-dataset (generalization) test: does the architecture hold on the
# structurally different PaySim dataset (interpretable mobile-money features,
# not PCA components)?
#
# PaySim-specific preprocessing (paper Section 2.4):
#   - drop identifier columns (nameOrig, nameDest) and the near-constant rule
#     flag (isFlaggedFraud);
#   - one-hot encode the transaction `type`;
#   - drop constant columns and one of each highly-correlated (>0.95) pair.
#
# Tractability: PaySim has ~1.05M rows. Only the LSTM's HybridUS runs a
# One-Class SVM on the (huge) NORMAL class, which is quadratic and infeasible at
# full size - so the LSTM trains on a CONTIGUOUS tail block of the training split
# (~200k rows). Contiguous (not random) is essential: the LSTM models transaction
# *sequences*, and a random subsample would shred temporal contiguity. LightGBM's
# HybridOS runs OCSVM only on the tiny fraud class, so LightGBM trains on the FULL
# train split (no subsample needed). The TEST split is left intact throughout.
#
# Run:  python -m fraud_detection.experiments.paysim_generalization --data paysim.csv
# =============================================================================

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd

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
DROP_COLS = ["nameOrig", "nameDest", "isFlaggedFraud"]
TRAIN_TAIL = 200_000  # contiguous tail of the training split (OCSVM-tractable, sequences intact)


def preprocess_paysim(df: pd.DataFrame) -> pd.DataFrame:
    """Turn raw PaySim into an all-numeric frame with a `Class` label column."""
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    if "type" in df.columns:
        df = pd.get_dummies(df, columns=["type"], prefix="type")

    # bool dummies -> int
    bool_cols = [c for c in df.columns if df[c].dtype == bool]
    df[bool_cols] = df[bool_cols].astype(int)

    label = config.LABEL_COL
    features = [c for c in df.columns if c != label]

    # drop constant columns
    const = [c for c in features if df[c].nunique() <= 1]
    if const:
        print(f"[paysim] dropping constant columns: {const}")
        df = df.drop(columns=const)

    # NOTE: we keep the interpretable balance features (incl. newbalanceOrig/
    # newbalanceDest). The thesis mentions >0.95 correlation pruning, but its
    # reported PaySim feature list retains these (they are its 2nd/5th most
    # important SHAP features), and F2Vote already performs selection - so
    # pre-pruning by correlation would only discard signal the thesis used.
    print(f"[paysim] preprocessed features: {[c for c in df.columns if c != label]}")
    return df


def _contiguous_tail(X, y, n):
    """Take the last n rows (contiguous) so LSTM transaction sequences stay intact."""
    if len(X) <= n:
        return X, y
    Xt, yt = X[-n:], y[-n:]
    print(f"[paysim] contiguous training tail: {len(X)} -> {len(Xt)} ({int(yt.sum())} fraud)")
    return Xt, yt


def run(data_file: str = "paysim.csv", out_dir: str = "experiments/results") -> list[dict]:
    df = load_data(data_file)
    df = preprocess_paysim(df)
    df = BasePreprocessor.remove_duplicates(df)

    feature_cols = [c for c in df.columns if c != config.LABEL_COL]
    X = df[feature_cols].values.astype(float)
    y = df[config.LABEL_COL].values

    split = int(len(X) * 0.7)
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]
    # Contiguous tail (sequences intact) for the LSTM, and for fitting the shared
    # scaler / F2Vote selectors tractably. LightGBM uses the FULL train split.
    tail_X, tail_y = _contiguous_tail(X_tr, y_tr, TRAIN_TAIL)
    print(
        f"[paysim] full train {len(y_tr)} ({int(y_tr.sum())} fraud), "
        f"LSTM tail {len(tail_y)} ({int(tail_y.sum())} fraud), "
        f"test {len(y_te)} ({int(y_te.sum())} fraud)"
    )

    pre = BasePreprocessor().fit(tail_X)
    f2 = F2VoteSelector().fit(pre.transform(tail_X), tail_y)

    def prep(A):
        return f2.transform(pre.transform(A))

    Xte_f = prep(X_te)

    # Subsystem 1: LightGBM on the FULL train split
    print("\n[paysim] LightGBM (Subsystem 1) on full train ...")
    X_os, y_os = HybridOS().fit_resample(prep(X_tr), y_tr)
    lgbm = LightGBMModel().fit(X_os, y_os)
    p1_te = lgbm.predict_proba(Xte_f)

    # Subsystem 2: LSTM on the contiguous tail (sequences preserved)
    print("\n[paysim] LSTM (Subsystem 2) on contiguous tail ...")
    X_us, y_us = HybridUS().fit_resample(prep(tail_X), tail_y)
    W = config.WINDOW_SIZE
    Xtr_win, ytr_win = create_windows(X_us, y_us, W)
    Xte_win, _ = create_windows(Xte_f, y_te, W)
    lstm = LSTMModel()
    lstm.build(input_shape=(W, Xtr_win.shape[2]))
    lstm.fit(Xtr_win, ytr_win, verbose=0)
    p2_te = lstm.predict_proba(Xte_win)

    # Evaluate
    results = []
    results.append({"config": "lightgbm", **_scalars(compute_metrics(y_te, (p1_te >= 0.5).astype(int), p1_te))})
    results.append({"config": "lstm", **_scalars(compute_metrics(y_te, (p2_te >= 0.5).astype(int), p2_te))})
    labels = engine.classify_batch(p1_te, p2_te, verbose=False)
    y_dec = (labels == engine.LABEL_FRAUD).astype(int)
    results.append({"config": "decision_level_fusion", **_scalars(compute_metrics(y_te, y_dec, (p1_te + p2_te) / 2.0))})

    _print_table(results)
    _save_csv(results, out_dir)
    _log_mlflow(results)
    return results


def _scalars(m: dict) -> dict:
    return {k: float(m[k]) for k in SCALAR_METRICS}


def _print_table(results: list[dict]) -> None:
    print("\n" + "=" * 92)
    print("  PAYSIM GENERALIZATION — test split (subsampled training)")
    print("=" * 92)
    print(f"  {'config':<24}" + "".join(f"{m[:9]:>10}" for m in SCALAR_METRICS))
    print("  " + "-" * 90)
    for r in results:
        print(f"  {r['config']:<24}" + "".join(f"{r[m]:>10.4f}" for m in SCALAR_METRICS))
    print("=" * 92 + "\n")


def _save_csv(results: list[dict], out_dir: str) -> Path:
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    path = d / "paysim_generalization.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["config", *SCALAR_METRICS])
        writer.writeheader()
        writer.writerows(results)
    print(f"[paysim] wrote {path}")
    return path


def _log_mlflow(results: list[dict]) -> None:
    import os

    try:
        import mlflow
    except ImportError:
        return
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("paysim_generalization")
    for r in results:
        with mlflow.start_run(run_name=r["config"]):
            mlflow.log_params({"config": r["config"], "dataset": "paysim"})
            mlflow.log_metrics({m: r[m] for m in SCALAR_METRICS})
    print(f"[paysim] logged {len(results)} runs to MLflow experiment 'paysim_generalization'.")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="PaySim generalization test.")
    parser.add_argument("--data", default="paysim.csv")
    parser.add_argument("--out", default="experiments/results")
    args = parser.parse_args(argv)
    run(args.data, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
