# =============================================================================
# experiments/calibration.py  -  Phase 2+: probability calibration
#
# Fraud teams act on SCORES, not just labels: "block above 0.9", "review 0.5-0.9".
# For that to be safe, a predicted probability of 0.8 must mean ~80% of such
# transactions really are fraud. That property is CALIBRATION.
#
# Our LightGBM is trained on HybridOS-RESAMPLED data (~5% fraud) while the real
# base rate is ~0.17%, so its raw probabilities are systematically inflated -
# they rank well (high AUC) but are NOT calibrated. This experiment quantifies
# that (Brier score, Expected Calibration Error) and fixes it with post-hoc
# isotonic regression fitted on a held-out slice at the TRUE base rate.
#
# Outputs a reliability diagram + a JSON summary; logs to MLflow.
#
# Run:  python -m fraud_detection.experiments.calibration --data creditcard.csv
# =============================================================================

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from fraud_detection import config
from fraud_detection.data.loader import load_data
from fraud_detection.models.lightgbm_model import LightGBMModel
from fraud_detection.preprocessing.base_preprocessor import BasePreprocessor
from fraud_detection.preprocessing.feature_selection import F2VoteSelector
from fraud_detection.preprocessing.hybrid_os import HybridOS

N_BINS = 12


def expected_calibration_error(y_true, y_prob, n_bins=N_BINS) -> float:
    """ECE: weighted average gap between confidence and accuracy across bins."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (y_prob >= lo) & (y_prob < hi if hi < 1.0 else y_prob <= hi)
        if not mask.any():
            continue
        conf = float(y_prob[mask].mean())
        acc = float(y_true[mask].mean())
        ece += (mask.sum() / n) * abs(conf - acc)
    return float(ece)


def run(data_file: str = "creditcard.csv", out_dir: str = "experiments/results") -> dict:
    from sklearn.calibration import calibration_curve
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import brier_score_loss

    df = BasePreprocessor.remove_duplicates(load_data(data_file))
    feature_cols = [c for c in df.columns if c != config.LABEL_COL]
    X = df[feature_cols].values
    y = df[config.LABEL_COL].values

    # chronological 60 / 20 / 20 : train / calibration / holdout
    n = len(X)
    a, b = int(n * 0.6), int(n * 0.8)
    X_tr, X_cal, X_te = X[:a], X[a:b], X[b:]
    y_tr, y_cal, y_te = y[:a], y[a:b], y[b:]

    pre = BasePreprocessor().fit(X_tr)
    f2 = F2VoteSelector().fit(pre.transform(X_tr), y_tr)

    def prep(A):
        return f2.transform(pre.transform(A))

    print("[calibration] training LightGBM on HybridOS-resampled data ...")
    X_os, y_os = HybridOS().fit_resample(prep(X_tr), y_tr)
    model = LightGBMModel().fit(X_os, y_os)

    p_raw_cal = model.predict_proba(prep(X_cal))
    p_raw_te = model.predict_proba(prep(X_te))

    # post-hoc isotonic calibration, fit on the calibration slice (true base rate)
    iso = IsotonicRegression(out_of_bounds="clip").fit(p_raw_cal, y_cal)
    p_cal_te = iso.predict(p_raw_te)

    # Aggregate calibration is dominated by the ~99.8% easy negatives (all scored
    # ~0), which masks miscalibration. What matters operationally is calibration
    # in the FLAGGED region (scores the model actually acts on), so we report ECE
    # there too - that is where HybridOS resampling inflates probabilities.
    flag = p_raw_te >= 0.05
    metrics = {
        "brier_raw": float(brier_score_loss(y_te, p_raw_te)),
        "brier_calibrated": float(brier_score_loss(y_te, p_cal_te)),
        "ece_raw": expected_calibration_error(y_te, p_raw_te),
        "ece_calibrated": expected_calibration_error(y_te, p_cal_te),
        "flagged_n": int(flag.sum()),
        "ece_raw_flagged": expected_calibration_error(y_te[flag], p_raw_te[flag]),
        "ece_calibrated_flagged": expected_calibration_error(y_te[flag], p_cal_te[flag]),
    }
    curve_raw = calibration_curve(y_te, p_raw_te, n_bins=N_BINS, strategy="quantile")
    curve_cal = calibration_curve(y_te, p_cal_te, n_bins=N_BINS, strategy="quantile")

    _print(metrics)
    _plot(curve_raw, curve_cal, metrics, out_dir)
    _save(metrics, out_dir)
    _log_mlflow(metrics)
    return metrics


def _print(m: dict) -> None:
    print("\n" + "=" * 66)
    print("  PROBABILITY CALIBRATION — LightGBM (holdout)")
    print("=" * 66)
    print(f"  Brier score            raw={m['brier_raw']:.5f}  ->  cal={m['brier_calibrated']:.5f}")
    print(f"  ECE (all)              raw={m['ece_raw']:.4f}   ->  cal={m['ece_calibrated']:.4f}")
    print(f"  ECE (flagged, n={m['flagged_n']:<5d}) raw={m['ece_raw_flagged']:.4f}   ->  cal={m['ece_calibrated_flagged']:.4f}")
    print("  Aggregate ECE is tiny (easy negatives dominate); the FLAGGED-region ECE is")
    print("  what matters — isotonic calibration corrects the resampling-induced inflation there.")
    print("=" * 66 + "\n")


def _plot(curve_raw, curve_cal, metrics, out_dir: str) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k:", label="perfectly calibrated")
    tr_true, tr_pred = curve_raw
    ca_true, ca_pred = curve_cal
    ax.plot(tr_pred, tr_true, "o-", color="#f85149",
            label=f"raw LightGBM (ECE {metrics['ece_raw']:.3f})")
    ax.plot(ca_pred, ca_true, "s-", color="#3fb950",
            label=f"isotonic-calibrated (ECE {metrics['ece_calibrated']:.3f})")
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed fraud frequency")
    ax.set_title("Reliability diagram — raw vs calibrated")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    path = d / "calibration.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"[calibration] wrote {path}")
    return path


def _save(metrics: dict, out_dir: str) -> Path:
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / "calibration.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with open(d / "calibration.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["metric", "raw", "calibrated"])
        w.writerow(["brier", metrics["brier_raw"], metrics["brier_calibrated"]])
        w.writerow(["ece", metrics["ece_raw"], metrics["ece_calibrated"]])
    print(f"[calibration] wrote {d / 'calibration.json'}")
    return d / "calibration.json"


def _log_mlflow(metrics: dict) -> None:
    import os

    try:
        import mlflow
    except ImportError:
        return
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("calibration")
    with mlflow.start_run(run_name="lightgbm_isotonic"):
        mlflow.log_metrics(metrics)
    print("[calibration] logged to MLflow experiment 'calibration'.")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="LightGBM probability calibration analysis.")
    parser.add_argument("--data", default="creditcard.csv")
    parser.add_argument("--out", default="experiments/results")
    args = parser.parse_args(argv)
    run(args.data, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
