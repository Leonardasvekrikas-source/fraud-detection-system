# =============================================================================
# experiments/threshold_cost.py  -  Phase 2, experiment #2
#
# Operating-point analysis: treat the decision threshold as a COST decision,
# not just a point on a PR curve.
#
# In fraud detection the two error types have very different costs:
#   - a false negative (missed fraud) costs the fraud loss,
#   - a false positive (blocked legit transaction) costs review/friction.
# Their ratio  R = cost(FN) / cost(FP)  encodes "how much worse is a miss than a
# false alarm". This experiment sweeps the LightGBM threshold, and for a range
# of cost ratios finds the threshold that MINIMISES expected cost — showing how
# the optimal operating point shifts as fraud loss dominates friction.
#
# Output: a per-ratio table (optimal threshold, resulting FP/FN, precision,
# recall) + a cost-vs-threshold plot, logged to MLflow and written to disk.
#
# Run:  python -m fraud_detection.experiments.threshold_cost --data creditcard.csv
# =============================================================================

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from fraud_detection import config
from fraud_detection.data.loader import load_data
from fraud_detection.models.lightgbm_model import LightGBMModel
from fraud_detection.preprocessing.base_preprocessor import BasePreprocessor
from fraud_detection.preprocessing.feature_selection import F2VoteSelector
from fraud_detection.preprocessing.hybrid_os import HybridOS

# cost(FN)/cost(FP): a miss is 1x .. 100x as costly as a false alarm.
COST_RATIOS = [1, 2, 5, 10, 20, 50, 100]


def _confusion_at(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    return tp, fp, fn, tn


def _train_and_score(data_file: str):
    """Chronological 70/30 split; train Subsystem 1; return (y_test, p1_test)."""
    df = BasePreprocessor.remove_duplicates(load_data(data_file))
    feature_cols = [c for c in df.columns if c != config.LABEL_COL]
    X = df[feature_cols].values
    y = df[config.LABEL_COL].values

    split = int(len(X) * 0.7)
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]

    pre = BasePreprocessor()
    Xtr_s = pre.fit_transform(X_tr)
    Xte_s = pre.transform(X_te)
    f2 = F2VoteSelector().fit(Xtr_s, y_tr)
    Xtr_f, Xte_f = f2.transform(Xtr_s), f2.transform(Xte_s)

    X_os, y_os = HybridOS().fit_resample(Xtr_f, y_tr)
    lgbm = LightGBMModel().fit(X_os, y_os)
    return y_te, lgbm.predict_proba(Xte_f)


def analyse(y_true, y_prob, ratios=COST_RATIOS, n_thresholds=500):
    """For each cost ratio, find the expected-cost-minimising threshold."""
    thresholds = np.linspace(0.0, 1.0, n_thresholds)
    fp_at = np.empty(n_thresholds, dtype=float)
    fn_at = np.empty(n_thresholds, dtype=float)
    conf = []
    for i, t in enumerate(thresholds):
        tp, fp, fn, tn = _confusion_at(y_true, y_prob, t)
        fp_at[i], fn_at[i] = fp, fn
        conf.append((tp, fp, fn, tn))

    rows = []
    cost_curves = {}
    for r in ratios:
        # cost(FP) = 1, cost(FN) = r
        cost = fp_at * 1.0 + fn_at * float(r)
        cost_curves[r] = cost
        best = int(np.argmin(cost))
        tp, fp, fn, tn = conf[best]
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        rows.append(
            {
                "cost_ratio": r,
                "opt_threshold": round(float(thresholds[best]), 4),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "expected_cost": round(float(cost[best]), 2),
            }
        )
    return thresholds, cost_curves, rows


def _print_table(rows: list[dict]) -> None:
    print("\n" + "=" * 88)
    print("  THRESHOLD-AS-COST — optimal operating point per cost ratio (LightGBM, test split)")
    print("=" * 88)
    cols = ["cost_ratio", "opt_threshold", "fp", "fn", "precision", "recall", "f1"]
    print("  " + "".join(f"{c:>14}" for c in cols))
    print("  " + "-" * 86)
    for r in rows:
        print("  " + "".join(f"{r[c]:>14}" for c in cols))
    print("=" * 88)
    print("  Read: as a missed fraud (FN) grows costlier vs a false alarm (FP), the optimal")
    print("  threshold drops — the model is tuned to catch more fraud at the cost of more review.\n")


def _plot(thresholds, cost_curves, rows, out_dir: str) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for r, row in zip(cost_curves, rows, strict=True):
        cost = cost_curves[r]
        ax.plot(thresholds, cost / cost.max(), label=f"R={r}")
        ax.axvline(row["opt_threshold"], color="gray", lw=0.5, ls=":")
    ax.set_xlabel("decision threshold")
    ax.set_ylabel("expected cost (normalised per ratio)")
    ax.set_title("Threshold as a cost decision — optimum shifts with cost(FN)/cost(FP)")
    ax.legend(title="cost(FN)/cost(FP)", fontsize=8)
    fig.tight_layout()
    path = d / "threshold_cost.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"[cost] wrote {path}")
    return path


def _save_csv(rows: list[dict], out_dir: str) -> Path:
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    path = d / "threshold_cost.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[cost] wrote {path}")
    return path


def _log_mlflow(rows: list[dict]) -> None:
    import os

    try:
        import mlflow
    except ImportError:
        return
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("threshold_cost")
    for r in rows:
        with mlflow.start_run(run_name=f"ratio_{r['cost_ratio']}"):
            mlflow.log_params({"cost_ratio": r["cost_ratio"], "dataset": "european"})
            mlflow.log_metrics(
                {k: float(r[k]) for k in ["opt_threshold", "fp", "fn", "precision", "recall", "f1", "expected_cost"]}
            )
    print(f"[cost] logged {len(rows)} runs to MLflow experiment 'threshold_cost'.")


def run(data_file: str = "creditcard.csv", out_dir: str = "experiments/results") -> list[dict]:
    y_test, p1_test = _train_and_score(data_file)
    thresholds, cost_curves, rows = analyse(y_test, p1_test)
    _print_table(rows)
    _plot(thresholds, cost_curves, rows, out_dir)
    _save_csv(rows, out_dir)
    _log_mlflow(rows)
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Threshold-as-cost operating-point analysis.")
    parser.add_argument("--data", default="creditcard.csv")
    parser.add_argument("--out", default="experiments/results")
    args = parser.parse_args(argv)
    run(args.data, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
