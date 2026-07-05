# =============================================================================
# monitoring/drift.py  -  Phase 3: data & target drift monitoring (Evidently)
#
# A fraud model is trained on a fixed snapshot, but production data keeps moving
# (new fraud tactics, changing spend patterns). The model doesn't crash - it
# silently degrades. This module compares a REFERENCE distribution (what the
# model was trained on) against a CURRENT batch and reports whether features or
# the target have DRIFTED, using Evidently's statistical tests (K-S, PSI, chi2).
#
# The drift here is SIMULATED and clearly labelled - this is offline public
# data, not a live feed. Two scenarios:
#   - "stable": current is another clean sample -> the monitor should NOT alarm.
#   - "drift" : current has shifted features + an inflated fraud rate -> the
#               monitor SHOULD flag it. A good monitor discriminates the two.
#
# Run:  python -m fraud_detection.monitoring.drift --scenario drift
#       python -m fraud_detection.monitoring.drift --scenario stable
# =============================================================================

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from fraud_detection import config
from fraud_detection.data.loader import load_data

# Features we perturb in the "drift" scenario, with the change applied. These
# are the strongest fraud-signal features (per the SHAP study) plus Amount, so
# the simulated drift is meaningful. Documented so the simulation is honest.
DRIFT_SHIFTS = {"V14": +2.0, "V10": +1.5, "V12": +1.5, "V4": -1.5}
SAMPLE_SIZE = 20_000


def make_scenario(df: pd.DataFrame, scenario: str, seed: int = 42):
    """Build (reference, current) frames for the chosen scenario.

    reference = a clean random sample (the 'training-time' distribution).
    current   = another sample; in the 'drift' scenario it is perturbed.
    """
    n = min(SAMPLE_SIZE, len(df) // 2)
    ref_raw = df.sample(n=n, random_state=seed)  # keep original index to exclude below
    reference = ref_raw.reset_index(drop=True)
    current = df.drop(ref_raw.index).sample(n=n, random_state=seed + 1).reset_index(drop=True)

    changes: dict[str, str] = {}
    if scenario == "drift":
        # (1) covariate / data drift: shift key features + scale Amount up.
        for col, delta in DRIFT_SHIFTS.items():
            if col in current:
                current[col] = current[col] + delta
                changes[col] = f"shifted by {delta:+g}"
        if "Amount" in current:
            current["Amount"] = current["Amount"] * 1.8 + 50.0
            changes["Amount"] = "scaled x1.8 + 50 (larger transactions)"

        # (2) target / prevalence drift: raise the fraud rate by upsampling fraud
        # rows (simulating a fraud wave).
        fraud = current[current[config.LABEL_COL] == config.FRAUD_LABEL]
        if len(fraud):
            extra = fraud.sample(n=len(fraud) * 15, replace=True, random_state=seed)
            current = pd.concat([current, extra], ignore_index=True)
            changes[config.LABEL_COL] = "fraud rate inflated (simulated fraud wave)"

    return reference, current, changes


def run_report(reference: pd.DataFrame, current: pd.DataFrame, out_dir: str, tag: str) -> dict:
    """Run an Evidently drift report; save HTML + a small JSON summary."""
    from evidently import DataDefinition, Dataset, Report
    from evidently.presets import DataDriftPreset

    feature_cols = [c for c in reference.columns if c != config.LABEL_COL]
    data_def = DataDefinition(
        numerical_columns=feature_cols,
        categorical_columns=[config.LABEL_COL],
    )
    ref_ds = Dataset.from_pandas(reference, data_definition=data_def)
    cur_ds = Dataset.from_pandas(current, data_definition=data_def)

    report = Report([DataDriftPreset()])
    result = report.run(current_data=cur_ds, reference_data=ref_ds)

    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    html_path = d / f"drift_report_{tag}.html"
    result.save_html(str(html_path))

    summary = _summarise(result.dict(), tag)
    (d / f"drift_summary_{tag}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[drift] full report:  {html_path}")
    print(f"[drift] summary json: {d / f'drift_summary_{tag}.json'}")
    return summary


def _summarise(report_dict: dict, tag: str) -> dict:
    """Extract a compact, reviewable summary from Evidently's report dict.

    Evidently auto-selects a drift method per column (distance-based like
    Wasserstein / Jensen-Shannon for large samples, or p-value tests otherwise).
    The metric name carries the method + threshold; a column has drifted when a
    distance EXCEEDS the threshold (or a p-value FALLS BELOW it).
    """
    drifted_share = None
    drifted_count = None
    columns = []
    for m in report_dict.get("metrics", []):
        name = m.get("metric_name", "")
        val = m.get("value", {})
        if name.startswith("DriftedColumnsCount"):
            if isinstance(val, dict):
                # cast numpy scalars -> native Python for JSON serialisation
                drifted_count = None if val.get("count") is None else float(val["count"])
                drifted_share = None if val.get("share") is None else float(val["share"])
        elif name.startswith("ValueDrift(column="):
            parts = {}
            for kv in name[len("ValueDrift(") :].rstrip(")").split(","):
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    parts[k.strip()] = v.strip()
            method = parts.get("method", "")
            threshold = float(parts["threshold"]) if "threshold" in parts else None
            score = float(val) if isinstance(val, (int, float)) else None
            is_pvalue = "p_value" in method or "p-value" in method
            drifted = bool(
                score is not None
                and threshold is not None
                and (score < threshold if is_pvalue else score > threshold)
            )
            columns.append(
                {
                    "column": parts.get("column", "?"),
                    "method": method,
                    "score": score,
                    "threshold": threshold,
                    "drifted": drifted,
                }
            )

    columns.sort(key=lambda c: (not c["drifted"], -(c["score"] or 0.0)))  # drifted first, score desc
    return {
        "scenario": tag,
        "note": "SIMULATED drift on offline public data — not a live production feed.",
        "drifted_columns_count": drifted_count,
        "drifted_columns_share": drifted_share,
        "columns": columns,
    }


def _print_summary(summary: dict, changes: dict) -> None:
    print("\n" + "=" * 74)
    print(f"  DRIFT MONITOR — scenario: {summary['scenario'].upper()}")
    print("=" * 74)
    if changes:
        print("  Simulated changes applied to the CURRENT batch:")
        for k, v in changes.items():
            print(f"    - {k}: {v}")
    share = summary["drifted_columns_share"]
    print(
        f"  Drifted columns: {summary['drifted_columns_count']} "
        f"({share:.0%} of columns)" if share is not None else "  Drifted columns: n/a"
    )
    flagged = [c for c in summary["columns"] if c["drifted"]]
    if flagged:
        print("  Flagged as drifted (distance > threshold):")
        for c in flagged[:12]:
            print(
                f"    - {c['column']:<8} {c['method']:<28} "
                f"score={c['score']:.3f} (thr {c['threshold']})"
            )
    else:
        print("  No columns flagged as drifted (as expected for a stable batch).")
    print("=" * 74 + "\n")


def run(data_file: str = "creditcard.csv", scenario: str = "drift", out_dir: str = "monitoring/reports") -> dict:
    df = load_data(data_file)
    reference, current, changes = make_scenario(df, scenario)
    print(
        f"[drift] scenario={scenario}  reference={len(reference)} rows, current={len(current)} rows "
        f"(ref fraud={reference[config.LABEL_COL].mean():.4f}, cur fraud={current[config.LABEL_COL].mean():.4f})"
    )
    summary = run_report(reference, current, out_dir, tag=scenario)
    _print_summary(summary, changes)
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Evidently drift monitoring (simulated).")
    parser.add_argument("--data", default="creditcard.csv")
    parser.add_argument("--scenario", choices=["stable", "drift"], default="drift")
    parser.add_argument("--out", default="monitoring/reports")
    args = parser.parse_args(argv)
    run(args.data, args.scenario, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
