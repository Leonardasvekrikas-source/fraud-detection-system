# =============================================================================
# scripts/build_demo_assets.py
#
# Generate the two static assets the demo UI uses, from the DEPLOYED fusion
# model + the real dataset. Run once after training; commit the outputs.
#
#   python scripts/build_demo_assets.py
#
# Outputs (written into the serving package so they ship inside the Docker image):
#   src/fraud_detection/serving/demo_examples.json
#       10 real transactions each of Normal / Fraud / Expert-Checking, verified
#       to produce that verdict through the SINGLE-ROW API path. The
#       Expert-Checking set mixes both disagreement directions.
#   src/fraud_detection/serving/feature_meta.json
#       Per raw feature: a human display name, an honest data/model-derived fraud
#       descriptor (direction + strength + model-importance rank), and - for the
#       top drivers only - a clearly-labelled ILLUSTRATIVE alias. V1-V28 are
#       anonymized PCA components, so nothing here invents their true meaning.
# =============================================================================

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd

from fraud_detection.artifacts.store import ModelBundle
from fraud_detection.fusion import engine

BUNDLE_DIR = "artifacts/model-fusion"
DATA = "data/creditcard.csv"
OUT_DIR = Path("src/fraud_detection/serving")
RNG = np.random.default_rng(42)

# Illustrative, human-friendly stand-ins for the top fraud-driving components.
# These are NOT the real (anonymized) field meanings - they are mnemonics to make
# the demo readable, and the UI labels them as such. Attached only to top drivers.
ALIAS_POOL = {
    "V14": "account-behavior anomaly",
    "V17": "spending-profile shift",
    "V12": "transaction-pattern deviation",
    "V10": "merchant-risk signal",
    "V16": "historical-consistency score",
    "V11": "location / context signal",
    "V4": "purchase-velocity signal",
    "V3": "amount-pattern signal",
    "V7": "recipient-risk signal",
    "V18": "session-behavior signal",
    "V9": "device / channel signal",
    "V2": "counterparty signal",
}
N_ALIASES = 8  # attach aliases to the top-N most model-important features


def tier_for(rank: int, n_selected: int) -> str:
    """How much the MODEL relies on the feature (from its SHAP-importance rank).

    Kept consistent with `rank` so the label never contradicts it - unlike the
    raw label correlation, which can be large for a feature the model barely uses
    (e.g. a component made redundant by a correlated one).
    """
    if rank <= 4:
        return "primary fraud driver"
    if rank <= 10:
        return "secondary driver"
    return "minor driver"


def main() -> None:
    b = ModelBundle.load(BUNDLE_DIR, with_lstm=True)
    cols = b.metadata["raw_feature_columns"]
    selected = b.metadata.get("selected_features", cols)
    theta = float(b.metadata.get("theta", 0.5))
    df = pd.read_csv(DATA)

    # -- model importance (mean |SHAP|) over a balanced sample -----------------
    import shap

    sample = pd.concat([df[df.Class == 1], df[df.Class == 0].sample(2000, random_state=0)])
    Xs = b.transform_features(sample[cols].values.astype(float))
    expl = shap.TreeExplainer(b.lightgbm)
    sv = expl.shap_values(Xs)
    sv = sv[-1] if isinstance(sv, list) else (sv[:, :, -1] if np.ndim(sv) == 3 else sv)
    imp = np.abs(sv).mean(axis=0)  # over selected features
    imp_by_feat = dict(zip(selected, imp, strict=False))
    # rank selected features by importance (1 = most important)
    rank = {f: i + 1 for i, (f, _) in enumerate(sorted(imp_by_feat.items(), key=lambda kv: -kv[1]))}
    top_feats = [f for f, _ in sorted(imp_by_feat.items(), key=lambda kv: -kv[1])[:N_ALIASES]]

    # -- per-feature metadata --------------------------------------------------
    meta = {}
    for c in cols:
        corr = float(np.corrcoef(df[c].values, df["Class"].values)[0, 1])
        entry = {"selected": c in selected}
        if c == "Time":
            entry.update(name="Time", unit="seconds since first transaction", kind="raw")
        elif c == "Amount":
            entry.update(name="Amount", unit="transaction value", kind="raw")
        else:
            entry.update(name=c, unit="anonymized PCA component", kind="pca")
        if c in selected:
            entry["direction"] = "lower → fraud" if corr < 0 else "higher → fraud"
            entry["tier"] = tier_for(rank[c], len(selected))
            entry["corr"] = round(corr, 3)
            entry["rank"] = rank[c]
            if c in top_feats and c in ALIAS_POOL:
                entry["alias"] = ALIAS_POOL[c]
        meta[c] = entry

    (OUT_DIR / "feature_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[assets] feature_meta.json: {len(meta)} features, "
          f"{sum('alias' in m for m in meta.values())} aliased, top driver = "
          f"{min(rank, key=rank.get)}")

    # -- curated examples (single-row verdicts) --------------------------------
    def score(row):
        X = row[cols].values.astype(float).reshape(1, -1)
        p1 = float(b.predict_p1(X)[0])
        p2 = float(b.predict_p2(X)[0])
        return p1, p2, engine.classify_single(p1, p2, theta)

    def pack(row, note):
        return {"features": {c: round(float(row[c]), 6) for c in cols}, "note": note}

    normal, fraud, ec_fraud, ec_normal = [], [], [], []

    # frauds: all 492 -> Fraud examples + fraud-origin Expert-Checking
    for _, r in df[df.Class == 1].sample(frac=1.0, random_state=1).iterrows():
        p1, p2, d = score(r)
        if d == "Fraud" and len(fraud) < 10:
            fraud.append(pack(r, "both models agree: fraud"))
        elif d == "Expert-Checking" and len(ec_fraud) < 5:
            ec_fraud.append(pack(r, "LightGBM flags fraud, LSTM unsure → route to human"))
        if len(fraud) >= 10 and len(ec_fraud) >= 5:
            break

    # normals: sample -> Normal examples + normal-origin Expert-Checking
    for _, r in df[df.Class == 0].sample(4000, random_state=2).iterrows():
        p1, p2, d = score(r)
        if d == "Normal" and len(normal) < 10 and p1 < 0.05:
            normal.append(pack(r, "both models agree: normal"))
        elif d == "Expert-Checking" and len(ec_normal) < 5:
            ec_normal.append(pack(r, "LSTM flags fraud, LightGBM clears it → route to human"))
        if len(normal) >= 10 and len(ec_normal) >= 5:
            break

    examples = {"Normal": normal, "Fraud": fraud, "Expert-Checking": ec_fraud + ec_normal}
    for k, v in examples.items():
        print(f"[assets] examples[{k}]: {len(v)}")
    (OUT_DIR / "demo_examples.json").write_text(json.dumps(examples, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
