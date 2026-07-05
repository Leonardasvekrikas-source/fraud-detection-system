# =============================================================================
# serving/benchmark.py  -  latency benchmark for the /score endpoint
#
# Measures the per-request latency of scoring + SHAP explanation (the real work
# a fraud service does), reported as p50 / p95 / p99 and throughput. Runs the
# app in-process (FastAPI TestClient) so the numbers are model+serving latency,
# not network noise.
#
# Run:  python -m fraud_detection.serving.benchmark --n 500
# =============================================================================

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path


def _percentiles(latencies: list[float]) -> dict:
    latencies = sorted(latencies)

    def pct(p: float) -> float:
        return latencies[min(int(len(latencies) * p), len(latencies) - 1)]

    mean = statistics.mean(latencies)
    return {
        "mean_ms": round(mean, 3),
        "p50_ms": round(pct(0.50), 3),
        "p95_ms": round(pct(0.95), 3),
        "p99_ms": round(pct(0.99), 3),
    }


def run(model_dir: str = "artifacts/model", n: int = 500, sample: str = "docs/sample_transactions.json") -> dict:
    import numpy as np
    from fastapi.testclient import TestClient

    from fraud_detection.artifacts.store import ModelBundle
    from fraud_detection.serving.app import create_app

    bundle = ModelBundle.load(model_dir, with_lstm=False)
    client = TestClient(create_app(bundle=bundle))
    payload = json.loads(Path(sample).read_text(encoding="utf-8"))["fraud_example"]

    raw_cols = bundle.metadata["raw_feature_columns"]
    X = np.array([[float(payload["features"][c]) for c in raw_cols]], dtype=float)

    for _ in range(25):  # warm up (JIT, caches)
        bundle.predict_p1(X)
        client.post("/score", json=payload)

    # (1) prediction only — the LightGBM score, no explanation
    predict_lat = []
    for _ in range(n):
        t0 = time.perf_counter()
        bundle.predict_p1(X)
        predict_lat.append((time.perf_counter() - t0) * 1000.0)

    # (2) full /score — prediction + SHAP explanation over HTTP
    score_lat = []
    for _ in range(n):
        t0 = time.perf_counter()
        r = client.post("/score", json=payload)
        score_lat.append((time.perf_counter() - t0) * 1000.0)
        r.raise_for_status()

    predict_stats = _percentiles(predict_lat)
    score_stats = _percentiles(score_lat)
    stats = {
        "requests": n,
        "predict_only": predict_stats,
        "full_score_with_shap": score_stats,
        "explanation_overhead_ms_p50": round(score_stats["p50_ms"] - predict_stats["p50_ms"], 2),
        "score_throughput_rps": round(1000.0 / score_stats["mean_ms"], 1),
    }

    print("\n" + "=" * 62)
    print("  /score LATENCY (in-process; prediction vs full explanation)")
    print("=" * 62)
    print(f"  {'':<24}{'p50':>9}{'p95':>9}{'p99':>9}")
    print(f"  {'predict only (LightGBM)':<24}"
          f"{predict_stats['p50_ms']:>9.2f}{predict_stats['p95_ms']:>9.2f}{predict_stats['p99_ms']:>9.2f}")
    print(f"  {'full /score (+ SHAP)':<24}"
          f"{score_stats['p50_ms']:>9.2f}{score_stats['p95_ms']:>9.2f}{score_stats['p99_ms']:>9.2f}")
    print(f"  -> the SHAP explanation adds ~{stats['explanation_overhead_ms_p50']} ms (p50);")
    print("     prediction itself is sub-millisecond. Score everything fast; explain what's reviewed.")
    print("=" * 62 + "\n")
    return stats


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark the /score endpoint latency.")
    parser.add_argument("--model-dir", default="artifacts/model")
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)
    stats = run(args.model_dir, args.n)
    if args.out:
        Path(args.out).write_text(json.dumps(stats, indent=2), encoding="utf-8")
        print(f"[benchmark] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
