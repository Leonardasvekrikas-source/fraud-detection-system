# =============================================================================
# cli.py  -  `fraud-detect` command-line entry point.
#
#   fraud-detect evaluate --subsystem 1        # LightGBM branch, 5-fold CV
#   fraud-detect evaluate --subsystem 2        # LSTM branch, 5-fold CV
#   fraud-detect evaluate --subsystem 0        # both
#   fraud-detect train --save artifacts/       # train once and persist a model
#
# Common options:
#   --data   CSV filename inside data/ (or an explicit path). Default creditcard.csv
#   --config path to an alternative config YAML (overrides config/default.yaml)
# =============================================================================

from __future__ import annotations

import argparse
import sys

from fraud_detection import config
from fraud_detection.data.loader import DEFAULT_DATA_FILE, load_data


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--data",
        default=DEFAULT_DATA_FILE,
        help=f"CSV filename inside data/ or an explicit path. Default: {DEFAULT_DATA_FILE}",
    )
    p.add_argument(
        "--config",
        default=None,
        help="Path to an alternative config YAML (overrides config/default.yaml).",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fraud-detect",
        description="LightGBM-LSTM decision-level fusion fraud detection.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_eval = sub.add_parser("evaluate", help="Cross-validated evaluation of a subsystem.")
    p_eval.add_argument(
        "--subsystem",
        type=int,
        choices=[0, 1, 2],
        default=0,
        help="1=LightGBM, 2=LSTM, 0=both. Default: 0",
    )
    _add_common(p_eval)

    p_train = sub.add_parser("train", help="Train once on the full dataset and persist.")
    p_train.add_argument(
        "--save",
        default="artifacts/model",
        help="Directory to write the model bundle. Default: artifacts/model",
    )
    p_train.add_argument(
        "--no-lstm",
        action="store_true",
        help="Train only the LightGBM subsystem (skip the heavy LSTM).",
    )
    _add_common(p_train)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if getattr(args, "config", None):
        config.reload(args.config)
    print(f"[cli] Using config: {config.CONFIG_FILE or '<built-in defaults>'}")

    df = load_data(args.data)

    if args.command == "evaluate":
        results = {}
        if args.subsystem in (0, 1):
            from fraud_detection.pipelines.subsystem1 import run_subsystem1

            results["subsystem1"] = run_subsystem1(df)
        if args.subsystem in (0, 2):
            from fraud_detection.pipelines.subsystem2 import run_subsystem2

            results["subsystem2"] = run_subsystem2(df)

        if "subsystem1" in results and "subsystem2" in results:
            print("\n" + "=" * 70)
            print("  Both subsystems evaluated. Decision-level fusion (Algorithm 1)")
            print(f"  combines P1 + P2 with theta = {config.INFERENCE_THETA}.")
            print("  Note: the CV splits differ per subsystem, so a fused score on")
            print("  aligned predictions is a Phase 2 experiment (see docs/).")
            print("=" * 70)
        return 0

    if args.command == "train":
        from fraud_detection.pipelines.train import train_and_save

        train_and_save(df, args.save, include_lstm=not args.no_lstm)
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
