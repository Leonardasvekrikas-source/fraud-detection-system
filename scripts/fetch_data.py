#!/usr/bin/env python
"""Fetch the public datasets into ``data/`` (both are public Kaggle datasets).

Usage:
    python scripts/fetch_data.py --dataset european
    python scripts/fetch_data.py --dataset paysim
    python scripts/fetch_data.py --dataset all

If the Kaggle CLI is installed and configured (``pip install kaggle`` and a
``~/.kaggle/kaggle.json`` API token), this downloads and extracts automatically.
Otherwise it prints manual download instructions. Nothing here is committed to
git - the CSVs are gitignored.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# Registry: dataset key -> (Kaggle slug, file inside the zip, local target name)
DATASETS = {
    "european": {
        "slug": "mlg-ulb/creditcardfraud",
        "zip_member": "creditcard.csv",
        "target": "creditcard.csv",
        "url": "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud",
    },
    "paysim": {
        "slug": "ealaxi/paysim1",
        "zip_member": "PS_20174392719_1491204439457_log.csv",
        "target": "paysim.csv",
        "url": "https://www.kaggle.com/datasets/ealaxi/paysim1",
    },
}


def _kaggle_available() -> bool:
    return shutil.which("kaggle") is not None


def _download_with_kaggle(key: str, spec: dict) -> bool:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[fetch] downloading '{spec['slug']}' via the Kaggle CLI ...")
    try:
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", spec["slug"], "-p", str(DATA_DIR)],
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        print(f"[fetch] Kaggle CLI failed: {exc}")
        return False

    # Extract the expected member and rename to the local target.
    zips = sorted(DATA_DIR.glob("*.zip"))
    for zpath in zips:
        with zipfile.ZipFile(zpath) as zf:
            names = zf.namelist()
            member = spec["zip_member"] if spec["zip_member"] in names else names[0]
            zf.extract(member, DATA_DIR)
        extracted = DATA_DIR / member
        target = DATA_DIR / spec["target"]
        if extracted != target:
            extracted.replace(target)
        zpath.unlink()
        print(f"[fetch] wrote {target}")
        return True
    return False


def _manual_instructions(key: str, spec: dict) -> None:
    target = DATA_DIR / spec["target"]
    print(
        f"\n[fetch] Could not download '{key}' automatically.\n"
        f"        Install + configure the Kaggle CLI (pip install kaggle, then add\n"
        f"        an API token at ~/.kaggle/kaggle.json), or download manually:\n\n"
        f"          {spec['url']}\n\n"
        f"        Then place the CSV at:\n          {target}\n"
    )


def fetch(key: str) -> bool:
    spec = DATASETS[key]
    target = DATA_DIR / spec["target"]
    if target.is_file():
        print(f"[fetch] '{key}' already present: {target}")
        return True
    if _kaggle_available() and _download_with_kaggle(key, spec):
        return True
    _manual_instructions(key, spec)
    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch public fraud datasets.")
    parser.add_argument(
        "--dataset",
        choices=[*DATASETS.keys(), "all"],
        default="european",
        help="Which dataset to fetch. Default: european",
    )
    args = parser.parse_args(argv)

    keys = list(DATASETS) if args.dataset == "all" else [args.dataset]
    ok = all(fetch(k) for k in keys)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
