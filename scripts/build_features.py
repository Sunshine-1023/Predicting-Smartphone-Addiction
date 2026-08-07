#!/usr/bin/env python3
"""Build processed train/test parquet features from official competition CSVs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from smartphone_addiction.data.load import load_competition_frames
from smartphone_addiction.features.base import transform_competition_frames
from smartphone_addiction.features.io import write_processed_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=ROOT / "data" / "raw",
        help="Directory containing train.csv, test.csv, sample_submission.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "data" / "processed",
        help="Directory for parquet outputs and feature_manifest.json",
    )
    parser.add_argument("--version", default="v1", help="Feature pipeline version tag")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw = load_competition_frames(args.raw_dir)
    frames = transform_competition_frames(raw.train, raw.test)
    paths = write_processed_dataset(frames, args.out_dir, version=args.version)
    print(f"train_rows={len(frames.train)} test_rows={len(frames.test)}")
    print(f"feature_count={len(frames.feature_columns)}")
    for key, path in paths.items():
        print(f"{key}={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
