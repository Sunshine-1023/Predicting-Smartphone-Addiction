#!/usr/bin/env python3
"""CLI wrapper around smartphone_addiction.kaggle_bundle."""

from __future__ import annotations

import argparse
from pathlib import Path

from smartphone_addiction.kaggle_bundle import package_kaggle_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dist-dir", type=Path, default=None)
    args = parser.parse_args()
    paths = package_kaggle_bundle(config_path=args.config, dist_dir=args.dist_dir)
    for key, path in paths.items():
        print(f"{key}={path}")


if __name__ == "__main__":
    main()
