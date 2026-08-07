"""Write processed parquet datasets and feature manifests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from smartphone_addiction import __version__ as PACKAGE_VERSION
from smartphone_addiction.data.download import fingerprint_files
from smartphone_addiction.data.schema import FEATURE_COLUMNS, ID_COLUMN, TARGET_COLUMN
from smartphone_addiction.errors import DataValidationError
from smartphone_addiction.features.base import TransformedFrames
from smartphone_addiction.features.domain import (
    CATEGORY_INTERACTION_SEP,
    MISSING_TOKEN,
    SAFE_DIVIDE_EPS,
)

# Relative to the installed package root (src/smartphone_addiction/).
FEATURE_CODE_RELATIVE_PATHS = (
    "data/schema.py",
    "features/base.py",
    "features/domain.py",
    "features/io.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def feature_code_fingerprint() -> dict[str, Any]:
    """Return package version plus SHA-256 digests of feature-related source files."""
    package_root = Path(__file__).resolve().parents[1]
    files: dict[str, str] = {}
    for relative in FEATURE_CODE_RELATIVE_PATHS:
        path = package_root / relative
        if not path.is_file():
            raise DataValidationError(f"feature code file missing: {path}")
        files[relative] = sha256_file(path)

    combined = hashlib.sha256()
    for relative in sorted(files):
        combined.update(relative.encode("utf-8"))
        combined.update(b"\0")
        combined.update(files[relative].encode("utf-8"))
        combined.update(b"\0")
    return {
        "package_version": PACKAGE_VERSION,
        "files": files,
        "digest": combined.hexdigest(),
    }


def write_processed_dataset(
    frames: TransformedFrames,
    output_dir: Path,
    version: str = "v1",
    *,
    raw_directory: Path | str | None = None,
) -> dict[str, Path]:
    """Write train_features.parquet, test_features.parquet, feature_manifest.json.

    When ``raw_directory`` is provided, the manifest records SHA-256 digests of the
    official competition CSVs used as input. The manifest always records a fingerprint
    of the feature-code sources so stale parquet can be detected after code changes.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _assert_finite_features(frames.train, frames.feature_columns, "train")
    _assert_finite_features(frames.test, frames.feature_columns, "test")

    train_path = output_dir / "train_features.parquet"
    test_path = output_dir / "test_features.parquet"
    manifest_path = output_dir / "feature_manifest.json"

    frames.train.to_parquet(train_path, index=False)
    frames.test.to_parquet(test_path, index=False)

    source_hashes: dict[str, str] | None = None
    if raw_directory is not None:
        source_hashes = fingerprint_files(Path(raw_directory))

    manifest = {
        "version": version,
        "id_column": ID_COLUMN,
        "target_column": TARGET_COLUMN,
        "raw_features": list(FEATURE_COLUMNS),
        "feature_columns": list(frames.feature_columns),
        "categorical_columns": list(frames.categorical_columns),
        "numeric_columns": list(frames.numeric_columns),
        "feature_groups": list(frames.feature_groups),
        "source_hashes": source_hashes,
        "feature_code": feature_code_fingerprint(),
        "rules": {
            "numeric_missing": "keep_nan",
            "categorical_missing": MISSING_TOKEN,
            "safe_divide_eps": SAFE_DIVIDE_EPS,
            "category_interaction_sep": CATEGORY_INTERACTION_SEP,
            "missing_pattern": "pipe_joined_missing_column_names_in_feature_order",
        },
        "row_counts": {
            "train": len(frames.train),
            "test": len(frames.test),
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {
        "train": train_path,
        "test": test_path,
        "manifest": manifest_path,
    }


def read_processed_dataset(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Read processed parquet files and feature manifest."""
    output_dir = Path(output_dir)
    train = pd.read_parquet(output_dir / "train_features.parquet")
    test = pd.read_parquet(output_dir / "test_features.parquet")
    manifest = json.loads((output_dir / "feature_manifest.json").read_text(encoding="utf-8"))
    return train, test, manifest


def _assert_finite_features(
    frame: pd.DataFrame,
    feature_columns: list[str],
    name: str,
) -> None:
    numeric = frame[feature_columns].select_dtypes(include=[np.number])
    if numeric.empty:
        return
    if np.isinf(numeric.to_numpy(dtype=float, copy=False)).any():
        raise DataValidationError(f"{name} features contain infinite values before write")
