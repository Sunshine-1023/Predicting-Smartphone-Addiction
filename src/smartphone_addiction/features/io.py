"""Write processed parquet datasets and feature manifests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from smartphone_addiction.data.schema import FEATURE_COLUMNS, ID_COLUMN, TARGET_COLUMN
from smartphone_addiction.errors import DataValidationError
from smartphone_addiction.features.base import TransformedFrames
from smartphone_addiction.features.domain import (
    CATEGORY_INTERACTION_SEP,
    MISSING_TOKEN,
    SAFE_DIVIDE_EPS,
)


def write_processed_dataset(
    frames: TransformedFrames,
    output_dir: Path,
    version: str = "v1",
) -> dict[str, Path]:
    """Write train_features.parquet, test_features.parquet, feature_manifest.json."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _assert_finite_features(frames.train, frames.feature_columns, "train")
    _assert_finite_features(frames.test, frames.feature_columns, "test")

    train_path = output_dir / "train_features.parquet"
    test_path = output_dir / "test_features.parquet"
    manifest_path = output_dir / "feature_manifest.json"

    frames.train.to_parquet(train_path, index=False)
    frames.test.to_parquet(test_path, index=False)

    manifest = {
        "version": version,
        "id_column": ID_COLUMN,
        "target_column": TARGET_COLUMN,
        "raw_features": list(FEATURE_COLUMNS),
        "feature_columns": list(frames.feature_columns),
        "categorical_columns": list(frames.categorical_columns),
        "numeric_columns": list(frames.numeric_columns),
        "feature_groups": list(frames.feature_groups),
        "rules": {
            "numeric_missing": "keep_nan",
            "categorical_missing": MISSING_TOKEN,
            "safe_divide_eps": SAFE_DIVIDE_EPS,
            "category_interaction_sep": CATEGORY_INTERACTION_SEP,
            "missing_pattern": "pipe_joined_missing_column_names_in_feature_order",
        },
        "row_counts": {
            "train": int(len(frames.train)),
            "test": int(len(frames.test)),
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
