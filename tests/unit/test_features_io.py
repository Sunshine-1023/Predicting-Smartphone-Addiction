"""Unit tests for processed dataset IO."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from smartphone_addiction.features.base import transform_competition_frames
from smartphone_addiction.features.io import read_processed_dataset, write_processed_dataset


def test_write_and_read_processed_dataset(tmp_path: Path, competition_frames) -> None:
    train, test, _ = competition_frames
    frames = transform_competition_frames(train, test)
    paths = write_processed_dataset(frames, tmp_path, version="v1")

    assert paths["train"].is_file()
    assert paths["test"].is_file()
    assert paths["manifest"].is_file()

    loaded_train, loaded_test, manifest = read_processed_dataset(tmp_path)
    assert len(loaded_train) == len(frames.train)
    assert len(loaded_test) == len(frames.test)
    assert list(loaded_train.columns) == list(frames.train.columns)
    assert manifest["feature_columns"] == frames.feature_columns
    assert manifest["row_counts"]["train"] == len(frames.train)

    numeric = loaded_train[frames.numeric_columns]
    assert not np.isinf(numeric.to_numpy(dtype=float, copy=False)).any()
    assert pd.read_parquet(paths["train"]).equals(loaded_train)
