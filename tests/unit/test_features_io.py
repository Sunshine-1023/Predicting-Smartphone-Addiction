"""Unit tests for processed dataset IO."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from smartphone_addiction.errors import DataValidationError
from smartphone_addiction.features.base import transform_competition_frames
from smartphone_addiction.features.io import (
    feature_code_fingerprint,
    read_processed_dataset,
    validate_processed_manifest,
    write_processed_dataset,
)


def test_write_and_read_processed_dataset(tmp_path: Path, competition_frames) -> None:
    train, test, sample = competition_frames
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    train.to_csv(raw_dir / "train.csv", index=False)
    test.to_csv(raw_dir / "test.csv", index=False)
    sample.to_csv(raw_dir / "sample_submission.csv", index=False)

    frames = transform_competition_frames(train, test)
    paths = write_processed_dataset(
        frames, tmp_path / "processed", version="v1", raw_directory=raw_dir
    )

    assert paths["train"].is_file()
    assert paths["test"].is_file()
    assert paths["manifest"].is_file()

    loaded_train, loaded_test, manifest = read_processed_dataset(tmp_path / "processed")
    assert len(loaded_train) == len(frames.train)
    assert len(loaded_test) == len(frames.test)
    assert list(loaded_train.columns) == list(frames.train.columns)
    assert manifest["feature_columns"] == frames.feature_columns
    assert manifest["row_counts"]["train"] == len(frames.train)
    assert set(manifest["source_hashes"]) == {
        "train.csv",
        "test.csv",
        "sample_submission.csv",
    }
    assert all(len(value) == 64 for value in manifest["source_hashes"].values())
    assert manifest["feature_code"]["package_version"]
    assert len(manifest["feature_code"]["digest"]) == 64
    assert "features/domain.py" in manifest["feature_code"]["files"]
    assert set(manifest["content_hashes"]) == {"train", "test"}
    assert all(len(value) == 64 for value in manifest["content_hashes"].values())

    numeric = loaded_train[frames.numeric_columns]
    assert not np.isinf(numeric.to_numpy(dtype=float, copy=False)).any()
    assert pd.read_parquet(paths["train"]).equals(loaded_train)

    validate_processed_manifest(
        manifest,
        raw_directory=raw_dir,
        train=loaded_train,
        test=loaded_test,
    )


def test_validate_processed_manifest_requires_digest() -> None:
    with pytest.raises(DataValidationError, match="feature_code"):
        validate_processed_manifest({})


def test_validate_processed_manifest_rejects_digest_mismatch() -> None:
    manifest = {
        "feature_code": {"digest": "not-the-current-digest"},
        "source_hashes": {},
        "row_counts": {"train": 1, "test": 1},
    }
    with pytest.raises(DataValidationError, match="digest mismatch"):
        validate_processed_manifest(manifest)


def test_validate_processed_manifest_checks_source_hashes_and_rows(
    tmp_path: Path, competition_frames
) -> None:
    train, test, sample = competition_frames
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    train.to_csv(raw_dir / "train.csv", index=False)
    test.to_csv(raw_dir / "test.csv", index=False)
    sample.to_csv(raw_dir / "sample_submission.csv", index=False)
    frames = transform_competition_frames(train, test)
    write_processed_dataset(frames, tmp_path / "processed", raw_directory=raw_dir)
    _, _, manifest = read_processed_dataset(tmp_path / "processed")

    bad_hashes = dict(manifest)
    bad_hashes["source_hashes"] = {"train.csv": "0" * 64}
    with pytest.raises(DataValidationError, match="source_hashes"):
        validate_processed_manifest(
            bad_hashes,
            raw_directory=raw_dir,
            train=frames.train,
            test=frames.test,
        )

    bad_rows = dict(manifest)
    bad_rows["row_counts"] = {"train": 999, "test": len(frames.test)}
    with pytest.raises(DataValidationError, match="row_counts"):
        validate_processed_manifest(
            bad_rows,
            raw_directory=raw_dir,
            train=frames.train,
            test=frames.test,
        )

    assert feature_code_fingerprint()["digest"] == manifest["feature_code"]["digest"]


def test_validate_processed_manifest_rejects_missing_raw_directory() -> None:
    manifest = {
        "feature_code": feature_code_fingerprint(),
        "source_hashes": {"train.csv": "0" * 64},
        "row_counts": {"train": 1, "test": 1},
        "feature_columns": ["age"],
    }
    with pytest.raises(DataValidationError, match="raw data directory missing"):
        validate_processed_manifest(
            manifest,
            raw_directory=Path("/tmp/does-not-exist-for-features-io-test"),
            require_source_hashes=True,
        )
