"""Integration test for the feature build entrypoint."""

from __future__ import annotations

from pathlib import Path

from smartphone_addiction.data.load import load_competition_frames
from smartphone_addiction.features.base import transform_competition_frames
from smartphone_addiction.features.io import read_processed_dataset, write_processed_dataset


def test_build_features_end_to_end(tmp_path: Path, competition_frames) -> None:
    train, test, sample = competition_frames
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "processed"
    raw_dir.mkdir()
    train.to_csv(raw_dir / "train.csv", index=False)
    test.to_csv(raw_dir / "test.csv", index=False)
    sample.to_csv(raw_dir / "sample_submission.csv", index=False)

    frames = load_competition_frames(raw_dir)
    transformed = transform_competition_frames(frames.train, frames.test)
    write_processed_dataset(transformed, out_dir, version="v1", raw_directory=raw_dir)

    loaded_train, loaded_test, manifest = read_processed_dataset(out_dir)
    assert len(loaded_train) == len(train)
    assert len(loaded_test) == len(test)
    assert "addicted_label" not in manifest["feature_columns"]
    assert loaded_train["id"].tolist() == train["id"].tolist()
    assert set(manifest["source_hashes"]) == {
        "train.csv",
        "test.csv",
        "sample_submission.csv",
    }
    assert len(manifest["feature_code"]["digest"]) == 64
