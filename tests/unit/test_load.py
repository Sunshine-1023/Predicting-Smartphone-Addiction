"""Unit tests for official CSV loading."""

from __future__ import annotations

from pathlib import Path

from smartphone_addiction.data.load import load_competition_frames


def test_load_reads_three_csvs(tmp_path: Path, competition_frames) -> None:
    train, test, sample = competition_frames
    train.to_csv(tmp_path / "train.csv", index=False)
    test.to_csv(tmp_path / "test.csv", index=False)
    sample.to_csv(tmp_path / "sample_submission.csv", index=False)
    frames = load_competition_frames(tmp_path)
    assert len(frames.train) == len(train)
    assert list(frames.test.columns) == list(test.columns)
    assert frames.sample_submission["id"].equals(test["id"])
