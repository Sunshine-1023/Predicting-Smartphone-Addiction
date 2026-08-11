"""Unit tests for submission CSV validation and writing."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from smartphone_addiction.data.schema import ID_COLUMN, TARGET_COLUMN
from smartphone_addiction.errors import SubmissionValidationError
from smartphone_addiction.submission import (
    build_submission,
    build_submission_from_run,
    default_submission_csv,
    write_submission,
)


def test_submission_preserves_sample_ids(competition_frames) -> None:
    _, test, sample = competition_frames
    result = build_submission(sample, test["id"], np.full(len(test), 0.25))
    assert result.columns.tolist() == ["id", "addicted_label"]
    assert result["id"].equals(sample["id"])


def test_submission_rejects_nonfinite_probability(competition_frames) -> None:
    _, test, sample = competition_frames
    predictions = np.full(len(test), 0.25)
    predictions[0] = np.nan
    with pytest.raises(SubmissionValidationError, match="finite"):
        build_submission(sample, test["id"], predictions)


def test_submission_rejects_out_of_range(competition_frames) -> None:
    _, test, sample = competition_frames
    predictions = np.full(len(test), 0.25)
    predictions[0] = 1.5
    with pytest.raises(SubmissionValidationError, match=r"\[0, 1\]"):
        build_submission(sample, test["id"], predictions)


def test_submission_reorders_by_id(competition_frames) -> None:
    _, test, sample = competition_frames
    shuffled = test.sample(frac=1.0, random_state=0).reset_index(drop=True)
    preds = np.linspace(0.1, 0.9, len(shuffled))
    result = build_submission(sample, shuffled["id"], preds)
    assert result["id"].equals(sample["id"])
    expected = (
        pd.DataFrame({"id": shuffled["id"], "prediction": preds})
        .set_index("id")
        .loc[sample["id"].to_numpy(), "prediction"]
        .to_numpy()
    )
    assert result[TARGET_COLUMN].to_numpy() == pytest.approx(expected)


def test_submission_rejects_missing_id(competition_frames) -> None:
    _, test, sample = competition_frames
    bad_ids = test["id"].iloc[:-1].reset_index(drop=True)
    with pytest.raises(SubmissionValidationError, match="id set mismatch"):
        build_submission(sample, bad_ids, np.full(len(bad_ids), 0.25))


def test_submission_rejects_duplicate_ids(competition_frames) -> None:
    _, test, sample = competition_frames
    bad_ids = test["id"].copy()
    bad_ids.iloc[1] = bad_ids.iloc[0]
    with pytest.raises(SubmissionValidationError, match="unique"):
        build_submission(sample, bad_ids, np.full(len(test), 0.25))


def test_write_submission_creates_sidecar(tmp_path: Path, competition_frames) -> None:
    _, test, sample = competition_frames
    frame = build_submission(sample, test["id"], np.full(len(test), 0.25))
    out = tmp_path / "submission.csv"
    paths = write_submission(
        frame,
        out,
        sample=sample,
        metadata={"oof_auc": 0.81, "run_dir": "artifacts/runs/demo"},
    )
    assert paths["csv"].is_file()
    assert paths["meta"].is_file()
    reloaded = pd.read_csv(paths["csv"])
    assert list(reloaded.columns) == [ID_COLUMN, TARGET_COLUMN]
    meta = json.loads(paths["meta"].read_text(encoding="utf-8"))
    assert meta["n_rows"] == len(sample)
    assert meta["oof_auc"] == 0.81
    assert len(meta["sha256"]) == 64


def test_build_submission_from_run(tmp_path: Path, competition_frames) -> None:
    _, test, sample = competition_frames
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(
        json.dumps({"status": "completed", "run_id": "demo"}),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            ID_COLUMN: test[ID_COLUMN].to_numpy(),
            "prediction": np.full(len(test), 0.33),
        }
    ).to_parquet(run_dir / "test_predictions.parquet")
    (run_dir / "metrics.json").write_text(
        json.dumps({"oof_auc": 0.77, "model_name": "catboost", "seeds": [42]}),
        encoding="utf-8",
    )
    paths = build_submission_from_run(
        run_dir=run_dir,
        sample=sample,
        output_csv=tmp_path / "out" / "sub.csv",
    )
    meta = json.loads(paths["meta"].read_text(encoding="utf-8"))
    assert meta["oof_auc"] == 0.77
    assert meta["model_name"] == "catboost"


def test_default_submission_csv_uses_run_name() -> None:
    assert default_submission_csv("artifacts/runs/demo-run") == Path("submissions/demo-run.csv")


def test_build_submission_from_run_defaults_and_refuses_overwrite(
    tmp_path: Path, competition_frames, monkeypatch
) -> None:
    _, test, sample = competition_frames
    run_dir = tmp_path / "20260809T022155Z-lightgbm-final-demo"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(
        json.dumps({"status": "completed", "run_id": run_dir.name}),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            ID_COLUMN: test[ID_COLUMN].to_numpy(),
            "prediction": np.full(len(test), 0.33),
        }
    ).to_parquet(run_dir / "test_predictions.parquet")
    monkeypatch.chdir(tmp_path)
    paths = build_submission_from_run(run_dir=run_dir, sample=sample)
    expected = Path("submissions") / f"{run_dir.name}.csv"
    assert paths["csv"] == expected
    assert paths["csv"].is_file()
    with pytest.raises(SubmissionValidationError, match="already exists"):
        build_submission_from_run(run_dir=run_dir, sample=sample)
    paths2 = build_submission_from_run(run_dir=run_dir, sample=sample, force=True)
    assert paths2["csv"].is_file()


def test_build_submission_from_run_rejects_incomplete(tmp_path: Path, competition_frames) -> None:
    _, test, sample = competition_frames
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(
        json.dumps({"status": "running"}),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            ID_COLUMN: test[ID_COLUMN].to_numpy(),
            "prediction": np.full(len(test), 0.33),
        }
    ).to_parquet(run_dir / "test_predictions.parquet")
    with pytest.raises(SubmissionValidationError, match="completed"):
        build_submission_from_run(
            run_dir=run_dir,
            sample=sample,
            output_csv=tmp_path / "out" / "sub.csv",
        )


def test_build_submission_from_run_rejects_missing_manifest(
    tmp_path: Path, competition_frames
) -> None:
    _, test, sample = competition_frames
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    pd.DataFrame(
        {
            ID_COLUMN: test[ID_COLUMN].to_numpy(),
            "prediction": np.full(len(test), 0.33),
        }
    ).to_parquet(run_dir / "test_predictions.parquet")
    with pytest.raises(SubmissionValidationError, match="manifest"):
        build_submission_from_run(
            run_dir=run_dir,
            sample=sample,
            output_csv=tmp_path / "out" / "sub.csv",
        )
