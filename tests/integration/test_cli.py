"""CLI integration tests for help, validate, train wiring, and submission build."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from smartphone_addiction.cli import app
from smartphone_addiction.data.schema import ID_COLUMN, TARGET_COLUMN

runner = CliRunner()


def test_cli_help_lists_primary_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "train" in result.stdout
    assert "data" in result.stdout
    assert "features" in result.stdout
    assert "submission" in result.stdout
    assert "tune" in result.stdout
    assert "blend" in result.stdout
    assert "evaluate-candidates" in result.stdout
    assert "promote" in result.stdout
    assert "importance" in result.stdout
    assert "report" in result.stdout
    assert "package" in result.stdout


def test_cli_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.stdout.strip()


def test_cli_data_validate_ok(tmp_path: Path, competition_frames) -> None:
    train, test, sample = competition_frames
    train.to_csv(tmp_path / "train.csv", index=False)
    test.to_csv(tmp_path / "test.csv", index=False)
    sample.to_csv(tmp_path / "sample_submission.csv", index=False)
    result = runner.invoke(app, ["data", "validate", "--data-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "OK" in result.stdout


def test_cli_data_validate_missing_files(tmp_path: Path) -> None:
    result = runner.invoke(app, ["data", "validate", "--data-dir", str(tmp_path)])
    assert result.exit_code != 0
    assert "missing competition files" in result.output


def test_cli_submission_build(tmp_path: Path, competition_frames) -> None:
    _, test, sample = competition_frames
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(
        json.dumps({"status": "completed", "run_id": "cli-demo"}),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            ID_COLUMN: test[ID_COLUMN].to_numpy(),
            "prediction": np.full(len(test), 0.4),
        }
    ).to_parquet(run_dir / "test_predictions.parquet")
    (run_dir / "metrics.json").write_text(
        json.dumps({"oof_auc": 0.7, "model_name": "lightgbm"}),
        encoding="utf-8",
    )
    sample_path = tmp_path / "sample_submission.csv"
    sample.to_csv(sample_path, index=False)
    out = tmp_path / "submissions" / "submission.csv"
    result = runner.invoke(
        app,
        [
            "submission",
            "build",
            "--run",
            str(run_dir),
            "--sample",
            str(sample_path),
            "--output",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert out.is_file()
    assert out.with_name("submission.meta.json").is_file()
    frame = pd.read_csv(out)
    assert list(frame.columns) == [ID_COLUMN, TARGET_COLUMN]
    assert "never auto-uploads" in result.stdout


@pytest.mark.model
def test_cli_train_smoke_raw(tmp_path: Path, competition_frames) -> None:
    train, test, sample = competition_frames
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    train.to_csv(raw_dir / "train.csv", index=False)
    test.to_csv(raw_dir / "test.csv", index=False)
    sample.to_csv(raw_dir / "sample_submission.csv", index=False)
    artifact_dir = tmp_path / "artifacts"

    result = runner.invoke(
        app,
        [
            "train",
            "--raw",
            "--profile",
            "configs/profiles/smoke.yaml",
            "--model-config",
            "configs/models/catboost.yaml",
            "--override",
            f"data.directory={raw_dir}",
            "--override",
            f"artifacts.directory={artifact_dir}",
            "--override",
            "data.sample_rows=80",
            "--override",
            "cv.n_splits=2",
            "--override",
            "model.params.iterations=20",
            "--override",
            "model.params.depth=3",
            "--override",
            "model.params.early_stopping_rounds=5",
            "--override",
            "runtime.threads=2",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "run_dir=" in result.stdout
    assert "oof_auc=" in result.stdout
