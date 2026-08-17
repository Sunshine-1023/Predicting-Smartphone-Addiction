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


def test_cli_submission_build(tmp_path: Path, competition_frames, monkeypatch) -> None:
    _, test, sample = competition_frames
    (tmp_path / ".smartphone_addiction_root").write_text("", encoding="utf-8")
    monkeypatch.setenv("SMARTPHONE_ADDICTION_ROOT", str(tmp_path))

    run_dir = tmp_path / "cli-demo-run"
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
    result = runner.invoke(
        app,
        [
            "submission",
            "build",
            "--run",
            str(run_dir),
            "--sample",
            str(sample_path),
        ],
    )
    assert result.exit_code == 0, result.output
    out = tmp_path / "submissions" / f"{run_dir.name}.csv"
    assert out.is_file()
    assert out.with_name(f"{run_dir.name}.meta.json").is_file()
    frame = pd.read_csv(out)
    assert list(frame.columns) == [ID_COLUMN, TARGET_COLUMN]
    assert "never auto-uploads" in result.stdout

    again = runner.invoke(
        app,
        [
            "submission",
            "build",
            "--run",
            str(run_dir),
            "--sample",
            str(sample_path),
        ],
    )
    assert again.exit_code != 0
    assert "already exists" in again.output

    forced = runner.invoke(
        app,
        [
            "submission",
            "build",
            "--run",
            str(run_dir),
            "--sample",
            str(sample_path),
            "--force",
        ],
    )
    assert forced.exit_code == 0, forced.output


@pytest.mark.model
def test_cli_train_smoke_raw(tmp_path: Path, competition_frames, monkeypatch) -> None:
    train, test, sample = competition_frames
    (tmp_path / ".smartphone_addiction_root").write_text("", encoding="utf-8")
    monkeypatch.setenv("SMARTPHONE_ADDICTION_ROOT", str(tmp_path))
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    train.to_csv(raw_dir / "train.csv", index=False)
    test.to_csv(raw_dir / "test.csv", index=False)
    sample.to_csv(raw_dir / "sample_submission.csv", index=False)
    artifact_dir = tmp_path / "artifacts"
    real_root = Path(__file__).resolve().parents[2]

    result = runner.invoke(
        app,
        [
            "train",
            "--raw",
            "--base",
            str(real_root / "configs/base.yaml"),
            "--profile",
            str(real_root / "configs/profiles/smoke.yaml"),
            "--model-config",
            str(real_root / "configs/models/catboost.yaml"),
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
    summary = tmp_path / "reports" / "experiment_summary.csv"
    assert summary.is_file()
    run_id = pd.read_csv(summary).iloc[0]["run_id"]
    assert str(run_id).startswith("20")
    real_summary = real_root / "reports" / "experiment_summary.csv"
    if real_summary.is_file():
        assert str(run_id) not in real_summary.read_text(encoding="utf-8")


def _write_cli_blend_runs(tmp_path: Path) -> tuple[Path, Path]:
    ids = np.arange(6)
    y = np.array([0, 0, 0, 1, 1, 1])
    oof = pd.DataFrame(
        {ID_COLUMN: ids, TARGET_COLUMN: y, "prediction": [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]}
    )
    test = pd.DataFrame({ID_COLUMN: [10, 11], "prediction": [0.4, 0.6]})
    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    for run in (run_a, run_b):
        run.mkdir()
        oof.to_parquet(run / "oof_predictions.parquet", index=False)
        test.to_parquet(run / "test_predictions.parquet", index=False)
        (run / "manifest.json").write_text(
            json.dumps({"status": "completed", "run_id": run.name, "n_train_rows": len(oof)}),
            encoding="utf-8",
        )
    (tmp_path / "reports").mkdir(exist_ok=True)
    return run_a, run_b


def test_cli_blend_fixed_weight_requires_both_flags(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / ".smartphone_addiction_root").write_text("", encoding="utf-8")
    monkeypatch.setenv("SMARTPHONE_ADDICTION_ROOT", str(tmp_path))
    run_a, run_b = _write_cli_blend_runs(tmp_path)
    method_only = runner.invoke(
        app,
        [
            "blend",
            "--runs",
            str(run_a),
            "--runs",
            str(run_b),
            "--output-dir",
            str(tmp_path / "blends" / "method-only"),
            "--method",
            "probability",
        ],
    )
    assert method_only.exit_code != 0
    weight_only = runner.invoke(
        app,
        [
            "blend",
            "--runs",
            str(run_a),
            "--runs",
            str(run_b),
            "--output-dir",
            str(tmp_path / "blends" / "weight-only"),
            "--first-weight",
            "0.6",
        ],
    )
    assert weight_only.exit_code != 0


def test_cli_blend_fixed_weight_writes_selection_mode(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / ".smartphone_addiction_root").write_text("", encoding="utf-8")
    monkeypatch.setenv("SMARTPHONE_ADDICTION_ROOT", str(tmp_path))
    run_a, run_b = _write_cli_blend_runs(tmp_path)
    out = tmp_path / "blends" / "fixed"
    result = runner.invoke(
        app,
        [
            "blend",
            "--runs",
            str(run_a),
            "--runs",
            str(run_b),
            "--output-dir",
            str(out),
            "--method",
            "probability",
            "--first-weight",
            "0.60",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads((out / "blend_result.json").read_text(encoding="utf-8"))
    assert payload["selection_mode"] == "fixed"
    assert payload["first_weight"] == pytest.approx(0.60)
    assert payload["method"] == "probability"
