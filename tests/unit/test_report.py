"""Unit tests for public experiment reporting helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from smartphone_addiction.errors import ArtifactError
from smartphone_addiction.evaluation.report import (
    FINAL_REPORT_SECTIONS,
    SUMMARY_COLUMNS,
    append_runs_to_summary,
    row_from_run_dir,
    write_experiment_summary,
    write_final_report_scaffold,
)


def _completed_run(tmp_path: Path, name: str = "run1") -> Path:
    run = tmp_path / name
    run.mkdir()
    (run / "manifest.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "git_sha": "abc1234",
                "seeds": [42],
                "n_splits": 5,
                "slug": "catboost-smoke",
                "duration_seconds": 12.5,
            }
        ),
        encoding="utf-8",
    )
    (run / "metrics.json").write_text(
        json.dumps({"model_name": "catboost", "oof_auc": 0.81, "oof_auc_std": 0.01}),
        encoding="utf-8",
    )
    return run


def test_summary_columns_and_append(tmp_path: Path) -> None:
    run = _completed_run(tmp_path)
    summary = tmp_path / "experiment_summary.csv"
    append_runs_to_summary([run], summary, feature_groups="raw", profile="smoke", notes="ok")
    frame = pd.read_csv(summary)
    assert list(frame.columns) == SUMMARY_COLUMNS
    assert frame.iloc[0]["run_id"] == "run1"
    assert frame.iloc[0]["model"] == "catboost"
    assert "artifacts/" not in frame.astype(str).to_numpy().ravel()[0]


def test_rejects_incomplete_run(tmp_path: Path) -> None:
    run = _completed_run(tmp_path, "bad")
    payload = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    payload["status"] = "failed"
    (run / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ArtifactError, match="non-completed"):
        row_from_run_dir(run)


def test_rejects_path_like_notes(tmp_path: Path) -> None:
    with pytest.raises(ArtifactError, match="must not include"):
        write_experiment_summary(
            [
                {
                    "run_id": "x",
                    "model": "catboost",
                    "feature_groups": "raw",
                    "profile": "smoke",
                    "seeds": "42",
                    "folds": 5,
                    "oof_auc_mean": 0.8,
                    "oof_auc_std": 0.01,
                    "duration_seconds": 1,
                    "git_sha": "abc",
                    "status": "completed",
                    "notes": "artifacts/runs/foo/models/seed42-fold0.cbm",
                }
            ],
            tmp_path / "summary.csv",
        )


def test_final_report_scaffold(tmp_path: Path) -> None:
    path = write_final_report_scaffold(tmp_path / "final_report.md")
    text = path.read_text(encoding="utf-8")
    for section in FINAL_REPORT_SECTIONS:
        assert section in text
