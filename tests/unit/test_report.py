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
    mark_submission_built,
    record_leaderboard_score,
    row_from_run_dir,
    sync_artifact_runs_to_summary,
    upsert_run_to_summary,
    write_experiment_summary,
    write_final_report_scaffold,
)


def _completed_run(tmp_path: Path, name: str = "run1") -> Path:
    run = tmp_path / "artifacts" / "runs" / name
    run.mkdir(parents=True)
    (run / "manifest.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "git_sha": "abc1234",
                "seeds": [42],
                "n_splits": 5,
                "slug": "catboost-smoke",
                "duration_seconds": 12.5,
                "config_hash": "cfg123",
            }
        ),
        encoding="utf-8",
    )
    (run / "metrics.json").write_text(
        json.dumps(
            {
                "model_name": "catboost",
                "oof_auc": 0.81,
                "seed_auc_mean": 0.80,
                "seed_auc_std": 0.01,
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame({"id": [1], "addicted_label": [0], "prediction": [0.2]}).to_parquet(
        run / "oof_predictions.parquet"
    )
    pd.DataFrame({"id": [10], "prediction": [0.3]}).to_parquet(run / "test_predictions.parquet")
    (tmp_path / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    return run


def test_summary_columns_and_append(tmp_path: Path) -> None:
    run = _completed_run(tmp_path)
    summary = tmp_path / "experiment_summary.csv"
    append_runs_to_summary(
        [run],
        summary,
        root=tmp_path,
        feature_groups="raw",
        profile="smoke",
        notes="ok",
    )
    frame = pd.read_csv(summary)
    assert list(frame.columns) == SUMMARY_COLUMNS
    assert frame.iloc[0]["run_id"] == "run1"
    assert frame.iloc[0]["model"] == "catboost"
    assert frame.iloc[0]["oof_auc"] == pytest.approx(0.81)
    assert frame.iloc[0]["seed_auc_mean"] == pytest.approx(0.80)
    assert frame.iloc[0]["seed_auc_std"] == pytest.approx(0.01)
    assert frame.iloc[0]["oof_path"] == "artifacts/runs/run1/oof_predictions.parquet"
    assert frame.iloc[0]["notes"] == "ok"


def test_upsert_preserves_leaderboard(tmp_path: Path) -> None:
    run = _completed_run(tmp_path)
    summary = tmp_path / "experiment_summary.csv"
    upsert_run_to_summary(run, summary, root=tmp_path, notes="first")
    frame = pd.read_csv(summary)
    frame.loc[0, "public_lb"] = 0.9
    frame.to_csv(summary, index=False)
    upsert_run_to_summary(run, summary, root=tmp_path)
    refreshed = pd.read_csv(summary)
    assert refreshed.iloc[0]["public_lb"] == pytest.approx(0.9)
    assert refreshed.iloc[0]["notes"] == "first"


def test_append_migrates_legacy_column_names(tmp_path: Path) -> None:
    summary = tmp_path / "experiment_summary.csv"
    pd.DataFrame(
        [
            {
                "run_id": "legacy",
                "model": "catboost",
                "feature_groups": "raw",
                "profile": "smoke",
                "seeds": "42",
                "folds": 5,
                "oof_auc_mean": 0.9,
                "oof_auc_std": 0.02,
                "duration_seconds": 1,
                "git_sha": "abc",
                "status": "completed",
                "notes": "old",
            }
        ]
    ).to_csv(summary, index=False)
    run = _completed_run(tmp_path, "new")
    append_runs_to_summary([run], summary, root=tmp_path)
    frame = pd.read_csv(summary)
    assert list(frame.columns) == SUMMARY_COLUMNS
    assert frame.iloc[0]["run_id"] == "legacy"
    assert frame.iloc[0]["oof_auc"] == pytest.approx(0.9)
    assert frame.iloc[0]["seed_auc_std"] == pytest.approx(0.02)
    assert frame.iloc[1]["run_id"] == "new"


def test_mark_submission_and_lb(tmp_path: Path) -> None:
    run = _completed_run(tmp_path)
    summary = tmp_path / "reports" / "experiment_summary.csv"
    submissions = tmp_path / "reports" / "submissions.csv"
    csv_path = tmp_path / "submissions" / "run1.csv"
    csv_path.parent.mkdir(parents=True)
    csv_path.write_text("id,addicted_label\n10,0.3\n", encoding="utf-8")
    mark_submission_built(
        run_dir=run,
        submission_csv=csv_path,
        summary_path=summary,
        submissions_path=submissions,
        root=tmp_path,
        notes="built",
    )
    record_leaderboard_score(
        run_id="run1",
        public_lb=0.95,
        summary_path=summary,
        submissions_path=submissions,
    )
    summary_frame = pd.read_csv(summary)
    ledger = pd.read_csv(submissions)
    assert summary_frame.iloc[0]["has_submission"] == "yes"
    assert summary_frame.iloc[0]["public_lb"] == pytest.approx(0.95)
    assert ledger.iloc[0]["public_lb"] == pytest.approx(0.95)
    assert ledger.iloc[0]["submission_csv"] == "submissions/run1.csv"


def test_sync_artifact_runs(tmp_path: Path) -> None:
    _completed_run(tmp_path, "a")
    _completed_run(tmp_path, "b")
    summary = tmp_path / "summary.csv"
    sync_artifact_runs_to_summary(root=tmp_path, summary_path=summary)
    frame = pd.read_csv(summary)
    assert set(frame["run_id"]) == {"a", "b"}


def test_rejects_incomplete_run(tmp_path: Path) -> None:
    run = _completed_run(tmp_path, "bad")
    payload = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    payload["status"] = "failed"
    (run / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ArtifactError, match="non-completed"):
        row_from_run_dir(run, root=tmp_path)


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
                    "oof_auc": 0.8,
                    "seed_auc_mean": 0.79,
                    "seed_auc_std": 0.01,
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
