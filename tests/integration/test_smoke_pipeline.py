"""Integration smoke tests for the OOF training runner."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from smartphone_addiction.errors import ArtifactError, TrainingError
from smartphone_addiction.features.base import transform_competition_frames
from smartphone_addiction.training.runner import (
    _build_model,
    compute_training_data_hashes,
    run_training,
)

pytestmark = pytest.mark.model

CATBOOST_SMOKE_PARAMS = {
    "iterations": 20,
    "depth": 4,
    "thread_count": 2,
    "early_stopping_rounds": 5,
    "learning_rate": 0.1,
}

LIGHTGBM_SMOKE_PARAMS = {
    "n_estimators": 30,
    "num_leaves": 15,
    "n_jobs": 2,
    "early_stopping_rounds": 5,
    "learning_rate": 0.1,
}


def test_catboost_smoke_pipeline_creates_complete_oof(
    tmp_path: Path,
    competition_frames,
) -> None:
    result = run_training(
        frames=competition_frames,
        model_name="catboost",
        model_params=CATBOOST_SMOKE_PARAMS,
        n_splits=2,
        seeds=[42],
        artifact_root=tmp_path,
        git_sha="smoke001",
    )
    oof = pd.read_parquet(result.run_dir / "oof_predictions.parquet")
    test_pred = pd.read_parquet(result.run_dir / "test_predictions.parquet")
    assert len(oof) == len(competition_frames[0])
    assert len(test_pred) == len(competition_frames[1])
    assert oof["prediction"].notna().all()
    assert result.metrics["oof_coverage"] == 1.0
    assert result.store.manifest().status == "completed"
    assert (result.run_dir / "metrics.json").is_file()
    assert (result.run_dir / "slice_metrics.json").is_file()
    slice_payload = json.loads((result.run_dir / "slice_metrics.json").read_text(encoding="utf-8"))
    assert "core_complete_auc" in slice_payload
    assert "core_incomplete_auc" in slice_payload
    assert (result.run_dir / "models" / "seed42-fold0.cbm").is_file()
    names = json.loads((result.run_dir / "feature_names.json").read_text(encoding="utf-8"))
    assert "fold" not in names
    assert "fold_feature_columns" not in names
    hashes = result.store.manifest().data_hashes
    assert set(hashes) >= {"train", "test", "feature_manifest"}
    assert hashes["train"] != "in-memory"


def test_lightgbm_smoke_pipeline_creates_complete_oof(
    tmp_path: Path,
    competition_frames,
) -> None:
    result = run_training(
        frames=competition_frames,
        model_name="lightgbm",
        model_params=LIGHTGBM_SMOKE_PARAMS,
        n_splits=2,
        seeds=[42],
        artifact_root=tmp_path,
        git_sha="smoke002",
    )
    oof = pd.read_parquet(result.run_dir / "oof_predictions.parquet")
    assert len(oof) == len(competition_frames[0])
    assert oof["prediction"].notna().all()
    assert result.metrics["oof_coverage"] == 1.0
    assert 0.0 <= result.metrics["oof_auc"] <= 1.0
    assert result.store.manifest().status == "completed"
    assert (result.run_dir / "models" / "seed42-fold0.joblib").is_file()


def test_rejects_unsupported_model(tmp_path: Path, competition_frames) -> None:
    with pytest.raises(TrainingError, match="unsupported model_name"):
        run_training(
            frames=competition_frames,
            model_name="logistic",
            artifact_root=tmp_path,
            n_splits=2,
            seeds=[42],
        )


def test_resume_rejects_flipped_labels(tmp_path: Path, competition_frames) -> None:
    result = run_training(
        frames=competition_frames,
        model_name="catboost",
        model_params=CATBOOST_SMOKE_PARAMS,
        n_splits=2,
        seeds=[42],
        artifact_root=tmp_path,
        git_sha="resume01",
    )
    assert result.store.manifest().status == "completed"

    train, test, sample = competition_frames
    flipped = train.copy()
    flipped["addicted_label"] = 1 - flipped["addicted_label"]
    with pytest.raises(ArtifactError, match="completed runs cannot be resumed"):
        run_training(
            frames=(flipped, test, sample),
            model_name="catboost",
            model_params=CATBOOST_SMOKE_PARAMS,
            n_splits=2,
            seeds=[42],
            artifact_root=tmp_path,
            resume_run_dir=result.run_dir,
            git_sha="resume01",
        )


def test_resume_rejects_label_change_on_interrupted_run(
    tmp_path: Path,
    competition_frames,
) -> None:
    # Start a run, interrupt after forcing one fold complete via a tiny custom path:
    # run full smoke then open and set status interrupted with same hashes, then
    # attempt resume with flipped labels -> data hash mismatch.
    result = run_training(
        frames=competition_frames,
        model_name="catboost",
        model_params=CATBOOST_SMOKE_PARAMS,
        n_splits=2,
        seeds=[42],
        artifact_root=tmp_path / "a",
        git_sha="resume02",
    )
    # Simulate an interrupted run that still has completed fold artifacts.
    store = result.store
    store.interrupt("simulated")
    assert store.manifest().status == "interrupted"

    train, test, sample = competition_frames
    flipped = train.copy()
    flipped["addicted_label"] = 1 - flipped["addicted_label"]
    with pytest.raises(ArtifactError, match="data hash mismatch"):
        run_training(
            frames=(flipped, test, sample),
            model_name="catboost",
            model_params=CATBOOST_SMOKE_PARAMS,
            n_splits=2,
            seeds=[42],
            artifact_root=tmp_path / "b",
            resume_run_dir=store.run_dir,
            git_sha="resume02",
        )


def test_build_model_injects_current_seed() -> None:
    cat = _build_model("catboost", [], {"iterations": 10, "random_seed": 42}, seed=2026)
    assert cat.random_seed == 2026
    lgbm = _build_model("lightgbm", [], {"n_estimators": 10, "random_state": 42}, seed=3407)
    assert lgbm.random_state == 3407


def test_data_hashes_change_when_labels_flip(competition_frames) -> None:
    train, test, _ = competition_frames
    frames = transform_competition_frames(train, test)
    first = compute_training_data_hashes(
        frames.train,
        frames.test,
        frames.feature_columns,
        frames.categorical_columns,
    )
    flipped = frames.train.copy()
    flipped["addicted_label"] = 1 - flipped["addicted_label"]
    second = compute_training_data_hashes(
        flipped,
        frames.test,
        frames.feature_columns,
        frames.categorical_columns,
    )
    assert first["train"] != second["train"]
    assert first["test"] == second["test"]
    assert first["feature_manifest"] == second["feature_manifest"]
