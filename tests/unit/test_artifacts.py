"""Unit tests for experiment artifact lifecycle."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from smartphone_addiction.artifacts.store import ArtifactStore
from smartphone_addiction.errors import ArtifactError

VALID_HASHES = {
    "train": "a" * 64,
    "test": "b" * 64,
    "feature_manifest": "c" * 64,
}


def test_run_lifecycle_is_recorded_atomically(tmp_path: Path) -> None:
    store = ArtifactStore.create(tmp_path, slug="logistic-raw", git_sha="abc1234", git_dirty=True)
    store.start(
        config={"model": {"name": "logistic"}},
        data_hashes=VALID_HASHES,
    )
    assert store.manifest().status == "running"
    assert store.manifest().git_dirty is True
    store.complete(metrics={"oof_auc": 0.75})
    assert store.manifest().status == "completed"
    assert not list(store.run_dir.glob("*.tmp"))
    assert (store.run_dir / "metrics.json").is_file()
    assert (store.run_dir / "resolved_config.yaml").is_file()
    assert (store.run_dir / "manifest.json").is_file()


def test_resume_rejects_config_hash_mismatch(tmp_path: Path) -> None:
    store = ArtifactStore.create(tmp_path, slug="catboost", git_sha="deadbee")
    config = {"model": {"name": "catboost"}, "depth": 6}
    store.start(config=config, data_hashes=VALID_HASHES)
    store.mark_fold_complete("seed42-fold0", {"auc": 0.7})
    store.interrupt()

    reopened = ArtifactStore.open(store.run_dir)
    with pytest.raises(ArtifactError, match="config hash"):
        reopened.resume_missing_folds(
            config={"model": {"name": "catboost"}, "depth": 8},
            data_hashes=VALID_HASHES,
            expected_fold_keys=["seed42-fold0", "seed42-fold1"],
        )


def test_resume_returns_missing_folds(tmp_path: Path) -> None:
    store = ArtifactStore.create(tmp_path, slug="lgbm", git_sha="1234567")
    config = {"model": {"name": "lightgbm"}}
    store.start(config=config, data_hashes=VALID_HASHES)
    store.mark_fold_complete("seed42-fold0")
    store.interrupt()

    reopened = ArtifactStore.open(store.run_dir)
    missing = reopened.resume_missing_folds(
        config=config,
        data_hashes=VALID_HASHES,
        expected_fold_keys=["seed42-fold0", "seed42-fold1", "seed42-fold2"],
    )
    assert missing == ["seed42-fold1", "seed42-fold2"]
    assert reopened.manifest().status == "running"


def test_resume_rejects_completed_run(tmp_path: Path) -> None:
    store = ArtifactStore.create(tmp_path, slug="done", git_sha="abcdef0")
    config = {"model": {"name": "catboost"}}
    store.start(config=config, data_hashes=VALID_HASHES)
    store.complete(metrics={"oof_auc": 0.8})
    reopened = ArtifactStore.open(store.run_dir)
    with pytest.raises(ArtifactError, match="completed runs cannot be resumed"):
        reopened.resume_missing_folds(
            config=config,
            data_hashes=VALID_HASHES,
            expected_fold_keys=["seed42-fold0"],
        )


def test_resume_rejects_placeholder_hashes(tmp_path: Path) -> None:
    store = ArtifactStore.create(tmp_path, slug="ph", git_sha="abcdef1")
    config = {"model": {"name": "catboost"}}
    store.start(config=config, data_hashes=VALID_HASHES)
    store.interrupt()
    reopened = ArtifactStore.open(store.run_dir)
    with pytest.raises(ArtifactError, match="placeholder data hashes"):
        reopened.resume_missing_folds(
            config=config,
            data_hashes={"source": "in-memory"},
            expected_fold_keys=["seed42-fold0"],
        )


def test_write_frame_parquet(tmp_path: Path) -> None:
    store = ArtifactStore.create(tmp_path, slug="frame", git_sha="aaaaaaa")
    store.start(config={"x": 1}, data_hashes=VALID_HASHES)
    frame = pd.DataFrame({"id": [1, 2], "prediction": [0.1, 0.9]})
    path = store.write_frame("oof_predictions.parquet", frame)
    loaded = pd.read_parquet(path)
    assert list(loaded.columns) == ["id", "prediction"]
    assert len(loaded) == 2
