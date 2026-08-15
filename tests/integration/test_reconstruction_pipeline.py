"""CPU reconstruction pipeline smoke using synthetic competition CSVs."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from smartphone_addiction.neural.config import (
    NeuralArtifactConfig,
    NeuralCVConfig,
    NeuralDataConfig,
    NeuralMaskingConfig,
    NeuralModelArchConfig,
    NeuralReconstructionConfig,
    NeuralTrainingConfig,
)
from smartphone_addiction.neural.reconstruction import run_reconstruction_cv


def _write_frames(tmp_path: Path, frames) -> Path:
    raw = tmp_path / "raw"
    raw.mkdir()
    train, test, sample = frames
    train.to_csv(raw / "train.csv", index=False)
    test.to_csv(raw / "test.csv", index=False)
    sample.to_csv(raw / "sample_submission.csv", index=False)
    return raw


def test_reconstruction_pipeline_creates_complete_oof(tmp_path: Path, competition_frames) -> None:
    raw = _write_frames(tmp_path, competition_frames)
    config = NeuralReconstructionConfig(
        data=NeuralDataConfig(directory=str(raw)),
        artifacts=NeuralArtifactConfig(directory=str(tmp_path / "recon")),
        cv=NeuralCVConfig(n_splits=2, seed=42),
        model=NeuralModelArchConfig(name="mlp", hidden_dim=32, latent_dim=8, n_blocks=1),
        training=NeuralTrainingConfig(
            batch_size=64,
            max_epochs=2,
            early_stopping_patience=2,
            holdout_fraction=0.2,
            seed=42,
        ),
        masking=NeuralMaskingConfig(valid_repeats=2, min_eval_per_field=1, field_balance_prob=0.0),
        device="cpu",
    )
    run_dir = run_reconstruction_cv(config, smoke=False)
    assert (run_dir / "gate_decision.json").is_file()
    assert (run_dir / "summary.md").is_file()
    assert (run_dir / "reconstruction_oof.parquet").is_file()
    assert (run_dir / "field_metrics.csv").is_file()
    assert (run_dir / "checkpoints" / "fold_0.pt").is_file()
    assert (run_dir / "environment.json").is_file()
    payload = (run_dir / "environment.json").read_text(encoding="utf-8")
    assert '"device": "cpu"' in payload
