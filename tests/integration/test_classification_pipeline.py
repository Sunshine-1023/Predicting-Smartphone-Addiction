"""CPU Lookup Transformer pipeline smoke using synthetic competition CSVs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from smartphone_addiction.data.schema import ID_COLUMN
from smartphone_addiction.neural.classification import run_classification_cv
from smartphone_addiction.neural.classification_config import (
    ClassificationGateConfig,
    ClassificationTrainingConfig,
    ExactValueEncodingConfig,
    LookupTransformerArchConfig,
    NeuralClassificationConfig,
)
from smartphone_addiction.neural.config import (
    NeuralArtifactConfig,
    NeuralCVConfig,
    NeuralDataConfig,
)


def _write_frames(tmp_path: Path, frames) -> Path:
    raw = tmp_path / "raw"
    raw.mkdir()
    train, test, sample = frames
    train.to_csv(raw / "train.csv", index=False)
    test.to_csv(raw / "test.csv", index=False)
    sample.to_csv(raw / "sample_submission.csv", index=False)
    return raw


def test_lookup_pipeline_refits_full_outer_train(tmp_path: Path, competition_frames) -> None:
    train, test, _ = competition_frames
    raw = _write_frames(tmp_path, competition_frames)
    config = NeuralClassificationConfig(
        data=NeuralDataConfig(directory=str(raw)),
        artifacts=NeuralArtifactConfig(directory=str(tmp_path / "lookup")),
        cv=NeuralCVConfig(n_splits=2, seed=42),
        model=LookupTransformerArchConfig(
            hidden_dim=16,
            n_blocks=1,
            n_heads=4,
            feedforward_dim=32,
            dropout=0.0,
        ),
        training=ClassificationTrainingConfig(
            batch_size=64,
            max_epochs=2,
            early_stopping_patience=2,
            holdout_fraction=0.2,
            seed=42,
        ),
        encoding=ExactValueEncodingConfig(),
        gate=ClassificationGateConfig(oof_auc_min=0.9680, min_folds=5),
        device="cpu",
    )
    run_dir = run_classification_cv(config, smoke=False)
    checkpoint = torch.load(
        run_dir / "checkpoints" / "fold_0.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert checkpoint["fit_scope"] == "full_outer_train"
    history = pd.read_csv(run_dir / "training_history.csv")
    assert set(history["phase"]) == {"selection", "full_refit"}
    oof = pd.read_parquet(run_dir / "oof_predictions.parquet")
    test_pred = pd.read_parquet(run_dir / "test_predictions.parquet")
    assert oof[ID_COLUMN].tolist() == train[ID_COLUMN].tolist()
    assert test_pred[ID_COLUMN].tolist() == test[ID_COLUMN].tolist()
    assert oof["prediction"].notna().all()
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["model_name"] == "lookup_transformer"
    assert metrics["oof_coverage"] == 1.0
    gate = json.loads((run_dir / "gate_decision.json").read_text(encoding="utf-8"))
    assert gate["used_for_official_gate"] is True
