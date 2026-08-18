"""Trainer checkpointing and holdout-AUC selection for Lookup Transformer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from smartphone_addiction.neural.classification_config import LookupTransformerArchConfig
from smartphone_addiction.neural.classification_features import EncodedClassificationTable
from smartphone_addiction.neural.classification_trainer import (
    fit_classification_fixed_epochs,
    predict_classifier,
    train_classifier,
)
from smartphone_addiction.neural.lookup_transformer import build_lookup_transformer


def _table(n_rows: int, seed: int) -> EncodedClassificationTable:
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, size=n_rows).astype(np.float32)
    labels[0] = 0.0
    labels[1] = 1.0
    return EncodedClassificationTable(
        cat_indices=rng.integers(0, 4, size=(n_rows, 2), dtype=np.int64),
        row_ids=np.arange(n_rows),
        labels=labels,
    )


def test_trainer_writes_checkpoint_and_predicts(tmp_path: Path) -> None:
    train_table = _table(32, seed=1)
    hold_table = _table(12, seed=2)
    config = LookupTransformerArchConfig(
        hidden_dim=16,
        n_blocks=1,
        n_heads=4,
        feedforward_dim=32,
        dropout=0.0,
    )
    model = build_lookup_transformer([4, 4], config)
    checkpoint = tmp_path / "fold_0.pt"
    result = train_classifier(
        model=model,
        train_table=train_table,
        holdout_table=hold_table,
        device=torch.device("cpu"),
        batch_size=16,
        max_epochs=3,
        patience=2,
        learning_rate=0.01,
        weight_decay=0.0,
        clip_norm=1.0,
        seed=42,
        checkpoint_path=checkpoint,
        encoder_state={"ok": True},
    )
    assert checkpoint.is_file()
    assert result.best_epoch >= 1
    eval_table = EncodedClassificationTable(
        cat_indices=hold_table.cat_indices,
        row_ids=hold_table.row_ids,
        labels=None,
    )
    probs = predict_classifier(
        model,
        eval_table,
        device=torch.device("cpu"),
        batch_size=16,
    )
    assert probs.shape == (12,)
    assert np.isfinite(probs).all()
    assert ((probs >= 0.0) & (probs <= 1.0)).all()


def test_fixed_epoch_refit_uses_all_rows_and_writes_checkpoint(tmp_path: Path) -> None:
    table = _table(40, seed=3)
    config = LookupTransformerArchConfig(
        hidden_dim=16,
        n_blocks=1,
        n_heads=4,
        feedforward_dim=32,
        dropout=0.0,
    )
    model = build_lookup_transformer([4, 4], config)
    checkpoint = tmp_path / "lookup_fold_0.pt"
    result = fit_classification_fixed_epochs(
        model=model,
        train_table=table,
        device=torch.device("cpu"),
        batch_size=16,
        epochs=2,
        learning_rate=0.01,
        weight_decay=0.0,
        clip_norm=1.0,
        seed=42,
        checkpoint_path=checkpoint,
        encoder_state={"scope": "outer_train"},
    )
    assert result.epochs == 2
    assert len(result.history) == 2
    assert result.n_steps == 6
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["fit_scope"] == "full_outer_train"
    assert payload["encoder"] == {"scope": "outer_train"}
