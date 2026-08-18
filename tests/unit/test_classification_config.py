"""Lookup Transformer classification config loading."""

from __future__ import annotations

import pytest

from smartphone_addiction.errors import ConfigurationError
from smartphone_addiction.neural.classification_config import load_classification_config
from smartphone_addiction.paths import project_root


def test_lookup_transformer_yaml_loads() -> None:
    root = project_root()
    config = load_classification_config(
        [
            root / "configs/neural/lookup_transformer_v1.yaml",
            root / "configs/experiments/lookup_transformer_v1.yaml",
        ],
        resolve=False,
    )
    assert config.model.name == "lookup_transformer"
    assert config.model.hidden_dim == 128
    assert config.model.n_blocks == 4
    assert config.model.n_heads == 8
    assert config.model.feedforward_dim == 512
    assert config.training.batch_size == 2048
    assert config.training.max_epochs == 80
    assert config.gate.probe_auc_min == 0.9650
    assert config.cv.n_splits == 5
    assert config.cv.seed == 42


def test_classification_config_rejects_unknown_key() -> None:
    root = project_root()
    with pytest.raises(ConfigurationError):
        load_classification_config(
            [root / "configs/neural/lookup_transformer_v1.yaml"],
            ["training.typo=1"],
            resolve=False,
        )
