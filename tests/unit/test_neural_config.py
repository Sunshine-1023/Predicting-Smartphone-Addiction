"""Unit tests for NeuralReconstructionConfig."""

from __future__ import annotations

import pytest

from smartphone_addiction.errors import ConfigurationError
from smartphone_addiction.neural.config import load_neural_config
from smartphone_addiction.paths import project_root


def test_neural_yaml_loads_with_defaults() -> None:
    root = project_root()
    config = load_neural_config(
        [
            root / "configs/neural/masked_autoencoder.yaml",
            root / "configs/experiments/masked_autoencoder_reconstruction_v1.yaml",
        ],
        resolve=False,
    )
    assert config.model.latent_dim == 32
    assert config.model.ensemble_size == 4
    assert config.training.batch_size == 4096
    assert config.training.max_epochs == 50
    assert config.masking.max_fields == 3
    assert config.gate.r2_min == 0.10
    assert config.device == "auto"
    assert config.cv.n_splits == 5
    assert config.cv.seed == 42


def test_neural_config_rejects_unknown_key() -> None:
    root = project_root()
    with pytest.raises(ConfigurationError):
        load_neural_config(
            [root / "configs/neural/masked_autoencoder.yaml"],
            ["training.typo=1"],
            resolve=False,
        )
