"""CPU shape, missing-token, and gradient tests for Lookup Transformer."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from smartphone_addiction.neural.classification_config import LookupTransformerArchConfig
from smartphone_addiction.neural.lookup_transformer import build_lookup_transformer
from smartphone_addiction.neural.losses import binary_bce_with_logits


def test_lookup_transformer_shapes_and_gradients() -> None:
    config = LookupTransformerArchConfig(
        hidden_dim=16,
        n_blocks=2,
        n_heads=4,
        feedforward_dim=32,
        dropout=0.0,
    )
    model = build_lookup_transformer([5, 7, 4], config)
    indices = torch.tensor(
        [
            [0, 2, 1],
            [2, 3, 0],
            [3, 1, 2],
            [4, 6, 3],
        ],
        dtype=torch.long,
    )
    output = model(indices)
    assert tuple(output.logits.shape) == (4,)
    assert tuple(output.probability.shape) == (4,)
    labels = torch.tensor([0.0, 1.0, 0.0, 1.0])
    loss = binary_bce_with_logits(output.logits, labels)
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_lookup_config_rejects_incompatible_attention_width() -> None:
    with pytest.raises(ValueError, match="divisible"):
        LookupTransformerArchConfig(
            hidden_dim=15,
            n_heads=4,
        )
