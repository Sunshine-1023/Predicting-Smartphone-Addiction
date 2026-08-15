"""Unit tests for masked-only Huber loss."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from smartphone_addiction.errors import TrainingError
from smartphone_addiction.neural.losses import masked_huber_loss


def test_empty_loss_mask_raises() -> None:
    predictions = torch.zeros(4, 5)
    targets = torch.ones(4, 5)
    mask = torch.zeros(4, 5, dtype=torch.bool)
    with pytest.raises(TrainingError, match="empty loss_mask"):
        masked_huber_loss(predictions, targets, mask)


def test_targets_outside_mask_do_not_change_loss() -> None:
    predictions = torch.zeros(4, 5)
    targets = torch.zeros(4, 5)
    mask = torch.zeros(4, 5, dtype=torch.bool)
    mask[0, 0] = True
    targets[0, 0] = 1.0
    baseline = masked_huber_loss(predictions, targets, mask)
    targets = targets.clone()
    targets[1, :] = 99.0
    assert torch.allclose(masked_huber_loss(predictions, targets, mask), baseline)


def test_ensemble_predictions_broadcast_mask() -> None:
    predictions = torch.zeros(3, 4, 5)
    targets = torch.zeros(3, 5)
    mask = torch.zeros(3, 5, dtype=torch.bool)
    mask[0, 1] = True
    targets[0, 1] = 2.0
    loss = masked_huber_loss(predictions, targets, mask, delta=1.0)
    assert torch.isfinite(loss)
    assert loss.item() > 0
