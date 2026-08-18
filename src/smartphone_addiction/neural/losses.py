"""Masked-only Huber loss for core-field reconstruction."""

from __future__ import annotations

from smartphone_addiction.errors import TrainingError
from smartphone_addiction.neural.device import require_torch


def masked_huber_loss(
    predictions,
    targets,
    loss_mask,
    *,
    delta: float = 1.0,
):
    torch = require_torch()
    if delta <= 0:
        raise TrainingError("huber delta must be > 0")
    if predictions.shape[-1] != targets.shape[-1] or predictions.shape[0] != targets.shape[0]:
        raise TrainingError(
            "prediction/target shape mismatch: "
            f"{tuple(predictions.shape)} vs {tuple(targets.shape)}"
        )
    if loss_mask.shape != targets.shape:
        raise TrainingError("loss_mask must match target shape")

    pred = predictions
    tgt = targets
    mask = loss_mask
    if pred.ndim == 3:
        if tgt.ndim != 2 or mask.ndim != 2:
            raise TrainingError("ensemble predictions require 2D targets and loss_mask")
        tgt = tgt.unsqueeze(1).expand_as(pred)
        mask = mask.unsqueeze(1).expand_as(pred)
    if pred.shape != tgt.shape or pred.shape != mask.shape:
        raise TrainingError("broadcasted reconstruction tensors have inconsistent shapes")
    if not bool(mask.any()):
        raise TrainingError("empty loss_mask")

    diff = pred - tgt
    abs_diff = diff.abs()
    quadratic = torch.clamp(abs_diff, max=delta)
    linear = abs_diff - quadratic
    element = 0.5 * quadratic.square() + delta * linear
    weights = mask.to(dtype=element.dtype)
    return element.mul(weights).sum() / weights.sum()


def binary_bce_with_logits(logits, labels):
    """Binary cross-entropy for one logit per row."""
    torch = require_torch()
    logits = logits.reshape(-1)
    target = labels.reshape(-1).to(dtype=logits.dtype)
    if target.shape != logits.shape:
        raise TrainingError(
            f"label/logit batch mismatch: {tuple(target.shape)} vs {tuple(logits.shape)}"
        )
    return torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
