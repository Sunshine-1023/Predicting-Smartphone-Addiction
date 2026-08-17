"""Shared reconstruction model outputs and input encoding."""

from __future__ import annotations

from dataclasses import dataclass

from smartphone_addiction.data.schema import NUMERIC_COLUMNS
from smartphone_addiction.errors import TrainingError
from smartphone_addiction.neural.config import CORE5_FIELDS
from smartphone_addiction.neural.device import require_torch
from smartphone_addiction.neural.masking import MaskBatch


@dataclass(frozen=True)
class ReconstructionOutput:
    mean_prediction: object
    mean_latent: object
    member_predictions: object | None = None
    member_latents: object | None = None
    member_std: object | None = None


def input_available_mask(natural_observed: object, artificial_mask: object) -> object:
    """Unify natural missing and artificial hiding into one availability mask."""
    torch = require_torch()
    natural = torch.as_tensor(natural_observed, dtype=torch.bool)
    artificial = torch.as_tensor(artificial_mask, dtype=torch.bool)
    if artificial.shape[0] != natural.shape[0]:
        raise TrainingError("availability masks have inconsistent row counts")
    if artificial.shape[1] != len(CORE5_FIELDS):
        raise TrainingError("artificial mask width must equal core5 field count")
    available = natural.clone()
    core_indices = [NUMERIC_COLUMNS.index(name) for name in CORE5_FIELDS]
    for field_i, column_index in enumerate(core_indices):
        available[:, column_index] = natural[:, column_index] & ~artificial[:, field_i]
    return available


def encode_reconstruction_inputs(batch: MaskBatch, embeddings: list) -> object:
    """Build encoder inputs with unified ``input_available`` (no separate artificial channel).

    ``input_available = natural_observed & ~artificial_mask`` on core5 columns.
    Natural missing and artificial hiding share (value=0, available=0). ``loss_mask``
    remains separate and only selects supervised reconstruction targets.
    """
    torch = require_torch()
    numeric = batch.masked_numeric
    if batch.categorical.shape[0] != numeric.shape[0]:
        raise TrainingError("categorical rows do not match numeric rows")
    if batch.categorical.shape[1] != len(embeddings):
        raise TrainingError("categorical width does not match embedding modules")
    available = input_available_mask(batch.natural_observed, batch.artificial_mask)
    if available.shape != numeric.shape:
        raise TrainingError("input_available shape does not match numeric inputs")
    available_f = available.to(dtype=numeric.dtype)
    pieces = [numeric, available_f]
    pieces.extend(module(batch.categorical[:, index]) for index, module in enumerate(embeddings))
    return torch.cat(pieces, dim=-1)


def input_dim(n_numeric: int, n_core: int, n_categorical: int, embedding_dim: int) -> int:
    # n_core kept for call-site compatibility; artificial is no longer an input channel.
    _ = n_core
    return n_numeric + n_numeric + n_categorical * embedding_dim
