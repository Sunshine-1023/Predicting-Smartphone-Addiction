"""Shared reconstruction model outputs and input encoding."""

from __future__ import annotations

from dataclasses import dataclass

from smartphone_addiction.errors import TrainingError
from smartphone_addiction.neural.device import require_torch
from smartphone_addiction.neural.masking import MaskBatch


@dataclass(frozen=True)
class ReconstructionOutput:
    mean_prediction: object
    mean_latent: object
    member_predictions: object | None = None
    member_latents: object | None = None
    member_std: object | None = None


def encode_reconstruction_inputs(batch: MaskBatch, embeddings: list) -> object:
    torch = require_torch()
    numeric = batch.masked_numeric
    observed = batch.natural_observed.to(dtype=numeric.dtype)
    artificial = batch.artificial_mask.to(dtype=numeric.dtype)
    if numeric.shape[0] != observed.shape[0] or numeric.shape[0] != artificial.shape[0]:
        raise TrainingError("reconstruction input tensors have inconsistent row counts")
    if batch.categorical.shape[0] != numeric.shape[0]:
        raise TrainingError("categorical rows do not match numeric rows")
    if batch.categorical.shape[1] != len(embeddings):
        raise TrainingError("categorical width does not match embedding modules")
    pieces = [numeric, observed, artificial]
    pieces.extend(module(batch.categorical[:, index]) for index, module in enumerate(embeddings))
    return torch.cat(pieces, dim=-1)


def input_dim(n_numeric: int, n_core: int, n_categorical: int, embedding_dim: int) -> int:
    return n_numeric + n_numeric + n_core + n_categorical * embedding_dim
