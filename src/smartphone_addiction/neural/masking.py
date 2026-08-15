"""Test-like artificial core5 masking with a fixed validation mask bank."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from smartphone_addiction.errors import TrainingError
from smartphone_addiction.neural.config import CORE5_FIELDS
from smartphone_addiction.neural.device import require_torch
from smartphone_addiction.neural.preprocessing import FoldTensorizer, TensorizedFrame


@dataclass(frozen=True)
class PatternDistribution:
    patterns: tuple[str, ...]
    probabilities: np.ndarray
    counts: dict[str, int]
    n_test_rows: int


@dataclass(frozen=True)
class MaskBatch:
    masked_numeric: object
    natural_observed: object
    artificial_mask: object
    categorical: object
    targets: object
    loss_mask: object
    row_ids: np.ndarray


@dataclass(frozen=True)
class ValidationMaskBank:
    batch: MaskBatch
    repeats: np.ndarray
    n_source_rows: int


def core5_pattern_key(frame: pd.DataFrame, fields: tuple[str, ...] = CORE5_FIELDS) -> pd.Series:
    missing = [name for name in fields if name not in frame.columns]
    if missing:
        raise TrainingError(f"frame missing core fields: {missing}")
    bits = frame.loc[:, list(fields)].notna().astype(int).astype(str)
    return bits.agg("".join, axis=1)


def pattern_distribution_from_test(
    test: pd.DataFrame,
    *,
    min_missing: int = 1,
    max_missing: int = 3,
    fields: tuple[str, ...] = CORE5_FIELDS,
) -> PatternDistribution:
    keys = core5_pattern_key(test, fields)
    counts = keys.value_counts().to_dict()
    eligible: list[str] = []
    weights: list[float] = []
    n_fields = len(fields)
    for pattern, count in counts.items():
        n_missing = n_fields - int(str(pattern).count("1"))
        if min_missing <= n_missing <= max_missing:
            eligible.append(str(pattern))
            weights.append(float(count))
    if not eligible:
        raise TrainingError("test has no core5 missing patterns with 1-3 hidden fields")
    probabilities = np.asarray(weights, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    return PatternDistribution(
        patterns=tuple(eligible),
        probabilities=probabilities,
        counts={str(key): int(value) for key, value in counts.items()},
        n_test_rows=len(test),
    )


def _sample_artificial_mask(
    core_observed: np.ndarray,
    *,
    rng: np.random.Generator,
    distribution: PatternDistribution,
    min_fields: int,
    max_fields: int,
    field_balance_prob: float,
    field_counts: np.ndarray | None = None,
) -> np.ndarray:
    n_rows, n_fields = core_observed.shape
    artificial = np.zeros((n_rows, n_fields), dtype=bool)
    pattern_missing = np.array(
        [[ch == "0" for ch in pattern] for pattern in distribution.patterns],
        dtype=bool,
    )
    chosen = rng.choice(len(distribution.patterns), size=n_rows, p=distribution.probabilities)
    use_balance = rng.random(n_rows) < field_balance_prob
    for row in range(n_rows):
        observed = core_observed[row]
        observed_idx = np.flatnonzero(observed)
        if observed_idx.size == 0:
            continue
        if use_balance[row] and field_counts is not None:
            weights = 1.0 / np.maximum(field_counts[observed_idx], 1.0)
            weights = weights / weights.sum()
            n_hide = int(rng.integers(min_fields, max_fields + 1))
            n_hide = min(n_hide, int(observed_idx.size))
            hide = rng.choice(observed_idx, size=n_hide, replace=False, p=weights)
        else:
            hide_flags = pattern_missing[chosen[row]] & observed
            hide = np.flatnonzero(hide_flags)
            if hide.size == 0:
                n_hide = int(rng.integers(min_fields, min(max_fields, int(observed_idx.size)) + 1))
                hide = rng.choice(observed_idx, size=n_hide, replace=False)
            elif hide.size > max_fields:
                hide = rng.choice(hide, size=max_fields, replace=False)
        artificial[row, hide] = True
    return artificial


def apply_artificial_mask(
    frame: TensorizedFrame,
    tensorizer: FoldTensorizer,
    artificial_core: np.ndarray,
) -> MaskBatch:
    torch = require_torch()
    if artificial_core.shape[0] != frame.numeric.shape[0]:
        raise TrainingError("artificial mask row count does not match tensorized frame")
    if artificial_core.shape[1] != len(CORE5_FIELDS):
        raise TrainingError("artificial mask must have one column per core5 field")

    core_idx = tensorizer.core_indices
    masked_numeric = frame.numeric.clone()
    artificial_t = torch.as_tensor(artificial_core, dtype=torch.bool)
    for i, column_index in enumerate(core_idx.tolist()):
        masked_numeric[:, column_index] = torch.where(
            artificial_t[:, i],
            torch.zeros_like(masked_numeric[:, column_index]),
            masked_numeric[:, column_index],
        )

    core_std = (
        (frame.core_raw - tensorizer.core_mean_) / tensorizer.core_std_
        if tensorizer.core_mean_ is not None and tensorizer.core_std_ is not None
        else np.zeros_like(frame.core_raw)
    )
    core_std = np.nan_to_num(core_std, nan=0.0).astype(np.float32)
    targets = torch.as_tensor(core_std, dtype=torch.float32)
    natural_core = torch.as_tensor(frame.core_observed, dtype=torch.bool)
    loss_mask = natural_core & artificial_t
    if masked_numeric.shape[0] != targets.shape[0] or masked_numeric.shape[0] != loss_mask.shape[0]:
        raise TrainingError("mask batch tensors have inconsistent row counts")
    return MaskBatch(
        masked_numeric=masked_numeric,
        natural_observed=frame.natural_observed,
        artificial_mask=artificial_t,
        categorical=frame.categorical,
        targets=targets,
        loss_mask=loss_mask,
        row_ids=np.asarray(frame.row_ids),
    )


def build_train_mask_batch(
    frame: TensorizedFrame,
    tensorizer: FoldTensorizer,
    *,
    generator: object,
    pattern_distribution: PatternDistribution,
    min_fields: int = 1,
    max_fields: int = 3,
    field_balance_prob: float = 0.20,
) -> MaskBatch:
    torch = require_torch()
    seed = int(generator.initial_seed()) if hasattr(generator, "initial_seed") else 0
    if hasattr(generator, "graphsafe") or type(generator).__name__ == "Generator":
        try:
            seed = int(generator.initial_seed())
        except Exception:
            seed = 0
        extra = int(torch.randint(0, 2**31 - 1, (1,), generator=generator).item())
        seed = (seed + extra) % (2**32)
    rng = np.random.default_rng(seed)
    artificial = _sample_artificial_mask(
        frame.core_observed,
        rng=rng,
        distribution=pattern_distribution,
        min_fields=min_fields,
        max_fields=max_fields,
        field_balance_prob=field_balance_prob,
    )
    return apply_artificial_mask(frame, tensorizer, artificial)


def build_fixed_validation_mask_bank(
    frame: TensorizedFrame,
    tensorizer: FoldTensorizer,
    *,
    seed: int,
    repeats: int,
    pattern_distribution: PatternDistribution,
    min_fields: int = 1,
    max_fields: int = 3,
    field_balance_prob: float = 0.20,
    min_eval_per_field: int = 1,
) -> ValidationMaskBank:
    torch = require_torch()
    rng = np.random.default_rng(seed)
    n_rows = int(frame.numeric.shape[0])
    n_fields = len(CORE5_FIELDS)
    masks = np.zeros((n_rows * repeats, n_fields), dtype=bool)
    field_counts = np.ones(n_fields, dtype=np.float64)
    for repeat in range(repeats):
        start = repeat * n_rows
        stop = start + n_rows
        masks[start:stop] = _sample_artificial_mask(
            frame.core_observed,
            rng=rng,
            distribution=pattern_distribution,
            min_fields=min_fields,
            max_fields=max_fields,
            field_balance_prob=field_balance_prob,
            field_counts=field_counts,
        )
        field_counts += (masks[start:stop] & frame.core_observed).sum(axis=0)

    eval_counts = (masks.reshape(repeats, n_rows, n_fields) & frame.core_observed).sum(axis=(0, 1))
    for field_i, count in enumerate(eval_counts.tolist()):
        if count >= min_eval_per_field:
            continue
        candidates = np.flatnonzero(frame.core_observed[:, field_i])
        if candidates.size == 0:
            continue
        need = int(min_eval_per_field - count)
        chosen = rng.choice(candidates, size=min(need, int(candidates.size)), replace=False)
        masks[chosen, field_i] = True

    numeric = frame.numeric.repeat(repeats, 1)
    natural_observed = frame.natural_observed.repeat(repeats, 1)
    categorical = frame.categorical.repeat(repeats, 1)
    # torch.repeat(repeats, 1) is repeat-major: ABC, ABC. np.repeat(axis=0) would be AAA, BBB.
    core_raw = np.tile(frame.core_raw, (repeats, 1))
    core_observed = np.tile(frame.core_observed, (repeats, 1))
    row_ids = np.tile(np.asarray(frame.row_ids), repeats)
    repeats_index = np.repeat(np.arange(repeats), n_rows)

    expanded = TensorizedFrame(
        numeric=numeric,
        categorical=categorical,
        natural_observed=natural_observed,
        row_ids=row_ids,
        core_raw=core_raw,
        core_observed=core_observed,
    )
    batch = apply_artificial_mask(expanded, tensorizer, masks)
    if batch.loss_mask.shape[0] != numeric.shape[0]:
        raise TrainingError("validation mask bank row mismatch")
    if int(batch.loss_mask.sum().item()) == 0:
        raise TrainingError("validation mask bank produced an empty loss_mask")
    _ = torch
    return ValidationMaskBank(batch=batch, repeats=repeats_index, n_source_rows=n_rows)


def subset_mask_batch(batch: MaskBatch, index: np.ndarray | slice) -> MaskBatch:
    return MaskBatch(
        masked_numeric=batch.masked_numeric[index],
        natural_observed=batch.natural_observed[index],
        artificial_mask=batch.artificial_mask[index],
        categorical=batch.categorical[index],
        targets=batch.targets[index],
        loss_mask=batch.loss_mask[index],
        row_ids=np.asarray(batch.row_ids)[index],
    )


def move_mask_batch(batch: MaskBatch, device: object) -> MaskBatch:
    return MaskBatch(
        masked_numeric=batch.masked_numeric.to(device),
        natural_observed=batch.natural_observed.to(device),
        artificial_mask=batch.artificial_mask.to(device),
        categorical=batch.categorical.to(device),
        targets=batch.targets.to(device),
        loss_mask=batch.loss_mask.to(device),
        row_ids=batch.row_ids,
    )
