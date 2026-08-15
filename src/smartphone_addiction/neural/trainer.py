"""Epoch loop, inner-holdout early stopping, and checkpoint helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from smartphone_addiction.errors import TrainingError
from smartphone_addiction.neural.device import require_torch, seed_everything
from smartphone_addiction.neural.losses import masked_huber_loss
from smartphone_addiction.neural.masking import (
    MaskBatch,
    PatternDistribution,
    ValidationMaskBank,
    build_train_mask_batch,
    move_mask_batch,
    subset_mask_batch,
)
from smartphone_addiction.neural.preprocessing import FoldTensorizer, TensorizedFrame


@dataclass
class TrainResult:
    best_epoch: int
    best_holdout_loss: float
    history: list[dict[str, float]]
    batch_size: int
    n_steps: int


def allowed_batch_sizes(requested: int) -> list[int]:
    if requested < 1024:
        return [requested]
    sizes: list[int] = []
    current = requested
    while current >= 1024:
        sizes.append(current)
        if current == 1024:
            break
        current = current // 2
        if current < 1024:
            sizes.append(1024)
            break
    return sizes


def _is_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "oom" in text or "mps backend out of memory" in text


def _minibatches(n_rows: int, batch_size: int, rng: np.random.Generator) -> list[np.ndarray]:
    order = rng.permutation(n_rows)
    return [order[start : start + batch_size] for start in range(0, n_rows, batch_size)]


def _epoch_loss(
    model,
    batch: MaskBatch,
    *,
    device: object,
    batch_size: int,
    huber_delta: float,
    train: bool,
    optimizer=None,
    clip_norm: float | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[float, int]:
    torch = require_torch()
    n_rows = int(batch.loss_mask.shape[0])
    if n_rows == 0:
        raise TrainingError("cannot evaluate an empty mask batch")
    rng = rng or np.random.default_rng(0)
    total = 0.0
    steps = 0
    n_eval = 0
    for index in _minibatches(n_rows, batch_size, rng):
        mini = move_mask_batch(subset_mask_batch(batch, index), device)
        if not bool(mini.loss_mask.any()):
            continue
        if train:
            model.train()
            optimizer.zero_grad(set_to_none=True)
        else:
            model.eval()
        with torch.set_grad_enabled(train):
            output = model(mini)
            loss = masked_huber_loss(
                output.member_predictions
                if output.member_predictions is not None
                else output.mean_prediction,
                mini.targets,
                mini.loss_mask,
                delta=huber_delta,
            )
        if train:
            loss.backward()
            if clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()
        weight = int(mini.loss_mask.sum().item())
        total += float(loss.detach().cpu()) * weight
        n_eval += weight
        steps += 1
    if n_eval == 0:
        raise TrainingError("empty loss_mask")
    return total / n_eval, steps


def train_reconstruction_model(
    *,
    model,
    train_frame: TensorizedFrame,
    tensorizer: FoldTensorizer,
    holdout_bank: ValidationMaskBank,
    pattern_distribution: PatternDistribution,
    device: object,
    batch_size: int,
    max_epochs: int,
    patience: int,
    learning_rate: float,
    weight_decay: float,
    huber_delta: float,
    clip_norm: float,
    seed: int,
    min_fields: int,
    max_fields: int,
    field_balance_prob: float,
    checkpoint_path: Path | None = None,
    epoch_callback: Callable[[dict[str, float]], None] | None = None,
) -> TrainResult:
    torch = require_torch()
    seed_everything(seed, torch_module=torch)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    generator = torch.Generator()
    generator.manual_seed(seed)
    rng = np.random.default_rng(seed)
    history: list[dict[str, float]] = []
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, Any] | None = None
    stale = 0
    n_steps = 0
    actual_batch = batch_size

    for size in allowed_batch_sizes(batch_size):
        try:
            _epoch_loss(
                model,
                holdout_bank.batch,
                device=device,
                batch_size=size,
                huber_delta=huber_delta,
                train=False,
                rng=np.random.default_rng(seed),
            )
            actual_batch = size
            break
        except RuntimeError as exc:
            if not _is_oom(exc):
                raise
            mps = getattr(torch, "mps", None)
            if mps is not None and hasattr(mps, "empty_cache"):
                mps.empty_cache()
    else:
        raise TrainingError("reconstruction batch does not fit on the selected device")

    for epoch in range(1, max_epochs + 1):
        train_batch = build_train_mask_batch(
            train_frame,
            tensorizer,
            generator=generator,
            pattern_distribution=pattern_distribution,
            min_fields=min_fields,
            max_fields=max_fields,
            field_balance_prob=field_balance_prob,
        )
        train_loss, steps = _epoch_loss(
            model,
            train_batch,
            device=device,
            batch_size=actual_batch,
            huber_delta=huber_delta,
            train=True,
            optimizer=optimizer,
            clip_norm=clip_norm,
            rng=rng,
        )
        n_steps += steps
        holdout_loss, _ = _epoch_loss(
            model,
            holdout_bank.batch,
            device=device,
            batch_size=actual_batch,
            huber_delta=huber_delta,
            train=False,
            rng=np.random.default_rng(seed + epoch),
        )
        row = {
            "epoch": float(epoch),
            "train_loss": float(train_loss),
            "holdout_loss": float(holdout_loss),
            "batch_size": float(actual_batch),
        }
        history.append(row)
        if epoch_callback is not None:
            epoch_callback(row)
        if holdout_loss + 1e-12 < best_loss:
            best_loss = holdout_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            stale = 0
            if checkpoint_path is not None:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "epoch": best_epoch,
                        "holdout_loss": best_loss,
                        "model_state": best_state,
                        "tensorizer": tensorizer.to_state(),
                    },
                    checkpoint_path,
                )
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is None:
        raise TrainingError("reconstruction training failed to produce a checkpoint")
    model.load_state_dict(best_state)
    return TrainResult(
        best_epoch=best_epoch,
        best_holdout_loss=best_loss,
        history=history,
        batch_size=actual_batch,
        n_steps=n_steps,
    )
