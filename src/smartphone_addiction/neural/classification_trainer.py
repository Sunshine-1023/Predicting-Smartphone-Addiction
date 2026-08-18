"""Lookup Transformer training with inner-holdout AUC selection and full refit."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm

from smartphone_addiction.errors import TrainingError
from smartphone_addiction.neural.classification_features import EncodedClassificationTable
from smartphone_addiction.neural.device import require_torch, seed_everything
from smartphone_addiction.neural.losses import binary_bce_with_logits
from smartphone_addiction.neural.trainer import _is_oom, allowed_batch_sizes


@dataclass
class ClassificationTrainResult:
    best_epoch: int
    best_holdout_auc: float
    best_holdout_loss: float
    history: list[dict[str, float]]
    batch_size: int
    n_steps: int


@dataclass
class FixedEpochFitResult:
    epochs: int
    history: list[dict[str, float]]
    batch_size: int
    n_steps: int


@dataclass
class PackedClassificationBatch:
    cat_indices: Any
    labels: Any | None
    row_ids: np.ndarray


def pack_classification_tensors(
    table: EncodedClassificationTable,
    *,
    include_labels: bool = True,
) -> PackedClassificationBatch:
    torch = require_torch()
    labels = None
    if include_labels:
        if table.labels is None:
            raise TrainingError("packed training tables require labels")
        labels = torch.tensor(np.asarray(table.labels), dtype=torch.float32)
    return PackedClassificationBatch(
        cat_indices=torch.tensor(table.cat_indices, dtype=torch.long),
        labels=labels,
        row_ids=np.asarray(table.row_ids),
    )


def _move(batch: PackedClassificationBatch, device: object, index: np.ndarray):
    torch = require_torch()
    idx = torch.as_tensor(index, dtype=torch.long)
    labels = None if batch.labels is None else batch.labels.index_select(0, idx).to(device)
    return PackedClassificationBatch(
        cat_indices=batch.cat_indices.index_select(0, idx).to(device),
        labels=labels,
        row_ids=np.asarray(batch.row_ids)[index],
    )


def _minibatches(n_rows: int, batch_size: int, rng: np.random.Generator) -> list[np.ndarray]:
    order = rng.permutation(n_rows)
    return [order[start : start + batch_size] for start in range(0, n_rows, batch_size)]


def _holdout_auc(labels: np.ndarray, probabilities: np.ndarray) -> float | None:
    if len(np.unique(labels)) < 2:
        return None
    return float(roc_auc_score(labels, probabilities))


def _epoch(
    model,
    batch: PackedClassificationBatch,
    *,
    device: object,
    batch_size: int,
    train: bool,
    optimizer=None,
    clip_norm: float | None = None,
    rng: np.random.Generator,
) -> tuple[float, np.ndarray, np.ndarray, int]:
    torch = require_torch()
    n_rows = len(batch.row_ids)
    if n_rows == 0:
        raise TrainingError("cannot evaluate an empty classification batch")
    total = 0.0
    steps = 0
    probs: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for index in _minibatches(n_rows, batch_size, rng):
        mini = _move(batch, device, index)
        if train:
            model.train()
            optimizer.zero_grad(set_to_none=True)
        else:
            model.eval()
        with torch.set_grad_enabled(train):
            output = model(mini.cat_indices)
            loss = binary_bce_with_logits(output.logits, mini.labels)
        if train:
            loss.backward()
            if clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()
        weight = len(index)
        total += float(loss.detach().cpu()) * weight
        steps += 1
        probs.append(output.probability.detach().cpu().numpy())
        labels.append(mini.labels.detach().cpu().numpy())
    return total / n_rows, np.concatenate(probs), np.concatenate(labels), steps


def train_classifier(
    *,
    model,
    train_table: EncodedClassificationTable,
    holdout_table: EncodedClassificationTable,
    device: object,
    batch_size: int,
    max_epochs: int,
    patience: int,
    learning_rate: float,
    weight_decay: float,
    clip_norm: float,
    seed: int,
    checkpoint_path: Path | None = None,
    encoder_state: dict[str, Any] | None = None,
    epoch_callback: Callable[[dict[str, float]], None] | None = None,
    show_progress: bool = False,
) -> ClassificationTrainResult:
    torch = require_torch()
    seed_everything(seed, torch_module=torch)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    rng = np.random.default_rng(seed)
    train_batch = pack_classification_tensors(train_table)
    hold_batch = pack_classification_tensors(holdout_table)
    history: list[dict[str, float]] = []
    best_auc = -np.inf
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, Any] | None = None
    stale = 0
    n_steps = 0
    actual_batch = batch_size

    for size in allowed_batch_sizes(batch_size):
        try:
            _epoch(
                model,
                hold_batch,
                device=device,
                batch_size=size,
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
        raise TrainingError("classification batch does not fit on the selected device")

    epoch_iter = range(1, max_epochs + 1)
    progress = tqdm(
        epoch_iter,
        total=max_epochs,
        desc="classification epochs",
        disable=not show_progress,
        leave=True,
    )
    for epoch in progress:
        train_loss, _, _, steps = _epoch(
            model,
            train_batch,
            device=device,
            batch_size=actual_batch,
            train=True,
            optimizer=optimizer,
            clip_norm=clip_norm,
            rng=rng,
        )
        n_steps += steps
        hold_loss, hold_prob, hold_y, _ = _epoch(
            model,
            hold_batch,
            device=device,
            batch_size=actual_batch,
            train=False,
            rng=np.random.default_rng(seed + epoch),
        )
        hold_auc = _holdout_auc(hold_y, hold_prob)
        monitor = hold_auc if hold_auc is not None else -hold_loss
        row = {
            "epoch": float(epoch),
            "train_loss": float(train_loss),
            "holdout_loss": float(hold_loss),
            "holdout_auc": float("nan") if hold_auc is None else float(hold_auc),
            "batch_size": float(actual_batch),
        }
        history.append(row)
        if epoch_callback is not None:
            epoch_callback(row)
        improved = monitor > best_auc + 1e-12 or (
            abs(monitor - best_auc) <= 1e-12 and hold_loss + 1e-12 < best_loss
        )
        if improved:
            best_auc = float(monitor)
            best_loss = float(hold_loss)
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
                        "holdout_auc": None if hold_auc is None else float(hold_auc),
                        "holdout_loss": best_loss,
                        "model_state": best_state,
                        "encoder": encoder_state,
                    },
                    checkpoint_path,
                )
        else:
            stale += 1
        progress.set_postfix(
            loss=f"{train_loss:.4f}",
            holdout_auc="NA" if hold_auc is None else f"{hold_auc:.5f}",
            best=str(best_epoch),
            stale=str(stale),
        )
        if not improved and stale >= patience:
            break

    if best_state is None:
        raise TrainingError("classification training failed to produce a checkpoint")
    model.load_state_dict(best_state)
    return ClassificationTrainResult(
        best_epoch=best_epoch,
        best_holdout_auc=float(best_auc),
        best_holdout_loss=best_loss,
        history=history,
        batch_size=actual_batch,
        n_steps=n_steps,
    )


def fit_classification_fixed_epochs(
    *,
    model,
    train_table: EncodedClassificationTable,
    device: object,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    clip_norm: float,
    seed: int,
    checkpoint_path: Path | None = None,
    encoder_state: dict[str, Any] | None = None,
    show_progress: bool = False,
) -> FixedEpochFitResult:
    """Fit a fresh model on all outer-train rows for a preselected epoch count."""
    if epochs < 1:
        raise TrainingError("fixed-epoch classification fit requires epochs >= 1")
    torch = require_torch()
    seed_everything(seed, torch_module=torch)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    packed = pack_classification_tensors(train_table)
    rng = np.random.default_rng(seed)
    actual_batch = batch_size
    for size in allowed_batch_sizes(batch_size):
        try:
            index = np.arange(min(size, len(packed.row_ids)))
            mini = _move(packed, device, index)
            model.eval()
            with torch.no_grad():
                output = model(mini.cat_indices)
                binary_bce_with_logits(output.logits, mini.labels)
            actual_batch = size
            break
        except RuntimeError as exc:
            if not _is_oom(exc):
                raise
            mps = getattr(torch, "mps", None)
            if mps is not None and hasattr(mps, "empty_cache"):
                mps.empty_cache()
    else:
        raise TrainingError("fixed-epoch batch does not fit on the selected device")

    history: list[dict[str, float]] = []
    n_steps = 0
    progress = tqdm(
        range(1, epochs + 1),
        total=epochs,
        desc="lookup full refit",
        disable=not show_progress,
        leave=True,
    )
    for epoch in progress:
        train_loss, _, _, steps = _epoch(
            model,
            packed,
            device=device,
            batch_size=actual_batch,
            train=True,
            optimizer=optimizer,
            clip_norm=clip_norm,
            rng=rng,
        )
        n_steps += steps
        history.append({"epoch": float(epoch), "train_loss": float(train_loss)})
        progress.set_postfix(loss=f"{train_loss:.4f}")

    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        torch.save(
            {
                "epoch": epochs,
                "model_state": state,
                "encoder": encoder_state,
                "fit_scope": "full_outer_train",
            },
            checkpoint_path,
        )
    return FixedEpochFitResult(
        epochs=epochs,
        history=history,
        batch_size=actual_batch,
        n_steps=n_steps,
    )


def predict_classifier(
    model,
    table: EncodedClassificationTable,
    *,
    device: object,
    batch_size: int,
) -> np.ndarray:
    torch = require_torch()
    model.eval()
    packed = pack_classification_tensors(table, include_labels=False)
    n_rows = len(packed.row_ids)
    probs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, n_rows, batch_size):
            index = np.arange(start, min(start + batch_size, n_rows))
            mini = _move(packed, device, index)
            output = model(mini.cat_indices)
            probs.append(output.probability.detach().cpu().numpy())
    return np.concatenate(probs, axis=0).astype(np.float64)
