"""Deterministic stratified cross-validation fold assignment."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedKFold

from smartphone_addiction.errors import TrainingError


def make_folds(y: np.ndarray, n_splits: int = 5, seed: int = 42) -> np.ndarray:
    """Return an integer fold id for every row using StratifiedKFold.

    The same (y, n_splits, seed) always yields the same assignment. Each row
    appears in exactly one validation fold.
    """
    labels = np.asarray(y)
    if labels.ndim != 1:
        raise TrainingError("target for fold assignment must be one-dimensional")
    if n_splits < 2:
        raise TrainingError("n_splits must be at least 2")
    if len(labels) < n_splits:
        raise TrainingError("not enough rows for the requested number of folds")
    if len(np.unique(labels)) < 2:
        raise TrainingError("target must contain more than one class for stratified folds")

    fold_ids = np.empty(len(labels), dtype=np.int32)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    # Feature matrix is unused by StratifiedKFold beyond length.
    dummy = np.zeros((len(labels), 1), dtype=np.float32)
    for fold_id, (_, valid_index) in enumerate(splitter.split(dummy, labels)):
        fold_ids[valid_index] = fold_id
    return fold_ids
