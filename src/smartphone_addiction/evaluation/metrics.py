"""Out-of-fold prediction metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score

from smartphone_addiction.errors import TrainingError


@dataclass(frozen=True)
class OOFSummary:
    """Summary statistics for one OOF prediction vector."""

    auc: float
    coverage: float
    min: float
    max: float
    mean: float
    std: float


def summarize_oof(y: np.ndarray, predictions: np.ndarray) -> OOFSummary:
    """Compute ROC-AUC and basic prediction stats for OOF scores.

    Rejects missing or non-finite predictions. Coverage is the fraction of rows
    with a finite prediction (1.0 when all predictions are accepted).
    """
    labels = np.asarray(y, dtype=float)
    preds = np.asarray(predictions, dtype=float)
    if labels.shape != preds.shape:
        raise TrainingError("y and predictions must have the same shape")
    if labels.size == 0:
        raise TrainingError("cannot summarize empty OOF predictions")

    finite = np.isfinite(preds)
    if not finite.all():
        raise TrainingError("OOF predictions must be finite (no NaN or infinity)")
    if not np.isfinite(labels).all():
        raise TrainingError("OOF labels must be finite")

    coverage = float(finite.mean())
    try:
        auc = float(roc_auc_score(labels, preds))
    except ValueError as exc:
        raise TrainingError(f"unable to compute ROC-AUC: {exc}") from exc

    return OOFSummary(
        auc=auc,
        coverage=coverage,
        min=float(preds.min()),
        max=float(preds.max()),
        mean=float(preds.mean()),
        std=float(preds.std(ddof=0)),
    )
