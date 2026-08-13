"""OOF slice metrics for core-field completeness and missingness patterns."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

CORE_FIELDS: tuple[str, ...] = (
    "daily_screen_time_hours",
    "weekend_screen_time",
    "social_media_hours",
    "work_study_hours",
    "gaming_hours",
)

TOP3_CORE_FIELDS: tuple[str, ...] = (
    "daily_screen_time_hours",
    "weekend_screen_time",
    "social_media_hours",
)


def core_observed_count(frame: pd.DataFrame, fields: tuple[str, ...] = CORE_FIELDS) -> pd.Series:
    """Count non-missing values among the configured core fields (row-wise)."""
    missing = [name for name in fields if name not in frame.columns]
    if missing:
        raise ValueError(f"frame missing core fields: {missing}")
    return frame.loc[:, list(fields)].notna().sum(axis=1)


def _safe_auc(y: np.ndarray, prediction: np.ndarray) -> float | None:
    labels = np.asarray(y)
    preds = np.asarray(prediction, dtype=float)
    if len(labels) == 0 or len(labels) != len(preds):
        return None
    if not np.isfinite(preds).all():
        return None
    if len(np.unique(labels)) < 2:
        return None
    return float(roc_auc_score(labels, preds))


def _pattern_key(frame: pd.DataFrame, fields: tuple[str, ...] = CORE_FIELDS) -> pd.Series:
    bits = frame.loc[:, list(fields)].notna().astype(int).astype(str)
    return bits.agg("".join, axis=1)


def compute_slice_metrics(
    features: pd.DataFrame,
    y: np.ndarray | pd.Series,
    prediction: np.ndarray | pd.Series,
    *,
    test_features: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Compute completeness / missingness slice AUCs without changing row order.

    When a slice has fewer than two label classes, its AUC is ``None``.
    ``test_pattern_weighted_auc`` weights train-pattern AUCs by the missingness
    pattern frequencies observed in ``test_features`` (falls back to train
    frequencies when test is omitted).
    """
    labels = np.asarray(y)
    preds = np.asarray(prediction, dtype=float)
    if len(features) != len(labels) or len(labels) != len(preds):
        raise ValueError("features, y, and prediction must share the same length")

    observed = core_observed_count(features, CORE_FIELDS)
    top3_observed = core_observed_count(features, TOP3_CORE_FIELDS)
    complete_mask = observed == len(CORE_FIELDS)
    incomplete_mask = ~complete_mask
    top3_incomplete_mask = top3_observed < len(TOP3_CORE_FIELDS)

    by_count: dict[str, float | None] = {}
    n_by_count: dict[str, int] = {}
    for count in range(len(CORE_FIELDS) + 1):
        mask = observed == count
        by_count[str(count)] = _safe_auc(labels[mask], preds[mask])
        n_by_count[str(count)] = int(mask.sum())

    patterns = _pattern_key(features, CORE_FIELDS)
    pattern_aucs: dict[str, float | None] = {}
    pattern_counts: dict[str, int] = {}
    for key, index in patterns.groupby(patterns).groups.items():
        idx = np.asarray(list(index))
        pattern_aucs[str(key)] = _safe_auc(labels[idx], preds[idx])
        pattern_counts[str(key)] = len(idx)

    weight_source = test_features if test_features is not None else features
    test_patterns = _pattern_key(weight_source, CORE_FIELDS)
    test_freq = test_patterns.value_counts(normalize=True).to_dict()
    weighted_num = 0.0
    weighted_den = 0.0
    for key, weight in test_freq.items():
        auc = pattern_aucs.get(str(key))
        if auc is None:
            continue
        weighted_num += float(weight) * auc
        weighted_den += float(weight)
    test_pattern_weighted = (weighted_num / weighted_den) if weighted_den > 0 else None

    return {
        "n_rows": len(features),
        "core_fields": list(CORE_FIELDS),
        "top3_core_fields": list(TOP3_CORE_FIELDS),
        "overall_auc": _safe_auc(labels, preds),
        "core_complete_auc": _safe_auc(labels[complete_mask], preds[complete_mask]),
        "core_incomplete_auc": _safe_auc(labels[incomplete_mask], preds[incomplete_mask]),
        "top3_incomplete_auc": _safe_auc(labels[top3_incomplete_mask], preds[top3_incomplete_mask]),
        "n_core_complete": int(complete_mask.sum()),
        "n_core_incomplete": int(incomplete_mask.sum()),
        "n_top3_incomplete": int(top3_incomplete_mask.sum()),
        "auc_by_core_observed_count": by_count,
        "n_by_core_observed_count": n_by_count,
        "test_pattern_weighted_auc": test_pattern_weighted,
        "pattern_aucs": pattern_aucs,
        "pattern_counts": pattern_counts,
    }
