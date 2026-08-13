"""Unit tests for OOF slice metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from smartphone_addiction.evaluation.slices import CORE_FIELDS, compute_slice_metrics


def _frame_with_core(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_slice_metrics_complete_and_incomplete() -> None:
    frame = _frame_with_core(
        [
            {name: 1.0 for name in CORE_FIELDS},
            {name: 1.0 for name in CORE_FIELDS},
            {name: np.nan if name == "gaming_hours" else 1.0 for name in CORE_FIELDS},
            {name: np.nan if name == "gaming_hours" else 2.0 for name in CORE_FIELDS},
        ]
    )
    y = np.array([0, 1, 0, 1])
    pred = np.array([0.1, 0.9, 0.2, 0.8])
    metrics = compute_slice_metrics(frame, y, pred)
    assert metrics["n_core_complete"] == 2
    assert metrics["n_core_incomplete"] == 2
    assert metrics["core_complete_auc"] == pytest.approx(1.0)
    assert metrics["core_incomplete_auc"] == pytest.approx(1.0)
    assert metrics["overall_auc"] == pytest.approx(1.0)


def test_slice_metrics_by_observed_count_and_single_class() -> None:
    frame = _frame_with_core(
        [
            {name: 1.0 for name in CORE_FIELDS},
            {name: np.nan for name in CORE_FIELDS},
            {name: np.nan for name in CORE_FIELDS},
        ]
    )
    y = np.array([1, 0, 0])
    pred = np.array([0.9, 0.1, 0.2])
    metrics = compute_slice_metrics(frame, y, pred)
    assert metrics["auc_by_core_observed_count"]["5"] is None  # single label
    assert metrics["auc_by_core_observed_count"]["0"] is None  # single label
    assert metrics["n_by_core_observed_count"]["5"] == 1
    assert metrics["n_by_core_observed_count"]["0"] == 2


def test_slice_metrics_does_not_reorder_rows() -> None:
    frame = _frame_with_core(
        [
            {name: 1.0 for name in CORE_FIELDS},
            {name: np.nan if name == "work_study_hours" else 1.0 for name in CORE_FIELDS},
        ]
    )
    original = frame.copy(deep=True)
    y = np.array([0, 1])
    pred = np.array([0.2, 0.8])
    compute_slice_metrics(frame, y, pred)
    pd.testing.assert_frame_equal(frame, original)


def test_test_pattern_weighted_auc_uses_test_frequencies() -> None:
    train = _frame_with_core(
        [
            {name: 1.0 for name in CORE_FIELDS},
            {name: 1.0 for name in CORE_FIELDS},
            {name: np.nan if name == "gaming_hours" else 1.0 for name in CORE_FIELDS},
            {name: np.nan if name == "gaming_hours" else 2.0 for name in CORE_FIELDS},
        ]
    )
    test = _frame_with_core(
        [
            {name: np.nan if name == "gaming_hours" else 1.0 for name in CORE_FIELDS},
            {name: np.nan if name == "gaming_hours" else 1.0 for name in CORE_FIELDS},
            {name: 1.0 for name in CORE_FIELDS},
        ]
    )
    y = np.array([0, 1, 0, 1])
    pred = np.array([0.1, 0.9, 0.2, 0.8])
    metrics = compute_slice_metrics(train, y, pred, test_features=test)
    assert metrics["test_pattern_weighted_auc"] is not None
    assert 0.0 <= metrics["test_pattern_weighted_auc"] <= 1.0
