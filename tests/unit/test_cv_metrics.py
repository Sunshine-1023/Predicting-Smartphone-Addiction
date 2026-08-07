"""Unit tests for CV fold assignment and OOF metrics."""

from __future__ import annotations

import numpy as np
import pytest

from smartphone_addiction.errors import TrainingError
from smartphone_addiction.evaluation.metrics import summarize_oof
from smartphone_addiction.training.cv import make_folds


def test_folds_are_deterministic_and_cover_every_row_once() -> None:
    y = np.array([0, 1] * 50)
    first = make_folds(y, n_splits=5, seed=42)
    second = make_folds(y, n_splits=5, seed=42)
    assert np.array_equal(first, second)
    assert sorted(np.unique(first).tolist()) == [0, 1, 2, 3, 4]
    assert len(first) == len(y)
    # Each fold appears; each row assigned exactly once (implicit by length + unique folds).
    assert np.bincount(first, minlength=5).sum() == len(y)


def test_oof_summary_reports_auc_and_coverage() -> None:
    y = np.array([0, 0, 1, 1])
    predictions = np.array([0.1, 0.2, 0.8, 0.9])
    summary = summarize_oof(y, predictions)
    assert summary.auc == 1.0
    assert summary.coverage == 1.0
    assert summary.min == 0.1
    assert summary.max == 0.9


def test_make_folds_rejects_single_class() -> None:
    with pytest.raises(TrainingError, match="more than one class"):
        make_folds(np.zeros(20, dtype=int), n_splits=5, seed=42)


def test_summarize_oof_rejects_nonfinite_predictions() -> None:
    y = np.array([0, 1, 0, 1])
    predictions = np.array([0.1, np.nan, 0.8, 0.9])
    with pytest.raises(TrainingError, match="finite"):
        summarize_oof(y, predictions)
