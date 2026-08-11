"""Unit tests for prediction ID alignment helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from smartphone_addiction.alignment import (
    align_predictions_to_ids,
    assert_same_id_order,
    assert_unique_ids,
)
from smartphone_addiction.errors import AlignmentError


def test_assert_unique_ids_rejects_duplicates() -> None:
    with pytest.raises(AlignmentError, match="unique"):
        assert_unique_ids([1, 1, 2], label="id")


def test_assert_same_id_order_detects_permutation() -> None:
    with pytest.raises(AlignmentError, match="order differs"):
        assert_same_id_order([1, 2, 3], [3, 2, 1], label="ids")


def test_align_predictions_to_ids_reorders() -> None:
    base = pd.Series([10, 20, 30])
    pred_ids = pd.Series([30, 10, 20])
    values = np.array([0.3, 0.1, 0.2])
    aligned = align_predictions_to_ids(base, pred_ids, values)
    assert aligned.tolist() == pytest.approx([0.1, 0.2, 0.3])


def test_align_predictions_rejects_missing_id() -> None:
    with pytest.raises(AlignmentError, match="id set mismatch"):
        align_predictions_to_ids([1, 2, 3], [1, 2, 4], np.array([0.1, 0.2, 0.3]))
