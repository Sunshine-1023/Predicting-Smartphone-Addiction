"""Unit tests for fold-local neural tensorizer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from smartphone_addiction.data.schema import ID_COLUMN, NUMERIC_COLUMNS
from smartphone_addiction.neural.config import UNKNOWN_TOKEN
from smartphone_addiction.neural.preprocessing import FoldTensorizer


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            ID_COLUMN: [1, 2, 3, 4],
            "age": [20.0, 30.0, np.nan, 40.0],
            "daily_screen_time_hours": [8.0, 4.0, 6.0, 10.0],
            "social_media_hours": [2.0, 1.0, np.nan, 3.0],
            "gaming_hours": [1.0, 0.5, 0.2, 1.5],
            "work_study_hours": [3.0, 2.0, 4.0, 1.0],
            "sleep_hours": [7.0, 6.0, 8.0, 5.0],
            "notifications_per_day": [40.0, 80.0, 20.0, 60.0],
            "app_opens_per_day": [10.0, 20.0, 30.0, 40.0],
            "weekend_screen_time": [9.0, 5.0, 7.0, 11.0],
            "gender": ["Male", "Female", np.nan, "Male"],
            "stress_level": ["Low", "High", "Medium", "Low"],
            "academic_work_impact": ["No", "Yes", "No", "Yes"],
            "entertainment_hours": [99.0, 99.0, 99.0, 99.0],
        }
    )


def test_valid_mutation_does_not_change_fitted_stats() -> None:
    train = _frame()
    tensorizer = FoldTensorizer().fit(train)
    mean_before = tensorizer.mean_.copy()
    valid = _frame()
    valid.loc[:, "age"] = 999.0
    tensorizer.transform(valid)
    np.testing.assert_allclose(tensorizer.mean_, mean_before)


def test_unseen_valid_category_maps_to_unknown() -> None:
    train = _frame()
    tensorizer = FoldTensorizer().fit(train)
    valid = _frame()
    valid.loc[0, "gender"] = "NonBinary"
    encoded = tensorizer.transform(valid)
    unknown = tensorizer.vocabs_["gender"][UNKNOWN_TOKEN]
    assert int(encoded.categorical[0, 0].item()) == unknown
    assert "NonBinary" not in tensorizer.vocabs_["gender"]


def test_transform_has_no_nan_or_inf_and_keeps_row_ids() -> None:
    tensorizer = FoldTensorizer().fit(_frame())
    encoded = tensorizer.transform(_frame())
    assert torch.isfinite(encoded.numeric).all()
    assert list(encoded.row_ids) == [1, 2, 3, 4]
    assert encoded.numeric.shape[1] == len(NUMERIC_COLUMNS)
    assert encoded.natural_observed.dtype == torch.bool


def test_derived_columns_are_not_neural_inputs() -> None:
    tensorizer = FoldTensorizer().fit(_frame())
    assert "entertainment_hours" not in tensorizer.numeric_columns
