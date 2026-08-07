"""Unit tests for safe_divide and missingness features."""

from __future__ import annotations

import numpy as np
import pandas as pd

from smartphone_addiction.features.domain import add_missingness_features, safe_divide


def test_safe_divide() -> None:
    out = safe_divide(pd.Series([4.0, 2.0, np.nan]), pd.Series([2.0, 0.0, 1.0]))
    assert out.iloc[0] == 2.0
    assert out.iloc[1:].isna().all()


def test_missingness_features() -> None:
    frame = pd.DataFrame(
        {
            "daily_screen_time_hours": [8.0, np.nan],
            "sleep_hours": [np.nan, 7.0],
            "gender": ["Female", None],
        }
    )
    cols = list(frame.columns)
    out = add_missingness_features(frame, cols)
    assert out.loc[0, "missing_count"] == 1
    assert out.loc[0, "sleep_hours_is_missing"] == 1
    assert out.loc[1, "gender_is_missing"] == 1
    assert out.loc[0, "missing_pattern"] == "sleep_hours"
    assert out.loc[1, "missing_pattern"] == "daily_screen_time_hours|gender"
    assert "missing_pattern" in out.columns
