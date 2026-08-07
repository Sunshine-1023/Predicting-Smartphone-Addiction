"""Unit tests for feature-group selection via build_features."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from smartphone_addiction.features.domain import build_features, safe_divide


def test_safe_divide_rejects_zero_and_preserves_missing() -> None:
    result = safe_divide(
        pd.Series([4.0, 2.0, np.nan]),
        pd.Series([2.0, 0.0, 1.0]),
    )
    assert result.iloc[0] == 2.0
    assert result.iloc[1:].isna().all()


def test_feature_groups_are_explicit() -> None:
    frame = pd.DataFrame(
        {
            "daily_screen_time_hours": [8.0],
            "social_media_hours": [2.0],
            "gaming_hours": [1.0],
            "work_study_hours": [3.0],
            "sleep_hours": [4.0],
            "notifications_per_day": [80.0],
            "app_opens_per_day": [40.0],
            "weekend_screen_time": [10.0],
            "age": [20.0],
            "gender": ["Female"],
            "stress_level": ["Medium"],
            "academic_work_impact": ["No"],
        }
    )
    result = build_features(frame, ["raw", "missingness", "behavioral_ratios"])
    assert result.loc[0, "screen_to_sleep_ratio"] == 2.0
    assert result.loc[0, "missing_count"] == 0
    assert "entertainment_hours" not in result.columns
    assert "weekend_minus_daily" not in result.columns


def test_unknown_feature_group_rejected() -> None:
    frame = pd.DataFrame(
        {
            "daily_screen_time_hours": [8.0],
            "social_media_hours": [2.0],
            "gaming_hours": [1.0],
            "work_study_hours": [3.0],
            "sleep_hours": [4.0],
            "notifications_per_day": [80.0],
            "app_opens_per_day": [40.0],
            "weekend_screen_time": [10.0],
            "age": [20.0],
            "gender": ["Female"],
            "stress_level": ["Medium"],
            "academic_work_impact": ["No"],
        }
    )
    with pytest.raises(ValueError, match="unknown feature groups"):
        build_features(frame, ["raw", "magic"])


def test_build_features_does_not_mutate_input() -> None:
    frame = pd.DataFrame(
        {
            "daily_screen_time_hours": [8.0],
            "social_media_hours": [2.0],
            "gaming_hours": [1.0],
            "work_study_hours": [3.0],
            "sleep_hours": [4.0],
            "notifications_per_day": [80.0],
            "app_opens_per_day": [40.0],
            "weekend_screen_time": [10.0],
            "age": [20.0],
            "gender": ["Female"],
            "stress_level": [None],
            "academic_work_impact": ["No"],
        }
    )
    original = frame.copy()
    _ = build_features(frame, ["raw", "missingness"])
    assert frame.equals(original)


def test_select_feature_columns_allows_subset_of_available() -> None:
    from smartphone_addiction.features.base import select_feature_columns_from_groups
    from smartphone_addiction.features.domain import ALL_FEATURE_GROUPS, columns_for_groups

    available = columns_for_groups(ALL_FEATURE_GROUPS)
    selected = select_feature_columns_from_groups(available, ["raw"])
    assert selected == columns_for_groups(["raw"])


def test_select_feature_columns_requires_configured_groups() -> None:
    from smartphone_addiction.errors import DataValidationError
    from smartphone_addiction.features.base import select_feature_columns_from_groups
    from smartphone_addiction.features.domain import columns_for_groups

    available = columns_for_groups(["raw"])
    with pytest.raises(DataValidationError, match="missing columns"):
        select_feature_columns_from_groups(available, ["raw", "missingness"])


def test_select_feature_columns_can_disable_require_all() -> None:
    from smartphone_addiction.features.base import select_feature_columns_from_groups
    from smartphone_addiction.features.domain import columns_for_groups

    available = columns_for_groups(["raw"])
    selected = select_feature_columns_from_groups(
        available,
        ["raw", "missingness"],
        require_all=False,
    )
    assert selected == available
