"""Unit tests for domain behavioral, ratio, log, and interaction features."""

from __future__ import annotations

import numpy as np
import pandas as pd

from smartphone_addiction.features.domain import (
    MISSING_TOKEN,
    add_behavioral_totals,
    add_categorical_interactions,
    add_log_count_features,
    add_ratio_and_delta_features,
    fill_categorical_missing,
)


def test_fill_categorical_missing() -> None:
    frame = pd.DataFrame({"gender": ["Female", None], "stress_level": [None, "High"]})
    out = fill_categorical_missing(frame, ["gender", "stress_level"])
    assert out.loc[0, "gender"] == "Female"
    assert out.loc[1, "gender"] == MISSING_TOKEN
    assert out.loc[0, "stress_level"] == MISSING_TOKEN


def test_behavioral_totals_preserve_missing_and_allow_negative() -> None:
    frame = pd.DataFrame(
        {
            "social_media_hours": [2.0, np.nan, 1.0],
            "gaming_hours": [1.0, 1.0, 1.0],
            "work_study_hours": [5.0, 2.0, 0.5],
            "daily_screen_time_hours": [4.0, 5.0, 2.0],
        }
    )
    out = add_behavioral_totals(frame)
    assert out.loc[0, "entertainment_hours"] == 3.0
    assert out.loc[0, "work_minus_entertainment"] == 2.0
    assert out.loc[0, "known_usage_hours"] == 8.0
    assert out.loc[0, "unaccounted_screen_time"] == -4.0
    assert pd.isna(out.loc[1, "entertainment_hours"])
    assert pd.isna(out.loc[1, "work_minus_entertainment"])
    assert pd.isna(out.loc[1, "unaccounted_screen_time"])
    # work 0.5 - entertainment 2.0 = -1.5 (negative gap kept)
    assert out.loc[2, "work_minus_entertainment"] == -1.5


def test_ratio_features_use_safe_divide() -> None:
    frame = pd.DataFrame(
        {
            "daily_screen_time_hours": [8.0, 4.0, np.nan],
            "sleep_hours": [4.0, 0.0, 4.0],
            "weekend_screen_time": [10.0, 2.0, 3.0],
            "notifications_per_day": [80.0, 10.0, 5.0],
            "app_opens_per_day": [40.0, 5.0, 1.0],
            "entertainment_hours": [3.0, 1.0, 1.0],
            "work_study_hours": [2.0, 1.0, 1.0],
        }
    )
    out = add_ratio_and_delta_features(frame)
    assert out.loc[0, "screen_to_sleep_ratio"] == 2.0
    assert pd.isna(out.loc[1, "screen_to_sleep_ratio"])  # sleep denominator is 0
    assert pd.isna(out.loc[2, "weekend_minus_daily"])
    assert out.loc[0, "weekend_minus_daily"] == 2.0
    assert out.loc[0, "opens_per_notification"] == 0.5


def test_log_features_preserve_originals() -> None:
    frame = pd.DataFrame(
        {
            "notifications_per_day": [0.0, np.nan, 99.0],
            "app_opens_per_day": [1.0, 2.0, np.nan],
        }
    )
    out = add_log_count_features(frame)
    assert out.loc[0, "log_notifications"] == 0.0
    assert pd.isna(out.loc[1, "log_notifications"])
    assert "notifications_per_day" in out.columns
    assert "app_opens_per_day" in out.columns


def test_categorical_interactions() -> None:
    frame = pd.DataFrame(
        {
            "gender": [MISSING_TOKEN, "Male"],
            "stress_level": ["High", "Low"],
            "academic_work_impact": ["Yes", "No"],
        }
    )
    out = add_categorical_interactions(frame)
    assert out.loc[0, "gender_x_stress"] == f"{MISSING_TOKEN}_High"
    assert out.loc[1, "stress_x_impact"] == "Low_No"
