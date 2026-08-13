"""Unit tests for train-fold core-field masking augmentation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from smartphone_addiction.errors import ConfigurationError, TrainingError
from smartphone_addiction.evaluation.slices import CORE_FIELDS
from smartphone_addiction.features.domain import RAW_COLUMNS, add_missingness_features
from smartphone_addiction.training.masking import (
    MaskingSettings,
    apply_core_pattern_mask,
    augment_training_fold,
    core_pattern_keys,
    eligible_complete_indices,
)


def _features(n: int = 40, *, complete: bool = True, seed: int = 0) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    data = {
        "age": rng.normal(30, 5, size=n),
        "daily_screen_time_hours": rng.normal(5, 1, size=n),
        "social_media_hours": rng.normal(2, 0.5, size=n),
        "gaming_hours": rng.normal(1, 0.4, size=n),
        "work_study_hours": rng.normal(2, 0.5, size=n),
        "sleep_hours": rng.normal(7, 1, size=n),
        "notifications_per_day": rng.normal(50, 10, size=n),
        "app_opens_per_day": rng.normal(80, 15, size=n),
        "weekend_screen_time": rng.normal(6, 1, size=n),
        "gender": rng.choice(["Male", "Female"], size=n),
        "stress_level": rng.choice(["Low", "Medium", "High"], size=n),
        "academic_work_impact": rng.choice(["None", "Mild", "Severe"], size=n),
    }
    frame = pd.DataFrame(data)
    if not complete:
        frame.loc[0, "daily_screen_time_hours"] = np.nan
        frame.loc[1, ["weekend_screen_time", "social_media_hours"]] = np.nan
    return add_missingness_features(frame, RAW_COLUMNS)


def test_masking_settings_reject_bad_fraction() -> None:
    with pytest.raises(ConfigurationError, match="fraction"):
        MaskingSettings.from_mapping({"enabled": True, "fraction": 0.0})


def test_apply_core_pattern_mask_sets_nans_and_flags() -> None:
    frame = _features(3, seed=1)
    out = apply_core_pattern_mask(frame, ["01111", "10101", "11111"])
    assert pd.isna(out.loc[0, "daily_screen_time_hours"])
    assert out.loc[0, "daily_screen_time_hours_is_missing"] == 1
    assert out.loc[0, "weekend_screen_time_is_missing"] == 0
    assert pd.isna(out.loc[1, "weekend_screen_time"])
    assert pd.isna(out.loc[1, "work_study_hours"])
    assert out.loc[2, "daily_screen_time_hours"] == frame.loc[2, "daily_screen_time_hours"]
    assert out.loc[0, "missing_count"] >= frame.loc[0, "missing_count"]


def test_augment_preserves_original_and_aligns_labels() -> None:
    train = _features(30, seed=2)
    labels = np.arange(len(train))
    test = _features(20, complete=False, seed=3)
    original = train.copy(deep=True)
    settings = MaskingSettings(enabled=True, fraction=0.2)
    copies, y_copies = augment_training_fold(
        train,
        labels,
        test_features=test,
        settings=settings,
        seed=42,
        fold_id=1,
    )
    pd.testing.assert_frame_equal(train, original)
    assert len(copies) == 6  # round(0.2 * 30)
    assert len(y_copies) == len(copies)
    # Labels come from source rows; values are the original row indices we assigned.
    assert set(y_copies).issubset(set(labels))
    # Copies should no longer be fully core-complete on average.
    assert core_pattern_keys(copies).ne("11111").any()


def test_augment_disabled_returns_empty() -> None:
    train = _features(10, seed=4)
    labels = np.zeros(10, dtype=int)
    test = _features(5, complete=False, seed=5)
    copies, y_copies = augment_training_fold(
        train,
        labels,
        test_features=test,
        settings=MaskingSettings(enabled=False, fraction=0.2),
        seed=1,
        fold_id=0,
    )
    assert len(copies) == 0
    assert len(y_copies) == 0


def test_augment_is_deterministic_for_seed_fold() -> None:
    train = _features(25, seed=6)
    labels = np.ones(25, dtype=int)
    test = _features(15, complete=False, seed=7)
    settings = MaskingSettings(enabled=True, fraction=0.2)
    a_x, a_y = augment_training_fold(
        train, labels, test_features=test, settings=settings, seed=9, fold_id=2
    )
    b_x, b_y = augment_training_fold(
        train, labels, test_features=test, settings=settings, seed=9, fold_id=2
    )
    pd.testing.assert_frame_equal(a_x, b_x)
    np.testing.assert_array_equal(a_y, b_y)


def test_eligible_complete_indices_fallback() -> None:
    frame = _features(5, seed=8)
    # Make all rows miss one core field so complete set is empty; keep >=4 observed.
    frame.loc[:, CORE_FIELDS[0]] = np.nan
    frame = add_missingness_features(frame[RAW_COLUMNS], RAW_COLUMNS)
    idx = eligible_complete_indices(frame)
    assert idx.size == 5


def test_test_without_incomplete_patterns_raises() -> None:
    train = _features(10, seed=9)
    labels = np.zeros(10, dtype=int)
    test = _features(8, seed=10)  # all complete
    with pytest.raises(TrainingError, match="incomplete core missing patterns"):
        augment_training_fold(
            train,
            labels,
            test_features=test,
            settings=MaskingSettings(enabled=True, fraction=0.2),
            seed=1,
            fold_id=0,
        )
