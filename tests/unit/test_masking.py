"""Unit tests for configurable train-fold masking augmentation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from smartphone_addiction.errors import ConfigurationError, TrainingError
from smartphone_addiction.features.domain import (
    RAW_COLUMNS,
    add_log_count_features,
    add_missingness_features,
)
from smartphone_addiction.training.masking import (
    CORE5_FIELDS,
    MASK_FIELDS,
    TOP8_FIELDS,
    MaskingSettings,
    apply_core_pattern_mask,
    augment_training_fold,
    compatible_source_indices,
    concat_fit_sample_weight,
    mask_pattern_keys,
    pattern_bit_ints,
)

_ALL_OBSERVED_CORE5 = "1" * len(CORE5_FIELDS)
_ALL_OBSERVED_TOP8 = "1" * len(TOP8_FIELDS)


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


def _m1_settings(**overrides: object) -> MaskingSettings:
    payload = {
        "enabled": True,
        "fraction": 0.2,
        "fields": "core5",
        "compatible_sources": True,
        "sample_weight": False,
    }
    payload.update(overrides)
    return MaskingSettings.from_mapping(payload)


def test_masking_settings_reject_bad_fraction() -> None:
    with pytest.raises(ConfigurationError, match="fraction"):
        MaskingSettings.from_mapping({"enabled": True, "fraction": 0.0})


def test_masking_settings_reject_unknown_fields() -> None:
    with pytest.raises(ConfigurationError, match="fields"):
        MaskingSettings.from_mapping({"enabled": True, "fields": "all"})


def test_masking_settings_defaults_match_champion_v2() -> None:
    settings = MaskingSettings.from_mapping({"enabled": True, "fraction": 0.20})
    assert settings.fields == "core5"
    assert settings.compatible_sources is False
    assert settings.sample_weight is False
    assert settings.mask_fields == CORE5_FIELDS


def test_masking_settings_to_dict_includes_knobs() -> None:
    payload = _m1_settings(fields="top8", sample_weight=True).to_dict()
    assert payload["enabled"] is True
    assert payload["fraction"] == 0.2
    assert payload["fields"] == "top8"
    assert payload["field_names"] == list(TOP8_FIELDS) == list(MASK_FIELDS)
    assert payload["compatible_sources"] is True
    assert payload["sample_weight"] is True
    assert payload["version"] == 3
    config = MaskingSettings.from_mapping(payload).to_config_dict()
    assert config["fields"] == "top8"
    assert "field_names" not in config
    assert "version" not in config


def test_apply_core_pattern_mask_sets_nans_and_flags() -> None:
    frame = _features(3, seed=1)
    out = apply_core_pattern_mask(frame, ["01111111", "10101111", "11111111"], TOP8_FIELDS)
    assert pd.isna(out.loc[0, "daily_screen_time_hours"])
    assert out.loc[0, "daily_screen_time_hours_is_missing"] == 1
    assert out.loc[0, "weekend_screen_time_is_missing"] == 0
    assert pd.isna(out.loc[1, "weekend_screen_time"])
    assert pd.isna(out.loc[1, "work_study_hours"])
    assert out.loc[2, "daily_screen_time_hours"] == frame.loc[2, "daily_screen_time_hours"]
    assert out.loc[0, "missing_count"] >= frame.loc[0, "missing_count"]


def test_apply_mask_refreshes_log_counts() -> None:
    frame = add_log_count_features(_features(2, seed=11))
    original_log = float(frame.loc[1, "log_notifications"])
    out = apply_core_pattern_mask(frame, ["11111011", "11111111"], TOP8_FIELDS)
    assert pd.isna(out.loc[0, "notifications_per_day"])
    assert pd.isna(out.loc[0, "log_notifications"])
    assert out.loc[1, "log_notifications"] == original_log


def test_augment_preserves_original_and_aligns_labels() -> None:
    train = _features(30, seed=2)
    labels = np.arange(len(train))
    test = _features(20, complete=False, seed=3)
    original = train.copy(deep=True)
    settings = _m1_settings()
    copies, y_copies, weights = augment_training_fold(
        train,
        labels,
        test_features=test,
        settings=settings,
        seed=42,
        fold_id=1,
    )
    pd.testing.assert_frame_equal(train, original)
    assert len(copies) == 6  # round(0.2 * 30)
    assert len(y_copies) == len(copies) == len(weights)
    assert set(y_copies).issubset(set(labels))
    assert mask_pattern_keys(copies, CORE5_FIELDS).ne(_ALL_OBSERVED_CORE5).any()
    np.testing.assert_array_equal(weights, np.ones(len(copies)))


def test_augment_disabled_returns_empty() -> None:
    train = _features(10, seed=4)
    labels = np.zeros(10, dtype=int)
    test = _features(5, complete=False, seed=5)
    copies, y_copies, weights = augment_training_fold(
        train,
        labels,
        test_features=test,
        settings=MaskingSettings(enabled=False, fraction=0.2),
        seed=1,
        fold_id=0,
    )
    assert len(copies) == 0
    assert len(y_copies) == 0
    assert len(weights) == 0


def test_augment_is_deterministic_for_seed_fold() -> None:
    train = _features(25, seed=6)
    labels = np.ones(25, dtype=int)
    test = _features(15, complete=False, seed=7)
    settings = _m1_settings()
    a_x, a_y, a_w = augment_training_fold(
        train, labels, test_features=test, settings=settings, seed=9, fold_id=2
    )
    b_x, b_y, b_w = augment_training_fold(
        train, labels, test_features=test, settings=settings, seed=9, fold_id=2
    )
    pd.testing.assert_frame_equal(a_x, b_x)
    np.testing.assert_array_equal(a_y, b_y)
    np.testing.assert_allclose(a_w, b_w)


def test_compatible_source_allows_incomplete_superset() -> None:
    train = _features(8, seed=12)
    train.loc[0, "gaming_hours"] = np.nan
    train = add_missingness_features(train[RAW_COLUMNS], RAW_COLUMNS)
    target = "11010111"
    bits = pattern_bit_ints(train, TOP8_FIELDS)
    compatible = compatible_source_indices(bits, target)
    assert 0 in set(compatible.tolist())
    needs_gaming = "11011111"
    compatible_strict = compatible_source_indices(bits, needs_gaming)
    assert 0 not in set(compatible_strict.tolist())


def test_augment_can_mask_additional_field_on_incomplete_source() -> None:
    train = _features(12, seed=13)
    train.loc[:, "gaming_hours"] = np.nan
    train = add_missingness_features(train[RAW_COLUMNS], RAW_COLUMNS)
    test = _features(8, seed=14)
    test.loc[:, ["social_media_hours", "gaming_hours"]] = np.nan
    test = add_missingness_features(test[RAW_COLUMNS], RAW_COLUMNS)
    copies, _, _ = augment_training_fold(
        train,
        np.ones(len(train), dtype=int),
        test_features=test,
        settings=_m1_settings(fraction=0.5),
        seed=3,
        fold_id=0,
    )
    assert len(copies) > 0
    assert copies["gaming_hours"].isna().all()
    assert copies["social_media_hours"].isna().any()


def test_complete_sources_reject_incomplete_rows() -> None:
    train = _features(12, seed=13)
    train.loc[:, "gaming_hours"] = np.nan
    train = add_missingness_features(train[RAW_COLUMNS], RAW_COLUMNS)
    test = _features(8, seed=14)
    test.loc[:, ["social_media_hours", "gaming_hours"]] = np.nan
    test = add_missingness_features(test[RAW_COLUMNS], RAW_COLUMNS)
    with pytest.raises(TrainingError, match="no complete train rows"):
        augment_training_fold(
            train,
            np.ones(len(train), dtype=int),
            test_features=test,
            settings=_m1_settings(compatible_sources=False, fraction=0.5),
            seed=3,
            fold_id=0,
        )


def test_sample_weight_scales_copies_when_enabled() -> None:
    train = _features(30, seed=2)
    test = _features(20, complete=False, seed=3)
    copies, _, weights = augment_training_fold(
        train,
        np.ones(len(train), dtype=int),
        test_features=test,
        settings=_m1_settings(sample_weight=True),
        seed=42,
        fold_id=1,
    )
    assert len(copies) == 6
    assert np.all(weights > 0)
    assert float(weights.mean()) < 1.0
    assert float(weights.sum()) < len(train)
    fit_w = concat_fit_sample_weight(len(train), weights, _m1_settings(sample_weight=True))
    assert fit_w is not None
    assert len(fit_w) == len(train) + len(copies)
    assert concat_fit_sample_weight(len(train), weights, _m1_settings()) is None


def test_top8_patterns_are_longer_than_core5() -> None:
    train = _features(20, seed=8)
    test = _features(12, complete=False, seed=9)
    core5, _, _ = augment_training_fold(
        train,
        np.ones(len(train), dtype=int),
        test_features=test,
        settings=_m1_settings(),
        seed=1,
        fold_id=0,
    )
    top8, _, _ = augment_training_fold(
        train,
        np.ones(len(train), dtype=int),
        test_features=test,
        settings=_m1_settings(fields="top8"),
        seed=1,
        fold_id=0,
    )
    assert mask_pattern_keys(core5, CORE5_FIELDS).str.len().eq(5).all()
    assert mask_pattern_keys(top8, TOP8_FIELDS).str.len().eq(8).all()
    assert mask_pattern_keys(core5, CORE5_FIELDS).ne(_ALL_OBSERVED_CORE5).any()
    assert mask_pattern_keys(top8, TOP8_FIELDS).ne(_ALL_OBSERVED_TOP8).any()


def test_test_without_incomplete_patterns_raises() -> None:
    train = _features(10, seed=9)
    labels = np.zeros(10, dtype=int)
    test = _features(8, seed=10)  # all complete
    with pytest.raises(TrainingError, match="incomplete missing patterns"):
        augment_training_fold(
            train,
            labels,
            test_features=test,
            settings=_m1_settings(),
            seed=1,
            fold_id=0,
        )
