"""Unified train/test feature transform pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from smartphone_addiction.data.schema import (
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    ID_COLUMN,
    TARGET_COLUMN,
)
from smartphone_addiction.errors import DataValidationError
from smartphone_addiction.features.domain import (
    add_behavioral_totals,
    add_categorical_interactions,
    add_log_count_features,
    add_missingness_features,
    add_ratio_and_delta_features,
    fill_categorical_missing,
)

MISSINGNESS_FLAG_COLUMNS: list[str] = [f"{column}_is_missing" for column in FEATURE_COLUMNS]
MISSINGNESS_SUMMARY_COLUMNS: list[str] = [
    "missing_count",
    "missing_ratio",
    "missing_pattern",
]
BEHAVIORAL_TOTAL_COLUMNS: list[str] = [
    "entertainment_hours",
    "work_minus_entertainment",
    "known_usage_hours",
    "unaccounted_screen_time",
]
RATIO_DELTA_COLUMNS: list[str] = [
    "screen_to_sleep_ratio",
    "entertainment_to_screen_ratio",
    "work_to_screen_ratio",
    "weekend_minus_daily",
    "weekend_to_daily_ratio",
    "notifications_per_screen_hour",
    "opens_per_screen_hour",
    "opens_per_notification",
    "notifications_minus_opens",
]
LOG_COLUMNS: list[str] = ["log_notifications", "log_app_opens"]
INTERACTION_COLUMNS: list[str] = [
    "gender_x_stress",
    "gender_x_impact",
    "stress_x_impact",
]

DERIVED_CATEGORICAL_COLUMNS: list[str] = [
    "missing_pattern",
    *INTERACTION_COLUMNS,
]

FEATURE_COLUMN_ORDER: list[str] = [
    *FEATURE_COLUMNS,
    *MISSINGNESS_FLAG_COLUMNS,
    *MISSINGNESS_SUMMARY_COLUMNS,
    *BEHAVIORAL_TOTAL_COLUMNS,
    *RATIO_DELTA_COLUMNS,
    *LOG_COLUMNS,
    *INTERACTION_COLUMNS,
]


@dataclass(frozen=True)
class TransformedFrames:
    train: pd.DataFrame
    test: pd.DataFrame
    feature_columns: list[str]
    categorical_columns: list[str]
    numeric_columns: list[str]


def transform_competition_frames(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> TransformedFrames:
    """Apply identical deterministic transforms to train and test. Never use addicted_label."""
    _require_columns(train, [ID_COLUMN, *FEATURE_COLUMNS, TARGET_COLUMN], "train")
    _require_columns(test, [ID_COLUMN, *FEATURE_COLUMNS], "test")

    train_ids = train[ID_COLUMN].copy()
    test_ids = test[ID_COLUMN].copy()
    train_target = train[TARGET_COLUMN].copy()

    train_features = _transform_feature_frame(train[FEATURE_COLUMNS])
    test_features = _transform_feature_frame(test[FEATURE_COLUMNS])

    if list(train_features.columns) != FEATURE_COLUMN_ORDER:
        raise DataValidationError("train transformed feature order is incorrect")
    if list(test_features.columns) != list(train_features.columns):
        raise DataValidationError("train and test feature columns must match exactly")
    if list(train_features.dtypes) != list(test_features.dtypes):
        raise DataValidationError("train and test feature dtypes must match exactly")

    _assert_no_infinities(train_features, "train")
    _assert_no_infinities(test_features, "test")
    _assert_categoricals_filled(train_features)
    _assert_categoricals_filled(test_features)

    if not train_ids.equals(train[ID_COLUMN]) or len(train_features) != len(train):
        raise DataValidationError("train id order or row count changed during transform")
    if not test_ids.equals(test[ID_COLUMN]) or len(test_features) != len(test):
        raise DataValidationError("test id order or row count changed during transform")

    train_out = pd.concat(
        [
            train_ids.reset_index(drop=True),
            train_features.reset_index(drop=True),
            train_target.reset_index(drop=True),
        ],
        axis=1,
    )
    test_out = pd.concat(
        [
            test_ids.reset_index(drop=True),
            test_features.reset_index(drop=True),
        ],
        axis=1,
    )

    categorical_columns = [
        column
        for column in FEATURE_COLUMN_ORDER
        if column in CATEGORICAL_COLUMNS or column in DERIVED_CATEGORICAL_COLUMNS
    ]
    numeric_columns = [
        column for column in FEATURE_COLUMN_ORDER if column not in categorical_columns
    ]

    return TransformedFrames(
        train=train_out,
        test=test_out,
        feature_columns=list(FEATURE_COLUMN_ORDER),
        categorical_columns=categorical_columns,
        numeric_columns=numeric_columns,
    )


def _transform_feature_frame(features: pd.DataFrame) -> pd.DataFrame:
    frame = features.copy()
    frame = add_missingness_features(frame, FEATURE_COLUMNS)
    frame = fill_categorical_missing(frame, CATEGORICAL_COLUMNS)
    frame = add_behavioral_totals(frame)
    frame = add_ratio_and_delta_features(frame)
    frame = add_log_count_features(frame)
    frame = add_categorical_interactions(frame)
    return frame[FEATURE_COLUMN_ORDER]


def _require_columns(frame: pd.DataFrame, expected: list[str], name: str) -> None:
    missing = [column for column in expected if column not in frame.columns]
    if missing:
        raise DataValidationError(f"{name} is missing required columns: {missing}")


def _assert_no_infinities(frame: pd.DataFrame, name: str) -> None:
    numeric = frame.select_dtypes(include=[np.number])
    if numeric.empty:
        return
    if np.isinf(numeric.to_numpy(dtype=float, copy=False)).any():
        raise DataValidationError(f"{name} features contain infinite values")


def _assert_categoricals_filled(frame: pd.DataFrame) -> None:
    for column in CATEGORICAL_COLUMNS:
        if frame[column].isna().any():
            raise DataValidationError(f"categorical column {column} still contains nulls")
