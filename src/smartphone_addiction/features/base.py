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
    ALL_FEATURE_GROUPS,
    INTERACTION_COLUMNS,
    build_features,
    columns_for_groups,
    normalize_feature_groups,
)

DERIVED_CATEGORICAL_COLUMNS: list[str] = [
    "missing_pattern",
    *INTERACTION_COLUMNS,
]

FEATURE_COLUMN_ORDER: list[str] = columns_for_groups(ALL_FEATURE_GROUPS)


@dataclass(frozen=True)
class TransformedFrames:
    train: pd.DataFrame
    test: pd.DataFrame
    feature_columns: list[str]
    categorical_columns: list[str]
    numeric_columns: list[str]
    feature_groups: list[str]


def transform_competition_frames(
    train: pd.DataFrame,
    test: pd.DataFrame,
    groups: list[str] | None = None,
) -> TransformedFrames:
    """Apply identical deterministic transforms to train and test.

    ``groups`` selects feature groups (see ``normalize_feature_groups``).
    ``None`` keeps the full production feature set. Never uses addicted_label.
    """
    _require_columns(train, [ID_COLUMN, *FEATURE_COLUMNS, TARGET_COLUMN], "train")
    _require_columns(test, [ID_COLUMN, *FEATURE_COLUMNS], "test")
    selected_groups = normalize_feature_groups(groups)
    feature_columns = columns_for_groups(selected_groups)

    train_ids = train[ID_COLUMN].copy()
    test_ids = test[ID_COLUMN].copy()
    train_target = train[TARGET_COLUMN].copy()

    try:
        train_features = build_features(train[FEATURE_COLUMNS], selected_groups)
        test_features = build_features(test[FEATURE_COLUMNS], selected_groups)
    except ValueError as exc:
        raise DataValidationError(str(exc)) from exc

    if list(train_features.columns) != feature_columns:
        raise DataValidationError("train transformed feature order is incorrect")
    if list(test_features.columns) != list(train_features.columns):
        raise DataValidationError("train and test feature columns must match exactly")
    if list(train_features.dtypes) != list(test_features.dtypes):
        raise DataValidationError("train and test feature dtypes must match exactly")

    _assert_no_infinities(train_features, "train")
    _assert_no_infinities(test_features, "test")
    _assert_categoricals_filled(train_features)

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
        for column in feature_columns
        if column in CATEGORICAL_COLUMNS or column in DERIVED_CATEGORICAL_COLUMNS
    ]
    numeric_columns = [column for column in feature_columns if column not in categorical_columns]

    return TransformedFrames(
        train=train_out,
        test=test_out,
        feature_columns=list(feature_columns),
        categorical_columns=categorical_columns,
        numeric_columns=numeric_columns,
        feature_groups=list(selected_groups),
    )


def select_feature_columns_from_groups(
    available_columns: list[str],
    groups: list[str] | None,
    *,
    require_all: bool = True,
) -> list[str]:
    """Filter an already-built feature list down to the requested groups.

    When ``require_all`` is True (default), raise if any column required by
    ``groups`` is missing from ``available_columns``. This prevents silently
    training on a raw-only parquet while the config claims domain features.
    """
    wanted = columns_for_groups(groups)
    available = set(available_columns)
    missing = [column for column in wanted if column not in available]
    if missing and require_all:
        raise DataValidationError(
            "processed features missing columns required by config.features.groups: "
            f"{missing}. Rebuild with matching --group flags, or narrow features.groups."
        )
    wanted_set = set(wanted)
    return [column for column in available_columns if column in wanted_set]


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
        if column not in frame.columns:
            continue
        if frame[column].isna().any():
            raise DataValidationError(f"categorical column {column} still contains nulls")
