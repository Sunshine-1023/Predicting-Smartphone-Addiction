"""Validate official train/test/sample_submission frames against the competition schema."""

from __future__ import annotations

import numpy as np
import pandas as pd

from smartphone_addiction.data.schema import (
    FEATURE_COLUMNS,
    ID_COLUMN,
    NUMERIC_COLUMNS,
    SAMPLE_SUBMISSION_COLUMNS,
    TARGET_COLUMN,
    TEST_COLUMNS,
    TRAIN_COLUMNS,
)
from smartphone_addiction.errors import DataValidationError


def validate_competition_frames(
    train: pd.DataFrame,
    test: pd.DataFrame,
    sample: pd.DataFrame,
) -> None:
    """Validate exact columns, IDs, target, values, and sample alignment."""
    _require_exact_columns(train, TRAIN_COLUMNS, "train")
    _require_exact_columns(test, TEST_COLUMNS, "test")
    _require_exact_columns(sample, SAMPLE_SUBMISSION_COLUMNS, "sample submission")

    if list(train[FEATURE_COLUMNS].columns) != FEATURE_COLUMNS:
        raise DataValidationError("train feature columns must match the expected order")
    if list(test[FEATURE_COLUMNS].columns) != FEATURE_COLUMNS:
        raise DataValidationError("test feature columns must match the expected order")
    if list(train[FEATURE_COLUMNS].columns) != list(test[FEATURE_COLUMNS].columns):
        raise DataValidationError("train and test feature columns must match")

    _require_unique_ids(train, "train")
    _require_unique_ids(test, "test")
    _require_unique_ids(sample, "sample submission")
    _validate_target(train[TARGET_COLUMN])
    _reject_infinities(train, "train")
    _reject_infinities(test, "test")

    if not sample[ID_COLUMN].equals(test[ID_COLUMN].reset_index(drop=True)):
        raise DataValidationError("sample submission ids must match test ids in the same order")


def _require_exact_columns(frame: pd.DataFrame, expected: list[str], name: str) -> None:
    actual = list(frame.columns)
    if actual != expected:
        raise DataValidationError(f"{name} columns must be exactly {expected}, got {actual}")


def _require_unique_ids(frame: pd.DataFrame, name: str) -> None:
    if frame[ID_COLUMN].isna().any():
        raise DataValidationError(f"{name} id must not contain missing values")
    if not frame[ID_COLUMN].is_unique:
        raise DataValidationError(f"{name} id must be unique")


def _validate_target(target: pd.Series) -> None:
    if target.isna().any():
        raise DataValidationError("addicted_label must not contain missing values")
    unique_values = set(pd.unique(target))
    if not unique_values.issubset({0, 1}):
        raise DataValidationError("addicted_label must contain only integer values 0 and 1")


def _reject_infinities(frame: pd.DataFrame, name: str) -> None:
    numeric_cols = [column for column in NUMERIC_COLUMNS if column in frame.columns]
    if not numeric_cols:
        return
    values = frame[numeric_cols].to_numpy(dtype=float, copy=False)
    if np.isinf(values).any():
        raise DataValidationError(f"{name} numeric columns must not contain infinite values")
