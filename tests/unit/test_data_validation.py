"""Unit tests for competition data validation."""

from __future__ import annotations

import pytest

from smartphone_addiction.data.validate import validate_competition_frames
from smartphone_addiction.errors import DataValidationError


def test_valid_frames_pass(competition_frames) -> None:
    train, test, sample = competition_frames
    validate_competition_frames(train, test, sample)


def test_duplicate_train_id_fails(competition_frames) -> None:
    train, test, sample = competition_frames
    train = train.copy()
    train.loc[1, "id"] = train.loc[0, "id"]
    with pytest.raises(DataValidationError, match="unique"):
        validate_competition_frames(train, test, sample)


def test_target_must_be_binary(competition_frames) -> None:
    train, test, sample = competition_frames
    train = train.copy()
    train.loc[0, "addicted_label"] = 2
    with pytest.raises(DataValidationError, match="addicted_label"):
        validate_competition_frames(train, test, sample)


def test_sample_order_must_match_test(competition_frames) -> None:
    train, test, sample = competition_frames
    reversed_sample = sample.iloc[::-1].reset_index(drop=True)
    with pytest.raises(DataValidationError, match="sample submission ids"):
        validate_competition_frames(train, test, reversed_sample)
