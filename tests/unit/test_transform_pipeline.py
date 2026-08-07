"""Unit tests for the unified transform pipeline."""

from __future__ import annotations

import numpy as np

from smartphone_addiction.data.schema import CATEGORICAL_COLUMNS
from smartphone_addiction.features.base import transform_competition_frames


def test_label_not_used_in_features(competition_frames) -> None:
    train, test, _ = competition_frames
    result = transform_competition_frames(train, test)
    assert "addicted_label" not in result.feature_columns
    assert result.feature_columns == [c for c in result.test.columns if c != "id"]
    assert list(result.train.columns) == ["id", *result.feature_columns, "addicted_label"]


def test_train_test_feature_alignment(competition_frames) -> None:
    train, test, _ = competition_frames
    result = transform_competition_frames(train, test)
    train_feats = result.train[result.feature_columns]
    test_feats = result.test[result.feature_columns]
    assert list(train_feats.columns) == list(test_feats.columns)
    assert list(train_feats.dtypes) == list(test_feats.dtypes)
    assert len(result.train) == len(train)
    assert len(result.test) == len(test)
    assert result.train["id"].equals(train["id"].reset_index(drop=True))


def test_no_infinities_and_categoricals_filled(competition_frames) -> None:
    train, test, _ = competition_frames
    result = transform_competition_frames(train, test)
    for frame in (result.train, result.test):
        numeric = frame[result.numeric_columns]
        assert not np.isinf(numeric.to_numpy(dtype=float, copy=False)).any()
        for column in CATEGORICAL_COLUMNS:
            assert frame[column].isna().sum() == 0
