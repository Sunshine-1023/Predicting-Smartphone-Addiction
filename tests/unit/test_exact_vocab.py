"""Exact-value vocabulary mapping and fold isolation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from smartphone_addiction.data.schema import FEATURE_COLUMNS, ID_COLUMN
from smartphone_addiction.neural.exact_vocab import (
    MISSING_INDEX,
    UNKNOWN_INDEX,
    ExactValueVocab,
    FoldExactVocabs,
)


def test_missing_and_unknown_indices() -> None:
    series = pd.Series([1.0, 1.0, np.nan, 2.0])
    vocab = ExactValueVocab(numeric=True).fit(series)
    encoded = vocab.transform(pd.Series([1.0, np.nan, 3.0, 2.0]))
    assert encoded[0] == encoded.tolist()[0]
    assert encoded[1] == MISSING_INDEX
    assert encoded[2] == UNKNOWN_INDEX
    assert encoded[0] != MISSING_INDEX
    assert encoded[0] != UNKNOWN_INDEX
    assert encoded[3] != encoded[2]
    assert vocab.size == 4


def test_fit_does_not_see_transform_values() -> None:
    train = pd.Series(["A", "B", "A"])
    valid = pd.Series(["A", "C"])
    vocab = ExactValueVocab(numeric=False).fit(train)
    encoded = vocab.transform(valid)
    assert encoded[0] != UNKNOWN_INDEX
    assert encoded[1] == UNKNOWN_INDEX


def test_fold_vocabs_encode_all_feature_columns(competition_frames) -> None:
    train, test, _ = competition_frames
    vocabs = FoldExactVocabs().fit(train)
    encoded = vocabs.transform(test)
    assert encoded.shape == (len(test), len(FEATURE_COLUMNS))
    assert encoded.dtype == np.int64
    assert (encoded >= MISSING_INDEX).all()
    assert vocabs.cardinalities()[FEATURE_COLUMNS.index("gender")] >= 2
    assert ID_COLUMN not in vocabs.columns
