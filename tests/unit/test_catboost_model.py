"""Unit tests for the CatBoost adapter."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from smartphone_addiction.features.base import transform_competition_frames
from smartphone_addiction.models.catboost import build_catboost

pytestmark = pytest.mark.model


def test_catboost_fit_predict_and_save(tmp_path: Path, competition_frames) -> None:
    train, test, _ = competition_frames
    frames = transform_competition_frames(train, test)
    x = frames.train[frames.feature_columns]
    y = frames.train["addicted_label"]
    categorical = [c for c in frames.categorical_columns if c in x.columns]

    model = build_catboost(
        categorical_columns=categorical,
        iterations=20,
        depth=4,
        thread_count=2,
        random_seed=42,
        early_stopping_rounds=5,
        learning_rate=0.1,
    )
    model.fit(x.iloc[:200], y.iloc[:200], x.iloc[200:260], y.iloc[200:260])
    prediction = model.predict_proba(x.iloc[200:260])
    assert prediction.shape == (60,)
    assert np.isfinite(prediction).all()
    assert prediction.min() >= 0.0
    assert prediction.max() <= 1.0

    path = model.save(tmp_path / "model.cbm")
    assert path.is_file()
    assert model.best_iteration is not None
    assert model.best_iteration >= 1
