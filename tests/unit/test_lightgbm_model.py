"""Unit tests for the LightGBM adapter."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from smartphone_addiction.features.base import transform_competition_frames
from smartphone_addiction.models.lightgbm import build_lightgbm

pytestmark = pytest.mark.model


def test_lightgbm_handles_unseen_categories(tmp_path: Path, competition_frames) -> None:
    train, test, _ = competition_frames
    frames = transform_competition_frames(train, test)
    x = frames.train[frames.feature_columns].copy()
    y = frames.train["addicted_label"]
    categorical = list(frames.categorical_columns)

    model = build_lightgbm(
        categorical_columns=categorical,
        n_estimators=30,
        num_leaves=15,
        n_jobs=2,
        random_state=42,
        early_stopping_rounds=5,
        learning_rate=0.1,
    )
    x_train = x.iloc[:200]
    y_train = y.iloc[:200]
    x_valid = x.iloc[200:260].copy()
    y_valid = y.iloc[200:260]
    # Inject a category level never seen in the training fold.
    if "gender" in x_valid.columns:
        x_valid.loc[x_valid.index[0], "gender"] = "UnseenGenderLevel"

    model.fit(x_train, y_train, x_valid, y_valid)
    prediction = model.predict_proba(x_valid)
    assert prediction.shape == (60,)
    assert np.isfinite(prediction).all()

    path = model.save(tmp_path / "model.joblib")
    assert path.is_file()
    assert model.best_iteration is not None
