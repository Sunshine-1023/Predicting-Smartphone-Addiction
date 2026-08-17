"""Tests for fold-native imputed_core feature construction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from smartphone_addiction.data.schema import FEATURE_COLUMNS, ID_COLUMN
from smartphone_addiction.neural.autoencoder import build_mlp_autoencoder
from smartphone_addiction.neural.config import CORE5_FIELDS, NeuralModelArchConfig
from smartphone_addiction.neural.fold_features import FoldEncoder, attach_encoder_features
from smartphone_addiction.neural.preprocessing import FoldTensorizer


def _frame(n: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    data = {ID_COLUMN: np.arange(n)}
    for column in FEATURE_COLUMNS:
        if column in {"gender", "stress_level", "academic_work_impact"}:
            data[column] = rng.choice(["A", "B"], n)
        else:
            data[column] = rng.normal(5, 1, n)
    frame = pd.DataFrame(data)
    frame.loc[0, "social_media_hours"] = np.nan
    frame.loc[1, ["work_study_hours", "gaming_hours"]] = np.nan
    return frame


def _encoder(frame: pd.DataFrame) -> FoldEncoder:
    tensorizer = FoldTensorizer().fit(frame)
    model = build_mlp_autoencoder(
        tensorizer.vocab_sizes(),
        NeuralModelArchConfig(hidden_dim=16, latent_dim=8, n_blocks=1, dropout=0.10),
    )
    return FoldEncoder(
        fold=0,
        model=model,
        tensorizer=tensorizer,
        device=torch.device("cpu"),
        batch_size=8,
        model_name="mlp",
    )


def test_imputed_core_keeps_observed_raw_and_fills_missing() -> None:
    frame = _frame()
    encoder = _encoder(frame)
    out = attach_encoder_features(frame, encoder, include=["imputed_core"])
    assert "imputed_social_media_hours" in out.columns
    assert out.loc[0, "social_media_hours_is_imputed"] == 1
    assert out.loc[2, "social_media_hours_is_imputed"] == 0
    assert out.loc[2, "imputed_daily_screen_time_hours"] == pytest.approx(
        frame.loc[2, "daily_screen_time_hours"]
    )
    assert np.isfinite(out.loc[0, "imputed_social_media_hours"])
    for field in CORE5_FIELDS:
        assert f"imputed_{field}" in out.columns
        assert f"{field}_is_imputed" in out.columns
        assert f"imputed_std_{field}" not in out.columns


def test_imputed_core_allows_duplicate_ids_by_row_position() -> None:
    frame = pd.concat([_frame(), _frame().iloc[:2]], ignore_index=True)
    encoder = _encoder(frame)
    out = attach_encoder_features(frame, encoder, include=["imputed_core"])
    assert len(out) == len(frame)
    assert list(out[ID_COLUMN]) == list(frame[ID_COLUMN])
    assert out["imputed_social_media_hours"].notna().all()
