"""Latent export keeps one row per id and covers the transformed frame."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from smartphone_addiction.data.schema import ID_COLUMN
from smartphone_addiction.neural.autoencoder import build_mlp_autoencoder
from smartphone_addiction.neural.config import NeuralModelArchConfig
from smartphone_addiction.neural.export import encode_latent
from smartphone_addiction.neural.preprocessing import FoldTensorizer


def test_latent_export_alignment() -> None:
    rng = np.random.default_rng(0)
    n = 20
    frame = pd.DataFrame(
        {
            ID_COLUMN: np.arange(n),
            "age": rng.normal(30, 5, n),
            "daily_screen_time_hours": rng.normal(6, 1, n),
            "social_media_hours": rng.normal(2, 0.5, n),
            "gaming_hours": rng.normal(1, 0.4, n),
            "work_study_hours": rng.normal(2, 0.5, n),
            "sleep_hours": rng.normal(7, 1, n),
            "notifications_per_day": rng.normal(50, 10, n),
            "app_opens_per_day": rng.normal(80, 15, n),
            "weekend_screen_time": rng.normal(8, 1, n),
            "gender": rng.choice(["Male", "Female"], n),
            "stress_level": rng.choice(["Low", "Medium"], n),
            "academic_work_impact": rng.choice(["No", "Yes"], n),
        }
    )
    tensorizer = FoldTensorizer().fit(frame)
    encoded = tensorizer.transform(frame)
    model = build_mlp_autoencoder(
        tensorizer.vocab_sizes(),
        NeuralModelArchConfig(hidden_dim=16, latent_dim=8, n_blocks=1),
    )
    latent = encode_latent(model, encoded, tensorizer, device=torch.device("cpu"), batch_size=8)
    assert len(latent) == n
    assert latent[ID_COLUMN].is_unique
    assert list(latent[ID_COLUMN]) == list(frame[ID_COLUMN])
    assert "latent_00" in latent.columns
    assert "recon_daily_screen_time_hours" in latent.columns
    assert latent.filter(regex=r"^latent_").shape[1] == 8
