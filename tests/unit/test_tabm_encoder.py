"""CPU shape tests for the TabM-style reconstruction model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from smartphone_addiction.data.schema import ID_COLUMN
from smartphone_addiction.neural.config import NeuralModelArchConfig
from smartphone_addiction.neural.masking import (
    build_train_mask_batch,
    pattern_distribution_from_test,
)
from smartphone_addiction.neural.preprocessing import FoldTensorizer
from smartphone_addiction.neural.tabm import build_tabm_autoencoder


def _frame(n: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(1)
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
    frame.loc[0, "daily_screen_time_hours"] = np.nan
    frame.loc[1, ["weekend_screen_time", "social_media_hours"]] = np.nan
    return frame


def test_tabm_output_has_member_axes() -> None:
    frame = _frame()
    tensorizer = FoldTensorizer().fit(frame)
    encoded = tensorizer.transform(frame)
    batch = build_train_mask_batch(
        encoded,
        tensorizer,
        generator=torch.Generator().manual_seed(1),
        pattern_distribution=pattern_distribution_from_test(frame),
    )
    model = build_tabm_autoencoder(
        tensorizer.vocab_sizes(),
        NeuralModelArchConfig(ensemble_size=4, n_blocks=2, hidden_dim=32, latent_dim=16),
    )
    output = model(batch)
    assert tuple(output.member_predictions.shape) == (len(frame), 4, 5)
    assert tuple(output.member_latents.shape) == (len(frame), 4, 16)
    assert tuple(output.mean_prediction.shape) == (len(frame), 5)
    assert tuple(output.mean_latent.shape) == (len(frame), 16)
    assert tuple(output.member_std.shape) == (len(frame), 5)
    output.mean_prediction.sum().backward()
    assert any(param.grad is not None for param in model.parameters())
