"""CPU shape and gradient tests for the MLP masked autoencoder."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from smartphone_addiction.data.schema import ID_COLUMN
from smartphone_addiction.neural.autoencoder import build_mlp_autoencoder
from smartphone_addiction.neural.config import NeuralModelArchConfig
from smartphone_addiction.neural.masking import (
    apply_artificial_mask,
    build_train_mask_batch,
    pattern_distribution_from_test,
)
from smartphone_addiction.neural.preprocessing import FoldTensorizer


def _frame(n: int = 16) -> pd.DataFrame:
    rng = np.random.default_rng(0)
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


def test_mlp_forward_shapes_and_optimizer_step() -> None:
    frame = _frame()
    tensorizer = FoldTensorizer().fit(frame)
    encoded = tensorizer.transform(frame)
    generator = torch.Generator().manual_seed(0)
    batch = build_train_mask_batch(
        encoded,
        tensorizer,
        generator=generator,
        pattern_distribution=pattern_distribution_from_test(frame),
    )
    model = build_mlp_autoencoder(tensorizer.vocab_sizes(), NeuralModelArchConfig())
    output = model(batch)
    assert tuple(output.mean_prediction.shape) == (len(frame), 5)
    assert tuple(output.mean_latent.shape) == (len(frame), 32)
    before = [param.detach().clone() for param in model.parameters()]
    loss = output.mean_prediction.sum()
    loss.backward()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    optimizer.step()
    changed = any(
        not torch.allclose(old, new)
        for old, new in zip(before, list(model.parameters()), strict=True)
    )
    assert changed


def test_apply_mask_rejects_row_mismatch() -> None:
    frame = _frame()
    tensorizer = FoldTensorizer().fit(frame)
    encoded = tensorizer.transform(frame)
    with pytest.raises(Exception, match="row count"):
        apply_artificial_mask(encoded, tensorizer, np.zeros((2, 5), dtype=bool))
