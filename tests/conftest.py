"""Shared pytest fixtures with synthetic competition frames."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from smartphone_addiction.data.schema import FEATURE_COLUMNS, ID_COLUMN, TARGET_COLUMN


@pytest.fixture
def competition_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return train, test, and sample_submission DataFrames with missing values."""
    rng = np.random.default_rng(42)
    n_train = 320
    n_test = 120

    def _build(n_rows: int, start_id: int, with_target: bool) -> pd.DataFrame:
        data: dict[str, object] = {ID_COLUMN: np.arange(start_id, start_id + n_rows)}
        data["age"] = rng.integers(14, 65, size=n_rows).astype(float)
        data["daily_screen_time_hours"] = rng.uniform(1, 14, size=n_rows)
        data["social_media_hours"] = rng.uniform(0, 6, size=n_rows)
        data["gaming_hours"] = rng.uniform(0, 5, size=n_rows)
        data["work_study_hours"] = rng.uniform(0, 8, size=n_rows)
        data["sleep_hours"] = rng.uniform(3, 10, size=n_rows)
        data["notifications_per_day"] = rng.integers(0, 400, size=n_rows).astype(float)
        data["app_opens_per_day"] = rng.integers(0, 300, size=n_rows).astype(float)
        data["weekend_screen_time"] = rng.uniform(1, 16, size=n_rows)
        data["gender"] = rng.choice(["Female", "Male", "Other"], size=n_rows)
        data["stress_level"] = rng.choice(["Low", "Medium", "High"], size=n_rows)
        data["academic_work_impact"] = rng.choice(["Yes", "No"], size=n_rows)

        frame = pd.DataFrame(data)
        # Inject numeric and categorical missingness.
        for column in [
            "daily_screen_time_hours",
            "sleep_hours",
            "social_media_hours",
            "notifications_per_day",
        ]:
            idx = rng.choice(n_rows, size=max(1, n_rows // 20), replace=False)
            frame.loc[idx, column] = np.nan
        for column in ["gender", "stress_level", "academic_work_impact"]:
            idx = rng.choice(n_rows, size=max(1, n_rows // 25), replace=False)
            frame.loc[idx, column] = np.nan

        if with_target:
            frame[TARGET_COLUMN] = rng.integers(0, 2, size=n_rows)
        return frame[[ID_COLUMN, *FEATURE_COLUMNS] + ([TARGET_COLUMN] if with_target else [])]

    train = _build(n_train, start_id=0, with_target=True)
    test = _build(n_test, start_id=10_000, with_target=False)
    sample = pd.DataFrame(
        {
            ID_COLUMN: test[ID_COLUMN].to_numpy(),
            TARGET_COLUMN: np.full(n_test, 0.5),
        }
    )
    return train, test, sample
