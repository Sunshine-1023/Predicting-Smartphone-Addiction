"""Shared model adapter protocol for binary classification."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from smartphone_addiction.features.domain import MISSING_TOKEN


class ModelAdapter(Protocol):
    """Common interface used by the OOF training runner."""

    def fit(
        self,
        x: pd.DataFrame,
        y: pd.Series | np.ndarray,
        x_valid: pd.DataFrame | None = None,
        y_valid: pd.Series | np.ndarray | None = None,
        *,
        sample_weight: np.ndarray | None = None,
    ) -> ModelAdapter:
        """Fit on the current training fold; optional validation fold for early stopping."""

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        """Return shape (n_samples,) positive-class probabilities."""

    def save(self, path: Path | str) -> Path:
        """Persist the fitted adapter to disk."""

    @property
    def best_iteration(self) -> int | None:
        """Best boosting iteration after early stopping, if available."""


def prepare_categorical_frame(
    frame: pd.DataFrame,
    categorical_columns: list[str],
) -> pd.DataFrame:
    """Copy frame and fill categorical nulls with __MISSING__ as strings."""
    out = frame.copy()
    for column in categorical_columns:
        if column not in out.columns:
            continue
        series = out[column]
        filled = series.where(series.notna(), MISSING_TOKEN).astype(str)
        filled = filled.replace({"nan": MISSING_TOKEN, "None": MISSING_TOKEN, "": MISSING_TOKEN})
        out[column] = filled
    return out
