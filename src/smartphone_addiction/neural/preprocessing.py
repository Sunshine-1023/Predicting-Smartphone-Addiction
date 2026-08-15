"""Fold-local numeric standardization and categorical vocabularies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from smartphone_addiction.data.schema import CATEGORICAL_COLUMNS, ID_COLUMN, NUMERIC_COLUMNS
from smartphone_addiction.errors import TrainingError
from smartphone_addiction.neural.config import CORE5_FIELDS, MISSING_TOKEN, UNKNOWN_TOKEN
from smartphone_addiction.neural.device import require_torch

FORBIDDEN_INPUT_COLUMNS = frozenset(
    {
        ID_COLUMN,
        "addicted_label",
        "entertainment_hours",
        "work_minus_entertainment",
        "known_usage_hours",
        "unaccounted_screen_time",
        "screen_to_sleep_ratio",
        "entertainment_to_screen_ratio",
        "work_to_screen_ratio",
        "weekend_to_daily_ratio",
        "notifications_per_screen_hour",
        "opens_per_screen_hour",
        "opens_per_notification",
        "weekend_minus_daily",
        "notifications_minus_opens",
        "log_notifications",
        "log_app_opens",
        "gender_x_stress",
        "gender_x_impact",
        "stress_x_impact",
        "missing_pattern",
        "missing_count",
        "missing_ratio",
    }
)


@dataclass(frozen=True)
class TensorizedFrame:
    numeric: Any
    categorical: Any
    natural_observed: Any
    row_ids: np.ndarray
    core_raw: np.ndarray
    core_observed: np.ndarray


class FoldTensorizer:
    """Fit numeric stats and category vocabs on outer-train only."""

    def __init__(self) -> None:
        self.numeric_columns: list[str] = list(NUMERIC_COLUMNS)
        self.categorical_columns: list[str] = list(CATEGORICAL_COLUMNS)
        self.core_fields: list[str] = list(CORE5_FIELDS)
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.median_: np.ndarray | None = None
        self.core_mean_: np.ndarray | None = None
        self.core_std_: np.ndarray | None = None
        self.core_median_: np.ndarray | None = None
        self.vocabs_: dict[str, dict[str, int]] | None = None
        self._fitted = False

    @property
    def core_indices(self) -> np.ndarray:
        return np.array(
            [self.numeric_columns.index(name) for name in self.core_fields],
            dtype=np.int64,
        )

    def fit(self, frame: pd.DataFrame) -> FoldTensorizer:
        self._validate_columns(frame)
        numeric = frame.loc[:, self.numeric_columns]
        self.mean_ = numeric.mean(axis=0, skipna=True).to_numpy(dtype=np.float64)
        self.std_ = numeric.std(axis=0, skipna=True, ddof=0).to_numpy(dtype=np.float64)
        self.median_ = numeric.median(axis=0, skipna=True).to_numpy(dtype=np.float64)
        self.std_ = np.where(self.std_ < 1e-6, 1.0, self.std_)
        self.mean_ = np.nan_to_num(self.mean_, nan=0.0)
        self.median_ = np.nan_to_num(self.median_, nan=0.0)

        core = frame.loc[:, self.core_fields]
        self.core_mean_ = core.mean(axis=0, skipna=True).to_numpy(dtype=np.float64)
        self.core_std_ = core.std(axis=0, skipna=True, ddof=0).to_numpy(dtype=np.float64)
        self.core_median_ = core.median(axis=0, skipna=True).to_numpy(dtype=np.float64)
        self.core_std_ = np.where(self.core_std_ < 1e-6, 1.0, self.core_std_)
        self.core_mean_ = np.nan_to_num(self.core_mean_, nan=0.0)
        self.core_median_ = np.nan_to_num(self.core_median_, nan=0.0)

        vocabs: dict[str, dict[str, int]] = {}
        for column in self.categorical_columns:
            mapping = {MISSING_TOKEN: 0, UNKNOWN_TOKEN: 1}
            observed = frame[column].dropna().astype(str)
            observed = observed[observed != ""].unique().tolist()
            for value in sorted(str(item) for item in observed):
                if value not in mapping:
                    mapping[value] = len(mapping)
            vocabs[column] = mapping
        self.vocabs_ = vocabs
        self._fitted = True
        return self

    def transform(self, frame: pd.DataFrame) -> TensorizedFrame:
        torch = require_torch()
        if not self._fitted:
            raise TrainingError("FoldTensorizer must be fitted before transform")
        self._validate_columns(frame)
        assert self.mean_ is not None
        assert self.std_ is not None
        assert self.vocabs_ is not None

        numeric_np = frame.loc[:, self.numeric_columns].to_numpy(dtype=np.float64)
        observed = np.isfinite(numeric_np)
        standardized = np.zeros_like(numeric_np, dtype=np.float32)
        rows, cols = np.where(observed)
        if len(rows):
            standardized[rows, cols] = (
                (numeric_np[rows, cols] - self.mean_[cols]) / self.std_[cols]
            ).astype(np.float32)

        core_raw = frame.loc[:, self.core_fields].to_numpy(dtype=np.float64)
        core_observed = np.isfinite(core_raw)

        cat_index = np.zeros((len(frame), len(self.categorical_columns)), dtype=np.int64)
        for col_i, column in enumerate(self.categorical_columns):
            vocab = self.vocabs_[column]
            values = frame[column].to_numpy()
            for row_i, value in enumerate(values):
                if value is None or (isinstance(value, float) and not np.isfinite(value)):
                    cat_index[row_i, col_i] = vocab[MISSING_TOKEN]
                    continue
                text = str(value)
                if text in {"", "nan", "None"}:
                    cat_index[row_i, col_i] = vocab[MISSING_TOKEN]
                else:
                    cat_index[row_i, col_i] = vocab.get(text, vocab[UNKNOWN_TOKEN])

        if ID_COLUMN not in frame.columns:
            raise TrainingError("frame must contain id")
        row_ids = frame[ID_COLUMN].to_numpy()
        numeric_t = torch.tensor(standardized, dtype=torch.float32)
        observed_t = torch.tensor(observed, dtype=torch.bool)
        categorical_t = torch.tensor(cat_index, dtype=torch.long)
        if not torch.isfinite(numeric_t).all():
            raise TrainingError("tensorizer produced NaN or Inf numeric values")
        return TensorizedFrame(
            numeric=numeric_t,
            categorical=categorical_t,
            natural_observed=observed_t,
            row_ids=row_ids,
            core_raw=core_raw.astype(np.float64, copy=False),
            core_observed=core_observed,
        )

    def inverse_core(self, standardized: np.ndarray) -> np.ndarray:
        if self.core_mean_ is None or self.core_std_ is None:
            raise TrainingError("FoldTensorizer must be fitted before inverse_core")
        return standardized.astype(np.float64) * self.core_std_ + self.core_mean_

    def vocab_sizes(self) -> list[int]:
        if self.vocabs_ is None:
            raise TrainingError("FoldTensorizer must be fitted before vocab_sizes")
        return [len(self.vocabs_[column]) for column in self.categorical_columns]

    def to_state(self) -> dict[str, Any]:
        if not self._fitted:
            raise TrainingError("cannot serialize an unfitted FoldTensorizer")
        return {
            "numeric_columns": list(self.numeric_columns),
            "categorical_columns": list(self.categorical_columns),
            "core_fields": list(self.core_fields),
            "mean": None if self.mean_ is None else self.mean_.tolist(),
            "std": None if self.std_ is None else self.std_.tolist(),
            "median": None if self.median_ is None else self.median_.tolist(),
            "core_mean": None if self.core_mean_ is None else self.core_mean_.tolist(),
            "core_std": None if self.core_std_ is None else self.core_std_.tolist(),
            "core_median": None if self.core_median_ is None else self.core_median_.tolist(),
            "vocabs": self.vocabs_,
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> FoldTensorizer:
        tensorizer = cls()
        tensorizer.numeric_columns = list(state["numeric_columns"])
        tensorizer.categorical_columns = list(state["categorical_columns"])
        tensorizer.core_fields = list(state["core_fields"])
        tensorizer.mean_ = np.asarray(state["mean"], dtype=np.float64)
        tensorizer.std_ = np.asarray(state["std"], dtype=np.float64)
        tensorizer.median_ = np.asarray(state["median"], dtype=np.float64)
        tensorizer.core_mean_ = np.asarray(state["core_mean"], dtype=np.float64)
        tensorizer.core_std_ = np.asarray(state["core_std"], dtype=np.float64)
        tensorizer.core_median_ = np.asarray(state["core_median"], dtype=np.float64)
        tensorizer.vocabs_ = dict(state["vocabs"])
        tensorizer._fitted = True
        return tensorizer

    def _validate_columns(self, frame: pd.DataFrame) -> None:
        missing = [
            name
            for name in [*self.numeric_columns, *self.categorical_columns]
            if name not in frame.columns
        ]
        if missing:
            raise TrainingError(f"tensorizer frame missing columns: {missing}")
        leaked = [name for name in FORBIDDEN_INPUT_COLUMNS if name in self.numeric_columns]
        if leaked:
            raise TrainingError(f"derived or forbidden columns selected as neural inputs: {leaked}")
