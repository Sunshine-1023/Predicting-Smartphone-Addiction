"""Fold-local exact-value vocabularies for numeric and categorical columns."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from smartphone_addiction.data.schema import FEATURE_COLUMNS, NUMERIC_COLUMNS
from smartphone_addiction.errors import TrainingError

MISSING_INDEX = 0
UNKNOWN_INDEX = 1
MISSING_KEY = "__MISSING__"


def is_missing_value(value: object) -> bool:
    """Return True for NaN / None / blank tokens that should share the missing index."""
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return True
    return isinstance(value, str) and value.strip() in {"", "nan", "None", "<NA>"}


def lookup_key(value: object, *, numeric: bool) -> object | None:
    """Return a hashable exact-value key, or None when the cell is missing."""
    if is_missing_value(value):
        return None
    if numeric:
        number = pd.to_numeric(value, errors="coerce")
        if is_missing_value(number):
            return None
        return float(number)
    text = str(value).strip()
    if text in {"", "nan", "None", "<NA>"}:
        return None
    return text


def series_keys(series: pd.Series, *, numeric: bool) -> pd.Series:
    """Map a column to lookup keys while preserving missing cells as NA."""
    if numeric:
        keys = pd.to_numeric(series, errors="coerce")
        return keys.astype("float64")
    mapped = [lookup_key(value, numeric=False) for value in series.to_numpy()]
    return pd.Series(mapped, index=series.index, dtype="object")


def _sorted_unique_keys(values: np.ndarray, *, numeric: bool) -> list[object]:
    present = [lookup_key(value, numeric=numeric) for value in values]
    keys = [key for key in present if key is not None]
    if numeric:
        return sorted(set(keys), key=lambda item: float(item))
    return sorted(set(keys), key=lambda item: str(item))


@dataclass
class ExactValueVocab:
    """Maps one column's exact values onto reserved missing/unknown plus seen indices."""

    numeric: bool
    mapping: dict[object, int] = field(default_factory=dict)
    _fitted: bool = False

    def fit(self, series: pd.Series, extra: pd.Series | None = None) -> ExactValueVocab:
        values = series.to_numpy()
        if extra is not None:
            values = np.concatenate([values, extra.to_numpy()], axis=0)
        keys = _sorted_unique_keys(values, numeric=self.numeric)
        mapping: dict[object, int] = {}
        for key in keys:
            mapping[key] = len(mapping) + 2
        self.mapping = mapping
        self._fitted = True
        return self

    @property
    def size(self) -> int:
        if not self._fitted:
            raise TrainingError("ExactValueVocab must be fitted before size")
        return len(self.mapping) + 2

    def transform(self, series: pd.Series) -> np.ndarray:
        if not self._fitted:
            raise TrainingError("ExactValueVocab must be fitted before transform")
        keys = series_keys(series, numeric=self.numeric)
        mapped = keys.map(self.mapping)
        out = np.full(len(series), UNKNOWN_INDEX, dtype=np.int64)
        known = mapped.notna().to_numpy()
        out[known] = mapped.to_numpy()[known].astype(np.int64)
        out[keys.isna().to_numpy()] = MISSING_INDEX
        return out

    def to_state(self) -> dict[str, Any]:
        if not self._fitted:
            raise TrainingError("cannot serialize an unfitted ExactValueVocab")
        return {
            "numeric": self.numeric,
            "keys": [[key, index] for key, index in self.mapping.items()],
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> ExactValueVocab:
        vocab = cls(numeric=bool(state["numeric"]))
        vocab.mapping = {item[0]: int(item[1]) for item in state["keys"]}
        vocab._fitted = True
        return vocab


class FoldExactVocabs:
    """Per-column exact-value vocabs fitted on outer-train only."""

    def __init__(self, columns: list[str] | None = None) -> None:
        self.columns: list[str] = list(columns or FEATURE_COLUMNS)
        self.vocabs_: dict[str, ExactValueVocab] | None = None

    def fit(
        self,
        frame: pd.DataFrame,
        extra: pd.DataFrame | None = None,
    ) -> FoldExactVocabs:
        self._validate(frame)
        if extra is not None:
            self._validate(extra)
        vocabs: dict[str, ExactValueVocab] = {}
        numeric = set(NUMERIC_COLUMNS)
        for column in self.columns:
            extra_series = None if extra is None else extra[column]
            vocabs[column] = ExactValueVocab(numeric=column in numeric).fit(
                frame[column], extra_series
            )
        self.vocabs_ = vocabs
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        vocabs = self._require()
        self._validate(frame)
        encoded = np.column_stack(
            [vocabs[column].transform(frame[column]) for column in self.columns]
        )
        return encoded.astype(np.int64, copy=False)

    def cardinalities(self) -> list[int]:
        vocabs = self._require()
        return [vocabs[column].size for column in self.columns]

    def to_state(self) -> dict[str, Any]:
        vocabs = self._require()
        return {
            "columns": list(self.columns),
            "vocabs": {column: vocabs[column].to_state() for column in self.columns},
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> FoldExactVocabs:
        bundle = cls(columns=list(state["columns"]))
        bundle.vocabs_ = {
            column: ExactValueVocab.from_state(payload)
            for column, payload in state["vocabs"].items()
        }
        return bundle

    def _require(self) -> dict[str, ExactValueVocab]:
        if self.vocabs_ is None:
            raise TrainingError("FoldExactVocabs must be fitted before use")
        return self.vocabs_

    def _validate(self, frame: pd.DataFrame) -> None:
        missing = [name for name in self.columns if name not in frame.columns]
        if missing:
            raise TrainingError(f"exact-value frame missing columns: {missing}")
        leaked = [name for name in self.columns if name not in FEATURE_COLUMNS]
        if leaked:
            raise TrainingError(
                f"derived or unknown columns selected as exact-value inputs: {leaked}"
            )
