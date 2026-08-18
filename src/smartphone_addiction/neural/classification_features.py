"""Pack fold-local exact-value indices for Lookup Transformer classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from smartphone_addiction.data.schema import FEATURE_COLUMNS, ID_COLUMN
from smartphone_addiction.errors import TrainingError
from smartphone_addiction.neural.exact_vocab import FoldExactVocabs
from smartphone_addiction.neural.preprocessing import FORBIDDEN_INPUT_COLUMNS


@dataclass
class EncodedClassificationTable:
    cat_indices: np.ndarray
    row_ids: np.ndarray
    labels: np.ndarray | None


def subset_encoded_table(
    table: EncodedClassificationTable, index: np.ndarray
) -> EncodedClassificationTable:
    """Select rows from an encoded table by positional index."""
    labels = None if table.labels is None else np.asarray(table.labels)[index]
    return EncodedClassificationTable(
        cat_indices=table.cat_indices[index],
        row_ids=np.asarray(table.row_ids)[index],
        labels=labels,
    )


class FoldClassificationEncoder:
    """Fit exact-value vocabularies on outer-train, then encode each split."""

    def __init__(self, columns: list[str] | None = None) -> None:
        selected = list(columns or FEATURE_COLUMNS)
        leaked = [name for name in selected if name in FORBIDDEN_INPUT_COLUMNS]
        if leaked:
            raise TrainingError(f"forbidden columns selected as classification inputs: {leaked}")
        self.columns = selected
        self.vocabs = FoldExactVocabs(self.columns)
        self._fitted = False

    def fit(
        self,
        frame: pd.DataFrame,
        extra_vocab_frame: pd.DataFrame | None = None,
    ) -> FoldClassificationEncoder:
        self.vocabs.fit(frame, extra=extra_vocab_frame)
        self._fitted = True
        return self

    def transform(
        self,
        frame: pd.DataFrame,
        y: np.ndarray | None = None,
    ) -> EncodedClassificationTable:
        if not self._fitted:
            raise TrainingError("FoldClassificationEncoder must be fitted before transform")
        if ID_COLUMN not in frame.columns:
            raise TrainingError("classification frame must contain id")
        labels = None if y is None else np.asarray(y, dtype=np.float32)
        if labels is not None and len(labels) != len(frame):
            raise TrainingError("classification labels must match frame rows")
        return EncodedClassificationTable(
            cat_indices=self.vocabs.transform(frame),
            row_ids=frame[ID_COLUMN].to_numpy(),
            labels=labels,
        )

    def transform_eval(self, frame: pd.DataFrame) -> EncodedClassificationTable:
        return self.transform(frame)

    def cardinalities(self) -> list[int]:
        return self.vocabs.cardinalities()

    def to_state(self) -> dict[str, Any]:
        if not self._fitted:
            raise TrainingError("cannot serialize an unfitted FoldClassificationEncoder")
        return {
            "columns": list(self.columns),
            "vocabs": self.vocabs.to_state(),
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> FoldClassificationEncoder:
        encoder = cls(columns=list(state["columns"]))
        encoder.vocabs = FoldExactVocabs.from_state(state["vocabs"])
        encoder._fitted = True
        return encoder
