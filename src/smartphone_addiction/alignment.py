"""ID alignment helpers for OOF, blend, and submission predictions."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from smartphone_addiction.data.schema import ID_COLUMN
from smartphone_addiction.errors import AlignmentError


def as_id_series(ids: pd.Series | np.ndarray | list[Any]) -> pd.Series:
    """Normalize ids to a 0-based pandas Series."""
    return pd.Series(ids).reset_index(drop=True)


def assert_unique_ids(ids: pd.Series | np.ndarray | list[Any], *, label: str = "id") -> pd.Series:
    """Require non-null unique ids."""
    series = as_id_series(ids)
    if series.isna().any():
        raise AlignmentError(f"{label} must not contain missing values")
    if not series.is_unique:
        raise AlignmentError(f"{label} must be unique")
    return series


def assert_same_id_order(
    base_ids: pd.Series | np.ndarray | list[Any],
    other_ids: pd.Series | np.ndarray | list[Any],
    *,
    label: str = "ids",
) -> None:
    """Require identical length, set, and order."""
    base = assert_unique_ids(base_ids, label=f"base {label}")
    other = assert_unique_ids(other_ids, label=f"other {label}")
    if len(base) != len(other):
        raise AlignmentError(f"{label} row count mismatch: base={len(base)} other={len(other)}")
    if not base.equals(other):
        if set(base.tolist()) != set(other.tolist()):
            raise AlignmentError(f"{label} sets differ")
        raise AlignmentError(f"{label} order differs from baseline")


def align_predictions_to_ids(
    base_ids: pd.Series | np.ndarray | list[Any],
    pred_ids: pd.Series | np.ndarray | list[Any],
    predictions: np.ndarray | pd.Series,
    *,
    label: str = "predictions",
) -> np.ndarray:
    """Reindex predictions onto ``base_ids`` order by ID (not by row position)."""
    base = assert_unique_ids(base_ids, label="base id")
    pred = assert_unique_ids(pred_ids, label=f"{label} id")
    values = np.asarray(predictions, dtype=float)
    if len(values) != len(pred):
        raise AlignmentError(f"{label} length {len(values)} != id rows {len(pred)}")
    if not np.isfinite(values).all():
        raise AlignmentError(f"{label} must be finite")
    if set(base.tolist()) != set(pred.tolist()):
        missing = sorted(set(base.tolist()) - set(pred.tolist()))
        extra = sorted(set(pred.tolist()) - set(base.tolist()))
        raise AlignmentError(f"{label} id set mismatch; missing={missing[:5]} extra={extra[:5]}")
    frame = pd.DataFrame({ID_COLUMN: pred.to_numpy(), "prediction": values})
    merged = pd.DataFrame({ID_COLUMN: base.to_numpy()}).merge(
        frame, on=ID_COLUMN, how="left", validate="one_to_one"
    )
    if merged["prediction"].isna().any():
        raise AlignmentError(f"{label} failed to align onto baseline ids")
    return merged["prediction"].to_numpy(dtype=float)


def align_frame_predictions(
    base_ids: pd.Series | np.ndarray | list[Any],
    frame: pd.DataFrame,
    *,
    prediction_column: str = "prediction",
    label: str = "frame",
) -> np.ndarray:
    """Align a prediction frame onto baseline ids."""
    if ID_COLUMN not in frame.columns or prediction_column not in frame.columns:
        raise AlignmentError(f"{label} must contain {ID_COLUMN} and {prediction_column}")
    return align_predictions_to_ids(
        base_ids,
        frame[ID_COLUMN],
        frame[prediction_column].to_numpy(),
        label=label,
    )
