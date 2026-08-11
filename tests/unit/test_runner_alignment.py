"""Unit tests for fold prediction ID alignment in the training runner."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from smartphone_addiction.errors import AlignmentError, ArtifactError
from smartphone_addiction.training.runner import _load_fold_predictions


def _write_fold(
    path: Path,
    *,
    valid_index: np.ndarray,
    valid_pred: np.ndarray,
    test_pred: np.ndarray,
    valid_ids: np.ndarray | None = None,
    test_ids: np.ndarray | None = None,
) -> None:
    payload: dict[str, np.ndarray] = {
        "valid_index": valid_index.astype(np.int64),
        "valid_pred": valid_pred.astype(np.float64),
        "test_pred": test_pred.astype(np.float64),
    }
    if valid_ids is not None:
        payload["valid_ids"] = valid_ids
    if test_ids is not None:
        payload["test_ids"] = test_ids
    np.savez_compressed(path, **payload)


def test_load_fold_reorders_test_predictions_by_id(tmp_path: Path) -> None:
    ids = np.array([1, 2, 3, 4])
    test_ids = np.array([10, 20, 30])
    path = tmp_path / "seed42-fold0.npz"
    _write_fold(
        path,
        valid_index=np.array([0, 1]),
        valid_ids=np.array([1, 2]),
        valid_pred=np.array([0.1, 0.2]),
        test_ids=np.array([30, 10, 20]),
        test_pred=np.array([0.3, 0.1, 0.2]),
    )
    _, valid_pred, test_pred = _load_fold_predictions(
        path,
        ids=ids,
        test_ids=test_ids,
        expected_valid_index=np.array([0, 1]),
        label="seed42-fold0",
    )
    assert valid_pred.tolist() == pytest.approx([0.1, 0.2])
    assert test_pred.tolist() == pytest.approx([0.1, 0.2, 0.3])


def test_load_fold_rejects_valid_id_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "bad.npz"
    _write_fold(
        path,
        valid_index=np.array([0, 1]),
        valid_ids=np.array([9, 8]),
        valid_pred=np.array([0.1, 0.2]),
        test_ids=np.array([10, 20]),
        test_pred=np.array([0.4, 0.6]),
    )
    with pytest.raises(AlignmentError, match="valid ids disagree"):
        _load_fold_predictions(
            path,
            ids=np.array([1, 2, 3]),
            test_ids=np.array([10, 20]),
            label="bad",
        )


def test_load_fold_rejects_legacy_test_length_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "legacy.npz"
    _write_fold(
        path,
        valid_index=np.array([0]),
        valid_pred=np.array([0.1]),
        test_pred=np.array([0.4]),
    )
    with pytest.raises(AlignmentError, match="test prediction length"):
        _load_fold_predictions(
            path,
            ids=np.array([1, 2]),
            test_ids=np.array([10, 20]),
            label="legacy",
        )


def test_load_fold_rejects_index_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "idx.npz"
    _write_fold(
        path,
        valid_index=np.array([0, 2]),
        valid_pred=np.array([0.1, 0.2]),
        test_pred=np.array([0.4, 0.6]),
        test_ids=np.array([10, 20]),
    )
    with pytest.raises(ArtifactError, match="prediction indices mismatch"):
        _load_fold_predictions(
            path,
            ids=np.array([1, 2, 3]),
            test_ids=np.array([10, 20]),
            expected_valid_index=np.array([0, 1]),
            label="idx",
        )
