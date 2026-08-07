"""Unit tests for OOF blending."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from smartphone_addiction.data.schema import ID_COLUMN, TARGET_COLUMN
from smartphone_addiction.errors import TrainingError
from smartphone_addiction.evaluation.blend import (
    blend_run_predictions,
    search_two_model_blend,
)


def test_blend_search_returns_valid_weights() -> None:
    y = np.array([0, 0, 1, 1])
    first = np.array([0.1, 0.4, 0.6, 0.9])
    second = np.array([0.2, 0.3, 0.8, 0.7])
    result = search_two_model_blend(y, first, second, step=0.05)
    assert 0.0 <= result.first_weight <= 1.0
    assert result.second_weight == pytest.approx(1.0 - result.first_weight)
    assert result.auc >= 0.5
    assert result.method in {"probability", "rank"}


def test_blend_rejects_misaligned_shapes() -> None:
    with pytest.raises(TrainingError, match="same shape"):
        search_two_model_blend(np.array([0, 1]), np.array([0.1]), np.array([0.2, 0.3]))


def test_blend_run_predictions_writes_artifacts(tmp_path: Path) -> None:
    ids = np.arange(6)
    y = np.array([0, 0, 0, 1, 1, 1])
    first_oof = pd.DataFrame(
        {ID_COLUMN: ids, TARGET_COLUMN: y, "prediction": [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]}
    )
    second_oof = pd.DataFrame(
        {ID_COLUMN: ids, TARGET_COLUMN: y, "prediction": [0.15, 0.25, 0.35, 0.65, 0.75, 0.85]}
    )
    first_test = pd.DataFrame({ID_COLUMN: [10, 11], "prediction": [0.4, 0.6]})
    second_test = pd.DataFrame({ID_COLUMN: [10, 11], "prediction": [0.45, 0.55]})

    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    for run, oof, test in (
        (run_a, first_oof, first_test),
        (run_b, second_oof, second_test),
    ):
        run.mkdir()
        oof.to_parquet(run / "oof_predictions.parquet", index=False)
        test.to_parquet(run / "test_predictions.parquet", index=False)

    out = tmp_path / "blend"
    payload = blend_run_predictions(first_run_dir=run_a, second_run_dir=run_b, output_dir=out)
    assert (out / "oof_predictions.parquet").is_file()
    assert (out / "test_predictions.parquet").is_file()
    assert (out / "blend_result.json").is_file()
    assert payload["auc"] >= 0.5
