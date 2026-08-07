"""Unit tests for sampled permutation importance."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from smartphone_addiction.errors import TrainingError
from smartphone_addiction.evaluation.importance import permutation_importance_auc


class _DummyModel:
    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        # Higher score when feature a is large and matches label signal.
        return 1.0 / (1.0 + np.exp(-(x["a"].to_numpy(dtype=float) * 2.0)))


def test_permutation_importance_ranks_signal_feature() -> None:
    rng = np.random.default_rng(0)
    n = 400
    a = rng.normal(size=n)
    noise = rng.normal(size=n)
    y = (a + rng.normal(scale=0.3, size=n) > 0).astype(int)
    x = pd.DataFrame({"a": a, "noise": noise})
    frame = permutation_importance_auc(
        _DummyModel(),
        x,
        y,
        ["a", "noise"],
        n_repeats=3,
        sample_rows=None,
        seed=0,
        run_id="demo",
    )
    assert {
        "feature",
        "importance_mean",
        "importance_std",
        "n_repeats",
        "sample_rows",
        "run_id",
        "seed",
    }.issubset(frame.columns)
    ranked = frame.set_index("feature")["importance_mean"]
    assert ranked.loc["a"] > ranked.loc["noise"]


def test_permutation_importance_rejects_bad_repeats() -> None:
    with pytest.raises(TrainingError, match="n_repeats"):
        permutation_importance_auc(
            _DummyModel(),
            pd.DataFrame({"a": [0.1, 0.2], "b": [0.3, 0.4]}),
            np.array([0, 1]),
            n_repeats=0,
        )


def test_validation_frame_for_fold_keeps_only_valid_rows(tmp_path: Path) -> None:
    from smartphone_addiction.data.schema import ID_COLUMN, TARGET_COLUMN
    from smartphone_addiction.evaluation.importance import validation_frame_for_fold

    train = pd.DataFrame(
        {
            ID_COLUMN: [1, 2, 3, 4],
            "a": [0.1, 0.2, 0.3, 0.4],
            TARGET_COLUMN: [0, 1, 0, 1],
        }
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    pd.DataFrame({ID_COLUMN: [1, 2, 3, 4], "fold": [0, 0, 1, 1]}).to_parquet(
        run_dir / "folds_seed42.parquet",
        index=False,
    )
    valid = validation_frame_for_fold(train, run_dir, "seed42-fold0")
    assert list(valid[ID_COLUMN]) == [1, 2]
