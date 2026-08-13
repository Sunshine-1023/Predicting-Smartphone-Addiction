"""Unit tests for OOF blending."""

from __future__ import annotations

import json
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


def _write_completed_run(
    run: Path,
    oof: pd.DataFrame,
    test: pd.DataFrame,
    *,
    data_hashes: dict[str, str] | None = None,
    n_train_rows: int | None = None,
) -> None:
    run.mkdir(parents=True, exist_ok=True)
    oof.to_parquet(run / "oof_predictions.parquet", index=False)
    test.to_parquet(run / "test_predictions.parquet", index=False)
    payload = {
        "status": "completed",
        "run_id": run.name,
        "data_hashes": data_hashes or {"train.csv": "abc"},
        "n_train_rows": n_train_rows if n_train_rows is not None else len(oof),
    }
    (run / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")


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
    hashes = {"train.csv": "same"}
    _write_completed_run(run_a, first_oof, first_test, data_hashes=hashes)
    _write_completed_run(run_b, second_oof, second_test, data_hashes=hashes)

    out = tmp_path / "blend"
    payload = blend_run_predictions(first_run_dir=run_a, second_run_dir=run_b, output_dir=out)
    assert (out / "oof_predictions.parquet").is_file()
    assert (out / "test_predictions.parquet").is_file()
    assert (out / "blend_result.json").is_file()
    assert (out / "manifest.json").is_file()
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["artifact_type"] == "blend"
    assert payload["auc"] >= 0.5


def test_blend_refuses_overwrite_without_force(tmp_path: Path) -> None:
    ids = np.arange(4)
    y = np.array([0, 0, 1, 1])
    oof = pd.DataFrame({ID_COLUMN: ids, TARGET_COLUMN: y, "prediction": [0.1, 0.2, 0.8, 0.9]})
    test = pd.DataFrame({ID_COLUMN: [10, 11], "prediction": [0.4, 0.6]})
    hashes = {"train.csv": "same"}
    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    _write_completed_run(run_a, oof, test, data_hashes=hashes)
    _write_completed_run(run_b, oof, test, data_hashes=hashes)
    out = tmp_path / "blend"
    blend_run_predictions(first_run_dir=run_a, second_run_dir=run_b, output_dir=out)
    with pytest.raises(TrainingError, match="already exists"):
        blend_run_predictions(first_run_dir=run_a, second_run_dir=run_b, output_dir=out)
    payload = blend_run_predictions(
        first_run_dir=run_a, second_run_dir=run_b, output_dir=out, force=True
    )
    assert payload["auc"] >= 0.5
    assert (out / "manifest.json").is_file()


def test_blend_aligns_permuted_ids(tmp_path: Path) -> None:
    ids = np.arange(6)
    y = np.array([0, 0, 0, 1, 1, 1])
    first_oof = pd.DataFrame(
        {ID_COLUMN: ids, TARGET_COLUMN: y, "prediction": [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]}
    )
    perm = np.array([5, 4, 3, 2, 1, 0])
    second_oof = pd.DataFrame(
        {
            ID_COLUMN: ids[perm],
            TARGET_COLUMN: y[perm],
            "prediction": np.array([0.15, 0.25, 0.35, 0.65, 0.75, 0.85])[perm],
        }
    )
    first_test = pd.DataFrame({ID_COLUMN: [10, 11], "prediction": [0.4, 0.6]})
    second_test = pd.DataFrame({ID_COLUMN: [11, 10], "prediction": [0.55, 0.45]})

    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    hashes = {"train.csv": "same"}
    _write_completed_run(run_a, first_oof, first_test, data_hashes=hashes)
    _write_completed_run(run_b, second_oof, second_test, data_hashes=hashes)

    out = tmp_path / "blend"
    blend_run_predictions(first_run_dir=run_a, second_run_dir=run_b, output_dir=out)
    blended_test = pd.read_parquet(out / "test_predictions.parquet")
    assert blended_test[ID_COLUMN].tolist() == [10, 11]
    assert np.isfinite(blended_test["prediction"]).all()
    blended_oof = pd.read_parquet(out / "oof_predictions.parquet")
    assert blended_oof[ID_COLUMN].tolist() == ids.tolist()
    assert np.isfinite(blended_oof["prediction"]).all()


def test_blend_rejects_duplicate_oof_ids(tmp_path: Path) -> None:
    y = np.array([0, 1, 1])
    first = pd.DataFrame({ID_COLUMN: [1, 2, 3], TARGET_COLUMN: y, "prediction": [0.1, 0.8, 0.9]})
    second = pd.DataFrame({ID_COLUMN: [1, 1, 3], TARGET_COLUMN: y, "prediction": [0.2, 0.3, 0.7]})
    test = pd.DataFrame({ID_COLUMN: [10, 11], "prediction": [0.4, 0.6]})
    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    hashes = {"train.csv": "same"}
    _write_completed_run(run_a, first, test, data_hashes=hashes)
    _write_completed_run(run_b, second, test, data_hashes=hashes)
    with pytest.raises(TrainingError, match="unique"):
        blend_run_predictions(
            first_run_dir=run_a,
            second_run_dir=run_b,
            output_dir=tmp_path / "blend",
        )


def test_blend_rejects_oof_id_set_mismatch(tmp_path: Path) -> None:
    y = np.array([0, 0, 1, 1])
    first = pd.DataFrame(
        {ID_COLUMN: [1, 2, 3, 4], TARGET_COLUMN: y, "prediction": [0.1, 0.2, 0.8, 0.9]}
    )
    second = pd.DataFrame(
        {ID_COLUMN: [1, 2, 3, 99], TARGET_COLUMN: y, "prediction": [0.15, 0.25, 0.75, 0.85]}
    )
    test = pd.DataFrame({ID_COLUMN: [10, 11], "prediction": [0.4, 0.6]})
    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    hashes = {"train.csv": "same"}
    _write_completed_run(run_a, first, test, data_hashes=hashes)
    _write_completed_run(run_b, second, test, data_hashes=hashes)
    with pytest.raises(TrainingError, match="id set mismatch"):
        blend_run_predictions(
            first_run_dir=run_a,
            second_run_dir=run_b,
            output_dir=tmp_path / "blend",
        )


def test_blend_allows_different_feature_hashes(tmp_path: Path) -> None:
    ids = np.arange(4)
    y = np.array([0, 0, 1, 1])
    oof = pd.DataFrame({ID_COLUMN: ids, TARGET_COLUMN: y, "prediction": [0.1, 0.2, 0.8, 0.9]})
    test = pd.DataFrame({ID_COLUMN: [10, 11], "prediction": [0.4, 0.6]})
    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    _write_completed_run(run_a, oof, test, data_hashes={"train": "aaa", "feature_manifest": "x"})
    _write_completed_run(run_b, oof, test, data_hashes={"train": "bbb", "feature_manifest": "y"})
    payload = blend_run_predictions(
        first_run_dir=run_a,
        second_run_dir=run_b,
        output_dir=tmp_path / "blend",
    )
    assert payload["auc"] >= 0.5


def test_blend_rejects_incomplete_source_run(tmp_path: Path) -> None:
    ids = np.arange(4)
    y = np.array([0, 0, 1, 1])
    oof = pd.DataFrame({ID_COLUMN: ids, TARGET_COLUMN: y, "prediction": [0.1, 0.2, 0.8, 0.9]})
    test = pd.DataFrame({ID_COLUMN: [10, 11], "prediction": [0.4, 0.6]})
    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    _write_completed_run(run_a, oof, test)
    run_b.mkdir()
    oof.to_parquet(run_b / "oof_predictions.parquet", index=False)
    test.to_parquet(run_b / "test_predictions.parquet", index=False)
    (run_b / "manifest.json").write_text(json.dumps({"status": "failed"}), encoding="utf-8")
    with pytest.raises(TrainingError, match="completed"):
        blend_run_predictions(
            first_run_dir=run_a,
            second_run_dir=run_b,
            output_dir=tmp_path / "blend",
        )
