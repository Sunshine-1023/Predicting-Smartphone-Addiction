"""Simple two-model OOF blending (probability and rank)."""

from __future__ import annotations

import json
import shutil
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from smartphone_addiction.alignment import (
    align_frame_predictions,
    align_predictions_to_ids,
    assert_unique_ids,
)
from smartphone_addiction.data.schema import ID_COLUMN, TARGET_COLUMN
from smartphone_addiction.errors import AlignmentError, TrainingError
from smartphone_addiction.evaluation.slices import CORE_FIELDS, compute_slice_metrics
from smartphone_addiction.paths import project_root

BlendMethod = Literal["probability", "rank"]


@dataclass(frozen=True)
class BlendResult:
    """Result of a weight search over two OOF prediction vectors."""

    first_weight: float
    second_weight: float
    method: BlendMethod
    auc: float
    first_auc: float
    second_auc: float
    correlation: float
    probability_auc: float
    rank_auc: float


def _validate_vectors(y: np.ndarray, first: np.ndarray, second: np.ndarray) -> None:
    labels = np.asarray(y, dtype=float)
    a = np.asarray(first, dtype=float)
    b = np.asarray(second, dtype=float)
    if labels.shape != a.shape or labels.shape != b.shape:
        raise TrainingError("y, first, and second must share the same shape")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise TrainingError("blend predictions must be finite")
    if not np.isfinite(labels).all():
        raise TrainingError("blend labels must be finite")


def apply_probability_blend(
    first: np.ndarray,
    second: np.ndarray,
    first_weight: float,
) -> np.ndarray:
    """Weighted average of probabilities."""
    w = float(first_weight)
    return w * np.asarray(first, dtype=float) + (1.0 - w) * np.asarray(second, dtype=float)


def apply_rank_blend(
    first: np.ndarray,
    second: np.ndarray,
    first_weight: float,
) -> np.ndarray:
    """Weighted average of percentile ranks (average method for ties)."""
    a = pd.Series(np.asarray(first, dtype=float)).rank(method="average", pct=True).to_numpy()
    b = pd.Series(np.asarray(second, dtype=float)).rank(method="average", pct=True).to_numpy()
    w = float(first_weight)
    return w * a + (1.0 - w) * b


def search_two_model_blend(
    y: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    step: float = 0.05,
) -> BlendResult:
    """Grid-search blend weights on OOF predictions; pick best method/AUC."""
    if step <= 0 or step > 1:
        raise TrainingError("step must be in (0, 1]")
    _validate_vectors(y, first, second)
    labels = np.asarray(y, dtype=float)
    a = np.asarray(first, dtype=float)
    b = np.asarray(second, dtype=float)

    first_auc = float(roc_auc_score(labels, a))
    second_auc = float(roc_auc_score(labels, b))
    correlation = float(np.corrcoef(a, b)[0, 1]) if len(a) > 1 else 0.0

    weights = np.round(np.arange(0.0, 1.0 + 1e-12, step), 10)
    best_prob = (-1.0, 0.5)
    best_rank = (-1.0, 0.5)
    for weight in weights:
        prob = apply_probability_blend(a, b, weight)
        rank = apply_rank_blend(a, b, weight)
        prob_auc = float(roc_auc_score(labels, prob))
        rank_auc = float(roc_auc_score(labels, rank))
        if prob_auc > best_prob[0]:
            best_prob = (prob_auc, float(weight))
        if rank_auc > best_rank[0]:
            best_rank = (rank_auc, float(weight))

    if best_rank[0] > best_prob[0]:
        method: BlendMethod = "rank"
        auc, weight = best_rank
    else:
        method = "probability"
        auc, weight = best_prob

    return BlendResult(
        first_weight=weight,
        second_weight=1.0 - weight,
        method=method,
        auc=auc,
        first_auc=first_auc,
        second_auc=second_auc,
        correlation=correlation,
        probability_auc=best_prob[0],
        rank_auc=best_rank[0],
    )


def apply_blend(
    first: np.ndarray,
    second: np.ndarray,
    *,
    first_weight: float,
    method: BlendMethod,
) -> np.ndarray:
    """Apply a previously selected blend method to any prediction vectors."""
    if method == "probability":
        return apply_probability_blend(first, second, first_weight)
    if method == "rank":
        return apply_rank_blend(first, second, first_weight)
    raise TrainingError(f"unknown blend method: {method!r}")


def blend_run_predictions(
    *,
    first_run_dir: Path | str,
    second_run_dir: Path | str,
    output_dir: Path | str,
    step: float = 0.05,
    force: bool = False,
) -> dict[str, Any]:
    """Blend OOF/test predictions from two completed runs and write artifacts.

    Writes into a staging directory first and publishes ``manifest.json`` last.
    Refuses to overwrite an existing non-empty ``output_dir`` unless ``force``.
    """
    first_dir = Path(first_run_dir)
    second_dir = Path(second_run_dir)
    output_dir = Path(output_dir)
    _assert_completed_run(first_dir, "first")
    _assert_completed_run(second_dir, "second")
    _assert_compatible_source_runs(first_dir, second_dir)

    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        raise TrainingError(
            f"blend output already exists: {output_dir}; pass force=True to replace"
        )

    staging = output_dir.parent / f".{output_dir.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir(parents=True, exist_ok=False)
    try:
        payload = _write_blend_artifacts(
            first_dir=first_dir,
            second_dir=second_dir,
            output_dir=staging,
            final_run_id=output_dir.name,
            step=step,
        )
        _publish_directory(staging, output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return payload


def _write_blend_artifacts(
    *,
    first_dir: Path,
    second_dir: Path,
    output_dir: Path,
    final_run_id: str,
    step: float,
) -> dict[str, Any]:
    first_oof = pd.read_parquet(first_dir / "oof_predictions.parquet")
    second_oof = pd.read_parquet(second_dir / "oof_predictions.parquet")
    base_ids, y, first_oof_pred, second_oof_pred = _aligned_oof_vectors(first_oof, second_oof)

    result = search_two_model_blend(y, first_oof_pred, second_oof_pred, step=step)

    oof_blend = apply_blend(
        first_oof_pred,
        second_oof_pred,
        first_weight=result.first_weight,
        method=result.method,
    )
    oof_frame = pd.DataFrame(
        {
            ID_COLUMN: base_ids.to_numpy(),
            TARGET_COLUMN: y,
            "prediction": oof_blend,
        }
    )
    oof_frame.to_parquet(output_dir / "oof_predictions.parquet", index=False)

    first_test = pd.read_parquet(first_dir / "test_predictions.parquet")
    second_test = pd.read_parquet(second_dir / "test_predictions.parquet")
    test_ids, first_test_pred, second_test_pred = _aligned_test_vectors(first_test, second_test)
    test_blend = apply_blend(
        first_test_pred,
        second_test_pred,
        first_weight=result.first_weight,
        method=result.method,
    )
    test_frame = pd.DataFrame({ID_COLUMN: test_ids.to_numpy(), "prediction": test_blend})
    test_frame.to_parquet(output_dir / "test_predictions.parquet", index=False)

    slice_metrics = _blend_slice_metrics(oof_frame)
    payload = {
        "first_run_dir": str(first_dir),
        "second_run_dir": str(second_dir),
        **asdict(result),
    }
    (output_dir / "blend_result.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    metrics = {
        "oof_auc": result.auc,
        "oof_auc_note": "in_sample_weight_search_on_oof",
        "model_name": f"blend-{result.method}",
        "first_weight": result.first_weight,
        "second_weight": result.second_weight,
        "method": result.method,
        "first_auc": result.first_auc,
        "second_auc": result.second_auc,
        "correlation": result.correlation,
    }
    if slice_metrics:
        for key in (
            "core_complete_auc",
            "core_incomplete_auc",
            "top3_incomplete_auc",
            "test_pattern_weighted_auc",
        ):
            if key in slice_metrics:
                metrics[key] = slice_metrics[key]
        (output_dir / "slice_metrics.json").write_text(
            json.dumps(slice_metrics, indent=2) + "\n",
            encoding="utf-8",
        )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    # Publish completed manifest last so partial staging directories are never "completed".
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": final_run_id,
                "slug": "blend",
                "status": "completed",
                "artifact_type": "blend",
                "source_runs": [str(first_dir), str(second_dir)],
                "metrics": metrics,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return payload


def _publish_directory(staging: Path, output_dir: Path) -> None:
    """Replace ``output_dir`` with ``staging`` after all artifacts are written."""
    backup: Path | None = None
    if output_dir.exists():
        backup = output_dir.parent / f".{output_dir.name}.backup-{uuid.uuid4().hex}"
        output_dir.rename(backup)
    try:
        staging.rename(output_dir)
    except Exception:
        if backup is not None and not output_dir.exists():
            backup.rename(output_dir)
        raise
    if backup is not None:
        shutil.rmtree(backup, ignore_errors=True)


def _assert_completed_run(run_dir: Path, label: str) -> None:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise TrainingError(f"{label} run missing manifest.json: {run_dir}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TrainingError(f"{label} run has invalid manifest.json: {run_dir}") from exc
    status = manifest.get("status")
    if status != "completed":
        raise TrainingError(f"{label} run status must be 'completed' (got {status!r}): {run_dir}")
    for name in ("oof_predictions.parquet", "test_predictions.parquet"):
        if not (run_dir / name).is_file():
            raise TrainingError(f"{label} run missing {name}: {run_dir}")


def _assert_compatible_source_runs(first_dir: Path, second_dir: Path) -> None:
    first = json.loads((first_dir / "manifest.json").read_text(encoding="utf-8"))
    second = json.loads((second_dir / "manifest.json").read_text(encoding="utf-8"))
    first_n = first.get("n_train_rows")
    second_n = second.get("n_train_rows")
    if first_n is not None and second_n is not None and int(first_n) != int(second_n):
        raise TrainingError(
            f"blend source runs have different n_train_rows ({first_n} vs {second_n})"
        )
    # Feature-column hashes may differ across models; identity is enforced by
    # aligned OOF IDs and labels below, not by processed-feature fingerprints.


def _aligned_oof_vectors(
    first: pd.DataFrame,
    second: pd.DataFrame,
) -> tuple[pd.Series, np.ndarray, np.ndarray, np.ndarray]:
    required = {ID_COLUMN, TARGET_COLUMN, "prediction"}
    for name, frame in (("first", first), ("second", second)):
        missing = required - set(frame.columns)
        if missing:
            raise TrainingError(f"{name} OOF missing columns: {sorted(missing)}")
    try:
        base_ids = assert_unique_ids(first[ID_COLUMN], label="first OOF id")
        first_pred = np.asarray(first["prediction"], dtype=float)
        if len(first_pred) != len(base_ids):
            raise AlignmentError("first OOF prediction length mismatch")
        if not np.isfinite(first_pred).all():
            raise AlignmentError("first OOF predictions must be finite")
        second_pred = align_frame_predictions(base_ids, second, label="second OOF")
        second_y = align_predictions_to_ids(
            base_ids,
            second[ID_COLUMN],
            second[TARGET_COLUMN].to_numpy(),
            label="second OOF target",
        )
    except AlignmentError as exc:
        raise TrainingError(str(exc)) from exc
    y = first[TARGET_COLUMN].to_numpy(dtype=float)
    if not np.array_equal(y, second_y):
        raise TrainingError("OOF targets must match between runs after ID alignment")
    return base_ids, y, first_pred, second_pred


def _aligned_test_vectors(
    first: pd.DataFrame,
    second: pd.DataFrame,
) -> tuple[pd.Series, np.ndarray, np.ndarray]:
    for name, frame in (("first", first), ("second", second)):
        if ID_COLUMN not in frame.columns or "prediction" not in frame.columns:
            raise TrainingError(f"{name} test predictions must contain id and prediction")
    try:
        base_ids = assert_unique_ids(first[ID_COLUMN], label="first test id")
        first_pred = np.asarray(first["prediction"], dtype=float)
        if len(first_pred) != len(base_ids):
            raise AlignmentError("first test prediction length mismatch")
        if not np.isfinite(first_pred).all():
            raise AlignmentError("first test predictions must be finite")
        second_pred = align_frame_predictions(base_ids, second, label="second test")
    except AlignmentError as exc:
        raise TrainingError(str(exc)) from exc
    return base_ids, first_pred, second_pred


def _blend_slice_metrics(oof_frame: pd.DataFrame) -> dict[str, Any] | None:
    """Compute completeness slices when processed core fields are available."""
    root = project_root()
    train_path = root / "data" / "processed" / "train_features.parquet"
    test_path = root / "data" / "processed" / "test_features.parquet"
    if not train_path.is_file():
        return None
    columns = [ID_COLUMN, *CORE_FIELDS]
    train = pd.read_parquet(train_path, columns=columns)
    merged = train.merge(oof_frame, on=ID_COLUMN, how="inner")
    if len(merged) != len(oof_frame):
        return None
    test = pd.read_parquet(test_path, columns=columns) if test_path.is_file() else None
    return compute_slice_metrics(
        merged[columns],
        merged[TARGET_COLUMN],
        merged["prediction"],
        test_features=test,
    )
