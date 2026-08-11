"""Build and validate Kaggle submission CSV files."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from smartphone_addiction.alignment import align_predictions_to_ids, assert_unique_ids
from smartphone_addiction.data.schema import ID_COLUMN, TARGET_COLUMN
from smartphone_addiction.errors import AlignmentError, SubmissionValidationError

REQUIRED_COLUMNS = [ID_COLUMN, TARGET_COLUMN]


def build_submission(
    sample: pd.DataFrame,
    ids: pd.Series | np.ndarray,
    predictions: np.ndarray,
) -> pd.DataFrame:
    """Build a submission frame aligned to sample_submission id order by ID join."""
    if ID_COLUMN not in sample.columns or TARGET_COLUMN not in sample.columns:
        raise SubmissionValidationError(
            f"sample submission must contain columns {REQUIRED_COLUMNS}"
        )

    try:
        sample_ids = assert_unique_ids(sample[ID_COLUMN], label="sample id")
        aligned = align_predictions_to_ids(
            sample_ids,
            ids,
            predictions,
            label="submission predictions",
        )
    except AlignmentError as exc:
        raise SubmissionValidationError(str(exc)) from exc

    if not np.isfinite(aligned).all():
        raise SubmissionValidationError("predictions must be finite")
    if np.any(aligned < 0.0) or np.any(aligned > 1.0):
        raise SubmissionValidationError("predictions must lie in [0, 1]")

    return pd.DataFrame(
        {
            ID_COLUMN: sample_ids.to_numpy(),
            TARGET_COLUMN: aligned,
        }
    )


def validate_submission_frame(frame: pd.DataFrame, sample: pd.DataFrame | None = None) -> None:
    """Validate an in-memory submission DataFrame."""
    if list(frame.columns) != REQUIRED_COLUMNS:
        raise SubmissionValidationError(
            f"submission columns must be exactly {REQUIRED_COLUMNS}, got {list(frame.columns)}"
        )
    try:
        assert_unique_ids(frame[ID_COLUMN], label="submission id")
    except AlignmentError as exc:
        raise SubmissionValidationError(str(exc)) from exc
    preds = frame[TARGET_COLUMN].to_numpy(dtype=float)
    if not np.isfinite(preds).all():
        raise SubmissionValidationError("predictions must be finite")
    if np.any(preds < 0.0) or np.any(preds > 1.0):
        raise SubmissionValidationError("predictions must lie in [0, 1]")
    if sample is not None:
        try:
            assert_unique_ids(sample[ID_COLUMN], label="sample id")
        except AlignmentError as exc:
            raise SubmissionValidationError(str(exc)) from exc
        if len(frame) != len(sample):
            raise SubmissionValidationError("submission row count must match sample submission")
        if not (
            frame[ID_COLUMN].reset_index(drop=True).equals(sample[ID_COLUMN].reset_index(drop=True))
        ):
            raise SubmissionValidationError(
                "submission ids must match sample submission ids in the same order"
            )


def write_submission(
    frame: pd.DataFrame,
    output_csv: Path | str,
    *,
    sample: pd.DataFrame | None = None,
    metadata: dict[str, Any] | None = None,
    force: bool = False,
) -> dict[str, Path]:
    """Write submission CSV and sidecar JSON as a best-effort transactional pair."""
    validate_submission_frame(frame, sample=sample)
    output_csv = Path(output_csv)
    sidecar = output_csv.with_name(output_csv.stem + ".meta.json")
    if not force and (output_csv.exists() or sidecar.exists()):
        raise SubmissionValidationError(
            f"submission output already exists: {output_csv}; pass force=True to replace"
        )
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    tmp_csv = output_csv.with_suffix(output_csv.suffix + ".tmp")
    tmp_json = sidecar.with_suffix(sidecar.suffix + ".tmp")
    backup_csv = output_csv.with_suffix(output_csv.suffix + ".bak")
    backup_meta = sidecar.with_suffix(sidecar.suffix + ".bak")

    frame.to_csv(tmp_csv, index=False)
    reloaded = pd.read_csv(tmp_csv)
    validate_submission_frame(reloaded, sample=sample)
    sha256 = hashlib.sha256(tmp_csv.read_bytes()).hexdigest()
    payload = {
        "created_at": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "csv_path": str(output_csv),
        "sha256": sha256,
        "n_rows": len(reloaded),
        "columns": list(reloaded.columns),
    }
    if metadata:
        payload.update(metadata)
    tmp_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Stage both temps first; publish CSV then meta; roll back CSV if meta publish fails.
    for path in (backup_csv, backup_meta):
        if path.exists():
            path.unlink()
    if output_csv.exists():
        output_csv.replace(backup_csv)
    if sidecar.exists():
        sidecar.replace(backup_meta)
    try:
        tmp_csv.replace(output_csv)
        try:
            tmp_json.replace(sidecar)
        except Exception:
            if output_csv.exists():
                output_csv.unlink()
            if backup_csv.exists():
                backup_csv.replace(output_csv)
            if backup_meta.exists() and not sidecar.exists():
                backup_meta.replace(sidecar)
            raise
    except Exception:
        if not output_csv.exists() and backup_csv.exists():
            backup_csv.replace(output_csv)
        if not sidecar.exists() and backup_meta.exists():
            backup_meta.replace(sidecar)
        raise
    finally:
        for path in (tmp_csv, tmp_json, backup_csv, backup_meta):
            if path.exists():
                path.unlink()

    return {"csv": output_csv, "meta": sidecar}


def default_submission_csv(run_dir: Path | str) -> Path:
    """Derive a stable local submission path from a run/blend directory name."""
    return Path("submissions") / f"{Path(run_dir).name}.csv"


def build_submission_from_run(
    *,
    run_dir: Path | str,
    sample: pd.DataFrame,
    output_csv: Path | str | None = None,
    force: bool = False,
) -> dict[str, Path]:
    """Create a submission from a completed training run directory."""
    run_dir = Path(run_dir)
    if output_csv is None:
        output_csv = default_submission_csv(run_dir)
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise SubmissionValidationError(f"missing run manifest: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SubmissionValidationError(f"invalid run manifest: {manifest_path}") from exc
    status = manifest.get("status")
    if status != "completed":
        raise SubmissionValidationError(
            f"run status must be 'completed' (got {status!r}): {run_dir}"
        )

    pred_path = run_dir / "test_predictions.parquet"
    metrics_path = run_dir / "metrics.json"
    if not pred_path.is_file():
        raise SubmissionValidationError(f"missing test predictions: {pred_path}")

    preds = pd.read_parquet(pred_path)
    if ID_COLUMN not in preds.columns or "prediction" not in preds.columns:
        raise SubmissionValidationError(
            "test_predictions.parquet must contain id and prediction columns"
        )

    frame = build_submission(sample, preds[ID_COLUMN], preds["prediction"].to_numpy())
    metadata: dict[str, Any] = {
        "run_dir": str(run_dir),
        "source_predictions": str(pred_path),
    }
    if metrics_path.is_file():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metadata["oof_auc"] = metrics.get("oof_auc")
        metadata["model_name"] = metrics.get("model_name")
        metadata["seeds"] = metrics.get("seeds")
    return write_submission(
        frame,
        output_csv,
        sample=sample,
        metadata=metadata,
        force=force,
    )
