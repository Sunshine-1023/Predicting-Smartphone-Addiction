"""Build and validate Kaggle submission CSV files."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from smartphone_addiction.data.schema import ID_COLUMN, TARGET_COLUMN
from smartphone_addiction.errors import SubmissionValidationError

REQUIRED_COLUMNS = [ID_COLUMN, TARGET_COLUMN]


def build_submission(
    sample: pd.DataFrame,
    ids: pd.Series | np.ndarray,
    predictions: np.ndarray,
) -> pd.DataFrame:
    """Build a submission frame aligned to sample_submission id order."""
    if ID_COLUMN not in sample.columns or TARGET_COLUMN not in sample.columns:
        raise SubmissionValidationError(
            f"sample submission must contain columns {REQUIRED_COLUMNS}"
        )

    sample_ids = sample[ID_COLUMN].reset_index(drop=True)
    pred_ids = pd.Series(ids).reset_index(drop=True)
    preds = np.asarray(predictions, dtype=float)

    if len(preds) != len(sample_ids):
        raise SubmissionValidationError(
            f"prediction length {len(preds)} != sample rows {len(sample_ids)}"
        )
    if not pred_ids.equals(sample_ids):
        raise SubmissionValidationError(
            "prediction ids must match sample submission ids in the same order"
        )
    if not np.isfinite(preds).all():
        raise SubmissionValidationError("predictions must be finite")
    if np.any(preds < 0.0) or np.any(preds > 1.0):
        raise SubmissionValidationError("predictions must lie in [0, 1]")

    return pd.DataFrame(
        {
            ID_COLUMN: sample_ids.to_numpy(),
            TARGET_COLUMN: preds,
        }
    )


def validate_submission_frame(frame: pd.DataFrame, sample: pd.DataFrame | None = None) -> None:
    """Validate an in-memory submission DataFrame."""
    if list(frame.columns) != REQUIRED_COLUMNS:
        raise SubmissionValidationError(
            f"submission columns must be exactly {REQUIRED_COLUMNS}, got {list(frame.columns)}"
        )
    if frame[ID_COLUMN].isna().any():
        raise SubmissionValidationError("submission id must not contain missing values")
    if not frame[ID_COLUMN].is_unique:
        raise SubmissionValidationError("submission id must be unique")
    preds = frame[TARGET_COLUMN].to_numpy(dtype=float)
    if not np.isfinite(preds).all():
        raise SubmissionValidationError("predictions must be finite")
    if np.any(preds < 0.0) or np.any(preds > 1.0):
        raise SubmissionValidationError("predictions must lie in [0, 1]")
    if sample is not None:
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
) -> dict[str, Path]:
    """Atomically write submission CSV and sidecar JSON metadata."""
    validate_submission_frame(frame, sample=sample)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    sidecar = output_csv.with_name(output_csv.stem + ".meta.json")

    tmp_csv = output_csv.with_suffix(output_csv.suffix + ".tmp")
    frame.to_csv(tmp_csv, index=False)
    tmp_csv.replace(output_csv)

    reloaded = pd.read_csv(output_csv)
    validate_submission_frame(reloaded, sample=sample)

    sha256 = hashlib.sha256(output_csv.read_bytes()).hexdigest()
    payload = {
        "created_at": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "csv_path": str(output_csv),
        "sha256": sha256,
        "n_rows": len(reloaded),
        "columns": list(reloaded.columns),
    }
    if metadata:
        payload.update(metadata)

    tmp_json = sidecar.with_suffix(sidecar.suffix + ".tmp")
    tmp_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp_json.replace(sidecar)

    return {"csv": output_csv, "meta": sidecar}


def build_submission_from_run(
    *,
    run_dir: Path | str,
    sample: pd.DataFrame,
    output_csv: Path | str,
) -> dict[str, Path]:
    """Create a submission from a completed training run directory."""
    run_dir = Path(run_dir)
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
    return write_submission(frame, output_csv, sample=sample, metadata=metadata)
