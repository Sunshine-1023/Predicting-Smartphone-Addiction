"""Simple two-model OOF blending (probability and rank)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from smartphone_addiction.data.schema import ID_COLUMN, TARGET_COLUMN
from smartphone_addiction.errors import TrainingError

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
) -> dict[str, Any]:
    """Blend OOF/test predictions from two completed runs and write artifacts."""
    first_dir = Path(first_run_dir)
    second_dir = Path(second_run_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    first_oof = pd.read_parquet(first_dir / "oof_predictions.parquet")
    second_oof = pd.read_parquet(second_dir / "oof_predictions.parquet")
    _assert_aligned_oof(first_oof, second_oof)

    y = first_oof[TARGET_COLUMN].to_numpy()
    result = search_two_model_blend(
        y,
        first_oof["prediction"].to_numpy(),
        second_oof["prediction"].to_numpy(),
        step=step,
    )

    oof_blend = apply_blend(
        first_oof["prediction"].to_numpy(),
        second_oof["prediction"].to_numpy(),
        first_weight=result.first_weight,
        method=result.method,
    )
    oof_frame = pd.DataFrame(
        {
            ID_COLUMN: first_oof[ID_COLUMN].to_numpy(),
            TARGET_COLUMN: y,
            "prediction": oof_blend,
        }
    )
    oof_frame.to_parquet(output_dir / "oof_predictions.parquet", index=False)

    first_test = pd.read_parquet(first_dir / "test_predictions.parquet")
    second_test = pd.read_parquet(second_dir / "test_predictions.parquet")
    if (
        not first_test[ID_COLUMN]
        .reset_index(drop=True)
        .equals(second_test[ID_COLUMN].reset_index(drop=True))
    ):
        raise TrainingError("test prediction ids must match between runs")
    test_blend = apply_blend(
        first_test["prediction"].to_numpy(),
        second_test["prediction"].to_numpy(),
        first_weight=result.first_weight,
        method=result.method,
    )
    test_frame = pd.DataFrame(
        {ID_COLUMN: first_test[ID_COLUMN].to_numpy(), "prediction": test_blend}
    )
    test_frame.to_parquet(output_dir / "test_predictions.parquet", index=False)

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
        "model_name": f"blend-{result.method}",
        "first_weight": result.first_weight,
        "second_weight": result.second_weight,
        "method": result.method,
        "first_auc": result.first_auc,
        "second_auc": result.second_auc,
        "correlation": result.correlation,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    # Minimal completed manifest so `submission build --run <blend_dir>` works.
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": output_dir.name,
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


def _assert_aligned_oof(first: pd.DataFrame, second: pd.DataFrame) -> None:
    required = {ID_COLUMN, TARGET_COLUMN, "prediction"}
    for name, frame in (("first", first), ("second", second)):
        missing = required - set(frame.columns)
        if missing:
            raise TrainingError(f"{name} OOF missing columns: {sorted(missing)}")
    if not first[ID_COLUMN].reset_index(drop=True).equals(second[ID_COLUMN].reset_index(drop=True)):
        raise TrainingError("OOF ids must match between runs in the same order")
    if (
        not first[TARGET_COLUMN]
        .reset_index(drop=True)
        .equals(second[TARGET_COLUMN].reset_index(drop=True))
    ):
        raise TrainingError("OOF targets must match between runs in the same order")
