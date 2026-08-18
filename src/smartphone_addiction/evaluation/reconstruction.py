"""Field-level reconstruction metrics and the reconstruction capability gate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from smartphone_addiction.neural.config import (
    CORE5_FIELDS,
    TOP3_CORE_FIELDS,
    ReconstructionGateConfig,
)


@dataclass(frozen=True)
class GateDecision:
    passed: bool
    passing_fields: tuple[str, ...]
    top3_passing: tuple[str, ...]
    n_passing_fields: int
    n_top3_passing: int
    reasons: tuple[str, ...]
    field_summary: dict[str, Any]


def _safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 3 or np.unique(y_true).size < 2:
        return float("nan")
    return float(pd.Series(y_true).corr(pd.Series(y_pred), method="spearman"))


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2 or np.unique(y_true).size < 2:
        return float("nan")
    ss_res = float(np.square(y_true - y_pred).sum())
    ss_tot = float(np.square(y_true - y_true.mean()).sum())
    if ss_tot <= 1e-12:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def compute_field_metrics(
    frame: pd.DataFrame,
    *,
    n_splits: int,
    core_fields: tuple[str, ...] = CORE5_FIELDS,
) -> pd.DataFrame:
    required = {"fold", "field", "y_true", "y_pred", "median_baseline"}
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise ValueError(f"reconstruction frame missing columns: {missing}")
    rows: list[dict[str, Any]] = []
    for field in core_fields:
        subset = frame.loc[frame["field"] == field]
        for fold in [*range(n_splits), "oof"]:
            part = subset if fold == "oof" else subset.loc[subset["fold"] == fold]
            y_true = part["y_true"].to_numpy(dtype=float)
            y_pred = part["y_pred"].to_numpy(dtype=float)
            baseline = part["median_baseline"].to_numpy(dtype=float)
            finite = np.isfinite(y_true) & np.isfinite(y_pred) & np.isfinite(baseline)
            y_true = y_true[finite]
            y_pred = y_pred[finite]
            baseline = baseline[finite]
            n_eval = len(y_true)
            if n_eval == 0:
                mae = rmse = nrmse = r2 = spearman = median_rmse = improvement = float("nan")
            else:
                residual = y_true - y_pred
                mae = float(np.abs(residual).mean())
                rmse = float(np.sqrt(np.square(residual).mean()))
                denom = float(np.std(y_true, ddof=0))
                nrmse = float(rmse / denom) if denom > 1e-12 else float("nan")
                r2 = _r2(y_true, y_pred)
                spearman = _safe_spearman(y_true, y_pred)
                median_rmse = float(np.sqrt(np.square(y_true - baseline).mean()))
                improvement = float("nan") if median_rmse <= 1e-12 else 1.0 - rmse / median_rmse
            rows.append(
                {
                    "field": field,
                    "fold": fold,
                    "mae": mae,
                    "rmse": rmse,
                    "nrmse": nrmse,
                    "r2": r2,
                    "spearman": spearman,
                    "median_rmse": median_rmse,
                    "rmse_improvement": improvement,
                    "n_eval": n_eval,
                }
            )
    return pd.DataFrame(rows)


def evaluate_reconstruction_gate(
    field_metrics: pd.DataFrame,
    gate: ReconstructionGateConfig,
    *,
    n_splits: int,
    core_fields: tuple[str, ...] = CORE5_FIELDS,
    top3_fields: tuple[str, ...] = TOP3_CORE_FIELDS,
) -> GateDecision:
    passing: list[str] = []
    reasons: list[str] = []
    summary: dict[str, Any] = {}
    for field in core_fields:
        oof = field_metrics.loc[
            (field_metrics["field"] == field) & (field_metrics["fold"] == "oof")
        ]
        folds = field_metrics.loc[
            (field_metrics["field"] == field) & (field_metrics["fold"] != "oof")
        ]
        if oof.empty:
            summary[field] = {"passed": False, "reason": "missing oof row"}
            reasons.append(f"{field}: missing OOF metrics")
            continue
        row = oof.iloc[0]
        fold_positive = int((folds["rmse_improvement"].to_numpy(dtype=float) > 0).sum())
        checks = {
            "r2": float(row["r2"]) >= gate.r2_min if np.isfinite(row["r2"]) else False,
            "spearman": (
                float(row["spearman"]) >= gate.spearman_min
                if np.isfinite(row["spearman"])
                else False
            ),
            "rmse_improvement": (
                float(row["rmse_improvement"]) >= gate.rmse_improvement_min
                if np.isfinite(row["rmse_improvement"])
                else False
            ),
            "positive_folds": fold_positive >= gate.min_positive_folds,
        }
        passed = all(checks.values())
        summary[field] = {
            "passed": passed,
            "r2": None if not np.isfinite(row["r2"]) else float(row["r2"]),
            "spearman": None if not np.isfinite(row["spearman"]) else float(row["spearman"]),
            "rmse_improvement": (
                None if not np.isfinite(row["rmse_improvement"]) else float(row["rmse_improvement"])
            ),
            "positive_folds": fold_positive,
            "n_splits": n_splits,
            "checks": checks,
        }
        if passed:
            passing.append(field)
        else:
            failed = [name for name, ok in checks.items() if not ok]
            reasons.append(f"{field}: failed {', '.join(failed)}")
    top3_passing = tuple(name for name in top3_fields if name in passing)
    overall = len(passing) >= gate.min_passing_fields and len(top3_passing) >= gate.min_top3_passing
    if len(passing) < gate.min_passing_fields:
        reasons.append(
            f"only {len(passing)}/{len(core_fields)} core fields passed "
            f"(need {gate.min_passing_fields})"
        )
    if len(top3_passing) < gate.min_top3_passing:
        reasons.append(
            f"only {len(top3_passing)}/{len(top3_fields)} top3 fields passed "
            f"(need {gate.min_top3_passing})"
        )
    if overall:
        reasons = ("all reconstruction gates passed",)
    return GateDecision(
        passed=overall,
        passing_fields=tuple(passing),
        top3_passing=top3_passing,
        n_passing_fields=len(passing),
        n_top3_passing=len(top3_passing),
        reasons=tuple(reasons),
        field_summary=summary,
    )
