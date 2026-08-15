"""Unit tests for reconstruction metrics and the capability gate."""

from __future__ import annotations

import numpy as np
import pandas as pd

from smartphone_addiction.evaluation.reconstruction import (
    compute_field_metrics,
    evaluate_reconstruction_gate,
)
from smartphone_addiction.neural.config import CORE5_FIELDS, ReconstructionGateConfig


def _frame(*, r2_easy: bool) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(0)
    for fold in range(5):
        for field in CORE5_FIELDS:
            y_true = rng.normal(6, 1, size=80)
            y_pred = y_true + rng.normal(0, 0.05, size=80) if r2_easy else rng.normal(0, 1, size=80)
            baseline = np.full(80, float(np.median(y_true)))
            for i in range(80):
                rows.append(
                    {
                        "fold": fold,
                        "field": field,
                        "y_true": float(y_true[i]),
                        "y_pred": float(y_pred[i]),
                        "median_baseline": float(baseline[i]),
                    }
                )
    return pd.DataFrame(rows)


def test_gate_passes_when_fields_beat_median_baseline() -> None:
    metrics = compute_field_metrics(_frame(r2_easy=True), n_splits=5)
    decision = evaluate_reconstruction_gate(metrics, ReconstructionGateConfig(), n_splits=5)
    assert decision.passed
    assert decision.n_passing_fields == 5
    assert decision.n_top3_passing == 3


def test_gate_fails_when_predictions_are_noise() -> None:
    metrics = compute_field_metrics(_frame(r2_easy=False), n_splits=5)
    decision = evaluate_reconstruction_gate(metrics, ReconstructionGateConfig(), n_splits=5)
    assert not decision.passed
    assert decision.n_passing_fields == 0
