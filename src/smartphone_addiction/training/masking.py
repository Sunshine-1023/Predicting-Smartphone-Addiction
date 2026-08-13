"""Train-fold core-field masking augmentation for missingness robustness."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from smartphone_addiction.errors import ConfigurationError, TrainingError
from smartphone_addiction.evaluation.slices import CORE_FIELDS, core_observed_count
from smartphone_addiction.features.domain import (
    BEHAVIORAL_DELTA_COLUMNS,
    BEHAVIORAL_RATIO_COLUMNS,
    BEHAVIORAL_TOTAL_COLUMNS,
    RAW_COLUMNS,
    add_behavioral_totals,
    add_ratio_and_delta_features,
)

# All-observed pattern is useless for masking augmentation.
_ALL_OBSERVED_PATTERN = "1" * len(CORE_FIELDS)


@dataclass(frozen=True)
class MaskingSettings:
    """Serializable train-fold masking configuration."""

    enabled: bool = False
    fraction: float = 0.20

    @classmethod
    def from_mapping(cls, payload: dict[str, Any] | None) -> MaskingSettings:
        data = dict(payload or {})
        fraction = float(data.get("fraction", 0.20))
        if not 0.0 < fraction <= 1.0:
            raise ConfigurationError("features.masking.fraction must be in (0, 1]")
        return cls(enabled=bool(data.get("enabled", False)), fraction=fraction)

    def to_dict(self) -> dict[str, Any]:
        return {"enabled": self.enabled, "fraction": self.fraction}


def core_pattern_keys(frame: pd.DataFrame) -> pd.Series:
    """Return 5-bit core presence keys (1=observed) matching slice metrics."""
    missing = [name for name in CORE_FIELDS if name not in frame.columns]
    if missing:
        raise TrainingError(f"masking requires core fields: {missing}")
    bits = frame.loc[:, list(CORE_FIELDS)].notna().astype(int).astype(str)
    return bits.agg("".join, axis=1)


def test_mask_pattern_distribution(
    test_features: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Return incomplete core patterns and sampling probabilities from test."""
    patterns = core_pattern_keys(test_features)
    counts = patterns.value_counts()
    counts = counts[counts.index != _ALL_OBSERVED_PATTERN]
    if counts.empty:
        raise TrainingError("test has no incomplete core missing patterns for masking")
    keys = counts.index.to_numpy()
    probs = counts.to_numpy(dtype=float)
    probs = probs / probs.sum()
    return keys, probs


def eligible_complete_indices(features: pd.DataFrame) -> np.ndarray:
    """Indices of rows with all core fields observed (fallback: >=4 observed)."""
    observed = core_observed_count(features, CORE_FIELDS)
    complete = np.flatnonzero(observed.to_numpy() == len(CORE_FIELDS))
    if complete.size > 0:
        return complete
    nearly = np.flatnonzero(observed.to_numpy() >= len(CORE_FIELDS) - 1)
    if nearly.size > 0:
        return nearly
    raise TrainingError("no complete/nearly-complete train rows available for masking")


def apply_core_pattern_mask(frame: pd.DataFrame, patterns: np.ndarray | list[str]) -> pd.DataFrame:
    """Mask core raw fields by pattern keys and refresh dependent features."""
    if len(frame) != len(patterns):
        raise TrainingError("mask patterns length must match frame rows")
    out = frame.copy().reset_index(drop=True)
    pattern_arr = np.asarray(patterns, dtype=object)
    for field_i, field in enumerate(CORE_FIELDS):
        drop = np.fromiter(
            (str(pattern)[field_i] == "0" for pattern in pattern_arr),
            dtype=bool,
            count=len(pattern_arr),
        )
        if drop.any():
            out.loc[drop, field] = np.nan
    return sync_features_after_core_mask(out)


def sync_features_after_core_mask(frame: pd.DataFrame) -> pd.DataFrame:
    """Update missingness flags/summaries and behavioral cols after core NaNs."""
    out = frame.copy()
    for field in CORE_FIELDS:
        if field not in out.columns:
            raise TrainingError(f"masking missing core column: {field}")
        flag = f"{field}_is_missing"
        if flag in out.columns:
            out[flag] = out[field].isna().astype("int8")

    flag_cols = [f"{column}_is_missing" for column in RAW_COLUMNS if f"{column}_is_missing" in out]
    if flag_cols and "missing_count" in out.columns:
        flags = out.loc[:, flag_cols]
        out["missing_count"] = flags.sum(axis=1).astype("int16")
        if "missing_ratio" in out.columns:
            out["missing_ratio"] = out["missing_count"] / float(len(flag_cols))
        if "missing_pattern" in out.columns:
            raw_for_pattern = [column for column in RAW_COLUMNS if f"{column}_is_missing" in out]
            missing_flags = out.loc[:, list(raw_for_pattern)].isna()
            pattern_parts = np.where(
                missing_flags.to_numpy(),
                np.array(raw_for_pattern, dtype=object),
                "",
            )
            patterns: list[str] = []
            for row in pattern_parts:
                names = [name for name in row if name]
                patterns.append("|".join(names))
            out["missing_pattern"] = patterns

    needs_behavioral = any(
        column in out.columns
        for column in (
            *BEHAVIORAL_TOTAL_COLUMNS,
            *BEHAVIORAL_RATIO_COLUMNS,
            *BEHAVIORAL_DELTA_COLUMNS,
        )
    )
    if needs_behavioral:
        rebuilt = add_ratio_and_delta_features(add_behavioral_totals(out))
        for column in (
            *BEHAVIORAL_TOTAL_COLUMNS,
            *BEHAVIORAL_RATIO_COLUMNS,
            *BEHAVIORAL_DELTA_COLUMNS,
        ):
            if column in out.columns:
                out[column] = rebuilt[column]
    return out


def augment_training_fold(
    features: pd.DataFrame,
    labels: np.ndarray | pd.Series,
    *,
    test_features: pd.DataFrame,
    settings: MaskingSettings,
    seed: int,
    fold_id: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Copy and mask a fraction of complete train rows; return only the copies.

    The original ``features`` / ``labels`` are never mutated. When masking is
    disabled or fraction yields zero rows, returns an empty frame and empty labels.
    """
    y = np.asarray(labels)
    if len(features) != len(y):
        raise TrainingError("masking features/labels length mismatch")
    if not settings.enabled:
        return features.iloc[0:0].copy(), np.asarray([], dtype=y.dtype)

    n_aug = round(settings.fraction * len(features))
    if n_aug <= 0:
        return features.iloc[0:0].copy(), np.asarray([], dtype=y.dtype)

    rng = np.random.RandomState(int(seed) + int(fold_id))
    pattern_keys, pattern_probs = test_mask_pattern_distribution(test_features)
    eligible = eligible_complete_indices(features)
    chosen = rng.choice(eligible, size=n_aug, replace=True)
    sampled_patterns = rng.choice(pattern_keys, size=n_aug, replace=True, p=pattern_probs)

    copies = features.iloc[chosen].copy()
    copies = apply_core_pattern_mask(copies, sampled_patterns)
    y_copies = y[chosen]
    return copies.reset_index(drop=True), np.asarray(y_copies)
