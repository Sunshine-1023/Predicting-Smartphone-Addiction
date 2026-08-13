"""Train-fold masking augmentation with configurable field set and sources."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from smartphone_addiction.errors import ConfigurationError, TrainingError
from smartphone_addiction.features.domain import (
    BEHAVIORAL_DELTA_COLUMNS,
    BEHAVIORAL_RATIO_COLUMNS,
    BEHAVIORAL_TOTAL_COLUMNS,
    LOG_COLUMNS,
    RAW_COLUMNS,
    add_behavioral_totals,
    add_log_count_features,
    add_ratio_and_delta_features,
)

FieldSetName = Literal["core5", "top8"]

CORE5_FIELDS: tuple[str, ...] = (
    "daily_screen_time_hours",
    "weekend_screen_time",
    "social_media_hours",
    "work_study_hours",
    "gaming_hours",
)

TOP8_FIELDS: tuple[str, ...] = (
    *CORE5_FIELDS,
    "notifications_per_day",
    "sleep_hours",
    "app_opens_per_day",
)

# Backward-compatible alias for the expanded masking field set.
MASK_FIELDS: tuple[str, ...] = TOP8_FIELDS

FIELD_SETS: dict[str, tuple[str, ...]] = {
    "core5": CORE5_FIELDS,
    "top8": TOP8_FIELDS,
}

_MASKING_VERSION = 3
_PATTERN_TRAIN_EPS_SCALE = 0.5
_ALLOWED_FIELD_SETS = frozenset(FIELD_SETS)


@dataclass(frozen=True)
class MaskingSettings:
    """Serializable train-fold masking configuration."""

    enabled: bool = False
    fraction: float = 0.20
    fields: FieldSetName = "core5"
    compatible_sources: bool = False
    sample_weight: bool = False

    @property
    def mask_fields(self) -> tuple[str, ...]:
        return FIELD_SETS[self.fields]

    @classmethod
    def from_mapping(cls, payload: dict[str, Any] | None) -> MaskingSettings:
        data = dict(payload or {})
        fraction = float(data.get("fraction", 0.20))
        if not 0.0 < fraction <= 1.0:
            raise ConfigurationError("features.masking.fraction must be in (0, 1]")
        fields = str(data.get("fields", "core5"))
        if fields not in _ALLOWED_FIELD_SETS:
            raise ConfigurationError("features.masking.fields must be 'core5' or 'top8'")
        return cls(
            enabled=bool(data.get("enabled", False)),
            fraction=fraction,
            fields=fields,  # type: ignore[arg-type]
            compatible_sources=bool(data.get("compatible_sources", False)),
            sample_weight=bool(data.get("sample_weight", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "fraction": self.fraction,
            "fields": self.fields,
            "field_names": list(self.mask_fields),
            "compatible_sources": self.compatible_sources,
            "sample_weight": self.sample_weight,
            "version": _MASKING_VERSION,
        }

    def to_config_dict(self) -> dict[str, Any]:
        """YAML-safe subset accepted by ``MaskingConfig`` (extra=forbid)."""
        return {
            "enabled": self.enabled,
            "fraction": self.fraction,
            "fields": self.fields,
            "compatible_sources": self.compatible_sources,
            "sample_weight": self.sample_weight,
        }


def mask_pattern_keys(frame: pd.DataFrame, fields: Sequence[str]) -> pd.Series:
    """Return presence keys (1=observed, 0=missing) in ``fields`` order."""
    names = tuple(fields)
    missing = [name for name in names if name not in frame.columns]
    if missing:
        raise TrainingError(f"masking requires fields: {missing}")
    bits = frame.loc[:, list(names)].notna().astype(int).astype(str)
    return bits.agg("".join, axis=1)


def core_pattern_keys(frame: pd.DataFrame) -> pd.Series:
    """Core5 presence keys (alias used by older call sites)."""
    return mask_pattern_keys(frame, CORE5_FIELDS)


def pattern_bit_ints(frame: pd.DataFrame, fields: Sequence[str]) -> np.ndarray:
    """Pack observed bits into integers (left bit = first field)."""
    names = tuple(fields)
    missing = [name for name in names if name not in frame.columns]
    if missing:
        raise TrainingError(f"masking requires fields: {missing}")
    bits = frame.loc[:, list(names)].notna().to_numpy(dtype=np.int32)
    n_fields = bits.shape[1]
    values = np.zeros(len(frame), dtype=np.int32)
    for index in range(n_fields):
        values |= bits[:, index] << (n_fields - 1 - index)
    return values


def compatible_source_indices(source_bits: np.ndarray, target_pattern: str) -> np.ndarray:
    """Rows whose observed fields are a superset of the target pattern's observed fields."""
    target_obs = _pattern_to_int(target_pattern)
    return np.flatnonzero((source_bits & target_obs) == target_obs)


def complete_source_indices(source_bits: np.ndarray, n_fields: int) -> np.ndarray:
    """Rows that observe every field in the masking set."""
    all_observed = (1 << n_fields) - 1
    return np.flatnonzero(source_bits == all_observed)


def mask_pattern_sampling_distribution(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    fields: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Incomplete patterns with sampling probs from test/train frequency ratios.

    Returns ``(keys, sampling_probs, importance)`` aligned to the same patterns.
    ``importance`` is ``p_test / max(p_train, eps)`` before converting to a
    sampling distribution. All-observed patterns are excluded.
    """
    names = tuple(fields)
    all_observed = "1" * len(names)
    test_keys = mask_pattern_keys(test_features, names)
    train_keys = mask_pattern_keys(train_features, names)
    test_counts = test_keys.value_counts()
    test_counts = test_counts[test_counts.index != all_observed]
    if test_counts.empty:
        raise TrainingError("test has no incomplete missing patterns for masking")

    n_train = max(len(train_keys), 1)
    n_test = max(len(test_keys), 1)
    train_counts = train_keys.value_counts()
    keys = test_counts.index.to_numpy()
    p_test = test_counts.to_numpy(dtype=float) / float(n_test)
    p_train = np.array(
        [float(train_counts.get(key, 0)) / float(n_train) for key in keys],
        dtype=float,
    )
    eps = _PATTERN_TRAIN_EPS_SCALE / float(n_train)
    importance = p_test / np.maximum(p_train, eps)
    probs = importance / importance.sum()
    return keys, probs, importance


def apply_core_pattern_mask(
    frame: pd.DataFrame,
    patterns: np.ndarray | list[str],
    fields: Sequence[str],
) -> pd.DataFrame:
    """Mask ``fields`` by pattern keys and refresh dependent features."""
    names = tuple(fields)
    if len(frame) != len(patterns):
        raise TrainingError("mask patterns length must match frame rows")
    out = frame.copy().reset_index(drop=True)
    pattern_arr = np.asarray(patterns, dtype=object)
    expected = len(names)
    for field_i, field in enumerate(names):
        drop = np.fromiter(
            (_pattern_bit_is_missing(pattern, field_i, expected) for pattern in pattern_arr),
            dtype=bool,
            count=len(pattern_arr),
        )
        if drop.any():
            out.loc[drop, field] = np.nan
    return sync_features_after_core_mask(out, names)


def sync_features_after_core_mask(frame: pd.DataFrame, fields: Sequence[str]) -> pd.DataFrame:
    """Update missingness flags/summaries and derived cols after masked NaNs."""
    out = frame.copy()
    for field in fields:
        if field not in out.columns:
            raise TrainingError(f"masking missing field: {field}")
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

    if any(column in out.columns for column in LOG_COLUMNS):
        rebuilt_logs = add_log_count_features(out)
        for column in LOG_COLUMNS:
            if column in out.columns:
                out[column] = rebuilt_logs[column]
    return out


def augment_training_fold(
    features: pd.DataFrame,
    labels: np.ndarray | pd.Series,
    *,
    test_features: pd.DataFrame,
    settings: MaskingSettings,
    seed: int,
    fold_id: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Copy train rows, apply test-like masks, return copies and copy weights.

    When ``compatible_sources`` is true, a source row is eligible if its observed
    mask fields are a superset of the target pattern. Otherwise only complete
    rows are used. Original ``features`` / ``labels`` are never mutated.
    Valid/test are not masked here.
    """
    y = np.asarray(labels)
    if len(features) != len(y):
        raise TrainingError("masking features/labels length mismatch")
    empty_x = features.iloc[0:0].copy()
    empty_y = np.asarray([], dtype=y.dtype)
    empty_w = np.asarray([], dtype=np.float64)
    if not settings.enabled:
        return empty_x, empty_y, empty_w

    n_original = len(features)
    n_aug = round(settings.fraction * n_original)
    if n_aug <= 0:
        return empty_x, empty_y, empty_w

    fields = settings.mask_fields
    keys, probs, importance = mask_pattern_sampling_distribution(features, test_features, fields)
    source_bits = pattern_bit_ints(features, fields)
    if settings.compatible_sources:
        sources_by_pattern = {
            str(key): compatible_source_indices(source_bits, str(key)) for key in keys
        }
    else:
        complete_idx = complete_source_indices(source_bits, n_fields=len(fields))
        sources_by_pattern = {str(key): complete_idx for key in keys}

    usable = np.array([sources_by_pattern[str(key)].size > 0 for key in keys], dtype=bool)
    if not usable.any():
        kind = "compatible" if settings.compatible_sources else "complete"
        raise TrainingError(f"no {kind} train rows available for any incomplete test mask pattern")
    keys = keys[usable]
    probs = probs[usable]
    importance = importance[usable]
    probs = probs / probs.sum()

    rng = np.random.RandomState(int(seed) + int(fold_id))
    sampled_patterns = rng.choice(keys, size=n_aug, replace=True, p=probs)
    chosen: list[int] = []
    for pattern in sampled_patterns:
        sources = sources_by_pattern[str(pattern)]
        chosen.append(int(rng.choice(sources)))
    chosen_idx = np.asarray(chosen, dtype=np.int64)

    copies = features.iloc[chosen_idx].copy()
    copies = apply_core_pattern_mask(copies, sampled_patterns, fields)
    y_copies = y[chosen_idx]

    if settings.sample_weight:
        importance_lookup = {
            str(key): float(value) for key, value in zip(keys, importance, strict=True)
        }
        raw_importance = np.array(
            [importance_lookup[str(pattern)] for pattern in sampled_patterns],
            dtype=np.float64,
        )
        raw_importance = raw_importance / float(raw_importance.mean())
        copy_scale = float(n_original) / float(n_original + n_aug)
        copy_weights = raw_importance * copy_scale
    else:
        copy_weights = np.ones(n_aug, dtype=np.float64)
    return copies.reset_index(drop=True), np.asarray(y_copies), copy_weights


def concat_fit_sample_weight(
    n_original: int,
    copy_weights: np.ndarray,
    settings: MaskingSettings,
) -> np.ndarray | None:
    """Return LightGBM/CatBoost sample weights, or None when weighting is off."""
    if not settings.sample_weight or n_original < 0:
        return None
    return np.concatenate(
        [
            np.ones(n_original, dtype=np.float64),
            np.asarray(copy_weights, dtype=np.float64),
        ]
    )


def _pattern_to_int(pattern: str) -> int:
    text = str(pattern)
    if not text or any(char not in {"0", "1"} for char in text):
        raise TrainingError(f"mask pattern must be bits of 0/1, got {text!r}")
    return int(text, 2)


def _pattern_bit_is_missing(pattern: object, field_i: int, expected: int) -> bool:
    text = str(pattern)
    if len(text) != expected:
        raise TrainingError(f"mask pattern must be {expected} bits, got {text!r}")
    return text[field_i] == "0"
