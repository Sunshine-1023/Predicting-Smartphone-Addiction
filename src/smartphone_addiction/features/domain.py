"""Deterministic domain feature transforms for the competition pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd

MISSING_TOKEN = "__MISSING__"
CATEGORY_INTERACTION_SEP = "_"
SAFE_DIVIDE_EPS = 1e-12


def safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
    eps: float = SAFE_DIVIDE_EPS,
) -> pd.Series:
    """Return NaN when either side is missing or |denominator| < eps; never ±inf."""
    num = pd.to_numeric(numerator, errors="coerce")
    den = pd.to_numeric(denominator, errors="coerce")
    result = pd.Series(np.nan, index=num.index, dtype="float64")
    valid = num.notna() & den.notna() & (den.abs() >= eps)
    result.loc[valid] = num.loc[valid] / den.loc[valid]
    return result


def add_missingness_features(
    frame: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Add missing_count, missing_ratio, missing_pattern, and {col}_is_missing."""
    out = frame.copy()
    missing_flags = out[feature_columns].isna()
    for column in feature_columns:
        out[f"{column}_is_missing"] = missing_flags[column].astype("int8")

    out["missing_count"] = missing_flags.sum(axis=1).astype("int16")
    out["missing_ratio"] = out["missing_count"] / float(len(feature_columns))

    # Join missing column names in fixed feature order with "|"; "" if none missing.
    pattern_parts = np.where(
        missing_flags.to_numpy(),
        np.array(feature_columns, dtype=object),
        "",
    )
    patterns: list[str] = []
    for row in pattern_parts:
        names = [name for name in row if name]
        patterns.append("|".join(names))
    out["missing_pattern"] = patterns
    return out


def fill_categorical_missing(
    frame: pd.DataFrame,
    categorical_columns: list[str],
) -> pd.DataFrame:
    """Replace categorical nulls with __MISSING__ and cast to string."""
    out = frame.copy()
    for column in categorical_columns:
        series = out[column]
        filled = series.where(series.notna(), MISSING_TOKEN).astype(str)
        # pandas may turn None into "None" via astype(str) before where; normalize empties.
        filled = filled.replace({"nan": MISSING_TOKEN, "None": MISSING_TOKEN, "": MISSING_TOKEN})
        out[column] = filled
    return out


def _sum_with_missing(*parts: pd.Series) -> pd.Series:
    stacked = pd.concat(parts, axis=1)
    total = stacked.sum(axis=1, min_count=len(parts))
    return total


def add_behavioral_totals(frame: pd.DataFrame) -> pd.DataFrame:
    """Add entertainment, work-entertainment gap, known usage, and unaccounted time."""
    out = frame.copy()
    out["entertainment_hours"] = _sum_with_missing(
        out["social_media_hours"],
        out["gaming_hours"],
    )
    work = pd.to_numeric(out["work_study_hours"], errors="coerce")
    entertainment = pd.to_numeric(out["entertainment_hours"], errors="coerce")
    out["work_minus_entertainment"] = (work - entertainment).where(
        work.notna() & entertainment.notna()
    )
    out["known_usage_hours"] = _sum_with_missing(
        out["social_media_hours"],
        out["gaming_hours"],
        out["work_study_hours"],
    )
    daily = pd.to_numeric(out["daily_screen_time_hours"], errors="coerce")
    known = pd.to_numeric(out["known_usage_hours"], errors="coerce")
    unaccounted = daily - known
    unaccounted = unaccounted.where(daily.notna() & known.notna())
    out["unaccounted_screen_time"] = unaccounted
    return out


def add_ratio_and_delta_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add ratio and delta behavioral features using safe_divide."""
    out = frame.copy()
    daily = out["daily_screen_time_hours"]
    sleep = out["sleep_hours"]
    weekend = out["weekend_screen_time"]
    notifications = out["notifications_per_day"]
    opens = out["app_opens_per_day"]
    entertainment = out["entertainment_hours"]
    work = out["work_study_hours"]

    out["screen_to_sleep_ratio"] = safe_divide(daily, sleep)
    out["entertainment_to_screen_ratio"] = safe_divide(entertainment, daily)
    out["work_to_screen_ratio"] = safe_divide(work, daily)
    out["weekend_to_daily_ratio"] = safe_divide(weekend, daily)
    out["notifications_per_screen_hour"] = safe_divide(notifications, daily)
    out["opens_per_screen_hour"] = safe_divide(opens, daily)
    out["opens_per_notification"] = safe_divide(opens, notifications)

    weekend_num = pd.to_numeric(weekend, errors="coerce")
    daily_num = pd.to_numeric(daily, errors="coerce")
    out["weekend_minus_daily"] = (weekend_num - daily_num).where(
        weekend_num.notna() & daily_num.notna()
    )

    notifications_num = pd.to_numeric(notifications, errors="coerce")
    opens_num = pd.to_numeric(opens, errors="coerce")
    out["notifications_minus_opens"] = (notifications_num - opens_num).where(
        notifications_num.notna() & opens_num.notna()
    )
    return out


def add_log_count_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add log1p transforms for notification and app-open counts."""
    out = frame.copy()
    notifications = pd.to_numeric(out["notifications_per_day"], errors="coerce")
    opens = pd.to_numeric(out["app_opens_per_day"], errors="coerce")
    out["log_notifications"] = np.log1p(notifications)
    out["log_app_opens"] = np.log1p(opens)
    return out


def add_categorical_interactions(frame: pd.DataFrame) -> pd.DataFrame:
    """Add pairwise categorical interaction features after missing fill."""
    out = frame.copy()
    gender = out["gender"].astype(str)
    stress = out["stress_level"].astype(str)
    impact = out["academic_work_impact"].astype(str)
    sep = CATEGORY_INTERACTION_SEP
    out["gender_x_stress"] = gender + sep + stress
    out["gender_x_impact"] = gender + sep + impact
    out["stress_x_impact"] = stress + sep + impact
    return out


# --- Feature groups (plan task 4 + domain extensions) ---

SUPPORTED_FEATURE_GROUPS: frozenset[str] = frozenset(
    {
        "raw",
        "missingness",
        "behavioral_totals",
        "behavioral_ratios",
        "behavioral_deltas",
        "log_counts",
        "categorical_interactions",
    }
)

RAW_COLUMNS: list[str] = [
    "age",
    "daily_screen_time_hours",
    "social_media_hours",
    "gaming_hours",
    "work_study_hours",
    "sleep_hours",
    "notifications_per_day",
    "app_opens_per_day",
    "weekend_screen_time",
    "gender",
    "stress_level",
    "academic_work_impact",
]

MISSINGNESS_FLAG_COLUMNS: list[str] = [f"{column}_is_missing" for column in RAW_COLUMNS]
MISSINGNESS_SUMMARY_COLUMNS: list[str] = [
    "missing_count",
    "missing_ratio",
    "missing_pattern",
]
BEHAVIORAL_TOTAL_COLUMNS: list[str] = [
    "entertainment_hours",
    "work_minus_entertainment",
    "known_usage_hours",
    "unaccounted_screen_time",
]
BEHAVIORAL_RATIO_COLUMNS: list[str] = [
    "screen_to_sleep_ratio",
    "entertainment_to_screen_ratio",
    "work_to_screen_ratio",
    "weekend_to_daily_ratio",
    "notifications_per_screen_hour",
    "opens_per_screen_hour",
    "opens_per_notification",
]
BEHAVIORAL_DELTA_COLUMNS: list[str] = [
    "weekend_minus_daily",
    "notifications_minus_opens",
]
LOG_COLUMNS: list[str] = ["log_notifications", "log_app_opens"]
INTERACTION_COLUMNS: list[str] = [
    "gender_x_stress",
    "gender_x_impact",
    "stress_x_impact",
]

GROUP_COLUMNS: dict[str, list[str]] = {
    "raw": list(RAW_COLUMNS),
    "missingness": [*MISSINGNESS_FLAG_COLUMNS, *MISSINGNESS_SUMMARY_COLUMNS],
    "behavioral_totals": list(BEHAVIORAL_TOTAL_COLUMNS),
    "behavioral_ratios": list(BEHAVIORAL_RATIO_COLUMNS),
    "behavioral_deltas": list(BEHAVIORAL_DELTA_COLUMNS),
    "log_counts": list(LOG_COLUMNS),
    "categorical_interactions": list(INTERACTION_COLUMNS),
}

ALL_FEATURE_GROUPS: list[str] = [
    "raw",
    "missingness",
    "behavioral_totals",
    "behavioral_ratios",
    "behavioral_deltas",
    "log_counts",
    "categorical_interactions",
]


def normalize_feature_groups(groups: list[str] | None) -> list[str]:
    """Validate and expand feature groups; None means the full production set."""
    if groups is None:
        return list(ALL_FEATURE_GROUPS)
    if not groups:
        raise ValueError("feature groups must be a non-empty list")
    unknown = [group for group in groups if group not in SUPPORTED_FEATURE_GROUPS]
    if unknown:
        raise ValueError(
            f"unknown feature groups: {unknown}; supported={sorted(SUPPORTED_FEATURE_GROUPS)}"
        )
    # Preserve caller order but drop duplicates.
    seen: set[str] = set()
    ordered: list[str] = []
    for group in groups:
        if group not in seen:
            ordered.append(group)
            seen.add(group)
    if "raw" not in seen:
        # Raw columns are always required as the base schema.
        ordered = ["raw", *ordered]
    return ordered


def columns_for_groups(groups: list[str] | None) -> list[str]:
    """Return feature column names for the selected groups in canonical order."""
    selected = set(normalize_feature_groups(groups))
    columns: list[str] = []
    for group in ALL_FEATURE_GROUPS:
        if group in selected:
            columns.extend(GROUP_COLUMNS[group])
    return columns


def build_features(frame: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    """Build selected feature groups without mutating the input frame.

    Supported groups: raw, missingness, behavioral_totals, behavioral_ratios,
    behavioral_deltas, plus optional log_counts and categorical_interactions.
    Intermediate columns needed by ratios are computed when required but only
    selected group columns are returned (plus always-filled raw categoricals).
    """
    from smartphone_addiction.data.schema import CATEGORICAL_COLUMNS

    selected = normalize_feature_groups(groups)
    selected_set = set(selected)

    working = frame.copy()
    missing_raw = [column for column in RAW_COLUMNS if column not in working.columns]
    if missing_raw:
        raise ValueError(f"frame missing raw columns: {missing_raw}")

    # Missingness must see original nulls before categorical fill.
    if "missingness" in selected_set:
        working = add_missingness_features(working, RAW_COLUMNS)

    working = fill_categorical_missing(working, CATEGORICAL_COLUMNS)

    need_totals = bool(
        selected_set & {"behavioral_totals", "behavioral_ratios", "behavioral_deltas"}
    )
    # Ratios depend on entertainment_hours from totals.
    if need_totals or "behavioral_ratios" in selected_set:
        working = add_behavioral_totals(working)

    if selected_set & {"behavioral_ratios", "behavioral_deltas"}:
        working = add_ratio_and_delta_features(working)

    if "log_counts" in selected_set:
        working = add_log_count_features(working)

    if "categorical_interactions" in selected_set:
        working = add_categorical_interactions(working)

    output_columns = columns_for_groups(selected)
    missing_out = [column for column in output_columns if column not in working.columns]
    if missing_out:
        raise RuntimeError(f"build_features failed to produce columns: {missing_out}")
    return working.loc[:, output_columns]
