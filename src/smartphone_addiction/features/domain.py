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
