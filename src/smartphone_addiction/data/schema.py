"""Competition field schema: feature, categorical, numeric, id, and target columns."""

from __future__ import annotations

FEATURE_COLUMNS: list[str] = [
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

CATEGORICAL_COLUMNS: list[str] = [
    "gender",
    "stress_level",
    "academic_work_impact",
]

NUMERIC_COLUMNS: list[str] = [
    column for column in FEATURE_COLUMNS if column not in CATEGORICAL_COLUMNS
]

ID_COLUMN = "id"
TARGET_COLUMN = "addicted_label"

TRAIN_COLUMNS: list[str] = [ID_COLUMN, *FEATURE_COLUMNS, TARGET_COLUMN]
TEST_COLUMNS: list[str] = [ID_COLUMN, *FEATURE_COLUMNS]
SAMPLE_SUBMISSION_COLUMNS: list[str] = [ID_COLUMN, TARGET_COLUMN]
