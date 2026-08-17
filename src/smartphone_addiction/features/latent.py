"""Join fold-local neural latent / reconstruction columns onto tabular frames."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd

from smartphone_addiction.data.schema import ID_COLUMN
from smartphone_addiction.errors import TrainingError

IncludeKind = Literal["latent", "recon"]

TRAIN_LATENT_NAME = "train_oof_latent.parquet"
TEST_LATENT_NAME = "test_latent_mean.parquet"


def select_latent_columns(columns: list[str], include: list[IncludeKind]) -> list[str]:
    selected: list[str] = []
    if "latent" in include:
        selected.extend(sorted(name for name in columns if name.startswith("latent_")))
    if "recon" in include:
        selected.extend(sorted(name for name in columns if name.startswith("recon_")))
    if not selected:
        raise TrainingError(f"latent include={include!r} selected zero columns from {columns}")
    return selected


def join_latent_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    directory: Path | str,
    include: list[IncludeKind] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Left-join OOF train / mean test latent tables by id and return extra columns."""
    include = list(include or ["latent"])
    root = Path(directory)
    train_path = root / TRAIN_LATENT_NAME
    test_path = root / TEST_LATENT_NAME
    if not train_path.is_file() or not test_path.is_file():
        raise TrainingError(
            f"latent directory missing {TRAIN_LATENT_NAME} or {TEST_LATENT_NAME}: {root}"
        )
    train_lat = pd.read_parquet(train_path)
    test_lat = pd.read_parquet(test_path)
    if ID_COLUMN not in train_lat.columns or ID_COLUMN not in test_lat.columns:
        raise TrainingError("latent parquet must contain id")
    if train_lat[ID_COLUMN].duplicated().any() or test_lat[ID_COLUMN].duplicated().any():
        raise TrainingError("latent parquet contains duplicate ids")

    extra = select_latent_columns(list(train_lat.columns), include)
    missing_test = [name for name in extra if name not in test_lat.columns]
    if missing_test:
        raise TrainingError(f"test latent missing columns: {missing_test}")

    train_out = train.merge(
        train_lat[[ID_COLUMN, *extra]], on=ID_COLUMN, how="left", validate="1:1"
    )
    test_out = test.merge(test_lat[[ID_COLUMN, *extra]], on=ID_COLUMN, how="left", validate="1:1")
    if len(train_out) != len(train) or len(test_out) != len(test):
        raise TrainingError("latent join changed row counts")
    if list(train_out[ID_COLUMN]) != list(train[ID_COLUMN]):
        raise TrainingError("latent join reordered train ids")
    if list(test_out[ID_COLUMN]) != list(test[ID_COLUMN]):
        raise TrainingError("latent join reordered test ids")
    if train_out[extra].isna().any().any():
        raise TrainingError("train latent join produced nulls; coverage must be 1.0")
    if test_out[extra].isna().any().any():
        raise TrainingError("test latent join produced nulls; coverage must be 1.0")
    return train_out, test_out, extra
