"""Load official competition CSV files and return validated frames."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from smartphone_addiction.data.validate import validate_competition_frames
from smartphone_addiction.errors import DataValidationError

REQUIRED_FILES = ("train.csv", "test.csv", "sample_submission.csv")


@dataclass(frozen=True)
class CompetitionFrames:
    train: pd.DataFrame
    test: pd.DataFrame
    sample_submission: pd.DataFrame


def load_competition_frames(directory: Path) -> CompetitionFrames:
    """Read the three official CSV files and validate them."""
    directory = Path(directory)
    missing = [name for name in REQUIRED_FILES if not (directory / name).is_file()]
    if missing:
        raise DataValidationError(f"missing competition files in {directory}: {', '.join(missing)}")

    train = pd.read_csv(directory / "train.csv")
    test = pd.read_csv(directory / "test.csv")
    sample = pd.read_csv(directory / "sample_submission.csv")
    validate_competition_frames(train, test, sample)
    return CompetitionFrames(train=train, test=test, sample_submission=sample)
