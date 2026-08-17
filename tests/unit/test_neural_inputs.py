"""Tests for unified input_available encoding."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from smartphone_addiction.data.schema import NUMERIC_COLUMNS
from smartphone_addiction.neural.config import CORE5_FIELDS
from smartphone_addiction.neural.outputs import input_available_mask, input_dim


def test_input_available_unifies_natural_and_artificial_missing() -> None:
    n_numeric = len(NUMERIC_COLUMNS)
    natural = torch.ones((2, n_numeric), dtype=torch.bool)
    artificial = torch.zeros((2, len(CORE5_FIELDS)), dtype=torch.bool)
    daily_i = NUMERIC_COLUMNS.index("daily_screen_time_hours")
    social_i = NUMERIC_COLUMNS.index("social_media_hours")
    # row0: natural missing daily
    natural[0, daily_i] = False
    # row1: artificial hide social
    artificial[1, CORE5_FIELDS.index("social_media_hours")] = True

    available = input_available_mask(natural, artificial)
    assert not bool(available[0, daily_i])
    assert bool(available[0, social_i])
    assert bool(available[1, daily_i])
    assert not bool(available[1, social_i])


def test_input_dim_drops_separate_artificial_channel() -> None:
    assert input_dim(9, 5, 3, 8) == 9 + 9 + 3 * 8
