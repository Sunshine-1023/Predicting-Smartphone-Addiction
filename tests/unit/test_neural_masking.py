"""Unit tests for test-like artificial core5 masking."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from smartphone_addiction.data.schema import ID_COLUMN
from smartphone_addiction.neural.config import CORE5_FIELDS
from smartphone_addiction.neural.masking import (
    build_fixed_validation_mask_bank,
    pattern_distribution_from_test,
)
from smartphone_addiction.neural.preprocessing import FoldTensorizer


def _raw(n: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            ID_COLUMN: np.arange(n),
            "age": rng.normal(30, 5, n),
            "daily_screen_time_hours": rng.normal(6, 1, n),
            "social_media_hours": rng.normal(2, 0.5, n),
            "gaming_hours": rng.normal(1, 0.4, n),
            "work_study_hours": rng.normal(2, 0.5, n),
            "sleep_hours": rng.normal(7, 1, n),
            "notifications_per_day": rng.normal(50, 10, n),
            "app_opens_per_day": rng.normal(80, 15, n),
            "weekend_screen_time": rng.normal(8, 1, n),
            "gender": rng.choice(["Male", "Female"], n),
            "stress_level": rng.choice(["Low", "Medium", "High"], n),
            "academic_work_impact": rng.choice(["No", "Yes"], n),
        }
    )
    frame.loc[0, "daily_screen_time_hours"] = np.nan
    frame.loc[1, ["weekend_screen_time", "social_media_hours"]] = np.nan
    return frame


def test_validation_mask_bank_is_deterministic() -> None:
    train = _raw()
    test = _raw(n=20, seed=1)
    tensorizer = FoldTensorizer().fit(train)
    encoded = tensorizer.transform(train)
    distribution = pattern_distribution_from_test(test)
    first = build_fixed_validation_mask_bank(
        encoded,
        tensorizer,
        seed=7,
        repeats=3,
        pattern_distribution=distribution,
        min_eval_per_field=1,
    )
    second = build_fixed_validation_mask_bank(
        encoded,
        tensorizer,
        seed=7,
        repeats=3,
        pattern_distribution=distribution,
        min_eval_per_field=1,
    )
    assert torch.equal(first.batch.artificial_mask, second.batch.artificial_mask)
    third = build_fixed_validation_mask_bank(
        encoded,
        tensorizer,
        seed=8,
        repeats=3,
        pattern_distribution=distribution,
        min_eval_per_field=1,
    )
    assert not torch.equal(first.batch.artificial_mask, third.batch.artificial_mask)


def test_loss_mask_is_natural_and_artificial() -> None:
    train = _raw()
    test = _raw(n=20, seed=1)
    tensorizer = FoldTensorizer().fit(train)
    encoded = tensorizer.transform(train)
    bank = build_fixed_validation_mask_bank(
        encoded,
        tensorizer,
        seed=3,
        repeats=2,
        pattern_distribution=pattern_distribution_from_test(test),
        min_eval_per_field=1,
    )
    core_obs = torch.as_tensor(np.tile(encoded.core_observed, (2, 1)))
    expected = core_obs & bank.batch.artificial_mask
    assert torch.equal(bank.batch.loss_mask, expected)


def test_artificial_mask_zeros_core_inputs_not_natural_missing_targets() -> None:
    train = _raw()
    tensorizer = FoldTensorizer().fit(train)
    encoded = tensorizer.transform(train)
    bank = build_fixed_validation_mask_bank(
        encoded,
        tensorizer,
        seed=4,
        repeats=1,
        pattern_distribution=pattern_distribution_from_test(_raw(n=20, seed=2)),
        min_eval_per_field=1,
    )
    core_idx = tensorizer.core_indices
    for field_i, column_index in enumerate(core_idx.tolist()):
        hidden = bank.batch.artificial_mask[:, field_i]
        if not bool(hidden.any()):
            continue
        values = bank.batch.masked_numeric[hidden, column_index]
        assert torch.count_nonzero(values) == 0
    natural_missing = ~torch.as_tensor(encoded.core_observed)
    assert not bool((natural_missing & bank.batch.loss_mask).any())
    assert set(CORE5_FIELDS)


def _unique_core_rows(n: int = 8) -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append(
            {
                ID_COLUMN: 100 + i,
                "age": 10.0 + i,
                "daily_screen_time_hours": 1000.0 + i,
                "social_media_hours": 2000.0 + i,
                "gaming_hours": 3000.0 + i,
                "work_study_hours": 4000.0 + i,
                "sleep_hours": 50.0 + i,
                "notifications_per_day": 60.0 + i,
                "app_opens_per_day": 70.0 + i,
                "weekend_screen_time": 5000.0 + i,
                "gender": "Male" if i % 2 == 0 else "Female",
                "stress_level": "Low",
                "academic_work_impact": "No",
            }
        )
    frame = pd.DataFrame(rows)
    frame.loc[0, "gaming_hours"] = np.nan
    return frame


def test_validation_bank_row_ids_use_repeat_major_tile() -> None:
    train = _unique_core_rows()
    tensorizer = FoldTensorizer().fit(train)
    encoded = tensorizer.transform(train)
    repeats = 3
    bank = build_fixed_validation_mask_bank(
        encoded,
        tensorizer,
        seed=5,
        repeats=repeats,
        pattern_distribution=pattern_distribution_from_test(_raw(n=20, seed=2)),
        min_eval_per_field=1,
    )
    assert np.array_equal(bank.batch.row_ids, np.tile(encoded.row_ids, repeats))
    assert np.array_equal(bank.repeats, np.repeat(np.arange(repeats), len(encoded.row_ids)))


def test_expanded_rows_keep_id_numeric_target_and_observed_aligned() -> None:
    train = _unique_core_rows()
    tensorizer = FoldTensorizer().fit(train)
    encoded = tensorizer.transform(train)
    repeats = 3
    bank = build_fixed_validation_mask_bank(
        encoded,
        tensorizer,
        seed=9,
        repeats=repeats,
        pattern_distribution=pattern_distribution_from_test(_raw(n=20, seed=3)),
        min_eval_per_field=1,
    )
    source_by_id = {int(row_id): i for i, row_id in enumerate(encoded.row_ids)}
    recovered = tensorizer.inverse_core(bank.batch.targets.cpu().numpy())
    core_idx = tensorizer.core_indices.tolist()
    core_cols = set(core_idx)
    for expanded_i, row_id in enumerate(bank.batch.row_ids.tolist()):
        source_i = source_by_id[int(row_id)]
        assert torch.equal(
            bank.batch.natural_observed[expanded_i],
            encoded.natural_observed[source_i],
        )
        for field_i, column_index in enumerate(core_idx):
            if encoded.core_observed[source_i, field_i]:
                assert recovered[expanded_i, field_i] == pytest.approx(
                    encoded.core_raw[source_i, field_i],
                    rel=1e-4,
                    abs=1e-3,
                )
            if bool(bank.batch.artificial_mask[expanded_i, field_i]):
                assert bank.batch.masked_numeric[expanded_i, column_index].item() == 0
            else:
                assert torch.allclose(
                    bank.batch.masked_numeric[expanded_i, column_index],
                    encoded.numeric[source_i, column_index],
                )
        for column_index in range(encoded.numeric.shape[1]):
            if column_index in core_cols:
                continue
            assert torch.allclose(
                bank.batch.masked_numeric[expanded_i, column_index],
                encoded.numeric[source_i, column_index],
            )
