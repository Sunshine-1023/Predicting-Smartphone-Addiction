"""Unit tests for latent parquet join helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from smartphone_addiction.data.schema import ID_COLUMN
from smartphone_addiction.errors import TrainingError
from smartphone_addiction.features.latent import join_latent_features


def test_join_latent_features_preserves_order_and_coverage(tmp_path: Path) -> None:
    train = pd.DataFrame({ID_COLUMN: [10, 20, 30], "x": [1.0, 2.0, 3.0]})
    test = pd.DataFrame({ID_COLUMN: [100, 200], "x": [4.0, 5.0]})
    train_lat = pd.DataFrame(
        {
            ID_COLUMN: [30, 10, 20],
            "latent_00": [0.3, 0.1, 0.2],
            "recon_daily_screen_time_hours": [1.0, 2.0, 3.0],
        }
    )
    test_lat = pd.DataFrame(
        {
            ID_COLUMN: [100, 200],
            "latent_00": [0.4, 0.5],
            "recon_daily_screen_time_hours": [4.0, 5.0],
        }
    )
    latent_dir = tmp_path / "latent"
    latent_dir.mkdir()
    train_lat.to_parquet(latent_dir / "train_oof_latent.parquet", index=False)
    test_lat.to_parquet(latent_dir / "test_latent_mean.parquet", index=False)

    out_train, out_test, extra = join_latent_features(
        train, test, directory=latent_dir, include=["latent"]
    )
    assert extra == ["latent_00"]
    assert list(out_train[ID_COLUMN]) == [10, 20, 30]
    assert list(out_train["latent_00"]) == pytest.approx([0.1, 0.2, 0.3])
    assert "recon_daily_screen_time_hours" not in out_train.columns
    assert list(out_test["latent_00"]) == pytest.approx([0.4, 0.5])


def test_join_latent_features_rejects_null_coverage(tmp_path: Path) -> None:
    train = pd.DataFrame({ID_COLUMN: [1, 2], "x": [1.0, 2.0]})
    test = pd.DataFrame({ID_COLUMN: [9], "x": [3.0]})
    latent_dir = tmp_path / "latent"
    latent_dir.mkdir()
    pd.DataFrame({ID_COLUMN: [1], "latent_00": [0.1]}).to_parquet(
        latent_dir / "train_oof_latent.parquet", index=False
    )
    pd.DataFrame({ID_COLUMN: [9], "latent_00": [0.9]}).to_parquet(
        latent_dir / "test_latent_mean.parquet", index=False
    )
    with pytest.raises(TrainingError, match="nulls"):
        join_latent_features(train, test, directory=latent_dir, include=["latent"])
