"""Unit tests for typed configuration loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from smartphone_addiction.config import CVConfig, load_config
from smartphone_addiction.errors import ConfigurationError
from smartphone_addiction.paths import project_root


def test_final_profile_has_expected_seeds() -> None:
    root = project_root()
    config = load_config(
        [
            root / "configs/base.yaml",
            root / "configs/profiles/final.yaml",
            root / "configs/models/catboost.yaml",
        ],
        resolve=False,
    )
    assert config.cv == CVConfig(n_splits=5, seeds=[42, 2026, 3407])
    assert config.model.name == "catboost"
    assert config.model.params["depth"] == 6


def test_cli_override_replaces_nested_value() -> None:
    root = project_root()
    config = load_config(
        [root / "configs/base.yaml"],
        ["runtime.threads=2"],
        resolve=False,
    )
    assert config.runtime.threads == 2


def test_unknown_override_is_rejected() -> None:
    root = project_root()
    with pytest.raises(ConfigurationError, match="Unknown configuration key"):
        load_config([root / "configs/base.yaml"], ["runtime.typo=2"], resolve=False)


def test_smoke_profile_sets_sample_and_folds() -> None:
    root = project_root()
    config = load_config(
        [
            root / "configs/base.yaml",
            root / "configs/profiles/smoke.yaml",
            root / "configs/models/lightgbm.yaml",
        ],
        resolve=False,
    )
    assert config.data.sample_rows == 5000
    assert config.cv.n_splits == 2
    assert config.cv.seeds == [42]
    assert config.model.name == "lightgbm"


def test_resolve_paths_are_absolute() -> None:
    root = project_root()
    config = load_config([root / "configs/base.yaml"], resolve=True)
    assert Path(config.data.directory).is_absolute()
    assert Path(config.artifacts.directory).is_absolute()
    assert str(config.data.directory).endswith("data/raw")


def test_rejects_unsupported_model_name() -> None:
    root = project_root()
    with pytest.raises(ConfigurationError):
        load_config(
            [root / "configs/base.yaml"],
            ["model.name=logistic"],
            resolve=False,
        )


def test_duplicate_seeds_rejected() -> None:
    root = project_root()
    with pytest.raises(ConfigurationError):
        load_config(
            [root / "configs/base.yaml"],
            ["cv.seeds=[42, 42]"],
            resolve=False,
        )
