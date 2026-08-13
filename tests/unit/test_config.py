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


def test_fold_feature_config_is_rejected() -> None:
    root = project_root()
    with pytest.raises(ConfigurationError, match="Unknown configuration key"):
        load_config(
            [root / "configs/base.yaml"],
            ["features.fold.enabled=true"],
            resolve=False,
        )


@pytest.mark.parametrize(
    "override",
    [
        "competition=playground-series-s6e8",
        "target=addicted_label",
        "id_column=id",
        "metric=roc_auc",
        "runtime.environment=kaggle",
    ],
)
def test_unused_identity_config_keys_are_rejected(override: str) -> None:
    root = project_root()
    with pytest.raises(ConfigurationError, match="Unknown configuration key"):
        load_config([root / "configs/base.yaml"], [override], resolve=False)


def test_leftover_identity_yaml_keys_are_rejected(tmp_path: Path) -> None:
    root = project_root()
    leftover = tmp_path / "leftover.yaml"
    leftover.write_text("competition: playground-series-s6e8\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="Extra inputs are not permitted"):
        load_config([root / "configs/base.yaml", leftover], resolve=False)


def test_base_config_has_masking_defaults() -> None:
    root = project_root()
    config = load_config([root / "configs/base.yaml"], resolve=False)
    assert config.features.exclude_columns == []
    assert config.features.masking.enabled is False
    assert config.features.masking.fraction == 0.20


def test_masked_experiment_loads() -> None:
    root = project_root()
    config = load_config(
        [
            root / "configs/base.yaml",
            root / "configs/experiments/lightgbm_masked_v2.yaml",
        ],
        resolve=False,
    )
    assert config.features.masking.enabled is True
    assert config.features.masking.fraction == 0.20
    assert config.features.exclude_columns == ["missing_pattern"]
    assert "categorical_interactions" not in config.features.groups
    assert "behavioral_ratios" not in config.features.groups
    assert config.model.name == "lightgbm"


def test_duplicate_seeds_rejected() -> None:
    root = project_root()
    with pytest.raises(ConfigurationError):
        load_config(
            [root / "configs/base.yaml"],
            ["cv.seeds=[42, 42]"],
            resolve=False,
        )


def test_tune_experiment_budget_fields_load() -> None:
    root = project_root()
    config = load_config(
        [
            root / "configs/base.yaml",
            root / "configs/experiments/archive/catboost_tune_v1.yaml",
        ],
        resolve=False,
    )
    assert config.cv.n_splits == 3
    assert config.cv.seeds == [42]
    assert config.tuning.sample_fraction == 0.5
    assert config.tuning.n_trials == 20
    assert config.tuning.n_candidates == 3


def test_tuning_override_changes_sample_fraction() -> None:
    root = project_root()
    config = load_config(
        [root / "configs/base.yaml"],
        ["tuning.sample_fraction=0.25", "tuning.n_trials=5"],
        resolve=False,
    )
    assert config.tuning.sample_fraction == 0.25
    assert config.tuning.n_trials == 5
