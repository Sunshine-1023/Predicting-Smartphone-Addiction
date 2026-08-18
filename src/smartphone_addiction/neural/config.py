"""Independent reconstruction config; not part of the tree-model RunConfig."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from smartphone_addiction.config import apply_overrides, deep_merge
from smartphone_addiction.errors import ConfigurationError
from smartphone_addiction.paths import project_root, resolve_path

CORE5_FIELDS: tuple[str, ...] = (
    "daily_screen_time_hours",
    "weekend_screen_time",
    "social_media_hours",
    "work_study_hours",
    "gaming_hours",
)
TOP3_CORE_FIELDS: tuple[str, ...] = (
    "daily_screen_time_hours",
    "weekend_screen_time",
    "social_media_hours",
)
MISSING_TOKEN = "__MISSING__"
UNKNOWN_TOKEN = "__UNKNOWN__"


class NeuralDataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    directory: str = "data/raw"
    sample_rows: int | None = None

    @field_validator("sample_rows")
    @classmethod
    def _sample_rows_positive(cls, value: int | None) -> int | None:
        if value is not None and value < 1:
            raise ValueError("sample_rows must be >= 1 when set")
        return value


class NeuralArtifactConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    directory: str = "artifacts/reconstruction"


class NeuralModelArchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Literal["mlp"] = "mlp"
    hidden_dim: int = 128
    latent_dim: int = 32
    n_blocks: int = 3
    dropout: float = 0.10
    activation: Literal["gelu"] = "gelu"
    normalization: Literal["layer_norm"] = "layer_norm"
    embedding_dim: int = 8

    @field_validator("hidden_dim", "latent_dim", "n_blocks", "embedding_dim")
    @classmethod
    def _positive(cls, value: int) -> int:
        if value < 1:
            raise ValueError("must be >= 1")
        return value

    @field_validator("dropout")
    @classmethod
    def _dropout_range(cls, value: float) -> float:
        if not 0.0 <= value < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        return value


class NeuralTrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    loss: Literal["huber"] = "huber"
    huber_delta: float = 1.0
    optimizer: Literal["adamw"] = "adamw"
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    batch_size: int = 4096
    max_epochs: int = 50
    early_stopping_patience: int = 5
    gradient_clip_norm: float = 1.0
    holdout_fraction: float = 0.10
    dtype: Literal["float32"] = "float32"
    seed: int = 42
    num_workers: int = 0
    pin_memory: bool = False

    @field_validator("huber_delta", "learning_rate", "gradient_clip_norm")
    @classmethod
    def _positive_float(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("must be > 0")
        return value

    @field_validator("weight_decay")
    @classmethod
    def _non_negative(cls, value: float) -> float:
        if value < 0:
            raise ValueError("must be >= 0")
        return value

    @field_validator("batch_size", "max_epochs", "early_stopping_patience")
    @classmethod
    def _positive_int(cls, value: int) -> int:
        if value < 1:
            raise ValueError("must be >= 1")
        return value

    @field_validator("holdout_fraction")
    @classmethod
    def _holdout_range(cls, value: float) -> float:
        if not 0.0 < value < 0.5:
            raise ValueError("holdout_fraction must be in (0, 0.5)")
        return value


class NeuralMaskingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_fields: int = 1
    max_fields: int = 3
    valid_repeats: int = 3
    field_balance_prob: float = 0.20
    min_eval_per_field: int = 50

    @field_validator("min_fields", "max_fields", "valid_repeats", "min_eval_per_field")
    @classmethod
    def _positive_int(cls, value: int) -> int:
        if value < 1:
            raise ValueError("must be >= 1")
        return value

    @field_validator("field_balance_prob")
    @classmethod
    def _prob_range(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("field_balance_prob must be in [0, 1]")
        return value

    @field_validator("max_fields")
    @classmethod
    def _max_ge_min(cls, value: int, info: Any) -> int:
        minimum = info.data.get("min_fields", 1)
        if value < minimum:
            raise ValueError("max_fields must be >= min_fields")
        if value > len(CORE5_FIELDS):
            raise ValueError("max_fields cannot exceed the number of core fields")
        return value


class ReconstructionGateConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    r2_min: float = 0.10
    spearman_min: float = 0.30
    rmse_improvement_min: float = 0.10
    min_positive_folds: int = 3
    min_passing_fields: int = 3
    min_top3_passing: int = 2


class NeuralCVConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n_splits: int = 5
    seed: int = 42

    @field_validator("n_splits")
    @classmethod
    def _n_splits_min(cls, value: int) -> int:
        if value < 2:
            raise ValueError("n_splits must be at least 2")
        return value


class NeuralReconstructionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data: NeuralDataConfig = Field(default_factory=NeuralDataConfig)
    artifacts: NeuralArtifactConfig = Field(default_factory=NeuralArtifactConfig)
    cv: NeuralCVConfig = Field(default_factory=NeuralCVConfig)
    model: NeuralModelArchConfig = Field(default_factory=NeuralModelArchConfig)
    training: NeuralTrainingConfig = Field(default_factory=NeuralTrainingConfig)
    masking: NeuralMaskingConfig = Field(default_factory=NeuralMaskingConfig)
    gate: ReconstructionGateConfig = Field(default_factory=ReconstructionGateConfig)
    device: Literal["auto", "mps", "cpu"] = "auto"

    def resolve_paths(self, root: Path | None = None) -> NeuralReconstructionConfig:
        base = root or project_root()
        payload = self.model_dump()
        payload["data"]["directory"] = str(resolve_path(self.data.directory, base))
        payload["artifacts"]["directory"] = str(resolve_path(self.artifacts.directory, base))
        return NeuralReconstructionConfig.model_validate(payload)


def load_neural_config(
    paths: list[Path],
    overrides: list[str] | None = None,
    *,
    resolve: bool = True,
) -> NeuralReconstructionConfig:
    """Merge YAML files left-to-right and validate NeuralReconstructionConfig."""
    if not paths:
        raise ConfigurationError("load_neural_config requires at least one YAML path")
    merged: dict[str, Any] = {}
    for path in paths:
        path = Path(path)
        if not path.is_file():
            raise ConfigurationError(f"configuration file not found: {path}")
        try:
            loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            raise ConfigurationError(f"invalid YAML in {path}: {exc}") from exc
        if not isinstance(loaded, dict):
            raise ConfigurationError(f"configuration root must be a mapping: {path}")
        merged = deep_merge(merged, loaded)
    if overrides:
        merged = apply_overrides(merged, overrides)
    try:
        config = NeuralReconstructionConfig.model_validate(merged)
    except Exception as exc:
        raise ConfigurationError(str(exc)) from exc
    if resolve:
        return config.resolve_paths()
    return config
