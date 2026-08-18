"""Independent Lookup Transformer classification configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from smartphone_addiction.config import apply_overrides, deep_merge
from smartphone_addiction.errors import ConfigurationError
from smartphone_addiction.neural.config import (
    NeuralArtifactConfig,
    NeuralCVConfig,
    NeuralDataConfig,
)
from smartphone_addiction.paths import project_root, resolve_path


class LookupTransformerArchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Literal["lookup_transformer"] = "lookup_transformer"
    hidden_dim: int = 128
    n_blocks: int = 4
    n_heads: int = 8
    feedforward_dim: int = 512
    dropout: float = 0.10

    @field_validator("hidden_dim", "n_blocks", "n_heads", "feedforward_dim")
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

    @model_validator(mode="after")
    def _attention_dimensions(self) -> LookupTransformerArchConfig:
        if self.hidden_dim % self.n_heads != 0:
            raise ValueError("lookup hidden_dim must be divisible by n_heads")
        return self


class ClassificationTrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    loss: Literal["bce"] = "bce"
    optimizer: Literal["adamw"] = "adamw"
    learning_rate: float = 0.0005
    weight_decay: float = 0.0001
    batch_size: int = 2048
    max_epochs: int = 80
    early_stopping_patience: int = 10
    gradient_clip_norm: float = 1.0
    holdout_fraction: float = 0.10
    dtype: Literal["float32"] = "float32"
    seed: int = 42

    @field_validator("learning_rate", "gradient_clip_norm")
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


class ExactValueEncodingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    include_unlabeled_vocab: bool = False


class ClassificationGateConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    oof_auc_min: float = 0.9680
    probe_auc_min: float = 0.9650
    coverage_min: float = 1.0
    min_folds: int = 5
    incomplete_auc_min: float | None = None

    @field_validator("oof_auc_min", "probe_auc_min", "coverage_min")
    @classmethod
    def _unit_range(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("must be in [0, 1]")
        return value

    @field_validator("min_folds")
    @classmethod
    def _min_folds_positive(cls, value: int) -> int:
        if value < 1:
            raise ValueError("min_folds must be >= 1")
        return value


class NeuralClassificationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data: NeuralDataConfig = Field(default_factory=NeuralDataConfig)
    artifacts: NeuralArtifactConfig = Field(
        default_factory=lambda: NeuralArtifactConfig(directory="artifacts/neural_classification")
    )
    cv: NeuralCVConfig = Field(default_factory=NeuralCVConfig)
    model: LookupTransformerArchConfig = Field(default_factory=LookupTransformerArchConfig)
    training: ClassificationTrainingConfig = Field(default_factory=ClassificationTrainingConfig)
    encoding: ExactValueEncodingConfig = Field(default_factory=ExactValueEncodingConfig)
    gate: ClassificationGateConfig = Field(default_factory=ClassificationGateConfig)
    device: Literal["auto", "mps", "cpu"] = "auto"

    def resolve_paths(self, root: Path | None = None) -> NeuralClassificationConfig:
        base = root or project_root()
        payload = self.model_dump()
        payload["data"]["directory"] = str(resolve_path(self.data.directory, base))
        payload["artifacts"]["directory"] = str(resolve_path(self.artifacts.directory, base))
        return NeuralClassificationConfig.model_validate(payload)


def load_classification_config(
    paths: list[Path],
    overrides: list[str] | None = None,
    *,
    resolve: bool = True,
) -> NeuralClassificationConfig:
    """Merge YAML files left-to-right and validate the Lookup Transformer config."""
    if not paths:
        raise ConfigurationError("load_classification_config requires at least one YAML path")
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
        config = NeuralClassificationConfig.model_validate(merged)
    except Exception as exc:
        raise ConfigurationError(str(exc)) from exc
    if resolve:
        return config.resolve_paths()
    return config
