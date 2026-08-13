"""Typed experiment configuration loading and merging."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from smartphone_addiction.errors import ConfigurationError
from smartphone_addiction.paths import project_root, resolve_path


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    directory: str = "data/raw"
    processed_directory: str = "data/processed"
    sample_rows: int | None = None

    @field_validator("sample_rows")
    @classmethod
    def _sample_rows_positive(cls, value: int | None) -> int | None:
        if value is not None and value < 1:
            raise ValueError("sample_rows must be >= 1 when set")
        return value


class ArtifactConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    directory: str = "artifacts/runs"


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    threads: int = 4

    @field_validator("threads")
    @classmethod
    def _threads_positive(cls, value: int) -> int:
        if value < 1:
            raise ValueError("threads must be positive")
        return value


class CVConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n_splits: int = 5
    seeds: list[int] = Field(default_factory=lambda: [42])

    @field_validator("n_splits")
    @classmethod
    def _n_splits_min(cls, value: int) -> int:
        if value < 2:
            raise ValueError("n_splits must be at least 2")
        return value

    @field_validator("seeds")
    @classmethod
    def _seeds_unique(cls, value: list[int]) -> list[int]:
        if not value:
            raise ValueError("seeds must be non-empty")
        if len(value) != len(set(value)):
            raise ValueError("seeds must not contain duplicates")
        return value


class MaskingConfig(BaseModel):
    """Train-fold core-field masking augmentation (disabled by default)."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    fraction: float = 0.20

    @field_validator("fraction")
    @classmethod
    def _fraction_range(cls, value: float) -> float:
        if not 0.0 < value <= 1.0:
            raise ValueError("features.masking.fraction must be in (0, 1]")
        return value


class FeatureConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    groups: list[str] = Field(default_factory=lambda: ["raw"])
    exclude_columns: list[str] = Field(default_factory=list)
    masking: MaskingConfig = Field(default_factory=MaskingConfig)


class TuningConfig(BaseModel):
    """Bounded Optuna search budget (distinct from full final CV)."""

    model_config = ConfigDict(extra="forbid")

    sample_fraction: float = 0.5
    n_trials: int = 20
    n_candidates: int = 3

    @field_validator("sample_fraction")
    @classmethod
    def _sample_fraction_range(cls, value: float) -> float:
        if not 0.0 < value <= 1.0:
            raise ValueError("tuning.sample_fraction must be in (0, 1]")
        return value

    @field_validator("n_trials", "n_candidates")
    @classmethod
    def _positive_int(cls, value: int) -> int:
        if value < 1:
            raise ValueError("must be >= 1")
        return value


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    params: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def _supported_model(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"catboost", "lightgbm"}:
            raise ValueError("model.name must be one of: catboost, lightgbm")
        return normalized


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data: DataConfig
    artifacts: ArtifactConfig
    runtime: RuntimeConfig
    cv: CVConfig
    features: FeatureConfig
    model: ModelConfig
    tuning: TuningConfig = Field(default_factory=TuningConfig)

    def resolve_paths(self, root: Path | None = None) -> RunConfig:
        """Return a copy with data/artifact directories resolved to absolute paths."""
        base = root or project_root()
        payload = self.model_dump()
        payload["data"]["directory"] = str(resolve_path(self.data.directory, base))
        payload["data"]["processed_directory"] = str(
            resolve_path(self.data.processed_directory, base)
        )
        payload["artifacts"]["directory"] = str(resolve_path(self.artifacts.directory, base))
        return RunConfig.model_validate(payload)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Return a recursive copy with override values taking precedence."""
    merged: dict[str, Any] = deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)  # type: ignore[arg-type]
        else:
            merged[key] = deepcopy(value)
    return merged


def apply_overrides(config: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply validated dotted key=value overrides parsed with yaml.safe_load."""
    result = deepcopy(config)
    for item in overrides:
        if "=" not in item:
            raise ConfigurationError(f"Invalid override {item!r}; expected dotted key=value")
        dotted_key, raw_value = item.split("=", 1)
        parts = [part for part in dotted_key.split(".") if part]
        if not parts:
            raise ConfigurationError(f"Invalid override key in {item!r}")

        try:
            value = yaml.safe_load(raw_value)
        except yaml.YAMLError as exc:
            raise ConfigurationError(f"Could not parse override value in {item!r}") from exc

        cursor: Any = result
        for part in parts[:-1]:
            if not isinstance(cursor, dict) or part not in cursor:
                raise ConfigurationError(f"Unknown configuration key: {dotted_key}")
            if not isinstance(cursor[part], dict):
                raise ConfigurationError(f"Unknown configuration key: {dotted_key}")
            cursor = cursor[part]
        leaf = parts[-1]
        if not isinstance(cursor, dict) or leaf not in cursor:
            raise ConfigurationError(f"Unknown configuration key: {dotted_key}")
        cursor[leaf] = value
    return result


def load_config(
    paths: list[Path],
    overrides: list[str] | None = None,
    *,
    resolve: bool = True,
) -> RunConfig:
    """Merge YAML files left-to-right, apply overrides, and validate RunConfig."""
    if not paths:
        raise ConfigurationError("load_config requires at least one YAML path")

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
        config = RunConfig.model_validate(merged)
    except Exception as exc:
        raise ConfigurationError(str(exc)) from exc

    if resolve:
        return config.resolve_paths()
    return config
