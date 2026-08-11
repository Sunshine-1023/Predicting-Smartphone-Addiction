"""Experiment run manifest schema and helpers."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


def utc_now_iso() -> str:
    """Return an ISO-8601 UTC timestamp with Z suffix."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def hash_mapping(payload: dict[str, Any]) -> str:
    """Stable SHA-256 over a JSON-serialized mapping."""
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass
class RunManifest:
    """Metadata for one training/evaluation run. Never store secrets here."""

    run_id: str
    slug: str
    status: str
    git_sha: str
    git_dirty: bool
    python_version: str
    platform: str
    package_versions: dict[str, str]
    config_hash: str | None = None
    data_hashes: dict[str, str] = field(default_factory=dict)
    n_train_rows: int | None = None
    n_features: int | None = None
    seeds: list[int] = field(default_factory=list)
    n_splits: int | None = None
    environment: str = "local"
    started_at: str | None = None
    ended_at: str | None = None
    duration_seconds: float | None = None
    artifacts: dict[str, str] = field(default_factory=dict)
    completed_folds: list[str] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RunManifest:
        return cls(**payload)


def collect_package_versions(names: list[str] | None = None) -> dict[str, str]:
    """Best-effort package version lookup without importing secrets."""
    import importlib.metadata

    default_names = [
        "numpy",
        "pandas",
        "pyarrow",
        "scikit-learn",
        "catboost",
        "lightgbm",
        "optuna",
        "pyyaml",
    ]
    versions: dict[str, str] = {}
    for name in names or default_names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def build_run_id(slug: str, git_sha: str, when: datetime | None = None) -> str:
    """Compose run id: UTC timestamp + experiment slug + short git sha."""
    moment = when or datetime.now(timezone.utc)
    stamp = moment.strftime("%Y%m%dT%H%M%SZ")
    short_sha = (git_sha or "unknown")[:7]
    safe_slug = slug.strip().replace(" ", "-")
    return f"{stamp}-{safe_slug}-{short_sha}"


def default_environment_fields(git_sha: str, git_dirty: bool = False) -> dict[str, Any]:
    return {
        "git_sha": git_sha,
        "git_dirty": git_dirty,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "package_versions": collect_package_versions(),
    }
