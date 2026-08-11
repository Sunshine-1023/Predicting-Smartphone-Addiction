"""Atomic experiment artifact storage and run lifecycle."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from smartphone_addiction.artifacts.manifest import (
    RunManifest,
    build_run_id,
    default_environment_fields,
    hash_mapping,
    utc_now_iso,
)
from smartphone_addiction.errors import ArtifactError


class ArtifactStore:
    """Manage one immutable-looking run directory under artifacts/runs/."""

    def __init__(self, run_dir: Path, manifest: RunManifest) -> None:
        self.run_dir = Path(run_dir)
        self._manifest = manifest
        self._started_monotonic: datetime | None = None

    @classmethod
    def create(
        cls,
        artifact_root: Path,
        slug: str,
        git_sha: str,
        git_dirty: bool = False,
        environment: str = "local",
    ) -> ArtifactStore:
        """Create a new run directory. Refuses to overwrite an existing run id."""
        run_id = build_run_id(slug=slug, git_sha=git_sha)
        run_dir = Path(artifact_root) / run_id
        if run_dir.exists():
            raise ArtifactError(f"run directory already exists: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "models").mkdir()
        (run_dir / "importance").mkdir()

        env = default_environment_fields(git_sha=git_sha, git_dirty=git_dirty)
        manifest = RunManifest(
            run_id=run_id,
            slug=slug,
            status="created",
            environment=environment,
            **env,
        )
        store = cls(run_dir=run_dir, manifest=manifest)
        store._write_manifest()
        return store

    @classmethod
    def open(cls, run_dir: Path) -> ArtifactStore:
        """Open an existing run directory and load its manifest."""
        run_dir = Path(run_dir)
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.is_file():
            raise ArtifactError(f"manifest.json missing in {run_dir}")
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        return cls(run_dir=run_dir, manifest=RunManifest.from_dict(payload))

    def manifest(self) -> RunManifest:
        return self._manifest

    def start(
        self,
        config: dict[str, Any],
        data_hashes: dict[str, str],
        n_train_rows: int | None = None,
        n_features: int | None = None,
        seeds: list[int] | None = None,
        n_splits: int | None = None,
    ) -> None:
        """Mark the run as running and persist the resolved config."""
        if self._manifest.status not in {"created", "interrupted"}:
            raise ArtifactError(f"cannot start run from status={self._manifest.status!r}")
        self.write_yaml("resolved_config.yaml", config)
        self._manifest.config_hash = hash_mapping(config)
        self._manifest.data_hashes = dict(data_hashes)
        self._manifest.n_train_rows = n_train_rows
        self._manifest.n_features = n_features
        self._manifest.seeds = list(seeds or [])
        self._manifest.n_splits = n_splits
        self._manifest.status = "running"
        self._manifest.started_at = utc_now_iso()
        self._manifest.error = None
        self._started_monotonic = datetime.now(timezone.utc)
        self._write_manifest()

    def mark_fold_complete(self, fold_key: str, fold_metrics: dict[str, Any] | None = None) -> None:
        """Record that a seed/fold combination finished."""
        if fold_key not in self._manifest.completed_folds:
            self._manifest.completed_folds.append(fold_key)
        if fold_metrics is not None:
            path = self.run_dir / "fold_metrics.csv"
            frame = pd.DataFrame([{**fold_metrics, "fold_key": fold_key}])
            if path.exists():
                existing = pd.read_csv(path)
                frame = pd.concat([existing, frame], ignore_index=True)
            self.write_frame("fold_metrics.csv", frame)
        self._write_manifest()

    def complete(self, metrics: dict[str, Any] | None = None) -> None:
        """Mark the run completed and optionally write metrics.json."""
        if metrics is not None:
            self.write_json("metrics.json", metrics)
        self._manifest.status = "completed"
        self._finalize_timing()
        self._write_manifest()

    def fail(self, error: str) -> None:
        """Mark the run failed."""
        self._manifest.status = "failed"
        self._manifest.error = error
        self._finalize_timing()
        self._write_manifest()

    def interrupt(self, error: str = "interrupted") -> None:
        """Mark the run interrupted (e.g. KeyboardInterrupt)."""
        self._manifest.status = "interrupted"
        self._manifest.error = error
        self._finalize_timing()
        self._write_manifest()

    def resume_missing_folds(
        self,
        config: dict[str, Any],
        data_hashes: dict[str, str],
        expected_fold_keys: list[str],
        *,
        allow_completed: bool = False,
    ) -> list[str]:
        """Return fold keys still missing after validating config/data hashes."""
        if self._manifest.status == "completed" and not allow_completed:
            raise ArtifactError("completed runs cannot be resumed by default")
        if "source" in data_hashes and data_hashes.get("source") == "in-memory":
            raise ArtifactError("placeholder data hashes cannot be used for resume")
        required = {"train", "test", "feature_manifest"}
        missing_keys = required - set(data_hashes)
        if missing_keys:
            raise ArtifactError(
                f"resume requires data hashes for {sorted(required)}; missing {sorted(missing_keys)}"
            )
        if self._manifest.config_hash != hash_mapping(config):
            raise ArtifactError("config hash mismatch; refusing to resume")
        if self._manifest.data_hashes != data_hashes:
            raise ArtifactError("data hash mismatch; refusing to resume")
        missing = [key for key in expected_fold_keys if key not in self._manifest.completed_folds]
        self._manifest.status = "running"
        if self._manifest.started_at is None:
            self._manifest.started_at = utc_now_iso()
        self._started_monotonic = datetime.now(timezone.utc)
        self._write_manifest()
        return missing

    def write_json(self, relative_name: str, payload: dict[str, Any] | list[Any]) -> Path:
        text = json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n"
        return self._atomic_write_text(relative_name, text)

    def write_yaml(self, relative_name: str, payload: dict[str, Any]) -> Path:
        text = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)
        return self._atomic_write_text(relative_name, text)

    def write_frame(self, relative_name: str, frame: pd.DataFrame) -> Path:
        path = self.run_dir / relative_name
        tmp = path.with_suffix(path.suffix + ".tmp")
        if path.suffix == ".csv":
            frame.to_csv(tmp, index=False)
        else:
            frame.to_parquet(tmp, index=False)
        tmp.replace(path)
        self._manifest.artifacts[relative_name] = "written"
        self._cleanup_tmp_files()
        self._write_manifest()
        return path

    def _finalize_timing(self) -> None:
        self._manifest.ended_at = utc_now_iso()
        if self._manifest.started_at:
            started = datetime.fromisoformat(self._manifest.started_at.replace("Z", "+00:00"))
            ended = datetime.fromisoformat(self._manifest.ended_at.replace("Z", "+00:00"))
            self._manifest.duration_seconds = (ended - started).total_seconds()

    def _write_manifest(self) -> None:
        self._atomic_write_text(
            "manifest.json",
            json.dumps(self._manifest.to_dict(), indent=2, ensure_ascii=False) + "\n",
        )
        self._manifest.artifacts["manifest.json"] = "written"

    def _atomic_write_text(self, relative_name: str, text: str) -> Path:
        path = self.run_dir / relative_name
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)
        self._manifest.artifacts[relative_name] = "written"
        self._cleanup_tmp_files()
        return path

    def _cleanup_tmp_files(self) -> None:
        for tmp in self.run_dir.rglob("*.tmp"):
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                continue
