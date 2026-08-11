"""Write processed parquet datasets and feature manifests."""

from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from smartphone_addiction import __version__ as PACKAGE_VERSION
from smartphone_addiction.data.download import fingerprint_files
from smartphone_addiction.data.schema import FEATURE_COLUMNS, ID_COLUMN, TARGET_COLUMN
from smartphone_addiction.errors import DataValidationError
from smartphone_addiction.features.base import TransformedFrames
from smartphone_addiction.features.domain import (
    CATEGORY_INTERACTION_SEP,
    MISSING_TOKEN,
    SAFE_DIVIDE_EPS,
)

# Relative to the installed package root (src/smartphone_addiction/).
FEATURE_CODE_RELATIVE_PATHS = (
    "data/schema.py",
    "features/base.py",
    "features/domain.py",
    "features/io.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def feature_code_fingerprint() -> dict[str, Any]:
    """Return package version plus SHA-256 digests of feature-related source files."""
    package_root = Path(__file__).resolve().parents[1]
    files: dict[str, str] = {}
    for relative in FEATURE_CODE_RELATIVE_PATHS:
        path = package_root / relative
        if not path.is_file():
            raise DataValidationError(f"feature code file missing: {path}")
        files[relative] = sha256_file(path)

    combined = hashlib.sha256()
    for relative in sorted(files):
        combined.update(relative.encode("utf-8"))
        combined.update(b"\0")
        combined.update(files[relative].encode("utf-8"))
        combined.update(b"\0")
    return {
        "package_version": PACKAGE_VERSION,
        "files": files,
        "digest": combined.hexdigest(),
    }


def validate_processed_manifest(
    manifest: dict[str, Any],
    *,
    raw_directory: Path | str | None = None,
    train: pd.DataFrame | None = None,
    test: pd.DataFrame | None = None,
    train_path: Path | str | None = None,
    test_path: Path | str | None = None,
    require_source_hashes: bool = True,
) -> None:
    """Reject stale or incomplete processed feature manifests.

    Always requires a matching ``feature_code.digest``. When ``raw_directory`` is
    provided, also requires ``source_hashes`` to match the official CSV fingerprints
    (and fails if the directory is missing). When train/test frames are provided,
    ``row_counts``, ``feature_columns``, and optional ``content_hashes`` must match.
    """
    feature_code = manifest.get("feature_code")
    if not isinstance(feature_code, dict):
        raise DataValidationError(
            "processed feature_manifest.json missing feature_code; "
            "rebuild with: smartphone-addiction features build"
        )
    saved_digest = feature_code.get("digest")
    if not saved_digest:
        raise DataValidationError(
            "processed feature_manifest.json missing feature_code.digest; "
            "rebuild with: smartphone-addiction features build"
        )
    current_digest = feature_code_fingerprint()["digest"]
    if saved_digest != current_digest:
        raise DataValidationError(
            "processed feature_code digest mismatch; "
            "rebuild with: smartphone-addiction features build"
        )

    raw_path = Path(raw_directory) if raw_directory is not None else None
    if raw_path is not None:
        if not raw_path.is_dir():
            if require_source_hashes:
                raise DataValidationError(
                    f"raw data directory missing: {raw_path}; "
                    "download official CSVs with: smartphone-addiction data download"
                )
        else:
            saved_hashes = manifest.get("source_hashes")
            if not isinstance(saved_hashes, dict) or not saved_hashes:
                if require_source_hashes:
                    raise DataValidationError(
                        "processed feature_manifest.json missing source_hashes; "
                        "rebuild with: smartphone-addiction features build"
                    )
            else:
                current_hashes = fingerprint_files(raw_path)
                if saved_hashes != current_hashes:
                    raise DataValidationError(
                        "processed source_hashes do not match data/raw CSVs; "
                        "rebuild with: smartphone-addiction features build"
                    )

    row_counts = manifest.get("row_counts")
    if isinstance(row_counts, dict):
        if train is not None and "train" in row_counts and int(row_counts["train"]) != len(train):
            raise DataValidationError(
                f"processed train row_counts mismatch: manifest={row_counts['train']} "
                f"parquet={len(train)}; rebuild with: smartphone-addiction features build"
            )
        if test is not None and "test" in row_counts and int(row_counts["test"]) != len(test):
            raise DataValidationError(
                f"processed test row_counts mismatch: manifest={row_counts['test']} "
                f"parquet={len(test)}; rebuild with: smartphone-addiction features build"
            )
    elif train is not None or test is not None:
        raise DataValidationError(
            "processed feature_manifest.json missing row_counts; "
            "rebuild with: smartphone-addiction features build"
        )

    if train is not None or test is not None:
        _assert_manifest_columns(manifest, train=train, test=test)

    content_hashes = manifest.get("content_hashes")
    if isinstance(content_hashes, dict) and content_hashes:
        if train_path is not None and "train" in content_hashes:
            current = sha256_file(Path(train_path))
            if current != content_hashes["train"]:
                raise DataValidationError(
                    "processed train parquet content hash mismatch; "
                    "rebuild with: smartphone-addiction features build"
                )
        if test_path is not None and "test" in content_hashes:
            current = sha256_file(Path(test_path))
            if current != content_hashes["test"]:
                raise DataValidationError(
                    "processed test parquet content hash mismatch; "
                    "rebuild with: smartphone-addiction features build"
                )


def write_processed_dataset(
    frames: TransformedFrames,
    output_dir: Path,
    version: str = "v1",
    *,
    raw_directory: Path | str | None = None,
) -> dict[str, Path]:
    """Write train_features.parquet, test_features.parquet, feature_manifest.json.

    Writes into a staging directory first and publishes atomically. The manifest
    is written last so incomplete directories never look valid.
    """
    output_dir = Path(output_dir)
    staging = output_dir.parent / f".{output_dir.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir(parents=True, exist_ok=False)
    try:
        _write_processed_to_staging(
            frames,
            staging,
            version=version,
            raw_directory=raw_directory,
        )
        _publish_processed_directory(staging, output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "train": output_dir / "train_features.parquet",
        "test": output_dir / "test_features.parquet",
        "manifest": output_dir / "feature_manifest.json",
    }


def read_processed_dataset(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Read processed parquet files and feature manifest."""
    output_dir = Path(output_dir)
    train_path = output_dir / "train_features.parquet"
    test_path = output_dir / "test_features.parquet"
    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)
    manifest = json.loads((output_dir / "feature_manifest.json").read_text(encoding="utf-8"))
    validate_processed_manifest(
        manifest,
        train=train,
        test=test,
        train_path=train_path,
        test_path=test_path,
        require_source_hashes=False,
    )
    return train, test, manifest


def _write_processed_to_staging(
    frames: TransformedFrames,
    staging: Path,
    *,
    version: str,
    raw_directory: Path | str | None,
) -> dict[str, Path]:
    _assert_finite_features(frames.train, frames.feature_columns, "train")
    _assert_finite_features(frames.test, frames.feature_columns, "test")

    train_path = staging / "train_features.parquet"
    test_path = staging / "test_features.parquet"
    frames.train.to_parquet(train_path, index=False)
    frames.test.to_parquet(test_path, index=False)

    source_hashes: dict[str, str] | None = None
    if raw_directory is not None:
        source_hashes = fingerprint_files(Path(raw_directory))

    manifest = {
        "version": version,
        "id_column": ID_COLUMN,
        "target_column": TARGET_COLUMN,
        "raw_features": list(FEATURE_COLUMNS),
        "feature_columns": list(frames.feature_columns),
        "categorical_columns": list(frames.categorical_columns),
        "numeric_columns": list(frames.numeric_columns),
        "feature_groups": list(frames.feature_groups),
        "source_hashes": source_hashes,
        "feature_code": feature_code_fingerprint(),
        "content_hashes": {
            "train": sha256_file(train_path),
            "test": sha256_file(test_path),
        },
        "rules": {
            "numeric_missing": "keep_nan",
            "categorical_missing": MISSING_TOKEN,
            "safe_divide_eps": SAFE_DIVIDE_EPS,
            "category_interaction_sep": CATEGORY_INTERACTION_SEP,
            "missing_pattern": "pipe_joined_missing_column_names_in_feature_order",
        },
        "row_counts": {
            "train": len(frames.train),
            "test": len(frames.test),
        },
    }
    manifest_path = staging / "feature_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {"train": train_path, "test": test_path, "manifest": manifest_path}


def _publish_processed_directory(staging: Path, output_dir: Path) -> None:
    """Replace ``output_dir`` with ``staging`` after all artifacts are written."""
    backup: Path | None = None
    if output_dir.exists():
        backup = output_dir.parent / f".{output_dir.name}.backup-{uuid.uuid4().hex}"
        output_dir.rename(backup)
    try:
        staging.rename(output_dir)
    except Exception:
        if backup is not None and not output_dir.exists():
            backup.rename(output_dir)
        raise
    if backup is not None:
        shutil.rmtree(backup, ignore_errors=True)


def _assert_manifest_columns(
    manifest: dict[str, Any],
    *,
    train: pd.DataFrame | None,
    test: pd.DataFrame | None,
) -> None:
    feature_columns = manifest.get("feature_columns")
    if not isinstance(feature_columns, list) or not feature_columns:
        raise DataValidationError(
            "processed feature_manifest.json missing feature_columns; "
            "rebuild with: smartphone-addiction features build"
        )
    id_col = str(manifest.get("id_column", ID_COLUMN))
    target_col = str(manifest.get("target_column", TARGET_COLUMN))
    for name, frame in (("train", train), ("test", test)):
        if frame is None:
            continue
        required = [id_col, *feature_columns]
        if name == "train":
            required.append(target_col)
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise DataValidationError(
                f"processed {name} parquet missing manifest columns: {missing}; "
                "rebuild with: smartphone-addiction features build"
            )
        actual_features = [column for column in frame.columns if column in feature_columns]
        if actual_features != list(feature_columns):
            raise DataValidationError(
                f"processed {name} feature column order does not match manifest; "
                "rebuild with: smartphone-addiction features build"
            )


def _assert_finite_features(
    frame: pd.DataFrame,
    feature_columns: list[str],
    name: str,
) -> None:
    numeric = frame[feature_columns].select_dtypes(include=[np.number])
    if numeric.empty:
        return
    if np.isinf(numeric.to_numpy(dtype=float, copy=False)).any():
        raise DataValidationError(f"{name} features contain infinite values before write")
