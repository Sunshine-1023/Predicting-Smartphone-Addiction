"""Deterministic offline Kaggle bundle helpers."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import yaml

from smartphone_addiction.git_info import git_sha
from smartphone_addiction.paths import ROOT_MARKER, project_root

FORBIDDEN_NAME_PARTS = (
    ".git/",
    "kaggle.json",
    "access_token",
    "data/raw",
    "data/processed",
    "/artifacts/",
    "submissions/",
)

# Stable timestamp when SOURCE_DATE_EPOCH is unset (reproducible ZIP contents).
FIXED_CREATED_AT = "1970-01-01T00:00:00Z"
MANIFEST_NAME = "bundle_manifest.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bundle_created_at() -> str:
    """Return a reproducible ISO-8601 UTC timestamp for the bundle manifest."""
    epoch = os.environ.get("SOURCE_DATE_EPOCH")
    if epoch is None or epoch.strip() == "":
        return FIXED_CREATED_AT
    try:
        seconds = int(epoch)
    except ValueError as exc:
        raise RuntimeError(f"invalid SOURCE_DATE_EPOCH={epoch!r}") from exc
    return (
        datetime.fromtimestamp(seconds, tz=UTC)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def build_wheel(root: Path, dist_dir: Path) -> Path:
    dist_dir.mkdir(parents=True, exist_ok=True)
    before = {path.name for path in dist_dir.glob("*.whl")}
    completed = subprocess.run(
        [
            os.environ.get("PYTHON", sys_executable()),
            "-m",
            "build",
            "--wheel",
            "--no-isolation",
            "--outdir",
            str(dist_dir),
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"wheel build failed:\n{completed.stdout}\n{completed.stderr}")
    after = [path for path in dist_dir.glob("*.whl") if path.name not in before]
    if after:
        return sorted(after)[0]
    wheels = sorted(dist_dir.glob("*.whl"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not wheels:
        raise RuntimeError("no wheel produced")
    return wheels[0]


def sys_executable() -> str:
    import sys

    return sys.executable


def write_deterministic_zip(staging: Path, zip_path: Path) -> None:
    tmp = zip_path.with_suffix(zip_path.suffix + ".tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(p for p in staging.rglob("*") if p.is_file()):
            rel = path.relative_to(staging).as_posix()
            info = zipfile.ZipInfo(rel, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, path.read_bytes())
    tmp.replace(zip_path)


def _infer_model_name(experiment_path: Path) -> str:
    loaded = yaml.safe_load(experiment_path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise RuntimeError(f"experiment config root must be a mapping: {experiment_path}")
    model = loaded.get("model") or {}
    if isinstance(model, dict) and model.get("name"):
        return str(model["name"]).strip().lower()
    return "catboost"


def _resolve_bundle_configs(
    *,
    root: Path,
    experiment_path: Path,
    base_path: Path | None,
    profile_path: Path | None,
    model_config_path: Path | None,
) -> dict[str, Path]:
    base = Path(base_path) if base_path is not None else root / "configs" / "base.yaml"
    profile = (
        Path(profile_path)
        if profile_path is not None
        else root / "configs" / "profiles" / "final.yaml"
    )
    if model_config_path is not None:
        model_cfg = Path(model_config_path)
    else:
        model_name = _infer_model_name(experiment_path)
        model_cfg = root / "configs" / "models" / f"{model_name}.yaml"
    for label, path in (
        ("base", base),
        ("profile", profile),
        ("model", model_cfg),
        ("experiment", experiment_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} config not found: {path}")
    return {
        "base": base,
        "profile": profile,
        "model": model_cfg,
        "experiment": experiment_path,
    }


def _copy_configs_tree(root: Path, staging: Path) -> None:
    source = root / "configs"
    if not source.is_dir():
        raise FileNotFoundError(f"configs directory not found: {source}")
    destination = staging / "configs"
    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns("README.md", "__pycache__", "*.pyc"),
    )


def _collect_file_records(staging: Path) -> list[dict[str, object]]:
    """Hash every staged file except the manifest (avoids self-referential digests)."""
    records: list[dict[str, object]] = []
    for path in sorted(p for p in staging.rglob("*") if p.is_file()):
        rel = path.relative_to(staging).as_posix()
        if rel == MANIFEST_NAME:
            continue
        lowered = rel.lower()
        for part in FORBIDDEN_NAME_PARTS:
            token = part.strip("/")
            if token and token in lowered:
                raise RuntimeError(f"forbidden path leaked into bundle: {rel}")
        records.append({"path": rel, "sha256": sha256_file(path), "bytes": path.stat().st_size})
    return records


def package_kaggle_bundle(
    *,
    config_path: Path,
    root: Path | None = None,
    dist_dir: Path | None = None,
    base_path: Path | None = None,
    profile_path: Path | None = None,
    model_config_path: Path | None = None,
    build_wheel_fn=None,
    git_sha_fn=None,
    created_at: str | None = None,
) -> dict[str, Path]:
    """Create wheel + configs + deterministic zip for offline Kaggle runs."""
    root = (root or project_root()).resolve()
    dist_dir = Path(dist_dir or (root / "dist"))
    config_path = Path(config_path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"config not found: {config_path}")

    configs = _resolve_bundle_configs(
        root=root,
        experiment_path=config_path,
        base_path=Path(base_path).resolve() if base_path is not None else None,
        profile_path=Path(profile_path).resolve() if profile_path is not None else None,
        model_config_path=(
            Path(model_config_path).resolve() if model_config_path is not None else None
        ),
    )

    builder = build_wheel_fn or build_wheel
    sha_fn = git_sha_fn or git_sha
    wheel = builder(root, dist_dir)
    sha = sha_fn(root)
    stamp = created_at if created_at is not None else bundle_created_at()

    staging = dist_dir / f"_bundle_staging_{sha}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    shutil.copy2(wheel, staging / wheel.name)
    _copy_configs_tree(root, staging)
    (staging / ROOT_MARKER).write_text(
        "Offline Kaggle bundle root marker for project_root().\n",
        encoding="utf-8",
    )

    experiment_rel = f"configs/experiments/{config_path.name}"
    if not (staging / experiment_rel).is_file():
        # Experiment may live outside configs/experiments; place a copy.
        target = staging / "configs" / "experiments" / config_path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(config_path, target)
        experiment_rel = target.relative_to(staging).as_posix()

    base_rel = configs["base"].relative_to(root).as_posix()
    profile_rel = configs["profile"].relative_to(root).as_posix()
    model_rel = configs["model"].relative_to(root).as_posix()
    for rel in (base_rel, profile_rel, model_rel):
        if not (staging / rel).is_file():
            raise RuntimeError(f"bundled config missing after copy: {rel}")

    launcher = staging / "run_offline.sh"
    launcher.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                'ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
                'export SMARTPHONE_ADDICTION_ROOT="$ROOT"',
                'cd "$ROOT"',
                'python -m pip install --no-deps "$(ls smartphone_addiction-*.whl | head -n 1)"',
                "python verify_environment.py --train-only",
                'DATA_DIR="${DATA_DIR:-}"',
                "TRAIN_ARGS=(",
                f'  --base "{base_rel}"',
                f'  --profile "{profile_rel}"',
                f'  --model-config "{model_rel}"',
                f'  --experiment "{experiment_rel}"',
                '  --override "runtime.environment=kaggle"',
                ")",
                'if [[ -n "$DATA_DIR" ]]; then',
                '  TRAIN_ARGS+=(--raw --override "data.directory=$DATA_DIR")',
                '  echo "Using competition data from $DATA_DIR"',
                "else",
                '  echo "Expect data/processed under $ROOT, or set DATA_DIR."',
                "fi",
                'smartphone-addiction train "${TRAIN_ARGS[@]}"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    launcher.chmod(0o755)
    shutil.copy2(root / "scripts" / "verify_environment.py", staging / "verify_environment.py")

    files = _collect_file_records(staging)
    manifest = {
        "created_at": stamp,
        "git_sha": sha,
        "wheel": wheel.name,
        "config": experiment_rel,
        "base": base_rel,
        "profile": profile_rel,
        "model_config": model_rel,
        "launcher": "run_offline.sh",
        "verify_environment": "verify_environment.py",
        "root_marker": ROOT_MARKER,
        "python": "3.11",
        "files": files,
    }
    # Write once. ``files`` excludes MANIFEST_NAME so every listed digest matches disk.
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    manifest_path = staging / MANIFEST_NAME
    manifest_path.write_bytes(manifest_bytes)

    zip_path = dist_dir / f"kaggle_bundle-{sha}.zip"
    write_deterministic_zip(staging, zip_path)
    outer_manifest = dist_dir / MANIFEST_NAME
    outer_manifest.write_bytes(manifest_bytes)
    shutil.rmtree(staging)
    return {"wheel": wheel, "zip": zip_path, "manifest": outer_manifest}
