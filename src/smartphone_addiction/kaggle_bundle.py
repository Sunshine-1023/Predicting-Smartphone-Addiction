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

from smartphone_addiction.paths import project_root

FORBIDDEN_NAME_PARTS = (
    ".git/",
    "kaggle.json",
    "access_token",
    "data/raw",
    "data/processed",
    "/artifacts/",
    "submissions/",
)


def git_sha(root: Path | None = None) -> str:
    root = root or project_root()
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except OSError:
        pass
    return "nogit"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_wheel(root: Path, dist_dir: Path) -> Path:
    dist_dir.mkdir(parents=True, exist_ok=True)
    before = {path.name for path in dist_dir.glob("*.whl")}
    completed = subprocess.run(
        [
            os.environ.get("PYTHON", sys_executable()),
            "-m",
            "build",
            "--wheel",
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


def package_kaggle_bundle(
    *,
    config_path: Path,
    root: Path | None = None,
    dist_dir: Path | None = None,
    build_wheel_fn=None,
    git_sha_fn=None,
) -> dict[str, Path]:
    """Create wheel + deterministic zip + manifest for offline Kaggle runs."""
    root = root or project_root()
    dist_dir = Path(dist_dir or (root / "dist"))
    config_path = Path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"config not found: {config_path}")

    builder = build_wheel_fn or build_wheel
    sha_fn = git_sha_fn or git_sha
    wheel = builder(root, dist_dir)
    sha = sha_fn(root)

    staging = dist_dir / f"_bundle_staging_{sha}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    shutil.copy2(wheel, staging / wheel.name)
    rel_config_name = config_path.name
    shutil.copy2(config_path, staging / rel_config_name)

    launcher = staging / "run_offline.sh"
    launcher.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                'python -m pip install --no-deps "$(ls smartphone_addiction-*.whl | head -n 1)"',
                "python verify_environment.py",
                f'echo "Train with: smartphone-addiction train --experiment {rel_config_name}"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    launcher.chmod(0o755)
    shutil.copy2(root / "scripts" / "verify_environment.py", staging / "verify_environment.py")

    files = []
    for path in sorted(p for p in staging.rglob("*") if p.is_file()):
        rel = path.relative_to(staging).as_posix()
        lowered = rel.lower()
        for part in FORBIDDEN_NAME_PARTS:
            token = part.strip("/")
            if token and token in lowered:
                raise RuntimeError(f"forbidden path leaked into bundle: {rel}")
        files.append({"path": rel, "sha256": sha256_file(path), "bytes": path.stat().st_size})

    manifest = {
        "created_at": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "git_sha": sha,
        "wheel": wheel.name,
        "config": rel_config_name,
        "launcher": "run_offline.sh",
        "verify_environment": "verify_environment.py",
        "python": "3.11",
        "files": files,
    }
    manifest_path = staging / "bundle_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    files = [
        {"path": rel, "sha256": sha256_file(staging / rel), "bytes": (staging / rel).stat().st_size}
        for rel in sorted(
            p.relative_to(staging).as_posix() for p in staging.rglob("*") if p.is_file()
        )
    ]
    manifest["files"] = files
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    zip_path = dist_dir / f"kaggle_bundle-{sha}.zip"
    write_deterministic_zip(staging, zip_path)
    outer_manifest = dist_dir / "bundle_manifest.json"
    outer_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    shutil.rmtree(staging)
    return {"wheel": wheel, "zip": zip_path, "manifest": outer_manifest}
