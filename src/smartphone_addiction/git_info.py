"""Best-effort git metadata for experiment manifests (never stores secrets)."""

from __future__ import annotations

import subprocess
from pathlib import Path

from smartphone_addiction.paths import project_root


def git_sha(root: Path | None = None) -> str:
    """Return short HEAD sha, or ``nogit`` when unavailable."""
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


def git_is_dirty(root: Path | None = None) -> bool:
    """Return True when the working tree has uncommitted changes."""
    root = root or project_root()
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return False
        return bool(result.stdout.strip())
    except OSError:
        return False
