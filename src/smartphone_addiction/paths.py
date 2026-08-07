"""Project path helpers."""

from __future__ import annotations

import os
from pathlib import Path

ROOT_MARKER = ".smartphone_addiction_root"
ROOT_ENV = "SMARTPHONE_ADDICTION_ROOT"


def _is_project_root(candidate: Path) -> bool:
    return (candidate / "pyproject.toml").is_file() or (candidate / ROOT_MARKER).is_file()


def project_root() -> Path:
    """Return the repository or offline-bundle root.

    Resolution order:
    1. ``SMARTPHONE_ADDICTION_ROOT`` environment variable
    2. Walk parents of this package looking for ``pyproject.toml``
    3. Walk parents of the current working directory for ``pyproject.toml``
       or ``.smartphone_addiction_root`` (Kaggle unzipped bundle)
    """
    env = os.environ.get(ROOT_ENV)
    if env:
        path = Path(env).expanduser().resolve()
        if not path.is_dir():
            raise RuntimeError(f"{ROOT_ENV} is not a directory: {path}")
        return path

    here = Path(__file__).resolve()
    for candidate in (here.parent, *here.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate

    cwd = Path.cwd().resolve()
    for candidate in (cwd, *cwd.parents):
        if _is_project_root(candidate):
            return candidate

    raise RuntimeError(
        "could not locate project root; set SMARTPHONE_ADDICTION_ROOT or "
        "run from a checkout / offline bundle containing pyproject.toml "
        f"or {ROOT_MARKER}"
    )


def resolve_path(path: str | Path, root: Path | None = None) -> Path:
    """Resolve a possibly relative path against the project root."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    base = root or project_root()
    return (base / candidate).resolve()
