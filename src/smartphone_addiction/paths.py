"""Project path helpers."""

from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """Return the repository root (directory containing pyproject.toml)."""
    here = Path(__file__).resolve()
    for candidate in (here.parent, *here.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise RuntimeError("could not locate project root with pyproject.toml")


def resolve_path(path: str | Path, root: Path | None = None) -> Path:
    """Resolve a possibly relative path against the project root."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    base = root or project_root()
    return (base / candidate).resolve()
