#!/usr/bin/env python3
"""Verify Python and package versions before training."""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class Requirement:
    module: str
    package: str
    minimum: tuple[int, ...]
    maximum: tuple[int, ...] | None = None


REQUIREMENTS = [
    Requirement("numpy", "numpy", (1, 26), (3,)),
    Requirement("pandas", "pandas", (2, 2), (3,)),
    Requirement("pyarrow", "pyarrow", (16,), (22,)),
    Requirement("sklearn", "scikit-learn", (1, 5), (2,)),
    Requirement("catboost", "catboost", (1, 2), (2,)),
    Requirement("lightgbm", "lightgbm", (4, 3), (5,)),
    Requirement("pydantic", "pydantic", (2, 7), (3,)),
    Requirement("yaml", "PyYAML", (6,), (7,)),
    Requirement("typer", "typer", (0, 12), (1,)),
    Requirement("optuna", "optuna", (4,), (5,)),
]


def _parse_version(version: str) -> tuple[int, ...]:
    parts: list[int] = []
    for chunk in version.replace("-", ".").split("."):
        digits = ""
        for char in chunk:
            if char.isdigit():
                digits += char
            else:
                break
        if digits:
            parts.append(int(digits))
        if len(parts) >= 3:
            break
    return tuple(parts) or (0,)


def _version_of(module_name: str) -> str:
    module = importlib.import_module(module_name)
    version = getattr(module, "__version__", None)
    if version is None and module_name == "yaml":
        version = "6.0.0"
    if version is None:
        raise RuntimeError(f"could not determine version for {module_name}")
    return str(version)


def _in_range(
    version: str,
    minimum: tuple[int, ...],
    maximum: tuple[int, ...] | None,
) -> bool:
    current = _parse_version(version)
    if current < minimum:
        return False
    if maximum is not None and current >= maximum:
        return False
    return True


def verify_environment(stream=sys.stdout) -> int:
    """Print a compact table and return process exit code."""
    rows: list[tuple[str, str, str]] = []
    ok = True

    py = f"{sys.version_info.major}.{sys.version_info.minor}"
    py_status = "ok" if sys.version_info[:2] == (3, 11) else "FAIL"
    if py_status != "ok":
        ok = False
    rows.append(("python", py, py_status))

    for req in REQUIREMENTS:
        try:
            version = _version_of(req.module)
            status = "ok" if _in_range(version, req.minimum, req.maximum) else "FAIL"
        except Exception as exc:
            version = f"missing ({type(exc).__name__})"
            status = "FAIL"
        if status != "ok":
            ok = False
        rows.append((req.package, version, status))

    width = max(len(name) for name, _, _ in rows)
    print(f"{'package':<{width}}  version           status", file=stream)
    print(f"{'-' * width}  {'-' * 16}  ------", file=stream)
    for name, version, status in rows:
        print(f"{name:<{width}}  {version:<16}  {status}", file=stream)
    return 0 if ok else 1


def main() -> None:
    raise SystemExit(verify_environment())


if __name__ == "__main__":
    main()
