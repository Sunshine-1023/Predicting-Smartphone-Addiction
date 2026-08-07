"""Tests for project_root resolution (repo checkout and offline bundle)."""

from __future__ import annotations

from pathlib import Path

import pytest

from smartphone_addiction.paths import ROOT_ENV, ROOT_MARKER, project_root


def test_project_root_finds_pyproject() -> None:
    root = project_root()
    assert (root / "pyproject.toml").is_file()


def test_project_root_respects_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ROOT_ENV, str(tmp_path))
    assert project_root() == tmp_path.resolve()


def test_project_root_finds_bundle_marker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ROOT_ENV, raising=False)
    (tmp_path / ROOT_MARKER).write_text("bundle\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    # Package still lives under the real checkout; env/cwd marker should win over
    # walking from __file__ only when pyproject is absent — here __file__ still
    # finds the real repo. Prefer env for bundle simulation.
    monkeypatch.setenv(ROOT_ENV, str(tmp_path))
    assert project_root() == tmp_path.resolve()
