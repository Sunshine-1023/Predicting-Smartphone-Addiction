"""Smoke test that the package exposes a version."""

from smartphone_addiction import __version__


def test_package_exposes_version() -> None:
    assert __version__ == "0.1.0"
