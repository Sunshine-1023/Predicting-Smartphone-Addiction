"""Unit tests for MPS/CPU device selection."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from smartphone_addiction.errors import ConfigurationError
from smartphone_addiction.neural.device import mps_status, resolve_device, seed_everything


class _Mps:
    def __init__(self, available: bool, built: bool = True) -> None:
        self._available = available
        self._built = built

    def is_available(self) -> bool:
        return self._available

    def is_built(self) -> bool:
        return self._built


def test_cpu_mode_always_returns_cpu() -> None:
    device = resolve_device("cpu", torch_module=torch)
    assert device.type == "cpu"


def test_auto_prefers_mps_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.backends, "mps", _Mps(True, True))
    device = resolve_device("auto", torch_module=torch)
    assert device.type == "mps"


def test_auto_falls_back_to_cpu_when_mps_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.backends, "mps", _Mps(False, False))
    device = resolve_device("auto", torch_module=torch)
    assert device.type == "cpu"


def test_mps_mode_errors_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.backends, "mps", _Mps(False, True))
    with pytest.raises(ConfigurationError, match="device=mps"):
        resolve_device("mps", torch_module=torch)


def test_unknown_device_is_rejected() -> None:
    with pytest.raises(ConfigurationError, match="device must be"):
        resolve_device("cuda", torch_module=torch)


def test_seed_everything_is_deterministic() -> None:
    seed_everything(42, torch_module=torch)
    first = torch.rand(3)
    seed_everything(42, torch_module=torch)
    second = torch.rand(3)
    assert torch.allclose(first, second)


def test_mps_status_keys() -> None:
    status = mps_status(torch)
    assert set(status) == {"mps_built", "mps_available"}
