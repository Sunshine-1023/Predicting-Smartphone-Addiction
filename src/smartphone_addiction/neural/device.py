"""MPS / CPU device selection, seeding, and environment reporting."""

from __future__ import annotations

import os
import platform
import random
import sys
from typing import Any, Literal

import numpy as np

from smartphone_addiction.errors import ConfigurationError

DeviceMode = Literal["auto", "mps", "cpu"]


def require_torch():
    """Import torch or raise an install hint. Kept lazy for the tree-model CLI."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for neural reconstruction. "
            "Install with: python -m pip install -e '.[neural]'"
        ) from exc
    return torch


def mps_status(torch_module: Any | None = None) -> dict[str, bool]:
    torch = torch_module or require_torch()
    backends = getattr(torch, "backends", None)
    mps = getattr(backends, "mps", None) if backends is not None else None
    built = bool(getattr(mps, "is_built", lambda: False)()) if mps is not None else False
    available = bool(getattr(mps, "is_available", lambda: False)()) if mps is not None else False
    return {"mps_built": built, "mps_available": available}


def resolve_device(mode: str = "auto", *, torch_module: Any | None = None) -> Any:
    """Return a torch.device honoring auto/mps/cpu. mps mode never silently falls back."""
    torch = torch_module or require_torch()
    normalized = str(mode).strip().lower()
    if normalized not in {"auto", "mps", "cpu"}:
        raise ConfigurationError("device must be one of: auto, mps, cpu")
    if normalized == "cpu":
        return torch.device("cpu")
    status = mps_status(torch)
    if normalized == "mps":
        if not status["mps_available"]:
            raise ConfigurationError("device=mps requested but MPS is not available")
        return torch.device("mps")
    if status["mps_available"]:
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int, *, torch_module: Any | None = None) -> None:
    torch = torch_module or require_torch()
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    mps = getattr(getattr(torch, "backends", None), "mps", None)
    if (
        mps is not None
        and hasattr(torch, "mps")
        and hasattr(torch.mps, "manual_seed")
        and mps.is_available()
    ):
        torch.mps.manual_seed(seed)


def torch_dtype(name: str = "float32", *, torch_module: Any | None = None) -> Any:
    torch = torch_module or require_torch()
    if name != "float32":
        raise ConfigurationError("neural reconstruction first version is float32-only")
    return torch.float32


def environment_info(
    *,
    device_mode: str,
    device: Any,
    dtype: str,
    batch_size: int,
    torch_module: Any | None = None,
) -> dict[str, Any]:
    torch = torch_module or require_torch()
    status = mps_status(torch)
    return {
        "python": sys.version.split()[0],
        "pytorch": torch.__version__,
        "macos": platform.mac_ver()[0] or None,
        "platform": platform.platform(),
        "device_mode": device_mode,
        "device": str(device),
        "mps_built": status["mps_built"],
        "mps_available": status["mps_available"],
        "dtype": dtype,
        "batch_size": int(batch_size),
        "amp": False,
        "num_workers": 0,
        "pin_memory": False,
    }
