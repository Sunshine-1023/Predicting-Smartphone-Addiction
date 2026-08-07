"""Training progress helpers built on tqdm."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from tqdm.auto import tqdm

_CATBOOST_ITER_RE = re.compile(r"^\s*(\d+)\s*:")


def make_iteration_bar(total: int, desc: str) -> tqdm:
    """Create a per-fold iteration progress bar (nested under the OOF fold bar)."""
    return tqdm(
        total=max(int(total), 1),
        desc=desc,
        unit="iter",
        leave=False,
        dynamic_ncols=True,
    )


def close_bar(bar: tqdm | None) -> None:
    if bar is not None:
        bar.close()


class CatBoostProgressStdout:
    """Capture CatBoost verbose lines and advance a tqdm iteration bar."""

    def __init__(self, bar: tqdm) -> None:
        self.bar = bar
        self._buf = ""

    def write(self, text: str) -> int:
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            match = _CATBOOST_ITER_RE.match(line)
            if not match:
                continue
            current = int(match.group(1)) + 1
            if self.bar.n < current:
                self.bar.update(current - self.bar.n)
            # Prefer validation score in the postfix when present.
            for token in ("test:", "best:"):
                if token in line:
                    try:
                        fragment = line.split(token, 1)[1].strip().split()[0]
                        value = float(fragment.rstrip(","))
                        self.bar.set_postfix_str(f"auc={value:.4f}")
                    except (TypeError, ValueError, IndexError):
                        pass
                    break
        return len(text)

    def flush(self) -> None:
        return None


def make_lightgbm_tqdm_callback(bar: tqdm) -> Callable[[Any], None]:
    """Return a LightGBM callback that syncs ``bar`` to the current iteration."""

    def _callback(env: Any) -> None:
        current = int(env.iteration) + 1
        if bar.n < current:
            bar.update(current - bar.n)
        results = getattr(env, "evaluation_result_list", None) or []
        if results:
            try:
                _data_name, metric_name, value, _ = results[-1]
                bar.set_postfix_str(f"{metric_name}={float(value):.4f}")
            except (TypeError, ValueError, IndexError):
                pass

    _callback.order = 90  # type: ignore[attr-defined]
    return _callback
