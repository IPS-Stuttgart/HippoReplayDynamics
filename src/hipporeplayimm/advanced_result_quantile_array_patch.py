"""Make advanced diagnostic quantiles robust to NumPy array inputs."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

_PATCHED_FLAG = "_advanced_result_quantile_array_patch_applied"
_WRAPPER_ATTR = "_advanced_result_quantile_array_wrapper"


def _current_patch_installed(diagnostics: object) -> bool:
    current = getattr(diagnostics, "_quantile", None)
    return bool(getattr(current, _WRAPPER_ATTR, False))


def apply_advanced_result_quantile_array_patch() -> None:
    """Patch ``_quantile`` so array and nullable inputs are handled uniformly."""

    from . import advanced_result_diagnostics as diagnostics

    if _current_patch_installed(diagnostics):
        setattr(diagnostics, _PATCHED_FLAG, True)
        return

    def _quantile(values: Sequence[float], q: float) -> float:
        raw = _flatten_quantile_values(values)
        arr = pd.to_numeric(pd.Series(raw), errors="coerce").to_numpy(dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return float("nan")
        return float(np.quantile(arr, q))

    setattr(_quantile, _WRAPPER_ATTR, True)
    diagnostics._quantile = _quantile
    setattr(diagnostics, _PATCHED_FLAG, True)


def _flatten_quantile_values(values: Sequence[float]) -> np.ndarray:
    """Return scalar values from possibly nested array-valued quantile input."""

    if isinstance(values, Iterable) and not isinstance(values, (str, bytes)):
        raw = np.asarray(list(values), dtype=object)
    else:
        raw = np.asarray(values, dtype=object)
    if raw.ndim == 0:
        raw = raw.reshape(1)
    flattened: list[object] = []
    for value in raw.reshape(-1):
        if isinstance(value, (str, bytes)):
            flattened.append(value)
            continue
        current = np.asarray(value, dtype=object)
        if current.ndim == 0:
            flattened.append(value)
        else:
            flattened.extend(current.reshape(-1).tolist())
    return np.asarray(flattened, dtype=object)


__all__ = ["apply_advanced_result_quantile_array_patch"]
