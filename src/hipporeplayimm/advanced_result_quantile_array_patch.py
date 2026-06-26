"""Make advanced diagnostic quantiles robust to NumPy array inputs."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

_PATCHED_FLAG = "_advanced_result_quantile_array_patch_applied"


def apply_advanced_result_quantile_array_patch() -> None:
    """Patch ``_quantile`` so array and nullable inputs are handled uniformly."""

    from . import advanced_result_diagnostics as diagnostics

    if getattr(diagnostics, _PATCHED_FLAG, False):
        return

    def _quantile(values: Sequence[float], q: float) -> float:
        raw = np.asarray(values, dtype=object).reshape(-1)
        arr = pd.to_numeric(pd.Series(raw), errors="coerce").to_numpy(dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return float("nan")
        return float(np.quantile(arr, q))

    diagnostics._quantile = _quantile
    setattr(diagnostics, _PATCHED_FLAG, True)


__all__ = ["apply_advanced_result_quantile_array_patch"]
