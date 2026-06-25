"""Make advanced diagnostic quantiles robust to NumPy array inputs."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

_PATCHED_FLAG = "_advanced_result_quantile_array_patch_applied"


def apply_advanced_result_quantile_array_patch() -> None:
    """Patch ``_quantile`` so array inputs do not hit ambiguous truth-value checks."""

    from . import advanced_result_diagnostics as diagnostics

    if getattr(diagnostics, _PATCHED_FLAG, False):
        return

    def _quantile(values: Sequence[float], q: float) -> float:
        arr = np.asarray(values, dtype=float)
        if arr.size == 0:
            return float("nan")
        return float(np.quantile(arr, q))

    diagnostics._quantile = _quantile
    setattr(diagnostics, _PATCHED_FLAG, True)


__all__ = ["apply_advanced_result_quantile_array_patch"]
