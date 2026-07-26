"""Prevent position extrapolation when assigning spikes to encoding bins."""

from __future__ import annotations

import sys
from functools import wraps

import numpy as np

_PATCH_MARKER = "_encoding_position_support_patch"
_PATCH_VERSION = 1
_ORIGINAL_ATTR = "__hipporeplayimm_original__"


def _synchronize_interpolator_aliases(previous: object, patched: object) -> None:
    """Refresh package-local aliases imported before the patch was installed."""

    for module in list(sys.modules.values()):
        module_name = getattr(module, "__name__", "")
        if not module_name.startswith("hipporeplayimm"):
            continue
        if getattr(module, "_interp_positions", None) is previous:
            module._interp_positions = patched


def apply_encoding_position_support_patch() -> None:
    """Mark position queries outside the measured timestamp support as invalid."""

    from . import encoding

    current = encoding._interp_positions
    if getattr(current, _PATCH_MARKER, None) == _PATCH_VERSION:
        previous = getattr(current, _ORIGINAL_ATTR, None)
        if previous is not None:
            _synchronize_interpolator_aliases(previous, current)
        return

    previous = current

    @wraps(previous)
    def _interp_positions(
        times: np.ndarray,
        xy: np.ndarray,
        query_times: np.ndarray,
    ) -> np.ndarray:
        interpolated = np.asarray(previous(times, xy, query_times), dtype=float)
        time_values = np.asarray(times, dtype=float)
        query_values = np.asarray(query_times, dtype=float).reshape(-1)
        if time_values.ndim != 1 or time_values.shape[0] == 0:
            return interpolated
        if interpolated.ndim != 2 or interpolated.shape[0] != query_values.shape[0]:
            return interpolated

        outside_support = (
            ~np.isfinite(query_values)
            | (query_values < time_values[0])
            | (query_values > time_values[-1])
        )
        interpolated[outside_support] = np.nan
        return interpolated

    setattr(_interp_positions, _PATCH_MARKER, _PATCH_VERSION)
    setattr(_interp_positions, _ORIGINAL_ATTR, previous)
    encoding._interp_positions = _interp_positions
    _synchronize_interpolator_aliases(previous, _interp_positions)


__all__ = ["apply_encoding_position_support_patch"]
