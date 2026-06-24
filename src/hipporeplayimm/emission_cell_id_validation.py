"""Validate spike-cell IDs during emission construction.

`build_emissions()` historically cast ripple spike cell IDs to ``int`` before
matching them against encoding rows.  That silently mapped corrupted fractional
IDs such as ``1.5`` onto a real cell ``1`` and counted the spike for the wrong
unit.  This patch validates both encoding and spike cell IDs as finite
integer-valued identifiers before any row lookup can truncate them.

"""

from __future__ import annotations

from functools import wraps
from typing import Any

import numpy as np

_PATCHED_FLAG = "_emission_cell_id_validation_patch_applied"


def _contains_boolean_ids(values: np.ndarray) -> bool:
    raw = np.asarray(values)
    if raw.size == 0:
        return False
    if np.issubdtype(raw.dtype, np.bool_):
        return True
    if raw.dtype == object:
        return any(isinstance(value, (bool, np.bool_)) for value in raw.reshape(-1))
    return False


def _coerce_integral_ids(values: Any, name: str) -> np.ndarray:
    ids = np.asarray(values)
    if ids.ndim == 0:
        ids = ids.reshape(1)
    if ids.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if ids.size == 0:
        return np.empty(0, dtype=int)
    if _contains_boolean_ids(ids):
        raise ValueError(f"{name} must not contain boolean identifiers")
    try:
        numeric = np.asarray(ids, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain finite integer identifiers") from exc
    if not np.all(np.isfinite(numeric)):
        raise ValueError(f"{name} must contain finite integer identifiers")
    rounded = np.rint(numeric)
    if not np.all(np.isclose(numeric, rounded, rtol=0.0, atol=1e-9)):
        raise ValueError(f"{name} must be integer-valued")
    integer_info = np.iinfo(np.dtype(int))
    if not np.all((rounded >= integer_info.min) & (rounded <= integer_info.max)):
        raise ValueError(f"{name} must fit into integer identifier range")
    return rounded.astype(int)


def _cell_id_row_indices(cell_ids: np.ndarray, spike_cell_ids: np.ndarray) -> np.ndarray:
    """Map spike cell IDs to encoding rows without lossy integer casts."""

    available = _coerce_integral_ids(cell_ids, "encoding.cell_ids")
    if np.unique(available).shape[0] != available.shape[0]:
        raise ValueError("encoding.cell_ids must be unique")

    requested = _coerce_integral_ids(spike_cell_ids, "spike cell IDs")
    row_by_cell_id = {int(cell_id): index for index, cell_id in enumerate(available)}
    return np.fromiter(
        (row_by_cell_id.get(int(cell_id), -1) for cell_id in requested),
        dtype=int,
        count=requested.shape[0],
    )


def apply_emission_cell_id_validation_patch() -> None:
    """Install integral-ID validation for emission row lookups."""

    from . import encoding as encoding_module
    from . import kd_reference as kd_module

    if not getattr(encoding_module, _PATCHED_FLAG, False):
        original_build_emissions = encoding_module.build_emissions

        @wraps(original_build_emissions)
        def build_emissions(session, encoding, ripple, config=None):
            encoding_cell_ids = _coerce_integral_ids(encoding.cell_ids, "encoding.cell_ids")

            spikes = np.asarray(session.spikes)
            if spikes.size and encoding_cell_ids.size > 0:
                if spikes.ndim != 2 or spikes.shape[1] < 2:
                    raise ValueError("spikes must be two-dimensional with at least time and cell-id columns")
                ripple_event = encoding_module._coerce_ripple_event(session, ripple)
                in_ripple = (spikes[:, 0] >= ripple_event.start) & (spikes[:, 0] < ripple_event.end)
                if np.any(in_ripple):
                    _coerce_integral_ids(spikes[in_ripple, 1], "spike cell IDs")

            return original_build_emissions(session, encoding, ripple, config)

        encoding_module._cell_id_row_indices = _cell_id_row_indices
        encoding_module.build_emissions = build_emissions
        setattr(encoding_module, _PATCHED_FLAG, True)

    if not getattr(kd_module, _PATCHED_FLAG, False):
        original_build_kd_emissions = kd_module.build_kd_emissions

        @wraps(original_build_kd_emissions)
        def build_kd_emissions(session, encoding, ripple, time_bin_s, spike_rate_scale=1.0):
            encoding_cell_ids = _coerce_integral_ids(encoding.cell_ids, "encoding.cell_ids")

            spikes = np.asarray(session.spikes)
            if spikes.size and encoding_cell_ids.size > 0:
                if spikes.ndim != 2 or spikes.shape[1] < 2:
                    raise ValueError("spikes must be two-dimensional with at least time and cell-id columns")
                ripple_event = kd_module._coerce_ripple_event(session, ripple)
                in_ripple = (spikes[:, 0] >= ripple_event.start) & (spikes[:, 0] < ripple_event.end)
                if np.any(in_ripple):
                    _coerce_integral_ids(spikes[in_ripple, 1], "spike cell IDs")

            return original_build_kd_emissions(
                session,
                encoding,
                ripple,
                time_bin_s,
                spike_rate_scale=spike_rate_scale,
            )

        kd_module.build_kd_emissions = build_kd_emissions
        setattr(kd_module, _PATCHED_FLAG, True)


__all__ = ["apply_emission_cell_id_validation_patch"]
