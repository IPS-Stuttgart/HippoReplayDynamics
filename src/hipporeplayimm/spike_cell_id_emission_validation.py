"""Runtime validation for spike cell IDs during emission binning.

Replay emission construction maps per-ripple spikes onto encoding rows.  The
loader-level cell-ID guard catches malformed identifiers for normal data loads,
but manually constructed sessions and downstream wrappers can still call
``build_emissions`` directly.  Validate at the row-mapping boundary as well so
fractional, nonfinite, or boolean spike identifiers never alias to a different
cell after NumPy integer casting.
"""

from __future__ import annotations

from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARK = "__hipporeplayimm_spike_cell_id_emission_validation_patch__"
_ORIGINAL_ATTR = "__hipporeplayimm_original__"


def apply_spike_cell_id_emission_validation_patch() -> None:
    """Install integral-ID validation on emission cell-row mapping."""

    from . import encoding

    current = encoding._cell_id_row_indices
    if bool(getattr(current, _PATCH_MARK, False)):
        return

    @wraps(current)
    def cell_id_row_indices(cell_ids: Any, spike_cell_ids: Any) -> np.ndarray:
        available = _coerce_integral_ids(cell_ids, "encoding cell IDs")
        requested = _coerce_integral_ids(spike_cell_ids, "spike cell IDs")
        return current(available, requested)

    setattr(cell_id_row_indices, _PATCH_MARK, True)
    setattr(cell_id_row_indices, _ORIGINAL_ATTR, current)
    encoding._cell_id_row_indices = cell_id_row_indices


def _coerce_integral_ids(values: Any, name: str) -> np.ndarray:
    raw = np.asarray(values)
    if raw.size == 0:
        return np.asarray(raw, dtype=int)
    if np.issubdtype(raw.dtype, np.bool_):
        raise ValueError(f"{name} must not contain boolean identifiers")
    if raw.dtype == object and any(isinstance(value, (bool, np.bool_)) for value in raw.reshape(-1)):
        raise ValueError(f"{name} must not contain boolean identifiers")

    try:
        numeric = np.asarray(raw, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric integer identifiers") from exc
    if not np.all(np.isfinite(numeric)):
        raise ValueError(f"{name} must contain finite integer identifiers")
    if not np.all(np.equal(numeric, np.round(numeric))):
        raise ValueError(f"{name} must contain integer-valued identifiers")

    integer_info = np.iinfo(np.dtype(int))
    if np.any(numeric < integer_info.min) or np.any(numeric > integer_info.max):
        raise ValueError(f"{name} must fit into integer identifier range")
    return numeric.astype(int)
