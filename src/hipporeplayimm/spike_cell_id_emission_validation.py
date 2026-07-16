"""Runtime validation for spike cell IDs during emission binning.

Replay emission construction maps per-ripple spikes onto encoding rows.  The
loader-level cell-ID guard catches malformed identifiers for normal data loads,
but manually constructed sessions and downstream wrappers can still call
``build_emissions`` directly.  Validate at the row-mapping boundary as well so
fractional, nonfinite, or boolean spike identifiers never alias to a different
cell after NumPy integer casting.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARK = "__hipporeplayimm_spike_cell_id_emission_validation_patch__"
_ORIGINAL_ATTR = "__hipporeplayimm_original__"


def apply_spike_cell_id_emission_validation_patch() -> None:
    """Install integral-ID validation on emission cell-row mapping."""

    from . import emission_cell_id_validation
    from . import encoding

    # The later emission-cell-ID patch installs its own row mapper.  Synchronize
    # the exact coercion helper into that active module before its wrapper is
    # applied so extended-precision and object-backed identifiers never pass
    # through binary64.
    emission_cell_id_validation._coerce_integral_ids = _coerce_integral_ids

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
    """Return exact platform integer IDs without a binary64 round trip."""

    raw = np.asarray(values)
    if raw.size == 0:
        return np.asarray(raw, dtype=int)

    integer_dtype = np.dtype(int)
    integer_info = np.iinfo(integer_dtype)
    coerced = np.asarray(
        [
            _coerce_integral_id(value, name, integer_info)
            for value in raw.reshape(-1)
        ],
        dtype=integer_dtype,
    )
    return coerced.reshape(raw.shape)


def _coerce_integral_text_id(value: str | bytes, name: str) -> int:
    if isinstance(value, bytes):
        try:
            text = value.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"{name} must contain numeric integer identifiers"
            ) from exc
    else:
        text = value
    text = text.strip()
    if not text:
        raise ValueError(f"{name} must contain numeric integer identifiers")
    try:
        return int(text, 10)
    except ValueError:
        pass
    try:
        numeric = Decimal(text)
    except InvalidOperation as exc:
        raise ValueError(
            f"{name} must contain numeric integer identifiers"
        ) from exc
    if not numeric.is_finite():
        raise ValueError(f"{name} must contain finite integer identifiers")
    integral = numeric.to_integral_value()
    if numeric != integral:
        raise ValueError(f"{name} must contain integer-valued identifiers")
    return int(integral)


def _coerce_integral_id(
    value: Any,
    name: str,
    integer_info: np.iinfo,
) -> int:
    try:
        item = np.asarray(value).item()
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"{name} must contain finite integer identifiers"
        ) from exc

    if isinstance(item, (bool, np.bool_)):
        raise ValueError(f"{name} must not contain boolean identifiers")
    if isinstance(item, (int, np.integer)):
        identifier = int(item)
    elif isinstance(item, Decimal):
        if not item.is_finite():
            raise ValueError(f"{name} must contain finite integer identifiers")
        integral = item.to_integral_value()
        if item != integral:
            raise ValueError(f"{name} must contain integer-valued identifiers")
        identifier = int(integral)
    elif isinstance(item, (str, bytes)):
        identifier = _coerce_integral_text_id(item, name)
    elif isinstance(item, (float, np.floating)):
        if not bool(np.isfinite(item)):
            raise ValueError(f"{name} must contain finite integer identifiers")
        if not bool(item.is_integer()):
            raise ValueError(f"{name} must contain integer-valued identifiers")
        identifier = int(item)
    else:
        try:
            identifier = int(item)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"{name} must contain numeric integer identifiers"
            ) from exc
        try:
            exact = bool(item == identifier)
        except (TypeError, ValueError, OverflowError):
            exact = False
        if not exact:
            raise ValueError(f"{name} must contain integer-valued identifiers")

    if identifier < int(integer_info.min) or identifier > int(integer_info.max):
        raise ValueError(f"{name} must fit into integer identifier range")
    return identifier
