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
import sys
from typing import Any

import numpy as np

_PATCH_MARK = "__hipporeplayimm_spike_cell_id_emission_validation_patch__"
_SPIKE_SELECTION_PATCH_MARK = (
    "__hipporeplayimm_spike_cell_id_encoding_selection_patch__"
)
_ORIGINAL_ATTR = "__hipporeplayimm_original__"


class _ExactSpikeTable(np.ndarray):
    """Spike table exposing times as floats and IDs as exact platform integers."""

    def __array_finalize__(self, obj: Any) -> None:
        del obj

    def __getitem__(self, key: Any) -> Any:
        result = super().__getitem__(key)
        column = _selected_scalar_column(key, self.shape)
        if column not in (0, 1):
            return result
        converted = np.asarray(result, dtype=float if column == 0 else int)
        return converted.item() if converted.ndim == 0 else converted


def apply_spike_cell_id_emission_validation_patch() -> None:
    """Install exact integral-ID handling for encoding and emission lookups."""

    from . import emission_cell_id_validation
    from . import encoding
    from . import log_emission_n_spikes_validation

    # The later emission-cell-ID patch installs its own row mapper.  Synchronize
    # the exact coercion helper into that active module before its wrapper is
    # applied so extended-precision and object-backed identifiers never pass
    # through binary64.
    emission_cell_id_validation._coerce_integral_ids = _coerce_integral_ids

    # LogEmissionTensor construction has a separate scalar-ID coercion path.
    # Keep it on the same exact parser so text and Decimal identifiers above
    # binary64's exact-integer range remain distinct.
    log_emission_n_spikes_validation._coerce_integer_identifier = _coerce_integral_id

    current_selection = encoding._spikes_and_cell_ids_for_encoding
    if bool(getattr(current_selection, _SPIKE_SELECTION_PATCH_MARK, False)):
        active_selection = current_selection
    else:
        original_selection = current_selection

        @wraps(original_selection)
        def spikes_and_cell_ids_for_encoding(session: Any, config: Any) -> tuple[np.ndarray, np.ndarray]:
            return _select_spikes_and_cell_ids_exactly(session, config)

        setattr(spikes_and_cell_ids_for_encoding, _SPIKE_SELECTION_PATCH_MARK, True)
        setattr(spikes_and_cell_ids_for_encoding, _ORIGINAL_ATTR, original_selection)
        active_selection = spikes_and_cell_ids_for_encoding
        encoding._spikes_and_cell_ids_for_encoding = active_selection

    _synchronize_spike_selection_aliases(active_selection)

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


def _select_spikes_and_cell_ids_exactly(
    session: Any,
    config: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Select encoding spikes without converting the ID column to binary64."""

    spikes = np.asarray(session.spikes)
    if spikes.size:
        if spikes.ndim != 2 or spikes.shape[1] < 2:
            raise ValueError("spikes must have at least two columns")
        spike_ids = _coerce_integral_ids(spikes[:, 1], "spike cell IDs").reshape(-1)
    else:
        spike_ids = np.empty(0, dtype=int)

    excitatory = np.asarray(session.excitatory_neurons)
    if bool(config.use_excitatory) and excitatory.size:
        cell_ids = _coerce_integral_ids(
            excitatory.reshape(-1),
            "excitatory neuron IDs",
        ).reshape(-1)
        selected_mask = np.isin(spike_ids, cell_ids)
        selected_spikes = spikes[selected_mask]
        selected_ids = spike_ids[selected_mask]
    else:
        cell_ids = np.unique(spike_ids)
        selected_spikes = spikes
        selected_ids = spike_ids

    exact_spikes = _exact_spike_table(selected_spikes, selected_ids)
    unique_cell_ids = np.asarray(sorted(np.unique(cell_ids)), dtype=int)
    return exact_spikes, unique_cell_ids


def _exact_spike_table(spikes: Any, spike_ids: Any) -> np.ndarray:
    raw = np.asarray(spikes)
    table = np.asarray(raw, dtype=object).copy()
    if table.size:
        if table.ndim != 2 or table.shape[1] < 2:
            raise ValueError("spikes must have at least two columns")
        ids = np.asarray(spike_ids, dtype=int).reshape(-1)
        if ids.shape[0] != table.shape[0]:
            raise ValueError("spike cell IDs must contain one ID per spike row")
        table[:, 1] = ids
    return table.view(_ExactSpikeTable)


def _selected_scalar_column(key: Any, shape: tuple[int, ...]) -> int | None:
    if len(shape) < 2 or not isinstance(key, tuple) or len(key) < 2:
        return None
    column = key[1]
    if not isinstance(column, (int, np.integer)):
        return None
    resolved = int(column)
    if resolved < 0:
        resolved += int(shape[1])
    return resolved


def _synchronize_spike_selection_aliases(active: Any) -> None:
    """Refresh modules that imported the encoding spike selector by value."""

    lineage: set[Any] = set()
    current = active
    while callable(current) and current not in lineage:
        lineage.add(current)
        current = getattr(current, _ORIGINAL_ATTR, None)

    for module in list(sys.modules.values()):
        if not getattr(module, "__name__", "").startswith("hipporeplayimm"):
            continue
        alias = getattr(module, "_spikes_and_cell_ids_for_encoding", None)
        if alias in lineage and alias is not active:
            setattr(module, "_spikes_and_cell_ids_for_encoding", active)


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
