"""Runtime validation for spike-cell identifier fields loaded from data files.

MATLAB datasets commonly store integer identifiers as floating-point values.  The
loader should accept integral floats such as ``1.0`` but must not silently truncate
corrupted fractional identifiers such as ``1.5`` to ``1``.  The same guard is
installed for manually constructed sessions before place-field encoding selects
spikes and cell IDs.  Replay-event selectors use the same integer-like MATLAB
values.  Integral scalar-array wrappers are valid replay-event indices, but
boolean flags and nonintegral scalar wrappers must not alias to event indices
0 or 1.
"""

from __future__ import annotations

from functools import wraps
from pathlib import Path
import sys
from typing import Any

import numpy as np

_PATCHED_FLAG = "_cell_id_validation_patch_applied"
_PATCH_MARK = "__hipporeplayimm_data_cell_id_validation_patch__"
_ORIGINAL_ATTR = "__hipporeplayimm_original__"


def apply_data_cell_id_validation_patch() -> None:
    """Install integral-ID validation on ``hipporeplayimm.data`` helpers."""

    from . import data, encoding

    current_cell_ids = getattr(getattr(data.ReplaySession, "cell_ids", None), "fget", None)
    current_excitatory_spikes = getattr(data.ReplaySession, "excitatory_spikes", None)
    current_ripple = getattr(data.ReplaySession, "ripple", None)
    original_load_replay_session = data.load_replay_session
    original_load_spike_marks = data._load_spike_marks
    original_mark_group_ids = data._mark_group_ids_from_tetrode_cell_ids
    original_coerce_ripple_event = data._coerce_ripple_event
    original_as_integer_vector = data._as_integer_vector
    original_spikes_and_cell_ids_for_encoding = encoding._spikes_and_cell_ids_for_encoding

    patch_targets = (
        current_cell_ids,
        current_excitatory_spikes,
        current_ripple,
        original_load_replay_session,
        original_load_spike_marks,
        original_mark_group_ids,
        original_coerce_ripple_event,
        original_as_integer_vector,
        original_spikes_and_cell_ids_for_encoding,
    )
    if all(_is_patched(target) for target in patch_targets):
        _synchronize_coerce_ripple_event_aliases(
            getattr(original_coerce_ripple_event, _ORIGINAL_ATTR, original_coerce_ripple_event),
            original_coerce_ripple_event,
        )
        setattr(data, _PATCHED_FLAG, True)
        return

    def replay_session_cell_ids(self):
        if self.spikes.size == 0:
            return np.array([], dtype=int)
        spikes = np.asarray(self.spikes)
        if spikes.ndim != 2 or spikes.shape[1] < 2:
            raise ValueError("spikes must have at least two columns")
        return np.unique(_coerce_integral_ids(spikes[:, 1], "spike cell IDs"))

    def replay_session_excitatory_spikes(self):
        spikes = np.asarray(self.spikes)
        excitatory = np.asarray(self.excitatory_neurons)
        if spikes.size == 0 or excitatory.size == 0:
            return np.empty((0, 2), dtype=float)
        if spikes.ndim != 2 or spikes.shape[1] < 2:
            raise ValueError("spikes must have at least two columns")
        spike_ids = _coerce_integral_ids(spikes[:, 1], "spike cell IDs")
        excitatory_ids = _coerce_integral_ids(excitatory.reshape(-1), "excitatory neuron IDs")
        keep = np.isin(spike_ids, excitatory_ids)
        return spikes[keep]

    def replay_session_ripple(self, index):
        index_value = _coerce_ripple_index(index, self.ripple_count)
        return current_ripple(self, index_value)

    def load_replay_session(session_path):
        path = Path(session_path)
        spike_data = data._load_mat_file(path / "Spike_Data.mat")
        _validate_optional_neuron_ids(spike_data, "Excitatory_Neurons", "excitatory neuron IDs")
        _validate_optional_neuron_ids(spike_data, "Inhibitory_Neurons", "inhibitory neuron IDs")
        return original_load_replay_session(session_path)

    def load_spike_marks(session_path, spike_data, spikes):
        spikes_arr = np.asarray(spikes)
        if spikes_arr.size and spikes_arr.ndim == 2 and spikes_arr.shape[1] > 1:
            _coerce_integral_ids(spikes_arr[:, 1], "spike cell IDs")
        return original_load_spike_marks(session_path, spike_data, spikes)

    def mark_group_ids_from_tetrode_cell_ids(cell_ids, tetrode_cell_ids):
        if cell_ids is not None:
            _coerce_integral_ids(cell_ids, "spike cell IDs")
        values = _tetrode_cell_id_values_for_validation(tetrode_cell_ids, cell_ids)
        if values.size:
            _coerce_integral_ids(values, "tetrode/cell IDs")
        out = original_mark_group_ids(cell_ids, tetrode_cell_ids)
        if out is not None:
            return _coerce_integral_ids(out, "tetrode group IDs")
        return out

    def coerce_ripple_event(session, ripple):
        if _is_boolean_scalar(ripple):
            raise TypeError("ripple index must be an integer, not boolean")
        if _is_ripple_index_scalar(ripple) or isinstance(ripple, np.ndarray):
            return session.ripple(_coerce_ripple_index(ripple, session.ripple_count))
        return original_coerce_ripple_event(session, ripple)

    def as_integer_vector(value, name):
        return _coerce_integral_ids(value, name).reshape(-1)

    def spikes_and_cell_ids_for_encoding(session, config):
        spikes = np.asarray(session.spikes)
        if spikes.size:
            if spikes.ndim != 2 or spikes.shape[1] < 2:
                raise ValueError("spikes must have at least two columns")
            _coerce_integral_ids(spikes[:, 1], "spike cell IDs")
        excitatory = np.asarray(session.excitatory_neurons)
        if excitatory.size:
            _coerce_integral_ids(excitatory.reshape(-1), "excitatory neuron IDs")
        return original_spikes_and_cell_ids_for_encoding(session, config)

    if not _is_patched(current_cell_ids):
        data.ReplaySession.cell_ids = property(_mark_patched(replay_session_cell_ids, current_cell_ids))
    if not _is_patched(current_excitatory_spikes):
        data.ReplaySession.excitatory_spikes = _mark_patched(replay_session_excitatory_spikes, current_excitatory_spikes)
    if not _is_patched(current_ripple):
        data.ReplaySession.ripple = _mark_patched(replay_session_ripple, current_ripple)
    if not _is_patched(original_load_replay_session):
        data.load_replay_session = _mark_patched(load_replay_session, original_load_replay_session)
    if not _is_patched(original_load_spike_marks):
        data._load_spike_marks = _mark_patched(load_spike_marks, original_load_spike_marks)
    if not _is_patched(original_mark_group_ids):
        data._mark_group_ids_from_tetrode_cell_ids = _mark_patched(mark_group_ids_from_tetrode_cell_ids, original_mark_group_ids)
    active_coerce_ripple_event = original_coerce_ripple_event
    if not _is_patched(original_coerce_ripple_event):
        active_coerce_ripple_event = _mark_patched(coerce_ripple_event, original_coerce_ripple_event)
        data._coerce_ripple_event = active_coerce_ripple_event
    _synchronize_coerce_ripple_event_aliases(
        original_coerce_ripple_event,
        active_coerce_ripple_event,
    )
    if not _is_patched(original_as_integer_vector):
        data._as_integer_vector = _mark_patched(as_integer_vector, original_as_integer_vector)
    if not _is_patched(original_spikes_and_cell_ids_for_encoding):
        encoding._spikes_and_cell_ids_for_encoding = _mark_patched(spikes_and_cell_ids_for_encoding, original_spikes_and_cell_ids_for_encoding)

    setattr(data, _PATCHED_FLAG, True)


def _is_patched(target: Any) -> bool:
    return bool(getattr(target, _PATCH_MARK, False))


def _mark_patched(wrapper: Any, original: Any) -> Any:
    if callable(original):
        wrapper = wraps(original)(wrapper)
    setattr(wrapper, _PATCH_MARK, True)
    setattr(wrapper, _ORIGINAL_ATTR, original)
    return wrapper


def _synchronize_coerce_ripple_event_aliases(original: Any, active: Any) -> None:
    """Refresh modules that imported the ripple-event coercer by value."""

    for module in list(sys.modules.values()):
        if not getattr(module, "__name__", "").startswith("hipporeplayimm"):
            continue
        current = getattr(module, "_coerce_ripple_event", None)
        if current is original:
            setattr(module, "_coerce_ripple_event", active)


def _validate_optional_neuron_ids(spike_data: dict[str, Any], variable_name: str, label: str) -> None:
    if variable_name not in spike_data:
        return
    values = np.asarray(spike_data[variable_name])
    if values.size == 0:
        return
    _coerce_integral_ids(values.reshape(-1), label)


def _tetrode_cell_id_values_for_validation(tetrode_cell_ids: Any, cell_ids: Any) -> np.ndarray:
    """Return only mapping ID fields that the loader will actually consume."""

    arr = np.asarray(tetrode_cell_ids)
    if arr.size == 0:
        return np.asarray([], dtype=object)
    arr = np.squeeze(arr)
    if arr.ndim == 1:
        if arr.shape[0] == 2:
            return _finite_complete_rows(arr.reshape(1, 2)).reshape(-1)
        if arr.shape[0] == np.asarray(cell_ids).shape[0]:
            return _finite_values(arr)
        return np.asarray([], dtype=object)
    if arr.ndim != 2:
        return np.asarray([], dtype=object)
    if arr.shape[0] == 2 and arr.shape[1] != 2:
        arr = arr.T
    if arr.shape[1] < 2:
        return np.asarray([], dtype=object)
    return _finite_complete_rows(arr[:, :2]).reshape(-1)


def _finite_values(values: Any) -> np.ndarray:
    raw = np.asarray(values)
    numeric = _as_numeric_ids(raw)
    if numeric.size == 0:
        return np.asarray([], dtype=object)
    return raw[np.isfinite(numeric)]


def _finite_complete_rows(values: Any) -> np.ndarray:
    raw = np.asarray(values)
    numeric = _as_numeric_ids(raw)
    if numeric.size == 0:
        return np.asarray([], dtype=object)
    finite_rows = np.isfinite(numeric).all(axis=1)
    if not np.any(finite_rows):
        return np.asarray([], dtype=object)
    return raw[finite_rows]


def _as_numeric_ids(values: Any) -> np.ndarray:
    try:
        return np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("tetrode/cell IDs must contain numeric integer identifiers") from exc


def _coerce_ripple_index(index: Any, ripple_count: int) -> int:
    if _is_boolean_scalar(index):
        raise TypeError("ripple index must be an integer, not boolean")
    item = _ripple_index_scalar_item(index)
    if item is None:
        raise TypeError("ripple index must be an integer")
    if isinstance(item, (int, np.integer)):
        resolved = int(item)
    else:
        try:
            numeric = float(item)
        except (TypeError, ValueError) as exc:
            raise TypeError("ripple index must be an integer") from exc
        if not np.isfinite(numeric) or not numeric.is_integer():
            raise TypeError("ripple index must be an integer")
        resolved = int(numeric)
    count = int(ripple_count)
    if resolved < 0 or resolved >= count:
        raise IndexError(f"ripple index {resolved} out of range for {count} ripple events")
    return resolved


def _is_ripple_index_scalar(value: Any) -> bool:
    return _ripple_index_scalar_item(value) is not None


def _ripple_index_scalar_item(value: Any) -> Any | None:
    """Return a numeric scalar candidate for replay-event indexing, if present."""

    if _is_boolean_scalar(value):
        return None
    if isinstance(value, (int, np.integer, float, np.floating)):
        return value
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError):
        return None
    if raw.ndim != 0:
        return None
    if np.issubdtype(raw.dtype, np.bool_):
        return None
    if np.issubdtype(raw.dtype, np.number):
        try:
            return raw.item()
        except (TypeError, ValueError):
            return None
    if raw.dtype == object:
        try:
            item = raw.item()
        except (TypeError, ValueError):
            return None
        if isinstance(item, (bool, np.bool_)):
            return None
        if isinstance(item, (int, np.integer, float, np.floating)):
            return item
    return None


def _is_boolean_scalar(value: Any) -> bool:
    """Return True for Python, NumPy, and object-wrapped boolean scalars."""

    if isinstance(value, (bool, np.bool_)):
        return True
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError):
        return False
    if raw.ndim != 0:
        return False
    if np.issubdtype(raw.dtype, np.bool_):
        return True
    if raw.dtype == object:
        try:
            return isinstance(raw.item(), (bool, np.bool_))
        except (TypeError, ValueError):
            return False
    return False


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
    raw = np.asarray(values)
    if _contains_boolean_ids(raw):
        raise ValueError(f"{name} must not contain boolean identifiers")
    if raw.size == 0:
        return np.asarray(raw, dtype=int)
    integer_dtype = np.dtype(int)
    integer_info = np.iinfo(integer_dtype)
    coerced = np.asarray(
        [_coerce_integral_id(value, name, integer_info) for value in raw.reshape(-1)],
        dtype=integer_dtype,
    )
    return coerced.reshape(raw.shape)


def _coerce_integral_id(value: Any, name: str, integer_info: np.iinfo) -> int:
    try:
        item = np.asarray(value).item()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain finite integer identifiers") from exc
    if isinstance(item, (bool, np.bool_)):
        raise ValueError(f"{name} must not contain boolean identifiers")
    if isinstance(item, (int, np.integer)):
        identifier = int(item)
    else:
        try:
            numeric = float(item)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must contain numeric integer identifiers") from exc
        if not np.isfinite(numeric):
            raise ValueError(f"{name} must be finite integer identifiers")
        if not numeric.is_integer():
            raise ValueError(f"{name} must be integer-valued")
        identifier = int(numeric)
    if identifier < int(integer_info.min) or identifier > int(integer_info.max):
        raise ValueError(f"{name} must fit into integer identifier range")
    return identifier
