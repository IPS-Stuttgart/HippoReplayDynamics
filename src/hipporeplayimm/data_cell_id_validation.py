"""Runtime validation for spike-cell identifier fields loaded from data files.

MATLAB datasets commonly store integer identifiers as floating-point values.  The
loader should accept integral floats such as ``1.0`` but must not silently truncate
corrupted fractional identifiers such as ``1.5`` to ``1``.  The same guard is
installed for manually constructed sessions before place-field encoding selects
spikes and cell IDs.  Replay-event selectors use the same integer-like MATLAB
values, but boolean flags must not alias to event indices 0 or 1.
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
    original_spikes_and_cell_ids_for_encoding = encoding._spikes_and_cell_ids_for_encoding

    patch_targets = (
        current_cell_ids,
        current_excitatory_spikes,
        current_ripple,
        original_load_replay_session,
        original_load_spike_marks,
        original_mark_group_ids,
        original_coerce_ripple_event,
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
        arr = np.asarray(tetrode_cell_ids)
        if arr.size:
            values = np.asarray(arr)
            finite_mask = np.isfinite(np.asarray(values, dtype=float))
            finite_values = values[finite_mask]
            if finite_values.size:
                _coerce_integral_ids(finite_values, "tetrode/cell IDs")
        out = original_mark_group_ids(cell_ids, tetrode_cell_ids)
        if out is not None:
            return _coerce_integral_ids(out, "tetrode group IDs")
        return out

    def coerce_ripple_event(session, ripple):
        if isinstance(ripple, (bool, np.bool_)):
            raise TypeError("ripple index must be an integer, not boolean")
        if isinstance(ripple, (int, np.integer)):
            return session.ripple(ripple)
        return original_coerce_ripple_event(session, ripple)

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


def _coerce_ripple_index(index: Any, ripple_count: int) -> int:
    if isinstance(index, (bool, np.bool_)):
        raise TypeError("ripple index must be an integer, not boolean")
    if not isinstance(index, (int, np.integer)):
        raise TypeError("ripple index must be an integer")
    resolved = int(index)
    count = int(ripple_count)
    if resolved < 0 or resolved >= count:
        raise IndexError(f"ripple index {resolved} out of range for {count} ripple events")
    return resolved


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
    ids = np.asarray(raw, dtype=float)
    if ids.size == 0:
        return np.asarray(ids, dtype=int)
    if not np.all(np.isfinite(ids)):
        raise ValueError(f"{name} must be finite integer identifiers")
    rounded = np.rint(ids)
    if not np.all(ids == rounded):
        raise ValueError(f"{name} must be integer-valued")
    integer_dtype = np.dtype(int)
    integer_info = np.iinfo(integer_dtype)
    if not np.all((rounded >= integer_info.min) & (rounded <= integer_info.max)):
        raise ValueError(f"{name} must fit into integer identifier range")
    return rounded.astype(integer_dtype)
