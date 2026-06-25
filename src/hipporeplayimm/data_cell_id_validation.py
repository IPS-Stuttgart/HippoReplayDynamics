"""Runtime validation for spike-cell identifier fields loaded from data files.

MATLAB datasets commonly store integer identifiers as floating-point values.  The
loader should accept integral floats such as ``1.0`` but must not silently truncate
corrupted fractional identifiers such as ``1.5`` to ``1``.  The same guard is
installed for manually constructed sessions before place-field encoding selects
spikes and cell IDs.
"""

from __future__ import annotations

from functools import wraps
from pathlib import Path
from typing import Any

import numpy as np


def apply_data_cell_id_validation_patch() -> None:
    """Install integral-ID validation on ``hipporeplayimm.data`` helpers."""

    from . import data, encoding

    if getattr(data, "_cell_id_validation_patch_applied", False):
        return

    original_load_replay_session = data.load_replay_session
    original_load_spike_marks = data._load_spike_marks
    original_mark_group_ids = data._mark_group_ids_from_tetrode_cell_ids
    original_spikes_and_cell_ids_for_encoding = encoding._spikes_and_cell_ids_for_encoding

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

    @wraps(original_load_replay_session)
    def load_replay_session(session_path):
        path = Path(session_path)
        spike_data = data._load_mat_file(path / "Spike_Data.mat")
        _validate_optional_neuron_ids(spike_data, "Excitatory_Neurons", "excitatory neuron IDs")
        _validate_optional_neuron_ids(spike_data, "Inhibitory_Neurons", "inhibitory neuron IDs")
        return original_load_replay_session(session_path)

    @wraps(original_load_spike_marks)
    def load_spike_marks(session_path, spike_data, spikes):
        spikes_arr = np.asarray(spikes)
        if spikes_arr.size and spikes_arr.ndim == 2 and spikes_arr.shape[1] > 1:
            _coerce_integral_ids(spikes_arr[:, 1], "spike cell IDs")
        return original_load_spike_marks(session_path, spike_data, spikes)

    @wraps(original_mark_group_ids)
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

    @wraps(original_spikes_and_cell_ids_for_encoding)
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

    data.ReplaySession.cell_ids = property(replay_session_cell_ids)
    data.ReplaySession.excitatory_spikes = replay_session_excitatory_spikes
    data.load_replay_session = load_replay_session
    data._load_spike_marks = load_spike_marks
    data._mark_group_ids_from_tetrode_cell_ids = mark_group_ids_from_tetrode_cell_ids
    encoding._spikes_and_cell_ids_for_encoding = spikes_and_cell_ids_for_encoding
    data._cell_id_validation_patch_applied = True


def _validate_optional_neuron_ids(spike_data: dict[str, Any], variable_name: str, label: str) -> None:
    if variable_name not in spike_data:
        return
    values = np.asarray(spike_data[variable_name])
    if values.size == 0:
        return
    _coerce_integral_ids(values.reshape(-1), label)


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
    if not np.all(np.isclose(ids, rounded, rtol=0.0, atol=1e-9)):
        raise ValueError(f"{name} must be integer-valued")
    integer_dtype = np.dtype(int)
    integer_info = np.iinfo(integer_dtype)
    if not np.all((rounded >= integer_info.min) & (rounded <= integer_info.max)):
        raise ValueError(f"{name} must fit into integer identifier range")
    return rounded.astype(integer_dtype)
