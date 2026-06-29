"""Dataset loading for Pfeiffer/Foster replay sessions."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io as sio

MARK_TERMS = ("amplitude", "amplitudes", "clusterless", "feature", "features", "mark", "marks", "waveform", "waveforms")
MARK_VAR_TERMS = MARK_TERMS + ("peak",)
MARK_EXCLUDE = {"spike_data", "tetrode_cell_ids", "excitatory_neurons", "inhibitory_neurons"}


@dataclass(frozen=True)
class RippleEvent:
    start: float
    end: float
    peak: float
    raw_power: float
    z_power_session: float
    z_power_epoch: float

    @classmethod
    def from_row(cls, row: np.ndarray) -> "RippleEvent":
        return cls(*(float(x) for x in row[:6]))


@dataclass(frozen=True)
class SpikeMarkData:
    times: np.ndarray
    marks: np.ndarray
    source_file: str
    source_variable: str
    feature_names: tuple[str, ...]
    cell_ids: np.ndarray | None = None
    group_ids: np.ndarray | None = None

    @property
    def n_spikes(self) -> int:
        return int(self.marks.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.marks.shape[1]) if self.marks.ndim == 2 else 0


@dataclass
class ReplaySession:
    rat: str
    name: str
    path: Path
    position: np.ndarray
    spikes: np.ndarray
    tetrode_cell_ids: np.ndarray
    excitatory_neurons: np.ndarray
    inhibitory_neurons: np.ndarray
    ripple_events: np.ndarray
    run_times: np.ndarray
    sleep_box_immobile_times: np.ndarray
    sleep_times: np.ndarray
    rem_times: np.ndarray
    well_sequence: np.ndarray | None
    metadata: dict[str, Any]
    spike_marks: SpikeMarkData | None = None

    @property
    def session_id(self) -> str:
        return f"{self.rat}/{self.name}"

    @property
    def ripple_count(self) -> int:
        return int(self.ripple_events.shape[0])

    @property
    def cell_ids(self) -> np.ndarray:
        if self.spikes.size == 0:
            return np.array([], dtype=int)
        return np.unique(self.spikes[:, 1].astype(int))

    @property
    def has_spike_marks(self) -> bool:
        return self.spike_marks is not None and self.spike_marks.n_features > 0

    def excitatory_spikes(self) -> np.ndarray:
        if self.spikes.size == 0 or self.excitatory_neurons.size == 0:
            return np.empty((0, 2), dtype=float)
        keep = np.isin(self.spikes[:, 1].astype(int), self.excitatory_neurons.astype(int))
        return self.spikes[keep]

    def ripple(self, index: int) -> RippleEvent:
        return RippleEvent.from_row(self.ripple_events[index])

    def ripple_indices_in_run(self) -> np.ndarray:
        if self.ripple_events.size == 0 or self.run_times.size == 0:
            return np.array([], dtype=int)
        return np.flatnonzero(_times_in_intervals(self.ripple_events[:, 2], _as_intervals(self.run_times)))


def _coerce_ripple_event(session: ReplaySession, ripple: RippleEvent | int) -> RippleEvent:
    """Return a ripple event for Python or NumPy integer indices."""

    if isinstance(ripple, (int, np.integer)):
        return session.ripple(int(ripple))
    return ripple


def load_open_field_sessions(root: str | Path, include_rats: tuple[int, ...] = (1, 2, 3, 4)) -> list[ReplaySession]:
    root_path = Path(root)
    sessions: list[ReplaySession] = []
    for rat_number in include_rats:
        rat_dir = root_path / f"Rat{rat_number}"
        if not rat_dir.exists():
            continue
        for session_dir in sorted(rat_dir.glob("Open*")):
            if session_dir.is_dir():
                sessions.append(load_replay_session(session_dir))
    return sessions


def load_replay_session(session_path: str | Path) -> ReplaySession:
    path = Path(session_path)
    spike_data = _load_mat_file(path / "Spike_Data.mat")
    spikes = _as_two_column_array(spike_data.get("Spike_Data", np.empty((0, 2))), "Spike_Data")
    spike_marks = _load_spike_marks(path, spike_data, spikes)
    epochs = _load_mat_file(path / "Epochs.mat")
    metadata = dict(_load_optional_metadata(path / "Experiment_Information.mat"))
    metadata["spike_mark_source"] = "" if spike_marks is None else f"{spike_marks.source_file}:{spike_marks.source_variable}"
    metadata["spike_mark_features"] = 0 if spike_marks is None else spike_marks.n_features
    well_sequence = _required_array(path / "Well_Sequence.mat", "Well_Sequence") if (path / "Well_Sequence.mat").exists() else None
    return ReplaySession(
        rat=path.parent.name,
        name=path.name,
        path=path,
        position=np.asarray(_required_array(path / "Position_Data.mat", "Position_Data"), dtype=float),
        spikes=spikes,
        tetrode_cell_ids=np.asarray(spike_data.get("Tetrode_Cell_IDs", np.empty((0, 2)))),
        excitatory_neurons=_as_integer_vector(spike_data.get("Excitatory_Neurons", np.array([])), "excitatory neuron IDs"),
        inhibitory_neurons=_as_integer_vector(spike_data.get("Inhibitory_Neurons", np.array([])), "inhibitory neuron IDs"),
        ripple_events=_as_two_dimensional(_required_array(path / "Ripple_Events.mat", "Ripple_Events"), "Ripple_Events"),
        run_times=_as_intervals(epochs.get("Run_Times", np.empty((0, 2)))),
        sleep_box_immobile_times=_as_intervals(epochs.get("Sleep_Box_Immobile_Times", np.empty((0, 2)))),
        sleep_times=_as_intervals(epochs.get("Sleep_Times", np.empty((0, 2)))),
        rem_times=_as_intervals(epochs.get("REM_Times", np.empty((0, 2)))),
        well_sequence=None if well_sequence is None else _as_two_dimensional(well_sequence, "Well_Sequence"),
        metadata=metadata,
        spike_marks=spike_marks,
    )


def load_mat_variable(path: str | Path, variable_name: str) -> Any:
    return _load_mat_file(Path(path))[variable_name]


def _load_mat_file(path: Path) -> dict[str, Any]:
    try:
        loaded = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
        return {key: value for key, value in loaded.items() if not key.startswith("__")}
    except (NotImplementedError, OSError, ValueError):
        if _is_hdf5_file(path):
            return _load_hdf5_mat_file(path)
        raise


def _load_hdf5_mat_file(path: Path) -> dict[str, Any]:
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(f"{path} is a MATLAB v7.3/HDF5 file. Install h5py to load it.") from exc
    output: dict[str, Any] = {}
    with h5py.File(path, "r") as handle:
        for key in handle.keys():
            if key != "#refs#":
                output[key] = _read_hdf5_value(handle[key])
    return output


def _is_hdf5_file(path: Path) -> bool:
    try:
        import h5py
    except ImportError:
        return False
    return bool(h5py.is_hdf5(path))


def _read_hdf5_value(obj: Any) -> Any:
    import h5py
    if isinstance(obj, h5py.Dataset):
        arr = np.array(obj)
        if h5py.check_dtype(ref=arr.dtype) is not None:
            return _read_hdf5_reference_array(arr, obj.file)
        if arr.dtype.kind in {"S", "U"}:
            return arr.astype(str)
        matlab_class = _hdf5_matlab_class(obj)
        if arr.dtype == np.uint16 and arr.ndim >= 1 and arr.size > 0 and matlab_class == "char":
            try:
                return "".join(chr(int(x)) for x in arr.reshape(-1) if int(x) != 0)
            except (TypeError, ValueError):
                pass
        return np.squeeze(arr.T if arr.ndim >= 2 else arr)
    if isinstance(obj, h5py.Group):
        return {key: _read_hdf5_value(value) for key, value in obj.items()}
    raise TypeError(f"Unsupported HDF5 MATLAB object: {type(obj)!r}")


def _hdf5_matlab_class(obj: Any) -> str | None:
    """Return a normalized MATLAB_class attribute for an HDF5 MATLAB object."""

    raw = obj.attrs.get("MATLAB_class")
    if raw is None:
        return None
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="ignore").strip().lower()
    if isinstance(raw, str):
        return raw.strip().lower()
    arr = np.asarray(raw)
    if arr.shape == ():
        value = arr.item()
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="ignore").strip().lower()
        return str(value).strip().lower()
    if arr.dtype.kind == "S":
        try:
            joined = b"".join(bytes(value) for value in arr.reshape(-1))
        except (TypeError, ValueError):
            return None
        return joined.decode("utf-8", errors="ignore").strip().lower()
    return str(raw).strip().lower()


def _read_hdf5_reference_array(arr: np.ndarray, handle: Any) -> Any:
    """Dereference MATLAB v7.3 cell/struct references stored in HDF5 datasets."""

    import h5py

    refs = np.asarray(arr)

    def read_reference(ref: Any) -> Any:
        if not isinstance(ref, h5py.Reference):
            raise TypeError(f"Expected HDF5 object reference, got {type(ref)!r}")
        if not ref:
            return None
        return _read_hdf5_value(handle[ref])

    if refs.ndim == 0:
        return read_reference(refs.item())

    values = np.empty(refs.shape, dtype=object)
    for index in np.ndindex(refs.shape):
        values[index] = read_reference(refs[index])

    aligned = values.T if values.ndim >= 2 else values
    squeezed = np.squeeze(aligned)
    if isinstance(squeezed, np.ndarray) and squeezed.ndim == 0:
        return squeezed.item()
    return squeezed


def _required_array(path: Path, variable_name: str) -> np.ndarray:
    data = _load_mat_file(path)
    if variable_name not in data:
        raise KeyError(f"{path} does not contain {variable_name!r}; available: {', '.join(sorted(data))}")
    return np.asarray(data[variable_name])


def _load_optional_metadata(path: Path) -> dict[str, Any]:
    return {} if not path.exists() else _load_mat_file(path)


def _as_integer_vector(value: Any, name: str) -> np.ndarray:
    """Return a flat integer ID vector without silently truncating bad values."""

    arr = np.asarray(value)
    if arr.size == 0:
        return np.array([], dtype=int)
    try:
        numeric = np.asarray(arr, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric integer IDs") from exc
    flat = numeric.reshape(-1)
    if not np.all(np.isfinite(flat)):
        raise ValueError(f"{name} must contain finite integer IDs")
    if not np.all(np.equal(flat, np.round(flat))):
        raise ValueError(f"{name} must contain integer-valued IDs")
    return flat.astype(int)


def _load_spike_marks(session_path: Path, spike_data: dict[str, Any], spikes: np.ndarray) -> SpikeMarkData | None:
    if spikes.size == 0:
        return None
    spike_count = int(spikes.shape[0])
    spike_times = spikes[:, 0]
    cell_ids = spikes[:, 1].astype(int) if spikes.shape[1] > 1 else None
    group_ids = _mark_group_ids_from_tetrode_cell_ids(
        cell_ids,
        spike_data.get("Tetrode_Cell_IDs", np.empty((0, 2))),
    )
    for source_file, data in [("Spike_Data.mat", spike_data), *[(p.name, _load_mat_file(p)) for p in _candidate_mark_files(session_path)]]:
        for variable_name, value in _candidate_mark_variables(data):
            marks = _coerce_mark_matrix(value, spike_count=spike_count, spike_times=spike_times)
            if marks is not None:
                return SpikeMarkData(
                    times=np.asarray(spike_times, dtype=float).copy(),
                    marks=marks,
                    source_file=source_file,
                    source_variable=variable_name,
                    feature_names=tuple(f"{variable_name}_{idx}" for idx in range(marks.shape[1])),
                    cell_ids=None if cell_ids is None else np.asarray(cell_ids, dtype=int).copy(),
                    group_ids=None if group_ids is None else np.asarray(group_ids, dtype=int).copy(),
                )
    return None


def _mark_group_ids_from_tetrode_cell_ids(cell_ids: np.ndarray | None, tetrode_cell_ids: Any) -> np.ndarray | None:
    """Map spike cell IDs to tetrode/group IDs when the dataset exposes them.

    Pfeiffer/Foster releases commonly provide a ``Tetrode_Cell_IDs`` matrix, but
    MATLAB files are not perfectly consistent about whether the first or second
    column stores the within-session cell ID. This helper chooses the column that
    best overlaps the spike cell IDs and falls back to the original cell ID for
    unmapped rows so clusterless grouping never silently drops marked spikes.
    """

    if cell_ids is None:
        return None
    arr = np.asarray(tetrode_cell_ids)
    if arr.size == 0:
        return None
    arr = np.squeeze(arr)
    if arr.ndim == 1:
        if arr.shape[0] == 2:
            arr = arr.reshape(1, 2)
        elif arr.shape[0] == np.asarray(cell_ids).shape[0]:
            return np.asarray(arr, dtype=int)
        else:
            return None
    if arr.ndim != 2:
        return None
    if arr.shape[0] == 2 and arr.shape[1] != 2:
        arr = arr.T
    if arr.shape[1] < 2:
        return None

    pairs = np.asarray(arr[:, :2], dtype=float)
    finite = np.isfinite(pairs).all(axis=1)
    if not np.any(finite):
        return None
    pairs = pairs[finite]
    spike_cell_ids = np.asarray(cell_ids, dtype=int)
    unique_spike_cells = np.unique(spike_cell_ids)
    first_column = pairs[:, 0].astype(int)
    second_column = pairs[:, 1].astype(int)
    second_matches_cells = int(np.isin(unique_spike_cells, second_column).sum())
    first_matches_cells = int(np.isin(unique_spike_cells, first_column).sum())
    if second_matches_cells == 0 and first_matches_cells == 0:
        return None

    if second_matches_cells >= first_matches_cells:
        group_column, cell_column = 0, 1
    else:
        group_column, cell_column = 1, 0
    mapping = {int(row[cell_column]): int(row[group_column]) for row in pairs}
    return np.asarray([mapping.get(int(cell_id), int(cell_id)) for cell_id in spike_cell_ids], dtype=int)


def _candidate_mark_files(session_path: Path) -> list[Path]:
    return [p for p in sorted(session_path.glob("*.mat")) if p.name != "Spike_Data.mat" and any(term in p.stem.lower() for term in MARK_TERMS)]


def _candidate_mark_variables(data: dict[str, Any]) -> list[tuple[str, Any]]:
    return [(name, value) for name, value in data.items() if name.lower() not in MARK_EXCLUDE and any(term in name.lower() for term in MARK_VAR_TERMS)]


def _coerce_mark_matrix(value: Any, *, spike_count: int, spike_times: np.ndarray) -> np.ndarray | None:
    arr = np.asarray(value)
    if arr.size == 0 or arr.dtype.kind not in {"b", "i", "u", "f", "c"}:
        return None
    if arr.dtype.kind == "c" and not np.all(np.isfinite(np.imag(arr)) & np.isclose(np.imag(arr), 0.0)):
        return None
    arr = np.real(arr).astype(float, copy=False)
    if arr.ndim == 0:
        marks = arr.reshape(1, 1) if spike_count == 1 else None
    elif arr.ndim == 1:
        if arr.shape[0] == spike_count:
            marks = arr.reshape(spike_count, 1)
        elif spike_count == 1:
            marks = arr.reshape(1, -1)
        else:
            marks = None
    else:
        axes = [axis for axis, size in enumerate(arr.shape) if size == spike_count]
        if not axes:
            return None
        aligned = np.moveaxis(arr, 0 if 0 in axes else axes[-1], 0)
        marks = aligned.reshape(spike_count, -1)
    if marks is not None and marks.shape[1] >= 2 and _looks_like_time_column(marks[:, 0], spike_times):
        marks = marks[:, 1:]
    if marks is None or marks.shape[1] == 0 or not np.any(np.isfinite(marks)):
        return None
    return np.asarray(marks, dtype=float)


def _looks_like_time_column(candidate: np.ndarray, spike_times: np.ndarray) -> bool:
    if candidate.shape != spike_times.shape:
        return False
    finite = np.isfinite(candidate) & np.isfinite(spike_times)
    return bool(np.any(finite) and np.allclose(candidate[finite], spike_times[finite], rtol=1e-6, atol=1e-6))


def _as_two_column_array(value: Any, name: str) -> np.ndarray:
    arr = _as_two_dimensional(value, name)
    if arr.size == 0:
        return np.empty((0, 2), dtype=float)
    if arr.shape[1] != 2 and arr.shape[0] == 2:
        arr = arr.T
    if arr.shape[1] != 2:
        raise ValueError(f"{name} must have two columns; got shape {arr.shape}")
    return np.asarray(arr, dtype=float)


def _as_two_dimensional(value: Any, name: str) -> np.ndarray:
    arr = np.asarray(value)
    if arr.size == 0:
        n_columns = 6 if name == "Ripple_Events" else 2
        return np.empty((0, n_columns), dtype=float)
    if arr.ndim == 1:
        if name == "Ripple_Events":
            if arr.shape[0] == 6:
                return arr.reshape(1, 6)
            raise ValueError(f"{name} must have six columns; got shape {arr.shape}")
        if name == "Well_Sequence":
            if arr.shape[0] == 2:
                return arr.reshape(1, 2)
            raise ValueError(f"{name} must have two columns; got shape {arr.shape}")
        if arr.shape[0] == 2:
            return arr.reshape(1, 2)
        return arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be one- or two-dimensional; got shape {arr.shape}")
    if name == "Ripple_Events":
        if arr.shape[1] != 6 and arr.shape[0] == 6:
            arr = arr.T
        if arr.shape[1] != 6:
            raise ValueError(f"{name} must have six columns; got shape {arr.shape}")
    if name == "Well_Sequence":
        if arr.shape[1] != 2 and arr.shape[0] == 2:
            arr = arr.T
        if arr.shape[1] != 2:
            raise ValueError(f"{name} must have two columns; got shape {arr.shape}")
    return arr


def _as_intervals(value: Any) -> np.ndarray:
    arr = np.asarray(value)
    if arr.size == 0:
        return np.empty((0, 2), dtype=float)
    if arr.ndim == 1:
        if arr.shape[0] == 2:
            return _validate_intervals(arr.reshape(1, 2).astype(float))
        raise ValueError(f"Intervals must have two columns; got shape {arr.shape}")
    if arr.ndim != 2:
        raise ValueError(f"Intervals must be one- or two-dimensional; got shape {arr.shape}")
    if arr.shape[1] != 2 and arr.shape[0] == 2:
        arr = arr.T
    if arr.shape[1] != 2:
        raise ValueError(f"Intervals must have two columns; got shape {arr.shape}")
    return _validate_intervals(arr.astype(float))


def _validate_intervals(intervals: np.ndarray) -> np.ndarray:
    if not np.all(np.isfinite(intervals)):
        raise ValueError("Intervals must contain finite start and end times")
    if np.any(intervals[:, 1] < intervals[:, 0]):
        raise ValueError("Intervals must have end times greater than or equal to start times")
    return intervals


def _times_in_intervals(times: np.ndarray, intervals: np.ndarray) -> np.ndarray:
    mask = np.zeros(times.shape, dtype=bool)
    for start, end in intervals:
        mask |= (times >= start) & (times <= end)
    return mask
