"""Dataset loading for Pfeiffer/Foster replay sessions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io as sio


@dataclass(frozen=True)
class RippleEvent:
    """One detected ripple event."""

    start: float
    end: float
    peak: float
    raw_power: float
    z_power_session: float
    z_power_epoch: float

    @classmethod
    def from_row(cls, row: np.ndarray) -> "RippleEvent":
        return cls(*(float(x) for x in row[:6]))


@dataclass
class ReplaySession:
    """Loaded open-field session with raw arrays kept in dataset units."""

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

    def excitatory_spikes(self) -> np.ndarray:
        """Return spikes from putative excitatory neurons."""

        if self.spikes.size == 0 or self.excitatory_neurons.size == 0:
            return np.empty((0, 2), dtype=float)
        keep = np.isin(self.spikes[:, 1].astype(int), self.excitatory_neurons.astype(int))
        return self.spikes[keep]

    def ripple(self, index: int) -> RippleEvent:
        return RippleEvent.from_row(self.ripple_events[index])

    def ripple_indices_in_run(self) -> np.ndarray:
        """Return ripple indices whose peak lies inside any run interval."""

        if self.ripple_events.size == 0 or self.run_times.size == 0:
            return np.array([], dtype=int)
        peaks = self.ripple_events[:, 2]
        return np.flatnonzero(_times_in_intervals(peaks, _as_intervals(self.run_times)))


def load_open_field_sessions(root: str | Path, include_rats: tuple[int, ...] = (1, 2, 3, 4)) -> list[ReplaySession]:
    """Load all open-field sessions for Rat1-4 by default."""

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
    """Load one open-field replay session directory."""

    path = Path(session_path)
    rat = path.parent.name
    name = path.name
    position = _required_array(path / "Position_Data.mat", "Position_Data")
    ripple_events = _required_array(path / "Ripple_Events.mat", "Ripple_Events")
    spike_data = _load_mat_file(path / "Spike_Data.mat")
    epochs = _load_mat_file(path / "Epochs.mat")
    metadata = _load_optional_metadata(path / "Experiment_Information.mat")
    well_sequence = None
    if (path / "Well_Sequence.mat").exists():
        well_sequence = _required_array(path / "Well_Sequence.mat", "Well_Sequence")

    return ReplaySession(
        rat=rat,
        name=name,
        path=path,
        position=np.asarray(position, dtype=float),
        spikes=_as_two_column_array(spike_data.get("Spike_Data", np.empty((0, 2))), "Spike_Data"),
        tetrode_cell_ids=np.asarray(spike_data.get("Tetrode_Cell_IDs", np.empty((0, 2)))),
        excitatory_neurons=np.asarray(spike_data.get("Excitatory_Neurons", np.array([])), dtype=int).reshape(-1),
        inhibitory_neurons=np.asarray(spike_data.get("Inhibitory_Neurons", np.array([])), dtype=int).reshape(-1),
        ripple_events=_as_two_dimensional(ripple_events, "Ripple_Events"),
        run_times=_as_intervals(epochs.get("Run_Times", np.empty((0, 2)))),
        sleep_box_immobile_times=_as_intervals(epochs.get("Sleep_Box_Immobile_Times", np.empty((0, 2)))),
        sleep_times=_as_intervals(epochs.get("Sleep_Times", np.empty((0, 2)))),
        rem_times=_as_intervals(epochs.get("REM_Times", np.empty((0, 2)))),
        well_sequence=None if well_sequence is None else _as_two_dimensional(well_sequence, "Well_Sequence"),
        metadata=metadata,
    )


def load_mat_variable(path: str | Path, variable_name: str) -> Any:
    """Load one variable from a MATLAB v5 or v7.3 file."""

    return _load_mat_file(Path(path))[variable_name]


def _load_mat_file(path: Path) -> dict[str, Any]:
    try:
        loaded = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
        return {key: value for key, value in loaded.items() if not key.startswith("__")}
    except (NotImplementedError, ValueError) as exc:
        if _is_hdf5_file(path):
            return _load_hdf5_mat_file(path)
        raise exc


def _load_hdf5_mat_file(path: Path) -> dict[str, Any]:
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            f"{path} is a MATLAB v7.3/HDF5 file. Install h5py to load it."
        ) from exc

    output: dict[str, Any] = {}
    with h5py.File(path, "r") as handle:
        for key in handle.keys():
            if key == "#refs#":
                continue
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
        if arr.dtype.kind in {"S", "U"}:
            return arr.astype(str)
        if arr.dtype == np.uint16 and arr.ndim >= 1 and arr.size > 0:
            try:
                return "".join(chr(int(x)) for x in arr.reshape(-1) if int(x) != 0)
            except (TypeError, ValueError):
                pass
        if arr.ndim >= 2:
            arr = arr.T
        return np.squeeze(arr)
    if isinstance(obj, h5py.Group):
        return {key: _read_hdf5_value(value) for key, value in obj.items()}
    raise TypeError(f"Unsupported HDF5 MATLAB object: {type(obj)!r}")


def _required_array(path: Path, variable_name: str) -> np.ndarray:
    data = _load_mat_file(path)
    if variable_name not in data:
        available = ", ".join(sorted(data))
        raise KeyError(f"{path} does not contain {variable_name!r}; available: {available}")
    return np.asarray(data[variable_name])


def _load_optional_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return _load_mat_file(path)


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
        return np.empty((0, 2), dtype=float)
    if arr.ndim == 1:
        if name == "Ripple_Events" and arr.shape[0] == 6:
            return arr.reshape(1, 6)
        if arr.shape[0] == 2:
            return arr.reshape(1, 2)
        return arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be one- or two-dimensional; got shape {arr.shape}")
    return arr


def _as_intervals(value: Any) -> np.ndarray:
    arr = np.asarray(value)
    if arr.size == 0:
        return np.empty((0, 2), dtype=float)
    if arr.ndim == 1:
        if arr.shape[0] == 2:
            return arr.reshape(1, 2).astype(float)
        raise ValueError(f"Intervals must have two columns; got shape {arr.shape}")
    if arr.shape[1] != 2 and arr.shape[0] == 2:
        arr = arr.T
    if arr.shape[1] != 2:
        raise ValueError(f"Intervals must have two columns; got shape {arr.shape}")
    return arr.astype(float)


def _times_in_intervals(times: np.ndarray, intervals: np.ndarray) -> np.ndarray:
    mask = np.zeros(times.shape, dtype=bool)
    for start, end in intervals:
        mask |= (times >= start) & (times <= end)
    return mask
