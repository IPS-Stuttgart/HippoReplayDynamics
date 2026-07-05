from pathlib import Path

import numpy as np
import pytest
import scipy.io as sio

from hipporeplayimm.data import _load_spike_marks, _mark_group_ids_from_tetrode_cell_ids, load_mat_variable, load_replay_session


def test_load_mat_v5_variable(tmp_path: Path):
    path = tmp_path / "Position_Data.mat"
    expected = np.array([[1.0, 2.0, 3.0, 4.0]])
    sio.savemat(path, {"Position_Data": expected})

    loaded = load_mat_variable(path, "Position_Data")

    assert np.allclose(loaded, expected)


def test_load_mat_v73_hdf5_variable(tmp_path: Path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "Spike_Data.mat"
    expected = np.array([[1.0, 11.0], [2.0, 12.0], [3.0, 13.0]])
    with h5py.File(path, "w") as handle:
        handle.create_dataset("Spike_Data", data=expected.T)

    loaded = load_mat_variable(path, "Spike_Data")

    assert np.allclose(loaded, expected)


def test_load_mat_v73_hdf5_variable_after_loadmat_oserror(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "Spike_Data.mat"
    expected = np.array([[1.0, 11.0], [2.0, 12.0], [3.0, 13.0]])
    with h5py.File(path, "w") as handle:
        handle.create_dataset("Spike_Data", data=expected.T)

    def raise_hdf5_load_error(*_args, **_kwargs):
        raise OSError("MATLAB v7.3/HDF5 files require an HDF5 reader")

    monkeypatch.setattr(sio, "loadmat", raise_hdf5_load_error)

    loaded = load_mat_variable(path, "Spike_Data")

    assert np.allclose(loaded, expected)


def test_single_tetrode_cell_id_row_maps_all_matching_spikes_to_the_tetrode() -> None:
    group_ids = _mark_group_ids_from_tetrode_cell_ids(
        np.array([42, 42], dtype=int),
        np.array([7, 42], dtype=int),
    )

    assert group_ids is not None
    np.testing.assert_array_equal(group_ids, np.array([7, 7], dtype=int))


def test_tetrode_cell_id_mapping_ignores_unused_extra_columns() -> None:
    group_ids = _mark_group_ids_from_tetrode_cell_ids(
        np.array([11, 12, 13], dtype=int),
        np.array(
            [
                [7.0, 11.0, 0.25],
                [8.0, 12.0, 0.50],
                [9.0, 13.0, 0.75],
            ]
        ),
    )

    assert group_ids is not None
    np.testing.assert_array_equal(group_ids, np.array([7, 8, 9], dtype=int))


def test_tetrode_cell_id_mapping_rejects_fractional_used_columns() -> None:
    with pytest.raises(ValueError, match="tetrode/cell IDs"):
        _mark_group_ids_from_tetrode_cell_ids(
            np.array([11, 12, 13], dtype=int),
            np.array(
                [
                    [7.0, 11.0, 0.0],
                    [8.5, 12.0, 1.0],
                    [9.0, 13.0, 2.0],
                ]
            ),
        )


def _write_minimal_session(path: Path, *, mark_variable: str | None = None) -> None:
    path.mkdir(parents=True)
    spikes = np.array([[1.0, 11.0], [2.0, 12.0], [3.0, 11.0]])
    spike_data = {
        "Spike_Data": spikes,
        "Tetrode_Cell_IDs": np.array([[1, 11], [1, 12]]),
        "Excitatory_Neurons": np.array([11, 12]),
        "Inhibitory_Neurons": np.array([]),
    }
    if mark_variable is not None:
        spike_data[mark_variable] = np.array(
            [
                [1.0, 40.0, 42.0, 43.0, 44.0],
                [2.0, 50.0, 52.0, 53.0, 54.0],
                [3.0, 60.0, 62.0, 63.0, 64.0],
            ]
        )
    sio.savemat(path / "Spike_Data.mat", spike_data)
    sio.savemat(path / "Position_Data.mat", {"Position_Data": np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])})
    sio.savemat(path / "Ripple_Events.mat", {"Ripple_Events": np.array([[0.1, 0.2, 0.15, 1.0, 2.0, 3.0]])})
    sio.savemat(path / "Epochs.mat", {"Run_Times": np.array([[0.0, 1.0]])})


def test_spike_mark_loader_prefers_embedded_marks_without_loading_optional_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "Amplitude_Marks.mat").touch()
    spikes = np.array([[1.0, 11.0], [2.0, 12.0], [3.0, 11.0]])
    embedded_marks = np.array(
        [
            [40.0, 42.0],
            [50.0, 52.0],
            [60.0, 62.0],
        ]
    )
    spike_data = {
        "Spike_Data": spikes,
        "Tetrode_Cell_IDs": np.array([[1, 11], [1, 12]]),
        "Spike_Amplitude_Marks": embedded_marks,
    }

    def fail_optional_load(path: Path):
        raise AssertionError(f"optional mark file was loaded eagerly: {path}")

    monkeypatch.setattr("hipporeplayimm.data._load_mat_file", fail_optional_load)

    marks = _load_spike_marks(tmp_path, spike_data, spikes)

    assert marks is not None
    assert marks.source_file == "Spike_Data.mat"
    assert marks.source_variable == "Spike_Amplitude_Marks"
    np.testing.assert_allclose(marks.marks, embedded_marks)


def test_load_replay_session_detects_row_aligned_spike_marks(tmp_path: Path):
    session_path = tmp_path / "Rat1" / "Open1"
    _write_minimal_session(session_path, mark_variable="Spike_Amplitude_Marks")

    session = load_replay_session(session_path)

    assert session.has_spike_marks
    assert session.spike_marks is not None
    assert session.spike_marks.source_file == "Spike_Data.mat"
    assert session.spike_marks.source_variable == "Spike_Amplitude_Marks"
    assert session.spike_marks.n_spikes == 3
    assert session.spike_marks.n_features == 4
    assert np.allclose(session.spike_marks.times, np.array([1.0, 2.0, 3.0]))
    assert np.allclose(session.spike_marks.marks[0], np.array([40.0, 42.0, 43.0, 44.0]))
    assert session.metadata["spike_mark_features"] == 4


def test_load_replay_session_ignores_nonreal_complex_spike_marks(tmp_path: Path):
    session_path = tmp_path / "Rat1" / "Open1"
    _write_minimal_session(session_path)
    sio.savemat(
        session_path / "Spike_Data.mat",
        {
            "Spike_Data": np.array([[1.0, 11.0], [2.0, 12.0], [3.0, 11.0]]),
            "Tetrode_Cell_IDs": np.array([[1, 11], [1, 12]]),
            "Excitatory_Neurons": np.array([11, 12]),
            "Inhibitory_Neurons": np.array([]),
            "Spike_Amplitude_Marks": np.array(
                [
                    [1.0 + 1.0j, 40.0 + 0.0j],
                    [2.0 + 0.0j, 50.0 + 0.0j],
                    [3.0 + 0.0j, 60.0 + 0.0j],
                ]
            ),
        },
    )

    session = load_replay_session(session_path)

    assert not session.has_spike_marks
    assert session.spike_marks is None
    assert session.metadata["spike_mark_features"] == 0


def test_load_replay_session_reports_missing_spike_marks(tmp_path: Path):
    session_path = tmp_path / "Rat1" / "Open1"
    _write_minimal_session(session_path)

    session = load_replay_session(session_path)

    assert not session.has_spike_marks
    assert session.spike_marks is None
    assert session.metadata["spike_mark_source"] == ""
    assert session.metadata["spike_mark_features"] == 0


def test_load_replay_session_rejects_fractional_excitatory_neuron_ids(tmp_path: Path):
    session_path = tmp_path / "Rat1" / "Open1"
    _write_minimal_session(session_path)
    sio.savemat(
        session_path / "Spike_Data.mat",
        {
            "Spike_Data": np.array([[1.0, 11.0], [2.0, 12.0]]),
            "Tetrode_Cell_IDs": np.array([[1, 11], [1, 12]]),
            "Excitatory_Neurons": np.array([11.5]),
            "Inhibitory_Neurons": np.array([]),
        },
    )

    with pytest.raises(ValueError, match="excitatory neuron IDs"):
        load_replay_session(session_path)


def test_load_replay_session_rejects_fractional_inhibitory_neuron_ids(tmp_path: Path):
    session_path = tmp_path / "Rat1" / "Open1"
    _write_minimal_session(session_path)
    sio.savemat(
        session_path / "Spike_Data.mat",
        {
            "Spike_Data": np.array([[1.0, 11.0], [2.0, 12.0]]),
            "Tetrode_Cell_IDs": np.array([[1, 11], [1, 12]]),
            "Excitatory_Neurons": np.array([11]),
            "Inhibitory_Neurons": np.array([12.25]),
        },
    )

    with pytest.raises(ValueError, match="inhibitory neuron IDs"):
        load_replay_session(session_path)
