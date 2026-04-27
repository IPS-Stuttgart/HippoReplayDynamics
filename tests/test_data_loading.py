from pathlib import Path

import numpy as np
import pytest
import scipy.io as sio

from hipporeplayimm.data import load_mat_variable


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
