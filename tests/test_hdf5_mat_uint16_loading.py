from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hipporeplayimm.data import load_mat_variable


def test_load_mat_v73_preserves_numeric_uint16_arrays(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "Spike_Data.mat"
    expected = np.array([[1, 11], [2, 12]], dtype=np.uint16)

    with h5py.File(path, "w") as handle:
        numeric = handle.create_dataset("Tetrode_Cell_IDs", data=expected.T)
        numeric.attrs["MATLAB_class"] = np.bytes_("uint16")
        label = handle.create_dataset("Session_Label", data=np.array([ord(char) for char in "Open1"], dtype=np.uint16))
        label.attrs["MATLAB_class"] = np.bytes_("char")

    loaded = load_mat_variable(path, "Tetrode_Cell_IDs")

    assert isinstance(loaded, np.ndarray)
    np.testing.assert_array_equal(loaded, expected)
    assert load_mat_variable(path, "Session_Label") == "Open1"
