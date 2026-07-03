import numpy as np
import pytest

from hipporeplayimm.data import _as_integer_vector


def test_data_loader_integer_vector_preserves_large_native_integer_ids() -> None:
    ids = np.array([2**53 + 1, 2**53 + 3], dtype=np.int64)
    if np.iinfo(np.dtype(int)).max < int(ids[-1]):
        pytest.skip("requires platform integers wide enough for large native IDs")

    loaded = _as_integer_vector(ids, "neuron IDs")

    np.testing.assert_array_equal(loaded, ids)
