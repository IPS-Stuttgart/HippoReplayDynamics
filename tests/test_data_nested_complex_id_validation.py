import numpy as np
import pytest

from hipporeplayimm.data import _as_integer_vector


def _nested_object_scalar(value: object) -> np.ndarray:
    values = np.empty(1, dtype=object)
    values[0] = np.asarray(value, dtype=object)
    return values


@pytest.mark.parametrize("value", [np.complex128(3.0 + 2.0j), np.complex128(3.0 + 0.0j)])
def test_as_integer_vector_rejects_nested_numpy_complex_ids(value: np.complex128) -> None:
    with pytest.raises(ValueError, match="real integer"):
        _as_integer_vector(_nested_object_scalar(value), "neuron IDs")


def test_as_integer_vector_still_accepts_nested_real_numpy_scalar_ids() -> None:
    loaded = _as_integer_vector(_nested_object_scalar(np.float64(3.0)), "neuron IDs")

    np.testing.assert_array_equal(loaded, np.array([3]))
