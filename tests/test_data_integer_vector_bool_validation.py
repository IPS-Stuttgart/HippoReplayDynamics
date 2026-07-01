import numpy as np
import pytest

from hipporeplayimm.data import _as_integer_vector


def test_as_integer_vector_rejects_boolean_dtype_ids():
    with pytest.raises(ValueError, match="boolean"):
        _as_integer_vector(np.array([True, False]), "neuron IDs")


def test_as_integer_vector_rejects_object_wrapped_boolean_ids():
    with pytest.raises(ValueError, match="boolean"):
        _as_integer_vector(np.array([1, True], dtype=object), "neuron IDs")


def test_as_integer_vector_still_accepts_integral_float_ids():
    ids = _as_integer_vector(np.array([1.0, 2.0]), "neuron IDs")

    np.testing.assert_array_equal(ids, np.array([1, 2]))
