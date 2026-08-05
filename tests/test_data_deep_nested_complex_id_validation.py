from __future__ import annotations

import warnings

import numpy as np
import pytest

from hipporeplayimm.data import _as_integer_vector


def _deeply_nested_object_scalar(value: object) -> np.ndarray:
    inner = np.empty((), dtype=object)
    inner[()] = value
    outer = np.empty((), dtype=object)
    outer[()] = inner
    values = np.empty(1, dtype=object)
    values[0] = outer
    return values


@pytest.mark.parametrize(
    "value",
    [np.complex128(3.0 + 2.0j), np.complex128(3.0 + 0.0j)],
)
def test_as_integer_vector_rejects_deeply_nested_complex_ids_without_cast_warning(
    value: np.complex128,
) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="real integer"):
            _as_integer_vector(_deeply_nested_object_scalar(value), "neuron IDs")


def test_as_integer_vector_accepts_deeply_nested_real_scalar_ids() -> None:
    loaded = _as_integer_vector(
        _deeply_nested_object_scalar(np.float64(3.0)),
        "neuron IDs",
    )

    np.testing.assert_array_equal(loaded, np.array([3]))


def test_as_integer_vector_rejects_self_referential_scalar_arrays() -> None:
    cyclic = np.empty((), dtype=object)
    cyclic[()] = cyclic
    values = np.empty(1, dtype=object)
    values[0] = cyclic

    with pytest.raises(ValueError, match="scalar integer"):
        _as_integer_vector(values, "neuron IDs")
