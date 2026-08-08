from __future__ import annotations

import warnings

import numpy as np
import pytest

from hipporeplayimm import state_space_first_order
from hipporeplayimm.state_space_utils import _coerce_valid_bin_mask


def _nested_scalar(value: object) -> np.ndarray:
    inner = np.empty((), dtype=object)
    inner[()] = value
    outer = np.empty((), dtype=object)
    outer[()] = inner
    return outer


def _mask_with_first(value: object) -> np.ndarray:
    mask = np.empty(3, dtype=object)
    mask[0] = value
    mask[1] = 0
    mask[2] = 1
    return mask


@pytest.mark.parametrize(
    "value",
    [
        np.complex128(1.0 + 2.0j),
        _nested_scalar(np.complex128(1.0 + 2.0j)),
    ],
)
def test_valid_bin_mask_rejects_object_backed_complex_values_without_warning(value: object) -> None:
    mask = _mask_with_first(value)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="valid_bin_mask"):
            _coerce_valid_bin_mask(mask, 3)
        with pytest.raises(ValueError, match="valid_bin_mask"):
            state_space_first_order._uniform_probabilities(3, mask)


def test_valid_bin_mask_rejects_nested_non_scalar_arrays() -> None:
    mask = _mask_with_first(np.array([1.0]))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="valid_bin_mask"):
            _coerce_valid_bin_mask(mask, 3)


def test_valid_bin_mask_accepts_nested_real_and_boolean_scalars() -> None:
    mask = np.empty(3, dtype=object)
    mask[0] = _nested_scalar(np.float64(1.0))
    mask[1] = _nested_scalar(np.bool_(False))
    mask[2] = _nested_scalar(np.int64(1))

    expected = np.array([True, False, True], dtype=bool)
    np.testing.assert_array_equal(_coerce_valid_bin_mask(mask, 3), expected)
    np.testing.assert_allclose(
        state_space_first_order._uniform_probabilities(3, mask),
        [0.5, 0.0, 0.5],
    )


def test_valid_bin_mask_rejects_cyclic_object_scalar() -> None:
    cyclic = np.empty((), dtype=object)
    cyclic[()] = cyclic
    mask = _mask_with_first(cyclic)

    with pytest.raises(ValueError, match="valid_bin_mask"):
        _coerce_valid_bin_mask(mask, 3)
