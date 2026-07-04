from __future__ import annotations

import numpy as np
import pytest

import hipporeplayimm.state_space  # noqa: F401 - importing the public surface applies runtime validation patches
from hipporeplayimm.state_space_sparse_momentum import _coerce_valid_bin_mask


def test_sparse_momentum_valid_bin_mask_accepts_boolean_and_binary_numeric_values() -> None:
    np.testing.assert_array_equal(
        _coerce_valid_bin_mask(np.array([True, False, True]), 3),
        np.array([True, False, True]),
    )
    np.testing.assert_array_equal(
        _coerce_valid_bin_mask(np.array([1, 0, 1], dtype=int), 3),
        np.array([True, False, True]),
    )


@pytest.mark.parametrize(
    ("mask", "message"),
    [
        (np.array([1.0, 0.5, 0.0], dtype=float), "boolean or 0/1"),
        (np.array([1.0, np.nan, 0.0], dtype=float), "finite"),
        (np.array(["1", "0", "1"], dtype=str), "boolean or 0/1"),
        (np.array([1.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j]), "boolean or 0/1"),
    ],
)
def test_sparse_momentum_valid_bin_mask_rejects_silent_bool_coercions(mask: np.ndarray, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _coerce_valid_bin_mask(mask, 3)
