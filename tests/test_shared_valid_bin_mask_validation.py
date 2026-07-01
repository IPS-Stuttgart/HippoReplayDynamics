from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.state_space_utils import _coerce_valid_bin_mask


def test_shared_valid_bin_mask_rejects_nonbinary_numeric_values() -> None:
    with pytest.raises(ValueError, match="boolean or 0/1"):
        _coerce_valid_bin_mask(np.array([2, 0], dtype=int), 2)

    with pytest.raises(ValueError, match="finite"):
        _coerce_valid_bin_mask(np.array([1.0, np.nan], dtype=float), 2)

    np.testing.assert_array_equal(
        _coerce_valid_bin_mask(np.array([1, 0], dtype=int), 2),
        np.array([True, False]),
    )


def test_shared_valid_bin_mask_rejects_textual_values() -> None:
    for mask in (
        np.array(["1", "0"], dtype=str),
        np.array([b"1", b"0"], dtype="S1"),
        np.array(["1", "0"], dtype=object),
    ):
        with pytest.raises(ValueError, match="boolean or 0/1"):
            _coerce_valid_bin_mask(mask, 2)


def test_shared_valid_bin_mask_rejects_complex_values() -> None:
    with pytest.raises(ValueError, match="boolean or 0/1"):
        _coerce_valid_bin_mask(np.array([1 + 1j, 0 + 0j]), 2)
