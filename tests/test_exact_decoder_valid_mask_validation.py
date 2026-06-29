from __future__ import annotations

import numpy as np
import pytest

import hipporeplayimm
from hipporeplayimm.state_space_displacement_momentum import (
    _coerce_valid_bin_mask as _coerce_displacement_valid_bin_mask,
)
from hipporeplayimm.state_space_sparse_momentum import (
    _coerce_valid_bin_mask as _coerce_sparse_valid_bin_mask,
)


@pytest.mark.parametrize(
    "coerce_mask",
    [
        _coerce_displacement_valid_bin_mask,
        _coerce_sparse_valid_bin_mask,
    ],
)
def test_exact_decoder_valid_masks_reject_nonbinary_numeric_values(coerce_mask) -> None:
    hipporeplayimm.apply_runtime_patches()

    with pytest.raises(ValueError, match="boolean or 0/1"):
        coerce_mask(np.array([2, 0], dtype=int), 2)

    with pytest.raises(ValueError, match="finite"):
        coerce_mask(np.array([1.0, np.nan], dtype=float), 2)

    np.testing.assert_array_equal(
        coerce_mask(np.array([1, 0], dtype=int), 2),
        np.array([True, False]),
    )
