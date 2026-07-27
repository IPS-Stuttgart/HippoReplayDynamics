from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.shuffle_spike_time_order import (
    _positive_integer_grid_dimension,
)


@pytest.mark.parametrize(
    "dimension",
    [
        2**53 + 1,
        np.int64(2**53 + 1),
        np.uint64(2**64 - 1),
    ],
)
def test_shuffle_grid_dimension_preserves_exact_integer_value(dimension) -> None:
    assert _positive_integer_grid_dimension(dimension) == int(dimension)


@pytest.mark.parametrize(
    "dimension",
    [
        "2",
        b"2",
        np.str_("2"),
        np.bytes_(b"2"),
        np.array("2"),
    ],
)
def test_shuffle_grid_dimension_rejects_text_scalars(dimension) -> None:
    with pytest.raises(ValueError, match="grid_shape dimensions"):
        _positive_integer_grid_dimension(dimension)
