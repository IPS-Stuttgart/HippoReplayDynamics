from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.shuffle_controls import _spatial_roll_rates


@pytest.mark.parametrize(
    "grid_shape",
    [
        ("2", 2),
        (b"2", 2),
        (np.str_("2"), 2),
        (np.bytes_(b"2"), 2),
        (np.array("2"), 2),
    ],
)
def test_spatial_roll_rejects_text_grid_shape_dimensions(grid_shape) -> None:
    rates = np.arange(4.0, dtype=float).reshape(1, 4)

    with pytest.raises(ValueError, match="grid_shape dimensions"):
        _spatial_roll_rates(rates, grid_shape, np.random.default_rng(1))  # type: ignore[arg-type]
