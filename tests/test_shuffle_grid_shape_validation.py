from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.shuffle_controls import _spatial_roll_rates


@pytest.mark.parametrize(
    "grid_shape",
    [
        (1.5, 2),
        (2, 1.5),
        (True, 2),
        (2, np.bool_(False)),
        (float("nan"), 2),
        ("2", 2),
        ([2], 2),
    ],
)
def test_spatial_roll_rejects_non_integral_grid_dimensions(grid_shape: object) -> None:
    rates = np.arange(4.0, dtype=float).reshape(1, 4)

    with pytest.raises(ValueError, match="grid_shape"):
        _spatial_roll_rates(rates, grid_shape, np.random.default_rng(1))  # type: ignore[arg-type]


def test_spatial_roll_accepts_numpy_integer_grid_dimensions() -> None:
    rates = np.arange(4.0, dtype=float).reshape(1, 4)

    rolled = _spatial_roll_rates(rates, (np.int64(2), np.int64(2)), np.random.default_rng(1))

    assert rolled.shape == rates.shape
