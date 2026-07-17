from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.accuracy_upgrades import valid_grid_graph_transition


@pytest.mark.parametrize(
    ("grid_shape", "stay_probability", "message"),
    [
        (
            (10**400, 2),
            0.0,
            "grid_shape must contain positive integer dimensions",
        ),
        (
            (2, 2),
            10**400,
            r"stay_probability must lie in \[0, 1\)",
        ),
    ],
)
def test_valid_grid_graph_transition_normalizes_numeric_overflow(
    grid_shape: tuple[object, object],
    stay_probability: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        valid_grid_graph_transition(
            grid_shape,
            np.ones(4, dtype=bool),
            stay_probability=stay_probability,
        )
