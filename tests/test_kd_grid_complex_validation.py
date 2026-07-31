from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.kd_reference import marginalize_grid_log_evidence


@pytest.mark.parametrize(
    ("grid", "prior", "message"),
    [
        (
            np.array([[0.0 + 1.0j, -1.0 + 0.0j]], dtype=complex),
            np.array([0.5, 0.5]),
            "grid must contain real values",
        ),
        (
            np.zeros((1, 2)),
            np.array([0.5 + 0.0j, 0.5 + 1.0j], dtype=complex),
            "prior must contain real values",
        ),
    ],
)
def test_grid_marginalization_rejects_complex_inputs(grid, prior, message):
    with pytest.raises(ValueError, match=message):
        marginalize_grid_log_evidence(grid, prior)
