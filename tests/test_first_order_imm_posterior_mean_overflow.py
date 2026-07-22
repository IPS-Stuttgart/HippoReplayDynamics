from __future__ import annotations

import numpy as np
import pytest

import hipporeplayimm.state_space_utils as state_space_utils


def test_first_order_imm_posterior_mean_preserves_extreme_convex_hull() -> None:
    max_float = np.finfo(float).max
    diagnostics = state_space_utils._first_order_imm_content_diagnostics(
        np.array(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=float,
        ),
        np.zeros((2, 7), dtype=float),
        np.full((7, 1), max_float, dtype=float),
        0.02,
    )

    assert diagnostics["state_space_imm_posterior_expected_path_length_cm"] == pytest.approx(0.0)
    assert diagnostics["state_space_imm_posterior_net_displacement_cm"] == pytest.approx(0.0)
    assert diagnostics["state_space_imm_posterior_path_speed_cm_s"] == pytest.approx(0.0)
    assert all(np.isfinite(float(value)) for value in diagnostics.values())
