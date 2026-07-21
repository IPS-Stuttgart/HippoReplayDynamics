from __future__ import annotations

import numpy as np
import pytest

import hipporeplayimm.state_space_utils as state_space_utils


def test_first_order_imm_diagnostics_are_invariant_to_log_offset() -> None:
    mode_posterior = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.25, 0.75, 0.0],
        ],
        dtype=float,
    )
    trajectory_log_posterior = np.log(
        np.array(
            [
                [0.8, 0.2],
                [0.1, 0.9],
            ],
            dtype=float,
        )
    )
    bin_centers = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=float,
    )

    baseline = state_space_utils._first_order_imm_content_diagnostics(
        mode_posterior,
        trajectory_log_posterior,
        bin_centers,
        0.02,
    )
    shifted = state_space_utils._first_order_imm_content_diagnostics(
        mode_posterior,
        trajectory_log_posterior - 1000.0,
        bin_centers,
        0.02,
    )

    assert shifted.keys() == baseline.keys()
    for key, value in baseline.items():
        if isinstance(value, int):
            assert shifted[key] == value
        else:
            assert shifted[key] == pytest.approx(value)
