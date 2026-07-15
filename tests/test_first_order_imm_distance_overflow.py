import numpy as np
import pytest

import hipporeplayimm.state_space_utils as state_space_utils


def test_first_order_imm_content_diagnostics_preserves_large_finite_distances() -> None:
    mode_posterior = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    trajectory_log_posterior = np.array(
        [
            [0.0, -np.inf],
            [-np.inf, 0.0],
        ],
        dtype=float,
    )
    bin_centers = np.array(
        [
            [0.0, 0.0],
            [1.0e200, 1.0e200],
        ],
        dtype=float,
    )

    with np.errstate(over="raise", invalid="raise"):
        diagnostics = state_space_utils._first_order_imm_content_diagnostics(
            mode_posterior,
            trajectory_log_posterior,
            bin_centers,
            0.5,
        )

    expected_distance = np.hypot(1.0e200, 1.0e200)
    assert np.isfinite(
        diagnostics["state_space_imm_posterior_expected_path_length_cm"]
    )
    assert diagnostics[
        "state_space_imm_posterior_expected_path_length_cm"
    ] == pytest.approx(expected_distance)
    assert diagnostics["state_space_imm_posterior_net_displacement_cm"] == pytest.approx(
        expected_distance
    )
    assert diagnostics["state_space_imm_posterior_path_speed_cm_s"] == pytest.approx(
        expected_distance / 0.5
    )
