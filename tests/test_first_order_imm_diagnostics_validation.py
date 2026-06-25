import numpy as np
import pytest

import hipporeplayimm.duration_occupancy as duration_occupancy
import hipporeplayimm.state_space_utils as state_space_utils
from hipporeplayimm import apply_runtime_patches


def _mode_posterior() -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.25, 0.75, 0.0],
        ],
        dtype=float,
    )


def _trajectory_log_posterior() -> np.ndarray:
    return np.log(
        np.array(
            [
                [0.8, 0.2],
                [0.1, 0.9],
            ],
            dtype=float,
        )
    )


def _deterministic_step_log_posterior() -> np.ndarray:
    return np.array([[0.0, -np.inf], [-np.inf, 0.0]], dtype=float)


def _bin_centers() -> np.ndarray:
    return np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)


def test_first_order_imm_content_diagnostics_rejects_nan_mode_posterior() -> None:
    mode_posterior = _mode_posterior()
    mode_posterior[1, 1] = np.nan

    with pytest.raises(ValueError, match="mode posterior must be finite"):
        state_space_utils._first_order_imm_content_diagnostics(
            mode_posterior,
            _trajectory_log_posterior(),
            _bin_centers(),
            0.02,
        )


@pytest.mark.parametrize("bad_value", [np.nan, np.inf])
def test_first_order_imm_content_diagnostics_rejects_bad_trajectory_values(bad_value: float) -> None:
    trajectory = _trajectory_log_posterior()
    trajectory[0, 1] = bad_value

    with pytest.raises(ValueError, match="trajectory posterior"):
        state_space_utils._first_order_imm_content_diagnostics(
            _mode_posterior(),
            trajectory,
            _bin_centers(),
            0.02,
        )


def test_first_order_imm_content_diagnostics_allows_impossible_log_bins() -> None:
    trajectory = _deterministic_step_log_posterior()

    diagnostics = state_space_utils._first_order_imm_content_diagnostics(
        _mode_posterior(),
        trajectory,
        _bin_centers(),
        0.02,
    )

    assert np.isfinite(diagnostics["state_space_imm_posterior_path_speed_cm_s"])
    assert diagnostics["state_space_imm_posterior_expected_path_length_cm"] == pytest.approx(1.0)


def test_runtime_patch_repairs_duration_occupancy_diagnostics_alias(monkeypatch) -> None:
    original_helper = getattr(
        state_space_utils._first_order_imm_content_diagnostics,
        "_first_order_imm_diagnostics_validation_original",
    )
    monkeypatch.setattr(
        duration_occupancy,
        "_first_order_imm_content_diagnostics",
        original_helper,
    )

    apply_runtime_patches()

    helper = duration_occupancy._first_order_imm_content_diagnostics
    assert helper is not original_helper
    assert getattr(helper, "_first_order_imm_duration_diagnostics_alias_patch", False)


def test_duration_occupancy_alias_uses_caller_transition_durations() -> None:
    apply_runtime_patches()
    helper = duration_occupancy._first_order_imm_content_diagnostics

    def call_like_duration_occupancy_scorer() -> dict[str, float | int]:
        durations = np.array([0.5], dtype=float)
        return helper(
            _mode_posterior(),
            _deterministic_step_log_posterior(),
            _bin_centers(),
            0.02,
        )

    diagnostics = call_like_duration_occupancy_scorer()

    assert diagnostics["state_space_imm_posterior_expected_path_length_cm"] == pytest.approx(1.0)
    assert diagnostics["state_space_imm_posterior_path_speed_cm_s"] == pytest.approx(2.0)
