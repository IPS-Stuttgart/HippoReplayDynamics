from __future__ import annotations

import contextvars

import numpy as np

from hipporeplayimm.first_order_imm_diagnostics_validation import (
    _DIAGNOSTIC_TRANSITION_DURATIONS,
    _compute_first_order_imm_content_diagnostics,
    _wrap_duration_occupancy_alias,
)


def _diagnostic_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mode_posterior = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    trajectory_log_posterior = np.array(
        [
            [0.0, -1.0e300],
            [-1.0e300, 0.0],
            [-1.0e300, 0.0],
        ],
        dtype=float,
    )
    bin_centers = np.array([[0.0], [10.0]], dtype=float)
    return mode_posterior, trajectory_log_posterior, bin_centers


def _path_speed_for_durations(durations: list[float]) -> float:
    alias = _wrap_duration_occupancy_alias(_compute_first_order_imm_content_diagnostics)
    _DIAGNOSTIC_TRANSITION_DURATIONS.set(np.asarray(durations, dtype=float))
    mode_posterior, trajectory_log_posterior, bin_centers = _diagnostic_inputs()
    diagnostics = alias(
        mode_posterior,
        trajectory_log_posterior,
        bin_centers,
        0.02,
    )
    return float(diagnostics["state_space_imm_posterior_path_speed_cm_s"])


def test_first_order_imm_duration_diagnostics_are_context_local() -> None:
    short_context = contextvars.Context()
    long_context = contextvars.Context()

    short_speed = short_context.run(_path_speed_for_durations, [0.02, 0.02])
    long_speed = long_context.run(_path_speed_for_durations, [0.02, 0.08])

    assert np.isclose(short_speed, 250.0)
    assert np.isclose(long_speed, 100.0)


def test_first_order_imm_duration_diagnostics_consume_recorded_durations_once() -> None:
    alias = _wrap_duration_occupancy_alias(_compute_first_order_imm_content_diagnostics)
    mode_posterior, trajectory_log_posterior, bin_centers = _diagnostic_inputs()

    _DIAGNOSTIC_TRANSITION_DURATIONS.set(np.array([0.02, 0.08], dtype=float))
    duration_aware = alias(
        mode_posterior,
        trajectory_log_posterior,
        bin_centers,
        0.02,
    )
    fallback = alias(
        mode_posterior,
        trajectory_log_posterior,
        bin_centers,
        0.02,
    )

    assert np.isclose(duration_aware["state_space_imm_posterior_path_speed_cm_s"], 100.0)
    assert np.isclose(fallback["state_space_imm_posterior_path_speed_cm_s"], 250.0)
