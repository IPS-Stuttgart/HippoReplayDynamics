from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import hipporeplayimm
from hipporeplayimm import state_space_trajectory_imm as trajectory_imm
from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.trajectory_imm_single_bin_diagnostics import (
    _duration_weighted_mode_summary,
    _evidence_only_mode_diagnostics,
    apply_trajectory_imm_single_bin_diagnostics_patch,
)


def _partial_bin_emissions() -> LogEmissionTensor:
    return LogEmissionTensor(
        log_likelihood=np.zeros((2, 2), dtype=float),
        spike_counts=np.zeros((2, 1), dtype=int),
        times=np.array([0.0, 0.01], dtype=float),
        dt=0.01,
        cell_ids=np.array([1], dtype=int),
        n_spikes=0,
        bin_durations=np.array([0.020, 0.005], dtype=float),
        transition_durations=np.array([0.010], dtype=float),
    )


def _trajectory_imm_config(*, momentum_initial_probability: float = 1.0) -> SimpleNamespace:
    return SimpleNamespace(
        stationary_sigma_cm=1.0,
        diffusion_sigma_cm_sqrt_s=1.0,
        momentum_sigma_cm_sqrt_s=1.0,
        momentum_initial_sigma_cm_sqrt_s=1.0,
        momentum_velocity_decay=0.0,
        momentum_velocity_decay_tau_s=0.0,
        max_step_sigma=4.0,
        imm_switch_tau_s=0.0,
        trajectory_imm_mode_stickiness=0.0,
        trajectory_imm_momentum_initial_probability=momentum_initial_probability,
        trajectory_imm_momentum_switch_probability=None,
    )


def test_trajectory_imm_mode_summary_respects_partial_final_bin() -> None:
    mode_posterior = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ]
    )

    summary = _duration_weighted_mode_summary(
        mode_posterior,
        np.asarray([0.020, 0.005]),
    )

    np.testing.assert_allclose(
        summary["event_probability"],
        [0.8, 0.2, 0.0, 0.0],
    )
    np.testing.assert_allclose(summary["mean_mode_entropy"], 0.0)


def test_smoothed_trajectory_imm_diagnostics_use_bin_exposure() -> None:
    hipporeplayimm.apply_runtime_patches()
    apply_trajectory_imm_single_bin_diagnostics_patch()

    emissions = _partial_bin_emissions()
    centers = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)
    _, trajectory, _, mode_posterior, diagnostics = (
        trajectory_imm._score_trajectory_imm_exact_sparse(
            emissions,
            centers,
            _trajectory_imm_config(),
            emissions.transition_durations,
            return_trajectory=True,
        )
    )

    assert trajectory is not None
    assert mode_posterior is not None
    weights = emissions.bin_durations / emissions.bin_durations.sum()
    expected_event = weights @ mode_posterior
    row_mean = np.mean(mode_posterior, axis=0)
    assert not np.allclose(expected_event, row_mean)
    for mode_index, mode_name in enumerate(trajectory_imm._TRAJECTORY_IMM_MODES):
        key = mode_name.replace("-", "_")
        np.testing.assert_allclose(
            diagnostics[f"state_space_mode_{key}_event_probability"],
            expected_event[mode_index],
        )
    np.testing.assert_allclose(
        diagnostics["state_space_trajectory_family_event_probability"],
        expected_event[1:].sum(),
    )
    assert diagnostics["state_space_trajectory_imm_time_weighting"] == "bin_duration"


def test_evidence_only_trajectory_imm_diagnostics_use_bin_exposure() -> None:
    hipporeplayimm.apply_runtime_patches()
    apply_trajectory_imm_single_bin_diagnostics_patch()

    emissions = _partial_bin_emissions()
    centers = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)
    _, trajectory, _, mode_posterior, diagnostics = (
        trajectory_imm._score_trajectory_imm_exact_sparse(
            emissions,
            centers,
            _trajectory_imm_config(),
            emissions.transition_durations,
            return_trajectory=False,
        )
    )

    assert trajectory is None
    assert mode_posterior is None
    assert diagnostics["state_space_trajectory_imm_mode_posterior"] == (
        "filtered_evidence_only_state"
    )
    np.testing.assert_allclose(
        diagnostics["state_space_mode_stationary_event_probability"],
        1.0 / 15.0,
    )
    np.testing.assert_allclose(
        diagnostics["state_space_mode_diffusion_event_probability"],
        1.0 / 15.0,
    )
    np.testing.assert_allclose(
        diagnostics["state_space_mode_fragmented_event_probability"],
        1.0 / 15.0,
    )
    np.testing.assert_allclose(
        diagnostics["state_space_mode_momentum_exact_sparse_event_probability"],
        4.0 / 5.0,
    )
    np.testing.assert_allclose(
        diagnostics["state_space_trajectory_family_event_probability"],
        14.0 / 15.0,
    )
    assert diagnostics["state_space_trajectory_imm_time_weighting"] == "bin_duration"


def test_evidence_only_history_is_used_without_momentum_mass() -> None:
    emissions = _partial_bin_emissions()
    forward_state = SimpleNamespace(
        first_order=np.asarray(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 0.0],
            ],
            dtype=float,
        ),
        momentum_alpha=np.empty(0, dtype=float),
    )

    diagnostics = _evidence_only_mode_diagnostics(
        trajectory_imm,
        _trajectory_imm_config(momentum_initial_probability=0.0),
        emissions,
        [forward_state],
        {
            "state_space_trajectory_family_terminal_probability": -1.0,
            "state_space_trajectory_family_event_probability": -1.0,
        },
    )

    np.testing.assert_allclose(
        diagnostics["state_space_mode_diffusion_terminal_probability"],
        1.0,
    )
    np.testing.assert_allclose(
        diagnostics["state_space_mode_diffusion_event_probability"],
        7.0 / 15.0,
    )
    assert diagnostics["state_space_mode_diffusion_event_probability"] != (
        diagnostics["state_space_mode_diffusion_terminal_probability"]
    )
    assert diagnostics["state_space_trajectory_imm_mode_posterior"] == (
        "filtered_evidence_only_state"
    )
