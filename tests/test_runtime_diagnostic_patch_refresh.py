from __future__ import annotations

import importlib
from types import SimpleNamespace

import numpy as np

import hipporeplayimm
from hipporeplayimm import duration_occupancy
from hipporeplayimm import state_space
from hipporeplayimm import state_space_trajectory_imm as trajectory_imm
from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.state_space import StateSpaceDecoderConfig, StateSpaceReplayModel


def _two_bin_emissions(*, bin_durations: np.ndarray | None = None) -> LogEmissionTensor:
    return LogEmissionTensor(
        log_likelihood=np.zeros((2, 2), dtype=float),
        spike_counts=np.zeros((2, 1), dtype=int),
        times=np.array([0.0, 0.02], dtype=float),
        dt=0.02,
        cell_ids=np.array([1], dtype=int),
        n_spikes=0,
        bin_durations=bin_durations,
        transition_durations=np.array([0.02], dtype=float),
    )


def test_runtime_refresh_restores_first_order_imm_time_weighting(monkeypatch) -> None:
    importlib.reload(duration_occupancy)
    assert not getattr(
        duration_occupancy._score_state_space_duration_with_occupancy,
        "_first_order_imm_time_weighting_aware",
        False,
    )

    hipporeplayimm.apply_runtime_patches()
    assert getattr(
        duration_occupancy._score_state_space_duration_with_occupancy,
        "_first_order_imm_time_weighting_aware",
        False,
    )

    mode_posterior = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    trajectory = np.log(
        np.array(
            [
                [0.75, 0.25],
                [0.25, 0.75],
            ],
            dtype=float,
        )
    )

    def fake_score_first_order_imm_variable(*args, **kwargs):
        return 0.0, trajectory, mode_posterior

    monkeypatch.setattr(
        duration_occupancy,
        "_score_first_order_imm_variable",
        fake_score_first_order_imm_variable,
    )
    result = StateSpaceReplayModel(
        mode="first-order-imm",
        config=StateSpaceDecoderConfig(mode="first-order-imm"),
    ).score(
        _two_bin_emissions(bin_durations=np.array([0.02, 0.005], dtype=float)),
        np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float),
    )

    assert result.diagnostics["state_space_imm_time_weighting"] == "bin_duration"
    np.testing.assert_allclose(
        result.diagnostics["state_space_mode_stationary_event_probability"],
        0.8,
    )
    np.testing.assert_allclose(
        result.diagnostics["state_space_mode_diffusion_event_probability"],
        0.2,
    )


def test_runtime_refresh_restores_trajectory_imm_evidence_only_history() -> None:
    importlib.reload(trajectory_imm)
    assert not getattr(
        trajectory_imm._score_trajectory_imm_exact_sparse,
        "_trajectory_imm_single_bin_diagnostics_patch_applied",
        False,
    )
    assert not getattr(
        trajectory_imm._advance_state,
        "_trajectory_imm_evidence_only_advance_recording_patch",
        False,
    )

    hipporeplayimm.apply_runtime_patches()
    assert getattr(
        trajectory_imm._score_trajectory_imm_exact_sparse,
        "_trajectory_imm_single_bin_diagnostics_patch_applied",
        False,
    )
    assert getattr(
        trajectory_imm._advance_state,
        "_trajectory_imm_evidence_only_advance_recording_patch",
        False,
    )
    assert (
        state_space._score_trajectory_imm_exact_sparse
        is trajectory_imm._score_trajectory_imm_exact_sparse
    )

    config = SimpleNamespace(
        stationary_sigma_cm=1.0,
        diffusion_sigma_cm_sqrt_s=1.0,
        momentum_sigma_cm_sqrt_s=1.0,
        momentum_initial_sigma_cm_sqrt_s=1.0,
        momentum_velocity_decay=0.0,
        momentum_velocity_decay_tau_s=0.0,
        max_step_sigma=4.0,
        imm_switch_tau_s=0.0,
        trajectory_imm_mode_stickiness=0.0,
        trajectory_imm_momentum_initial_probability=1.0,
        trajectory_imm_momentum_switch_probability=None,
    )
    emissions = _two_bin_emissions()
    _, trajectory, _, mode_posterior, diagnostics = (
        trajectory_imm._score_trajectory_imm_exact_sparse(
            emissions,
            np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float),
            config,
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
        diagnostics["state_space_trajectory_family_terminal_probability"],
        2.0 / 3.0,
    )
    np.testing.assert_allclose(
        diagnostics["state_space_trajectory_family_event_probability"],
        5.0 / 6.0,
    )
