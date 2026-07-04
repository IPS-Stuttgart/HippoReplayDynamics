from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import hipporeplayimm
from hipporeplayimm import state_space_trajectory_imm as trajectory_imm
from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.trajectory_imm_single_bin_diagnostics import apply_trajectory_imm_single_bin_diagnostics_patch


def test_trajectory_imm_evidence_only_event_probability_uses_filtered_history() -> None:
    hipporeplayimm.apply_runtime_patches()
    apply_trajectory_imm_single_bin_diagnostics_patch()

    emissions = LogEmissionTensor(
        log_likelihood=np.zeros((2, 2), dtype=float),
        spike_counts=np.zeros((2, 1), dtype=int),
        times=np.array([0.0, 0.01], dtype=float),
        dt=0.01,
        cell_ids=np.array([1], dtype=int),
        n_spikes=0,
    )
    bin_centers = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)
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

    _, trajectory, _, mode_posterior, diagnostics = trajectory_imm._score_trajectory_imm_exact_sparse(
        emissions,
        bin_centers,
        config,
        emissions.transition_durations,
        return_trajectory=False,
    )

    assert trajectory is None
    assert mode_posterior is None
    assert diagnostics["state_space_trajectory_imm_mode_posterior"] == "filtered_evidence_only_state"
    assert np.isclose(
        diagnostics["state_space_trajectory_family_terminal_probability"],
        2.0 / 3.0,
    )
    assert np.isclose(
        diagnostics["state_space_trajectory_family_event_probability"],
        5.0 / 6.0,
    )
    assert not np.isclose(
        diagnostics["state_space_trajectory_family_event_probability"],
        diagnostics["state_space_trajectory_family_terminal_probability"],
    )
