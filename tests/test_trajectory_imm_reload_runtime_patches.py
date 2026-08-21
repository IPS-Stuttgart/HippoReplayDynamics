from __future__ import annotations

import importlib
from types import SimpleNamespace

import numpy as np
import pytest

import hipporeplayimm
from hipporeplayimm.encoding import LogEmissionTensor


_VALIDATION_FLAG = "_trajectory_imm_parameter_validation_patch_applied"


def _reload_trajectory_imm():
    from hipporeplayimm import state_space_trajectory_imm as trajectory_imm

    # Reproduce the stale state deterministically. importlib.reload() executes
    # in the existing module namespace, so patch-only names survive unless the
    # module body explicitly replaces them.
    setattr(trajectory_imm, _VALIDATION_FLAG, True)
    return importlib.reload(trajectory_imm)


def test_runtime_patches_restore_trajectory_imm_validation_after_reload() -> None:
    trajectory_imm = _reload_trajectory_imm()

    assert getattr(trajectory_imm, _VALIDATION_FLAG, False)
    assert getattr(
        trajectory_imm._trajectory_imm_mode_stickiness,
        "__wrapped__",
        None,
    ) is None

    hipporeplayimm.apply_runtime_patches()

    config = SimpleNamespace(
        imm_switch_tau_s=True,
        trajectory_imm_momentum_switch_probability=None,
    )
    with pytest.raises(TypeError, match="imm_switch_tau_s"):
        trajectory_imm._trajectory_imm_mode_transition_matrices(
            config,
            0.95,
            np.asarray([0.01]),
        )


def test_runtime_patches_restore_trajectory_imm_evidence_only_diagnostics_after_reload() -> None:
    trajectory_imm = _reload_trajectory_imm()
    hipporeplayimm.apply_runtime_patches()

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


def test_runtime_patch_refresh_is_idempotent_after_trajectory_imm_reload() -> None:
    trajectory_imm = _reload_trajectory_imm()
    hipporeplayimm.apply_runtime_patches()

    validated_stickiness = trajectory_imm._trajectory_imm_mode_stickiness
    diagnostic_score = trajectory_imm._score_trajectory_imm_exact_sparse

    hipporeplayimm.apply_runtime_patches()

    assert trajectory_imm._trajectory_imm_mode_stickiness is validated_stickiness
    assert trajectory_imm._score_trajectory_imm_exact_sparse is diagnostic_score
