from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import hipporeplayimm
import hipporeplayimm.candidate_log_mass_validation as validation
import hipporeplayimm.duration_occupancy as duration_occupancy


def test_candidate_log_mass_patch_refreshes_stale_duration_helpers(monkeypatch) -> None:
    def stale_momentum(
        ss,
        emissions,
        bin_centers,
        candidates,
        *,
        sigmas_cm,
        initial_sigma_cm,
        velocity_decays,
        time_scales,
        valid_bin_mask=None,
    ):
        return 0.0, np.zeros((2, 2), dtype=float), [-10.0, -10.0]

    def stale_imm(
        ss,
        emissions,
        bin_centers,
        candidates,
        *,
        stationary_sigma_cm,
        diffusion_sigmas_cm,
        momentum_sigmas_cm,
        initial_momentum_sigma_cm,
        velocity_decays,
        time_scales,
        mode_stickiness,
        mode_transitions=None,
        valid_bin_mask=None,
    ):
        return (
            0.0,
            np.zeros((2, 2), dtype=float),
            np.full((2, 4), 0.25, dtype=float),
            [-10.0, -10.0],
        )

    monkeypatch.setattr(duration_occupancy, "_score_momentum_duration", stale_momentum)
    monkeypatch.setattr(duration_occupancy, "_score_imm_duration", stale_imm)
    monkeypatch.setattr(
        duration_occupancy,
        validation._DURATION_OCCUPANCY_PATCHED_FLAG,
        True,
        raising=False,
    )

    hipporeplayimm.apply_runtime_patches()

    patched_momentum = duration_occupancy._score_momentum_duration
    patched_imm = duration_occupancy._score_imm_duration
    assert getattr(patched_momentum, validation._DURATION_MOMENTUM_WRAPPER_ATTR, False)
    assert getattr(patched_imm, validation._DURATION_IMM_WRAPPER_ATTR, False)
    assert patched_momentum.__hipporeplayimm_original__ is stale_momentum
    assert patched_imm.__hipporeplayimm_original__ is stale_imm

    emissions = SimpleNamespace(
        log_likelihood=np.array([[0.0, 10.0], [0.0, 10.0]], dtype=float),
        n_time=2,
    )
    candidates = [np.array([0], dtype=int), np.array([0], dtype=int)]
    valid_bin_mask = np.array([True, False])
    common = {
        "ss": None,
        "emissions": emissions,
        "bin_centers": np.zeros((2, 1), dtype=float),
        "candidates": candidates,
    }

    _, _, momentum_masses = patched_momentum(
        **common,
        sigmas_cm=np.ones(1),
        initial_sigma_cm=1.0,
        velocity_decays=np.ones(1),
        time_scales=np.ones(1),
        valid_bin_mask=valid_bin_mask,
    )
    _, _, _, imm_masses = patched_imm(
        **common,
        stationary_sigma_cm=1.0,
        diffusion_sigmas_cm=np.ones(1),
        momentum_sigmas_cm=np.ones(1),
        initial_momentum_sigma_cm=1.0,
        velocity_decays=np.ones(1),
        time_scales=np.ones(1),
        mode_stickiness=0.9,
        valid_bin_mask=valid_bin_mask,
    )

    np.testing.assert_allclose(momentum_masses, [0.0, 0.0])
    np.testing.assert_allclose(imm_masses, [0.0, 0.0])

    hipporeplayimm.apply_runtime_patches()

    assert duration_occupancy._score_momentum_duration is patched_momentum
    assert duration_occupancy._score_imm_duration is patched_imm
