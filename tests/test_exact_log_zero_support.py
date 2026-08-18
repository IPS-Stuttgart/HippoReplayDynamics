from __future__ import annotations

import numpy as np

import hipporeplayimm
from hipporeplayimm import models
from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.models import CandidateKinematicModel


def test_candidate_pruned_posterior_does_not_fabricate_legacy_sentinel_mass() -> None:
    emissions = LogEmissionTensor(
        log_likelihood=np.array(
            [
                [-5.0e306, -6.0e306],
                [-5.0e306, -6.0e306],
            ],
            dtype=float,
        ),
        spike_counts=np.zeros((2, 1), dtype=int),
        times=np.array([0.0, 1.0]),
        dt=1.0,
        cell_ids=np.array([1]),
        n_spikes=0,
    )
    centers = np.array([[0.0, 0.0], [1.0, 0.0]])
    model = CandidateKinematicModel(
        mode="diffusion",
        top_k=1,
        diffusion_sigma_cm=1.0,
    )

    score = model.score(
        emissions,
        centers,
        candidate_indices=[np.array([0]), np.array([0])],
    )

    assert np.isfinite(score.log_likelihood)
    assert score.terminal_log_posterior is not None
    assert score.terminal_log_posterior[0] == 0.0
    assert np.isneginf(score.terminal_log_posterior[1])
    assert score.diagnostics["decoded_map_bin"] == 0


def test_runtime_refresh_restores_exact_log_zero_aliases(monkeypatch) -> None:
    from hipporeplayimm import state_space, state_space_utils, trajectory_metrics

    monkeypatch.setattr(models, "LOG_ZERO", -1.0e300)
    monkeypatch.setattr(state_space, "LOG_ZERO", -1.0e300)
    monkeypatch.setattr(state_space_utils, "LOG_ZERO", -1.0e300)
    monkeypatch.setattr(trajectory_metrics, "_LOG_ZERO_ROW_THRESHOLD", -5.0e299)

    hipporeplayimm.apply_runtime_patches()

    assert np.isneginf(models.LOG_ZERO)
    assert np.isneginf(state_space.LOG_ZERO)
    assert np.isneginf(state_space_utils.LOG_ZERO)
    assert np.isneginf(trajectory_metrics._LOG_ZERO_ROW_THRESHOLD)
