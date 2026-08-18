from __future__ import annotations

import numpy as np

from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.models import CandidateKinematicModel


def test_candidate_pruned_posterior_keeps_non_candidates_at_exact_zero() -> None:
    """A finite legacy sentinel must never become normalized posterior mass."""

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
    assert score.trajectory_log_posterior is not None
    assert score.terminal_log_posterior[0] == 0.0
    assert np.isneginf(score.terminal_log_posterior[1])
    assert np.all(score.trajectory_log_posterior[:, 0] == 0.0)
    assert np.all(np.isneginf(score.trajectory_log_posterior[:, 1]))
    assert score.diagnostics["decoded_map_bin"] == 0
