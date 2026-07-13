from __future__ import annotations

import numpy as np

from hipporeplayimm import models
from hipporeplayimm.encoding import LogEmissionTensor


def test_candidate_kinematic_gaussian_normalization_survives_sigma_square_underflow() -> None:
    predicted = np.array([[0.5]], dtype=float)
    centers = np.array([[0.0], [1.0]], dtype=float)

    log_prob = models._full_grid_normalized_pairwise_gaussian_log_prob(
        predicted,
        centers,
        centers,
        1.0e-200,
    )

    assert np.all(np.isfinite(log_prob))
    np.testing.assert_allclose(
        np.exp(log_prob),
        np.array([[0.5, 0.5]], dtype=float),
        atol=1.0e-12,
    )


def test_candidate_momentum_score_stays_finite_with_off_grid_prediction() -> None:
    centers = np.array([[0.0], [1.0]], dtype=float)
    emissions = LogEmissionTensor(
        log_likelihood=np.zeros((3, 2), dtype=float),
        spike_counts=np.zeros((3, 0), dtype=int),
        times=np.array([0.0, 1.0, 2.0], dtype=float),
        dt=1.0,
        cell_ids=np.empty(0, dtype=int),
        n_spikes=0,
    )
    model = models.CandidateKinematicModel(
        mode="momentum",
        top_k=0,
        momentum_sigma_cm=1.0e-200,
    )

    score = model.score(emissions, centers)

    assert np.isfinite(score.log_likelihood)
    assert score.trajectory_log_posterior is not None
    assert np.all(np.isfinite(score.trajectory_log_posterior))
