from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.models import (
    CandidateKinematicModel,
    _pairwise_gaussian_log_prob,
)


def _uniform_emissions() -> LogEmissionTensor:
    return LogEmissionTensor(
        log_likelihood=np.zeros((2, 2), dtype=float),
        spike_counts=np.empty((2, 0), dtype=int),
        times=np.array([0.0, 1.0]),
        dt=1.0,
        cell_ids=np.empty(0, dtype=int),
        n_spikes=0,
    )


@pytest.mark.parametrize(
    ("centers", "sigma_cm", "far_log_weight"),
    [
        (np.array([[0.0], [1.0e200]]), 1.0e200, -0.5),
        (np.array([[-1.0e308], [1.0e308]]), 1.0e308, -2.0),
    ],
)
def test_core_pairwise_gaussian_preserves_large_scale_log_weights(
    centers: np.ndarray,
    sigma_cm: float,
    far_log_weight: float,
) -> None:
    with np.errstate(over="raise", invalid="raise"):
        log_weights = _pairwise_gaussian_log_prob(centers, centers, sigma_cm)

    expected = np.array(
        [
            [0.0, far_log_weight],
            [far_log_weight, 0.0],
        ]
    )
    np.testing.assert_allclose(log_weights, expected, rtol=1.0e-12, atol=0.0)


def test_candidate_diffusion_scores_large_finite_grid_without_nan() -> None:
    model = CandidateKinematicModel(
        mode="diffusion",
        top_k=0,
        diffusion_sigma_cm=1.0e200,
    )
    centers = np.array([[0.0], [1.0e200]], dtype=float)

    with np.errstate(over="raise", invalid="raise"):
        score = model.score(_uniform_emissions(), centers)

    assert score.log_likelihood == pytest.approx(0.0, abs=1.0e-12)
    assert score.terminal_log_posterior is not None
    assert score.trajectory_log_posterior is not None
    np.testing.assert_allclose(
        np.exp(score.terminal_log_posterior),
        np.array([0.5, 0.5]),
        rtol=1.0e-12,
        atol=0.0,
    )
    np.testing.assert_allclose(
        np.exp(score.trajectory_log_posterior),
        np.full((2, 2), 0.5),
        rtol=1.0e-12,
        atol=0.0,
    )
