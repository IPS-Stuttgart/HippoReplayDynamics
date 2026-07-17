from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.models import DiffusionModel, _log_transition_matrix


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
    ("centers", "sigma_cm", "max_step_sigma", "standardized_distance"),
    [
        (np.array([[0.0], [1.0e200]]), 1.0e200, 2.0, 1.0),
        (np.array([[-1.0e308], [1.0e308]]), 1.0e308, 3.0, 2.0),
    ],
)
def test_core_diffusion_transition_preserves_large_scale_weights(
    centers: np.ndarray,
    sigma_cm: float,
    max_step_sigma: float,
    standardized_distance: float,
) -> None:
    with np.errstate(over="raise", invalid="raise"):
        transition = _log_transition_matrix(
            centers,
            sigma_cm=sigma_cm,
            max_step_sigma=max_step_sigma,
        )

    matrix = np.zeros((2, 2), dtype=float)
    for source, (destinations, log_weights) in enumerate(transition):
        matrix[source, destinations] = np.exp(log_weights)

    unnormalized_far_weight = np.exp(-0.5 * standardized_distance**2)
    far_weight = unnormalized_far_weight / (1.0 + unnormalized_far_weight)
    expected = np.array(
        [
            [1.0 - far_weight, far_weight],
            [far_weight, 1.0 - far_weight],
        ]
    )
    np.testing.assert_allclose(matrix, expected, rtol=1.0e-12, atol=0.0)


def test_diffusion_model_scores_large_finite_grid_without_overflow() -> None:
    model = DiffusionModel(sigma_cm=1.0e200, max_step_sigma=2.0)
    centers = np.array([[0.0], [1.0e200]], dtype=float)

    with np.errstate(over="raise", invalid="raise"):
        score = model.score(_uniform_emissions(), centers)

    assert np.isfinite(score.log_likelihood)
    np.testing.assert_allclose(
        np.exp(score.terminal_log_posterior),
        np.array([0.5, 0.5]),
        rtol=1.0e-12,
        atol=0.0,
    )
