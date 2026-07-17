from __future__ import annotations

import numpy as np

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


def test_core_diffusion_transition_preserves_large_scale_weights() -> None:
    centers = np.array([[0.0], [1.0e200]], dtype=float)

    with np.errstate(over="raise", invalid="raise"):
        transition = _log_transition_matrix(
            centers,
            sigma_cm=1.0e200,
            max_step_sigma=2.0,
        )

    matrix = np.zeros((2, 2), dtype=float)
    for source, (destinations, log_weights) in enumerate(transition):
        matrix[source, destinations] = np.exp(log_weights)

    far_weight = np.exp(-0.5) / (1.0 + np.exp(-0.5))
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
