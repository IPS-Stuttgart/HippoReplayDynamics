import warnings

import numpy as np
from scipy.special import logsumexp

from hipporeplayimm import models
from hipporeplayimm.encoding import LogEmissionTensor


_CENTERS = np.array(
    [
        [-1.0e308, 0.0],
        [0.0, 0.0],
        [1.0e308, 0.0],
    ],
    dtype=float,
)
_STANDARDIZED_DISTANCES = np.array(
    [
        [0.0, 1.0, 2.0],
        [1.0, 0.0, 1.0],
        [2.0, 1.0, 0.0],
    ],
    dtype=float,
)


def _expected_log_weights(distances: np.ndarray) -> np.ndarray:
    values = -0.5 * np.square(np.asarray(distances, dtype=float))
    return values - logsumexp(values)


def test_legacy_diffusion_transition_is_stable_at_extreme_finite_scale() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        transition = models._log_transition_matrix(
            _CENTERS,
            sigma_cm=1.0e308,
            max_step_sigma=3.0,
        )

    assert len(transition) == _CENTERS.shape[0]
    for source, (indices, log_weights) in enumerate(transition):
        np.testing.assert_array_equal(indices, np.arange(_CENTERS.shape[0]))
        assert np.all(np.isfinite(log_weights))
        np.testing.assert_allclose(
            log_weights,
            _expected_log_weights(_STANDARDIZED_DISTANCES[source]),
            rtol=1.0e-12,
            atol=1.0e-12,
        )


def test_legacy_candidate_pairwise_kernel_is_stable_at_extreme_finite_scale() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        log_prob = models._full_grid_normalized_pairwise_gaussian_log_prob(
            _CENTERS[[0]],
            _CENTERS,
            _CENTERS,
            1.0e308,
        )

    assert log_prob.shape == (1, _CENTERS.shape[0])
    assert np.all(np.isfinite(log_prob))
    np.testing.assert_allclose(
        log_prob[0],
        _expected_log_weights(_STANDARDIZED_DISTANCES[0]),
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_public_legacy_models_return_finite_evidence_at_extreme_finite_scale() -> None:
    emissions = LogEmissionTensor(
        log_likelihood=np.array(
            [
                [0.0, -1.0, -2.0],
                [-2.0, -1.0, 0.0],
            ],
            dtype=float,
        ),
        spike_counts=np.zeros((2, 1), dtype=int),
        times=np.array([0.0, 0.02], dtype=float),
        dt=0.02,
        cell_ids=np.array([1], dtype=int),
        n_spikes=0,
    )
    replay_models = (
        models.DiffusionModel(sigma_cm=1.0e308, max_step_sigma=3.0),
        models.CandidateKinematicModel(
            mode="diffusion",
            top_k=0,
            diffusion_sigma_cm=1.0e308,
        ),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        scores = [model.score(emissions, _CENTERS) for model in replay_models]

    for score in scores:
        assert np.isfinite(score.log_likelihood)
        assert score.terminal_log_posterior is not None
        assert np.all(np.isfinite(score.terminal_log_posterior))
