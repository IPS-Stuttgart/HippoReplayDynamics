from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.state_space_model import (
    StateSpaceDecoderConfig,
    StateSpaceReplayModel,
)
from hipporeplayimm.state_space_utils import (
    _full_grid_normalized_pairwise_gaussian_log_prob,
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
def test_state_space_pairwise_gaussian_preserves_large_scale_weights(
    centers: np.ndarray,
    sigma_cm: float,
    far_log_weight: float,
) -> None:
    with np.errstate(over="raise", invalid="raise"):
        log_prob = _full_grid_normalized_pairwise_gaussian_log_prob(
            centers[:1],
            centers,
            centers,
            sigma_cm,
        )

    expected = np.array([[0.0, far_log_weight]], dtype=float)
    expected -= np.logaddexp.reduce(expected, axis=1, keepdims=True)
    np.testing.assert_allclose(log_prob, expected, rtol=1.0e-12, atol=0.0)


def test_state_space_pairwise_gaussian_normalizes_when_sigma_square_underflows() -> None:
    predicted = np.array([[0.0]], dtype=float)
    support = np.array([[0.9], [1.0]], dtype=float)

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        log_prob = _full_grid_normalized_pairwise_gaussian_log_prob(
            predicted,
            support,
            support,
            1.0e-200,
        )

    assert not np.any(np.isnan(log_prob))
    np.testing.assert_allclose(np.exp(log_prob), np.array([[1.0, 0.0]]), atol=0.0)


def test_state_space_candidate_imm_scores_large_finite_grid() -> None:
    centers = np.array([[0.0], [1.0e200]], dtype=float)
    config = StateSpaceDecoderConfig(
        mode="imm",
        stationary_sigma_cm=1.0e200,
        diffusion_sigma_cm_sqrt_s=1.0e200,
        momentum_sigma_cm_sqrt_s=1.0e200,
        momentum_initial_sigma_cm_sqrt_s=1.0e200,
        momentum_candidate_top_k=0,
        momentum_predicted_candidate_top_k=0,
    )
    model = StateSpaceReplayModel(mode="imm", config=config)

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
