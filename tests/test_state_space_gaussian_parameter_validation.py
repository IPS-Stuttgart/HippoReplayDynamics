import numpy as np
import pytest

from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.state_space import StateSpaceDecoderConfig, StateSpaceReplayModel
from hipporeplayimm.state_space_utils import (
    _gaussian_transition_matrix,
    _pairwise_gaussian_log_prob,
    _per_bin_sigma,
)


def _toy_emissions() -> LogEmissionTensor:
    return LogEmissionTensor(
        log_likelihood=np.array(
            [
                [0.0, -1.0, -2.0],
                [-2.0, 0.0, -1.0],
                [-1.0, -2.0, 0.0],
            ],
            dtype=float,
        ),
        spike_counts=np.zeros((3, 1), dtype=int),
        times=np.array([0.0, 1.0, 2.0]),
        dt=1.0,
        cell_ids=np.array([1]),
        n_spikes=0,
    )


def _toy_centers() -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
        ],
        dtype=float,
    )


@pytest.mark.parametrize("value", [True, np.bool_(True), "85.0", np.asarray("85.0")])
def test_state_space_transition_helpers_reject_bool_and_string_sigmas(value) -> None:
    centers = _toy_centers()

    with pytest.raises(TypeError, match="sigma_cm"):
        _gaussian_transition_matrix(centers, value, 4.0)

    with pytest.raises(TypeError, match="sigma_cm"):
        _pairwise_gaussian_log_prob(centers, centers, value)

    with pytest.raises(TypeError, match="sigma_cm_sqrt_s"):
        _per_bin_sigma(value, 1.0)


def test_state_space_transition_helpers_reject_array_shaped_scalars() -> None:
    centers = _toy_centers()

    with pytest.raises(TypeError, match="numeric scalar"):
        _gaussian_transition_matrix(centers, np.array([12.0]), 4.0)

    with pytest.raises(TypeError, match="numeric scalar"):
        _gaussian_transition_matrix(centers, 12.0, np.array([4.0]))

    with pytest.raises(TypeError, match="numeric scalar"):
        _per_bin_sigma(85.0, np.array([1.0]))


def test_state_space_transition_helpers_accept_numeric_numpy_scalars() -> None:
    centers = _toy_centers()

    assert _per_bin_sigma(np.float64(85.0), np.float64(1.0)) > 0.0
    transition = _gaussian_transition_matrix(centers, np.float64(12.0), np.float64(4.0))
    assert transition.shape == (3, 3)
    log_prob = _pairwise_gaussian_log_prob(centers, centers, np.float64(12.0))
    assert log_prob.shape == (3, 3)
    assert np.all(np.isfinite(log_prob))


@pytest.mark.parametrize("value", [True, np.bool_(True), "85.0", np.asarray("85.0"), np.asarray([85.0])])
def test_diffusion_score_rejects_invalid_duration_aware_diffusion_sigma(value) -> None:
    model = StateSpaceReplayModel(
        mode="diffusion",
        config=StateSpaceDecoderConfig(
            mode="diffusion",
            diffusion_sigma_cm_sqrt_s=value,
        ),
    )

    with pytest.raises((TypeError, ValueError), match="sigma_cm_sqrt_s"):
        model.score(_toy_emissions(), _toy_centers())


@pytest.mark.parametrize("value", [True, np.bool_(True), "4.0", np.asarray("4.0"), np.asarray([4.0])])
def test_diffusion_score_rejects_invalid_max_step_sigma(value) -> None:
    model = StateSpaceReplayModel(
        mode="diffusion",
        config=StateSpaceDecoderConfig(
            mode="diffusion",
            max_step_sigma=value,
        ),
    )

    with pytest.raises((TypeError, ValueError), match="max_step_sigma"):
        model.score(_toy_emissions(), _toy_centers())
