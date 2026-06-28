import numpy as np
import pytest

from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.state_space import (
    StateSpaceDecoderConfig,
    StateSpaceReplayModel,
    _gaussian_transition_matrix,
    _score_fragmented,
    _score_imm_candidates,
)


def test_state_space_split_keeps_legacy_helper_exports():
    assert callable(_gaussian_transition_matrix)
    assert callable(_score_fragmented)
    assert callable(_score_imm_candidates)


def test_gaussian_transition_matrix_can_exclude_unvisited_bins():
    centers = np.array([[0.0, 0.0], [4.0, 0.0], [8.0, 0.0]])
    valid = np.array([True, False, True])

    transition = _gaussian_transition_matrix(
        centers,
        sigma_cm=10.0,
        max_step_sigma=4.0,
        valid_bin_mask=valid,
    ).toarray()

    assert np.allclose(transition[~valid], 0.0)
    assert np.allclose(transition.sum(axis=0), 1.0)


def test_state_space_occupancy_mask_excludes_invalid_terminal_bins():
    emissions = LogEmissionTensor(
        log_likelihood=np.log(
            np.array(
                [
                    [0.2, 0.6, 0.2],
                    [0.2, 0.6, 0.2],
                ]
            )
        ),
        spike_counts=np.zeros((2, 1), dtype=int),
        times=np.array([0.01, 0.03]),
        dt=0.02,
        cell_ids=np.array([1]),
        n_spikes=0,
    )
    centers = np.array([[0.0, 0.0], [4.0, 0.0], [8.0, 0.0]])
    occupancy = np.array([1.0, 0.0, 1.0])
    config = StateSpaceDecoderConfig(mode="fragmented", valid_occupancy_threshold_s=0.5)

    score = StateSpaceReplayModel(mode="fragmented", config=config).score(
        emissions,
        centers,
        occupancy_s=occupancy,
    )

    posterior = np.exp(score.terminal_log_posterior)
    assert np.isclose(posterior[1], 0.0)
    assert np.isclose(posterior[[0, 2]].sum(), 1.0)


def test_state_space_occupancy_rejects_boolean_mask_values():
    emissions = LogEmissionTensor(
        log_likelihood=np.log(
            np.array(
                [
                    [0.2, 0.6, 0.2],
                    [0.2, 0.6, 0.2],
                ]
            )
        ),
        spike_counts=np.zeros((2, 1), dtype=int),
        times=np.array([0.01, 0.03]),
        dt=0.02,
        cell_ids=np.array([1]),
        n_spikes=0,
    )
    centers = np.array([[0.0, 0.0], [4.0, 0.0], [8.0, 0.0]])
    occupancy = np.array([True, False, True])
    config = StateSpaceDecoderConfig(mode="fragmented", valid_occupancy_threshold_s=0.5)

    with pytest.raises(TypeError, match="occupancy_s.*not boolean"):
        StateSpaceReplayModel(mode="fragmented", config=config).score(
            emissions,
            centers,
            occupancy_s=occupancy,
        )


def test_state_space_model_scores_after_split():
    emissions = LogEmissionTensor(
        log_likelihood=np.log(
            np.array(
                [
                    [0.7, 0.2, 0.1],
                    [0.2, 0.6, 0.2],
                    [0.1, 0.2, 0.7],
                ]
            )
        ),
        spike_counts=np.zeros((3, 1), dtype=int),
        times=np.array([0.01, 0.03, 0.05]),
        dt=0.02,
        cell_ids=np.array([1]),
        n_spikes=0,
    )
    centers = np.array([[0.0, 0.0], [4.0, 0.0], [8.0, 0.0]])

    score = StateSpaceReplayModel(mode="diffusion").score(emissions, centers)

    assert np.isfinite(score.log_likelihood)
    assert score.trajectory_log_posterior is not None
    assert score.trajectory_log_posterior.shape == (3, 3)


def test_state_space_stationary_rejects_rows_without_finite_emission_mass():
    emissions = LogEmissionTensor(
        log_likelihood=np.array(
            [
                [0.0, -np.inf],
                [-np.inf, -np.inf],
            ]
        ),
        spike_counts=np.zeros((2, 1), dtype=int),
        times=np.array([0.0, 1.0]),
        dt=1.0,
        cell_ids=np.array([1]),
        n_spikes=0,
    )
    centers = np.array([[0.0, 0.0], [1.0, 0.0]])

    with pytest.raises(ValueError, match="emission row"):
        StateSpaceReplayModel(mode="stationary").score(emissions, centers)


def test_state_space_stationary_rejects_no_common_finite_stationary_path():
    emissions = LogEmissionTensor(
        log_likelihood=np.array(
            [
                [0.0, -np.inf],
                [-np.inf, 0.0],
            ]
        ),
        spike_counts=np.zeros((2, 1), dtype=int),
        times=np.array([0.0, 1.0]),
        dt=1.0,
        cell_ids=np.array([1]),
        n_spikes=0,
    )
    centers = np.array([[0.0, 0.0], [1.0, 0.0]])

    with pytest.raises(ValueError, match="stationary model has no finite path mass"):
        StateSpaceReplayModel(mode="stationary").score(emissions, centers)
