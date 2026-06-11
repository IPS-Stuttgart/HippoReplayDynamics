from __future__ import annotations

import numpy as np

from hipporeplayimm.duration_dynamics import attach_duration_metadata
from hipporeplayimm.duration_occupancy import _uniform_backward
from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.state_space import StateSpaceDecoderConfig, StateSpaceReplayModel
from hipporeplayimm.state_space_first_order import _forward_backward_first_order_time_varying
from hipporeplayimm.state_space_model import _candidate_support_log_values, _per_transition_sigmas_cm
from hipporeplayimm.state_space_utils import _gaussian_transition_matrix


def test_attach_duration_metadata_preserves_dt_duration_metadata() -> None:
    emissions = LogEmissionTensor(
        log_likelihood=np.zeros((3, 2), dtype=float),
        spike_counts=np.zeros((3, 1), dtype=int),
        times=np.array([0.0, 0.1, 0.25], dtype=float),
        dt=0.1,
        cell_ids=np.array([1], dtype=int),
        n_spikes=0,
    )

    attach_duration_metadata(emissions)

    assert type(emissions.dt) is float
    assert emissions.dt == 0.1
    np.testing.assert_allclose(emissions.transition_durations, np.array([0.1, 0.15], dtype=float))


def test_uniform_backward_zeroes_occupancy_masked_bins() -> None:
    values = np.array([1.0, 2.0, 4.0, 8.0], dtype=float)
    valid_mask = np.array([True, False, True, False])

    backward = _uniform_backward(values, valid_mask)

    np.testing.assert_allclose(backward, np.array([2.5, 0.0, 2.5, 0.0]))


def test_momentum_candidate_predictions_use_duration_adjusted_displacements() -> None:
    emissions = _variable_duration_candidate_emissions()
    centers = np.arange(8.0, dtype=float)[:, None]
    config = StateSpaceDecoderConfig(
        mode="momentum",
        momentum_candidate_top_k=1,
        momentum_predicted_candidate_top_k=1,
        momentum_velocity_decay=1.0,
        momentum_velocity_decay_tau_s=0.0,
    )
    model = StateSpaceReplayModel(mode="momentum", config=config)

    candidates = model.candidate_indices(emissions, centers)

    assert 4 in set(candidates[2])
    assert 6 in set(candidates[2])


def test_posterior_candidate_source_honors_occupancy_mask_in_diffusion_beam() -> None:
    valid_mask = np.array([True, True, False])
    log_likelihood = np.array(
        [
            [0.0, -0.1, -np.inf],
            [-0.2, 0.0, -np.inf],
            [-0.3, 0.0, -np.inf],
        ],
        dtype=float,
    )
    emissions = LogEmissionTensor(
        log_likelihood=log_likelihood,
        spike_counts=np.zeros((3, 1), dtype=int),
        times=np.array([0.0, 0.5, 2.0], dtype=float),
        dt=1.0,
        cell_ids=np.array([1], dtype=int),
        n_spikes=0,
    )
    centers = np.array([[0.0, 0.0], [1.0, 0.0], [1.1, 0.0]], dtype=float)
    config = StateSpaceDecoderConfig(
        mode="momentum",
        momentum_candidate_source="posterior",
        diffusion_sigma_cm_sqrt_s=2.0,
        max_step_sigma=10.0,
    )

    sigmas = _per_transition_sigmas_cm(
        config.diffusion_sigma_cm_sqrt_s,
        emissions.transition_durations,
    )
    transitions = [
        _gaussian_transition_matrix(
            centers,
            float(sigma_cm),
            config.max_step_sigma,
            valid_bin_mask=valid_mask,
        )
        for sigma_cm in sigmas
    ]
    _, expected = _forward_backward_first_order_time_varying(
        log_likelihood,
        transitions,
        valid_bin_mask=valid_mask,
    )

    actual = _candidate_support_log_values(
        emissions,
        centers,
        config,
        valid_bin_mask=valid_mask,
    )
    candidates = StateSpaceReplayModel(mode="momentum", config=config).candidate_indices(
        emissions,
        centers,
        valid_bin_mask=valid_mask,
    )

    np.testing.assert_allclose(actual[:, valid_mask], expected[:, valid_mask], rtol=1e-12, atol=1e-12)
    assert np.all(actual[:, ~valid_mask] <= -1.0e299)
    assert all(np.all(valid_mask[current]) for current in candidates)


def _variable_duration_candidate_emissions() -> LogEmissionTensor:
    log_likelihood = np.full((3, 8), -10.0, dtype=float)
    log_likelihood[0, 0] = 0.0
    log_likelihood[1, 2] = 0.0
    log_likelihood[2, 4] = 0.0
    emissions = LogEmissionTensor(
        log_likelihood=log_likelihood,
        spike_counts=np.zeros((3, 1), dtype=int),
        times=np.array([0.0, 1.0, 3.0], dtype=float),
        dt=1.0,
        cell_ids=np.array([1], dtype=int),
        n_spikes=0,
    )
    emissions.transition_durations = np.array([1.0, 2.0], dtype=float)
    attach_duration_metadata(emissions)
    return emissions
