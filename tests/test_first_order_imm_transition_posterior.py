import itertools
import json

import numpy as np

from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.sorted_spike_state_space import SortedSpikeStateSpaceReplayModel
from hipporeplayimm.state_space import StateSpaceDecoderConfig
from hipporeplayimm.state_space import _gaussian_transition_matrix
from hipporeplayimm.state_space_utils import _mode_transition_matrix


def test_first_order_imm_exports_normalized_transition_posteriors() -> None:
    emissions = LogEmissionTensor(
        log_likelihood=np.log(
            np.array(
                [
                    [0.85, 0.10, 0.05],
                    [0.10, 0.80, 0.10],
                    [0.05, 0.15, 0.80],
                    [0.75, 0.20, 0.05],
                ],
                dtype=float,
            )
        ),
        spike_counts=np.zeros((4, 1), dtype=int),
        times=np.array([0.0, 0.004, 0.008, 0.012]),
        dt=0.004,
        cell_ids=np.array([1]),
        n_spikes=0,
    )
    centers = np.array([[0.0, 0.0], [6.0, 0.0], [12.0, 0.0]])
    model = SortedSpikeStateSpaceReplayModel(
        mode="first-order-imm",
        config=StateSpaceDecoderConfig(
            mode="first-order-imm",
            stationary_sigma_cm=2.0,
            diffusion_sigma_cm_sqrt_s=60.0,
            max_step_sigma=3.0,
            imm_mode_stickiness=0.95,
            imm_switch_tau_s=0.06,
        ),
    )

    score = model.score(emissions, centers)
    mode = np.asarray(
        json.loads(score.diagnostics["state_space_imm_mode_posterior_over_time"]),
        dtype=float,
    )
    transition = np.asarray(
        json.loads(
            score.diagnostics["state_space_imm_mode_transition_posterior_over_time"]
        ),
        dtype=float,
    )
    switch = np.asarray(
        json.loads(score.diagnostics["state_space_imm_switch_probability_over_time"]),
        dtype=float,
    )

    assert mode.shape == (4, 3)
    assert transition.shape == (3, 3, 3)
    assert switch.shape == (3,)
    assert np.allclose(mode.sum(axis=1), 1.0)
    assert np.allclose(transition.sum(axis=(1, 2)), 1.0)
    assert np.allclose(switch, 1.0 - np.trace(transition, axis1=1, axis2=2))
    assert np.all((switch >= 0.0) & (switch <= 1.0))


def test_first_order_imm_transition_posterior_matches_bruteforce() -> None:
    likelihood = np.array(
        [
            [0.8, 0.2],
            [0.3, 0.7],
            [0.6, 0.4],
        ],
        dtype=float,
    )
    emissions = LogEmissionTensor(
        log_likelihood=np.log(likelihood),
        spike_counts=np.zeros((3, 1), dtype=int),
        times=np.array([0.0, 1.0, 2.0]),
        dt=1.0,
        cell_ids=np.array([1]),
        n_spikes=0,
    )
    centers = np.array([[0.0, 0.0], [1.0, 0.0]])
    config = StateSpaceDecoderConfig(
        mode="first-order-imm",
        stationary_sigma_cm=0.5,
        diffusion_sigma_cm_sqrt_s=1.5,
        max_step_sigma=10.0,
        imm_mode_stickiness=0.8,
        imm_switch_tau_s=0.0,
    )
    score = SortedSpikeStateSpaceReplayModel(
        mode="first-order-imm",
        config=config,
    ).score(emissions, centers)
    observed = np.asarray(
        json.loads(
            score.diagnostics["state_space_imm_mode_transition_posterior_over_time"]
        ),
        dtype=float,
    )

    mode_transition = _mode_transition_matrix(3, config.imm_mode_stickiness)
    spatial = (
        _gaussian_transition_matrix(
            centers,
            config.stationary_sigma_cm,
            config.max_step_sigma,
        ),
        _gaussian_transition_matrix(
            centers,
            config.diffusion_sigma_cm_sqrt_s,
            config.max_step_sigma,
        ),
        np.full((2, 2), 0.5),
    )
    expected = np.zeros((2, 3, 3), dtype=float)
    total = 0.0
    for modes in itertools.product(range(3), repeat=3):
        for bins in itertools.product(range(2), repeat=3):
            probability = (1.0 / 6.0) * likelihood[0, bins[0]]
            for time_index in (1, 2):
                probability *= mode_transition[modes[time_index - 1], modes[time_index]]
                probability *= spatial[modes[time_index]][bins[time_index], bins[time_index - 1]]
                probability *= likelihood[time_index, bins[time_index]]
            total += probability
            for transition_index in (0, 1):
                expected[
                    transition_index,
                    modes[transition_index],
                    modes[transition_index + 1],
                ] += probability
    expected /= total

    assert np.allclose(observed, expected, atol=1e-12)
