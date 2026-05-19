from argparse import Namespace
import itertools

import numpy as np
import pandas as pd
import pytest
from scipy.special import logsumexp

from hipporeplayimm.benchmarks import BenchmarkConfig, _build_models
from hipporeplayimm.cli import _goal_state_space_kwargs
from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.goal_state_space import GoalStateSpaceReplayModel, _goal_transition_matrix
from hipporeplayimm.ground_truth import _goal_state_space_kwargs_for_scores


def test_goal_state_space_model_scores_synthetic_event():
    centers = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    log_likelihood = np.log(
        np.array(
            [
                [0.70, 0.20, 0.08, 0.02],
                [0.10, 0.20, 0.60, 0.10],
                [0.02, 0.08, 0.20, 0.70],
            ]
        )
    )
    emissions = LogEmissionTensor(
        log_likelihood=log_likelihood,
        spike_counts=np.zeros((3, 1), dtype=int),
        times=np.array([0.0, 1.0, 2.0]),
        dt=1.0,
        cell_ids=np.array([1]),
        n_spikes=0,
    )
    model = GoalStateSpaceReplayModel(
        candidate_goals=np.array([[0.0, 0.0], [3.0, 0.0]]),
        transition_sigma_cm_sqrt_s=1.0,
        drift_speed_cm_s=1.0,
        max_step_sigma=10.0,
    )

    score = model.score(emissions, centers)

    assert np.isfinite(score.log_likelihood)
    assert score.terminal_log_posterior is not None
    assert score.trajectory_log_posterior is not None
    assert score.trajectory_log_posterior.shape == emissions.log_likelihood.shape
    assert np.allclose(logsumexp(score.terminal_log_posterior), 0.0)
    assert score.diagnostics['goal_state_space_candidate_goals'] == 2
    assert score.diagnostics['goal_state_space_lateral_sigma_scale'] == 1.0
    assert score.diagnostics['goal_state_space_diffusion_mixture_weight'] == 0.0
    assert score.diagnostics['goal_state_space_most_likely_goal_x'] == 3.0
    assert score.diagnostics['goal_state_space_most_likely_goal_probability'] > 0.5
    assert score.diagnostics['goal_state_space_evidence_support'] == 'exact_full_grid'


def test_goal_state_space_zero_drift_matches_diffusion_bruteforce():
    centers = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    log_likelihood = np.log(
        np.array(
            [
                [0.60, 0.30, 0.10],
                [0.20, 0.60, 0.20],
                [0.10, 0.30, 0.60],
            ]
        )
    )
    emissions = LogEmissionTensor(
        log_likelihood=log_likelihood,
        spike_counts=np.zeros((3, 1), dtype=int),
        times=np.array([0.0, 1.0, 2.0]),
        dt=1.0,
        cell_ids=np.array([1]),
        n_spikes=0,
    )
    score = GoalStateSpaceReplayModel(
        candidate_goals=np.array([[2.0, 0.0]]),
        transition_sigma_cm_sqrt_s=1.0,
        drift_speed_cm_s=0.0,
        max_step_sigma=10.0,
    ).score(emissions, centers)

    transition = np.empty((centers.shape[0], centers.shape[0]), dtype=float)
    for src, center in enumerate(centers):
        weights = np.exp(-0.5 * np.sum((centers - center[None, :]) ** 2, axis=1))
        transition[:, src] = weights / float(weights.sum())
    brute_terms = []
    for path in itertools.product(range(centers.shape[0]), repeat=emissions.n_time):
        logp = -np.log(centers.shape[0]) + log_likelihood[0, path[0]]
        for time_index in range(1, emissions.n_time):
            logp += np.log(transition[path[time_index], path[time_index - 1]])
            logp += log_likelihood[time_index, path[time_index]]
        brute_terms.append(logp)

    assert np.allclose(score.log_likelihood, logsumexp(brute_terms))


def test_goal_state_space_lateral_sigma_scale_sharpens_goal_axis_transition():
    centers = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
    isotropic = _goal_transition_matrix(
        centers,
        np.array([2.0, 0.0]),
        drift_step_cm=1.0,
        sigma_cm=1.0,
        max_step_sigma=10.0,
    )
    narrow = _goal_transition_matrix(
        centers,
        np.array([2.0, 0.0]),
        drift_step_cm=1.0,
        sigma_cm=1.0,
        max_step_sigma=10.0,
        lateral_sigma_scale=0.25,
    )

    isotropic_off_axis_ratio = float(isotropic[2, 0] / isotropic[1, 0])
    narrow_off_axis_ratio = float(narrow[2, 0] / narrow[1, 0])

    assert narrow_off_axis_ratio < isotropic_off_axis_ratio


def test_goal_state_space_lateral_sigma_scale_rewards_straight_sweep():
    centers = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
    emissions = LogEmissionTensor(
        log_likelihood=np.log(
            np.array(
                [
                    [1.0, 0.01, 0.01, 0.01],
                    [0.01, 1.0, 0.40, 0.01],
                ]
            )
        ),
        spike_counts=np.zeros((2, 1), dtype=int),
        times=np.array([0.0, 1.0]),
        dt=1.0,
        cell_ids=np.array([1]),
        n_spikes=0,
    )
    base = GoalStateSpaceReplayModel(
        candidate_goals=np.array([[2.0, 0.0]]),
        transition_sigma_cm_sqrt_s=1.0,
        drift_speed_cm_s=1.0,
        max_step_sigma=10.0,
    ).score(emissions, centers)
    narrow = GoalStateSpaceReplayModel(
        candidate_goals=np.array([[2.0, 0.0]]),
        transition_sigma_cm_sqrt_s=1.0,
        lateral_sigma_scale=0.25,
        drift_speed_cm_s=1.0,
        max_step_sigma=10.0,
    ).score(emissions, centers)

    assert narrow.log_likelihood > base.log_likelihood
    assert narrow.diagnostics['goal_state_space_lateral_sigma_scale'] == 0.25


def test_goal_state_space_diffusion_mixture_weight_allows_pause():
    centers = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    transition = _goal_transition_matrix(
        centers,
        np.array([2.0, 0.0]),
        drift_step_cm=1.0,
        sigma_cm=0.25,
        max_step_sigma=10.0,
        diffusion_mixture_weight=0.5,
    )
    base = _goal_transition_matrix(
        centers,
        np.array([2.0, 0.0]),
        drift_step_cm=1.0,
        sigma_cm=0.25,
        max_step_sigma=10.0,
    )

    assert transition[0, 0] > base[0, 0]
    assert np.allclose(np.asarray(transition.sum(axis=0)).ravel(), 1.0)


def test_goal_state_space_diffusion_mixture_weight_rewards_stalled_event():
    centers = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    emissions = LogEmissionTensor(
        log_likelihood=np.log(
            np.array(
                [
                    [1.0, 0.01, 0.01],
                    [1.0, 0.01, 0.01],
                ]
            )
        ),
        spike_counts=np.zeros((2, 1), dtype=int),
        times=np.array([0.0, 1.0]),
        dt=1.0,
        cell_ids=np.array([1]),
        n_spikes=0,
    )
    base = GoalStateSpaceReplayModel(
        candidate_goals=np.array([[2.0, 0.0]]),
        transition_sigma_cm_sqrt_s=0.25,
        drift_speed_cm_s=1.0,
        max_step_sigma=10.0,
    ).score(emissions, centers)
    mixed = GoalStateSpaceReplayModel(
        candidate_goals=np.array([[2.0, 0.0]]),
        transition_sigma_cm_sqrt_s=0.25,
        diffusion_mixture_weight=0.5,
        drift_speed_cm_s=1.0,
        max_step_sigma=10.0,
    ).score(emissions, centers)

    assert mixed.log_likelihood > base.log_likelihood
    assert mixed.diagnostics['goal_state_space_diffusion_mixture_weight'] == 0.5


def test_goal_state_space_component_switch_probability_rewards_goal_change():
    centers = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    log_likelihood = np.log(
        np.array(
            [
                [1.0, 0.01, 0.01],
                [0.01, 1.0, 0.01],
                [0.01, 0.01, 1.0],
                [0.01, 1.0, 0.01],
                [1.0, 0.01, 0.01],
            ]
        )
    )
    emissions = LogEmissionTensor(
        log_likelihood=log_likelihood,
        spike_counts=np.zeros((5, 1), dtype=int),
        times=np.arange(5.0),
        dt=1.0,
        cell_ids=np.array([1]),
        n_spikes=0,
    )
    fixed = GoalStateSpaceReplayModel(
        candidate_goals=np.array([[2.0, 0.0], [0.0, 0.0]]),
        transition_sigma_cm_sqrt_s=0.2,
        drift_speed_cm_s=1.0,
        max_step_sigma=10.0,
    ).score(emissions, centers)
    switching = GoalStateSpaceReplayModel(
        candidate_goals=np.array([[2.0, 0.0], [0.0, 0.0]]),
        transition_sigma_cm_sqrt_s=0.2,
        drift_speed_cm_s=1.0,
        max_step_sigma=10.0,
        component_switch_probability=0.25,
    ).score(emissions, centers)

    assert switching.log_likelihood > fixed.log_likelihood
    assert switching.diagnostics['goal_state_space_component_switch_probability'] == 0.25


def test_goal_state_space_uses_transition_durations():
    centers = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    transition_durations = np.array([1.0, 4.0])
    log_likelihood = np.log(
        np.array(
            [
                [0.70, 0.20, 0.10],
                [0.20, 0.60, 0.20],
                [0.05, 0.15, 0.80],
            ]
        )
    )
    emissions = LogEmissionTensor(
        log_likelihood=log_likelihood,
        spike_counts=np.zeros((3, 1), dtype=int),
        times=np.array([0.0, 1.0, 5.0]),
        dt=1.0,
        cell_ids=np.array([1]),
        n_spikes=0,
    )
    emissions.transition_durations = transition_durations
    model = GoalStateSpaceReplayModel(
        candidate_goals=np.array([[2.0, 0.0]]),
        transition_sigma_cm_sqrt_s=1.0,
        drift_speed_cm_s=1.0,
        max_step_sigma=10.0,
    )

    score = model.score(emissions, centers)

    transitions = [
        _goal_transition_matrix(
            centers,
            np.array([2.0, 0.0]),
            drift_step_cm=float(duration),
            sigma_cm=float(np.sqrt(duration)),
            max_step_sigma=10.0,
        ).toarray()
        for duration in transition_durations
    ]
    brute_terms = []
    for path in itertools.product(range(centers.shape[0]), repeat=emissions.n_time):
        logp = -np.log(centers.shape[0]) + log_likelihood[0, path[0]]
        for time_index in range(1, emissions.n_time):
            logp += np.log(transitions[time_index - 1][path[time_index], path[time_index - 1]])
            logp += log_likelihood[time_index, path[time_index]]
        brute_terms.append(logp)

    assert np.allclose(score.log_likelihood, logsumexp(brute_terms))
    assert score.diagnostics['goal_state_space_transition_durations'] == '1,4'
    assert score.diagnostics['goal_state_space_transition_sigma_cm'] == pytest.approx(1.5)
    assert score.diagnostics['goal_state_space_drift_step_cm'] == pytest.approx(2.5)


def test_goal_state_space_goal_prior_biases_ambiguous_event():
    centers = np.array([[0.0, 0.0], [1.0, 0.0]])
    log_likelihood = np.log(np.full((2, 2), 0.5))
    emissions = LogEmissionTensor(
        log_likelihood=log_likelihood,
        spike_counts=np.zeros((2, 1), dtype=int),
        times=np.array([0.0, 1.0]),
        dt=1.0,
        cell_ids=np.array([1]),
        n_spikes=0,
    )
    score = GoalStateSpaceReplayModel(
        candidate_goals=np.array([[0.0, 0.0], [1.0, 0.0]]),
        goal_prior_weights=np.array([9.0, 1.0]),
        transition_sigma_cm_sqrt_s=1.0,
        drift_speed_cm_s=0.0,
        max_step_sigma=10.0,
    ).score(emissions, centers)

    assert score.diagnostics['goal_state_space_goal_prior'] == 'provided'
    assert np.isclose(score.diagnostics['goal_state_space_goal_prior_max_probability'], 0.9)
    assert score.diagnostics['goal_state_space_most_likely_goal_index'] == 0
    assert np.isclose(score.diagnostics['goal_state_space_most_likely_goal_probability'], 0.9)


def test_goal_state_space_initial_position_prior_biases_ambiguous_event():
    centers = np.array([[0.0, 0.0], [1.0, 0.0]])
    log_likelihood = np.log(np.full((1, 2), 0.5))
    emissions = LogEmissionTensor(
        log_likelihood=log_likelihood,
        spike_counts=np.zeros((1, 1), dtype=int),
        times=np.array([0.0]),
        dt=1.0,
        cell_ids=np.array([1]),
        n_spikes=0,
    )
    score = GoalStateSpaceReplayModel(
        candidate_goals=np.array([[0.0, 0.0], [1.0, 0.0]]),
        initial_position_prior_weights=np.array([1.0, 9.0]),
        transition_sigma_cm_sqrt_s=1.0,
        drift_speed_cm_s=0.0,
        max_step_sigma=10.0,
    ).score(emissions, centers)

    assert score.diagnostics['goal_state_space_initial_position_prior'] == 'provided'
    assert np.isclose(
        score.diagnostics['goal_state_space_initial_position_prior_max_probability'],
        0.9,
    )
    assert np.isclose(np.exp(score.terminal_log_posterior[1]), 0.9)


def test_goal_state_space_reverse_terminal_position_prior_rewards_reverse_end():
    centers = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    emissions = LogEmissionTensor(
        log_likelihood=np.log(
            np.array(
                [
                    [0.05, 0.90, 0.05],
                    [0.05, 0.05, 0.90],
                ]
            )
        ),
        spike_counts=np.zeros((2, 1), dtype=int),
        times=np.array([0.0, 1.0]),
        dt=1.0,
        cell_ids=np.array([1]),
        n_spikes=0,
    )
    base = GoalStateSpaceReplayModel(
        candidate_goals=np.array([[0.0, 0.0]]),
        transition_sigma_cm_sqrt_s=0.5,
        drift_speed_cm_s=1.0,
        max_step_sigma=10.0,
        direction_mode='bidirectional',
        initial_goal_prior_sigma_cm=1.0,
    ).score(emissions, centers)
    terminal = GoalStateSpaceReplayModel(
        candidate_goals=np.array([[0.0, 0.0]]),
        reverse_terminal_position_prior_weights=np.array([0.05, 0.05, 0.90]),
        transition_sigma_cm_sqrt_s=0.5,
        drift_speed_cm_s=1.0,
        max_step_sigma=10.0,
        direction_mode='bidirectional',
        initial_goal_prior_sigma_cm=1.0,
    ).score(emissions, centers)

    assert terminal.log_likelihood > base.log_likelihood
    assert terminal.diagnostics['goal_state_space_reverse_terminal_position_prior'] == 'provided'
    assert terminal.diagnostics['goal_state_space_reverse_terminal_position_prior_weight'] == 1.0
    assert terminal.diagnostics['goal_state_space_reverse_terminal_position_prior_max_factor'] > 1.0


def test_goal_state_space_initial_position_prior_can_target_toward_components():
    centers = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    emissions = LogEmissionTensor(
        log_likelihood=np.log(
            np.array(
                [
                    [0.05, 0.90, 0.05],
                    [0.05, 0.05, 0.90],
                ]
            )
        ),
        spike_counts=np.zeros((2, 1), dtype=int),
        times=np.array([0.0, 1.0]),
        dt=1.0,
        cell_ids=np.array([1]),
        n_spikes=0,
    )
    all_components = GoalStateSpaceReplayModel(
        candidate_goals=np.array([[0.0, 0.0]]),
        initial_position_prior_weights=np.array([0.05, 0.05, 0.90]),
        reverse_terminal_position_prior_weights=np.array([0.05, 0.05, 0.90]),
        transition_sigma_cm_sqrt_s=0.5,
        drift_speed_cm_s=1.0,
        max_step_sigma=10.0,
        direction_mode='bidirectional',
    ).score(emissions, centers)
    toward_only = GoalStateSpaceReplayModel(
        candidate_goals=np.array([[0.0, 0.0]]),
        initial_position_prior_weights=np.array([0.05, 0.05, 0.90]),
        initial_position_prior_direction_mode='toward',
        reverse_terminal_position_prior_weights=np.array([0.05, 0.05, 0.90]),
        transition_sigma_cm_sqrt_s=0.5,
        drift_speed_cm_s=1.0,
        max_step_sigma=10.0,
        direction_mode='bidirectional',
    ).score(emissions, centers)

    assert toward_only.log_likelihood > all_components.log_likelihood
    assert (
        toward_only.diagnostics['goal_state_space_initial_position_prior_direction_mode']
        == 'toward'
    )
    assert toward_only.diagnostics['goal_state_space_initial_position_prior_max_factor'] > 1.0


def test_goal_state_space_reset_probability_improves_fragmented_jump():
    centers = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    log_likelihood = np.log(
        np.array(
            [
                [0.97, 0.01, 0.01, 0.01],
                [0.01, 0.01, 0.01, 0.97],
            ]
        )
    )
    emissions = LogEmissionTensor(
        log_likelihood=log_likelihood,
        spike_counts=np.zeros((2, 1), dtype=int),
        times=np.array([0.0, 1.0]),
        dt=1.0,
        cell_ids=np.array([1]),
        n_spikes=0,
    )
    base = GoalStateSpaceReplayModel(
        candidate_goals=np.array([[3.0, 0.0]]),
        transition_sigma_cm_sqrt_s=0.1,
        drift_speed_cm_s=0.0,
        max_step_sigma=1.0,
    ).score(emissions, centers)
    reset = GoalStateSpaceReplayModel(
        candidate_goals=np.array([[3.0, 0.0]]),
        transition_sigma_cm_sqrt_s=0.1,
        drift_speed_cm_s=0.0,
        max_step_sigma=1.0,
        reset_probability=0.5,
    ).score(emissions, centers)

    assert reset.log_likelihood > base.log_likelihood
    assert reset.diagnostics['goal_state_space_reset_probability'] == 0.5


def test_goal_state_space_reset_can_use_initial_position_prior():
    centers = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    log_likelihood = np.log(
        np.array(
            [
                [0.97, 0.01, 0.01, 0.01],
                [0.01, 0.01, 0.01, 0.97],
                [0.97, 0.01, 0.01, 0.01],
            ]
        )
    )
    emissions = LogEmissionTensor(
        log_likelihood=log_likelihood,
        spike_counts=np.zeros((3, 1), dtype=int),
        times=np.array([0.0, 1.0, 2.0]),
        dt=1.0,
        cell_ids=np.array([1]),
        n_spikes=0,
    )
    uniform_reset = GoalStateSpaceReplayModel(
        candidate_goals=np.array([[3.0, 0.0]]),
        initial_position_prior_weights=np.array([1.0, 0.0, 0.0, 0.0]),
        transition_sigma_cm_sqrt_s=0.1,
        drift_speed_cm_s=3.0,
        max_step_sigma=1.0,
        reset_probability=0.5,
    ).score(emissions, centers)
    prior_reset = GoalStateSpaceReplayModel(
        candidate_goals=np.array([[3.0, 0.0]]),
        initial_position_prior_weights=np.array([1.0, 0.0, 0.0, 0.0]),
        transition_sigma_cm_sqrt_s=0.1,
        drift_speed_cm_s=3.0,
        max_step_sigma=1.0,
        reset_probability=0.5,
        reset_initial_position_prior_weight=1.0,
    ).score(emissions, centers)

    assert prior_reset.log_likelihood > uniform_reset.log_likelihood
    assert prior_reset.diagnostics['goal_state_space_reset_initial_position_prior_weight'] == 1.0
    assert prior_reset.diagnostics['goal_state_space_reset_position_prior_max_probability'] == 1.0


def test_goal_state_space_bidirectional_mode_improves_reverse_goal_sweep():
    centers = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    log_likelihood = np.log(
        np.array(
            [
                [0.97, 0.01, 0.01, 0.01],
                [0.01, 0.97, 0.01, 0.01],
                [0.01, 0.01, 0.97, 0.01],
                [0.01, 0.01, 0.01, 0.97],
            ]
        )
    )
    emissions = LogEmissionTensor(
        log_likelihood=log_likelihood,
        spike_counts=np.zeros((4, 1), dtype=int),
        times=np.array([0.0, 1.0, 2.0, 3.0]),
        dt=1.0,
        cell_ids=np.array([1]),
        n_spikes=0,
    )
    toward = GoalStateSpaceReplayModel(
        candidate_goals=np.array([[0.0, 0.0]]),
        transition_sigma_cm_sqrt_s=0.2,
        drift_speed_cm_s=1.0,
        max_step_sigma=4.0,
    ).score(emissions, centers)
    bidirectional = GoalStateSpaceReplayModel(
        candidate_goals=np.array([[0.0, 0.0]]),
        transition_sigma_cm_sqrt_s=0.2,
        drift_speed_cm_s=1.0,
        max_step_sigma=4.0,
        direction_mode='bidirectional',
    ).score(emissions, centers)

    assert bidirectional.log_likelihood > toward.log_likelihood
    assert bidirectional.diagnostics['goal_state_space_direction_mode'] == 'bidirectional'
    assert bidirectional.diagnostics['goal_state_space_components'] == 2


def test_goal_state_space_toward_direction_prior_rewards_forward_component():
    centers = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    log_likelihood = np.log(
        np.array(
            [
                [0.97, 0.01, 0.01, 0.01],
                [0.01, 0.97, 0.01, 0.01],
                [0.01, 0.01, 0.97, 0.01],
                [0.01, 0.01, 0.01, 0.97],
            ]
        )
    )
    emissions = LogEmissionTensor(
        log_likelihood=log_likelihood,
        spike_counts=np.zeros((4, 1), dtype=int),
        times=np.array([0.0, 1.0, 2.0, 3.0]),
        dt=1.0,
        cell_ids=np.array([1]),
        n_spikes=0,
    )
    default = GoalStateSpaceReplayModel(
        candidate_goals=np.array([[3.0, 0.0]]),
        transition_sigma_cm_sqrt_s=0.5,
        drift_speed_cm_s=1.0,
        max_step_sigma=10.0,
        direction_mode='bidirectional',
    ).score(emissions, centers)
    toward = GoalStateSpaceReplayModel(
        candidate_goals=np.array([[3.0, 0.0]]),
        transition_sigma_cm_sqrt_s=0.5,
        drift_speed_cm_s=1.0,
        max_step_sigma=10.0,
        direction_mode='bidirectional',
        toward_direction_prior_weight=0.9,
    ).score(emissions, centers)

    assert toward.log_likelihood > default.log_likelihood
    assert toward.diagnostics['goal_state_space_toward_direction_prior_weight'] == 0.9
    assert toward.diagnostics['goal_state_space_component_prior_entropy'] < default.diagnostics[
        'goal_state_space_component_prior_entropy'
    ]


def test_goal_state_space_initial_goal_prior_rewards_reverse_start_at_goal():
    centers = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    log_likelihood = np.log(
        np.array(
            [
                [0.97, 0.01, 0.01, 0.01],
                [0.01, 0.97, 0.01, 0.01],
                [0.01, 0.01, 0.97, 0.01],
                [0.01, 0.01, 0.01, 0.97],
            ]
        )
    )
    emissions = LogEmissionTensor(
        log_likelihood=log_likelihood,
        spike_counts=np.zeros((4, 1), dtype=int),
        times=np.array([0.0, 1.0, 2.0, 3.0]),
        dt=1.0,
        cell_ids=np.array([1]),
        n_spikes=0,
    )
    base = GoalStateSpaceReplayModel(
        candidate_goals=np.array([[0.0, 0.0]]),
        transition_sigma_cm_sqrt_s=0.2,
        drift_speed_cm_s=1.0,
        max_step_sigma=10.0,
        direction_mode='bidirectional',
    ).score(emissions, centers)
    disabled = GoalStateSpaceReplayModel(
        candidate_goals=np.array([[0.0, 0.0]]),
        transition_sigma_cm_sqrt_s=0.2,
        drift_speed_cm_s=1.0,
        max_step_sigma=10.0,
        direction_mode='bidirectional',
        initial_goal_prior_sigma_cm=0.5,
        initial_goal_prior_weight=0.0,
    ).score(emissions, centers)
    initial = GoalStateSpaceReplayModel(
        candidate_goals=np.array([[0.0, 0.0]]),
        transition_sigma_cm_sqrt_s=0.2,
        drift_speed_cm_s=1.0,
        max_step_sigma=10.0,
        direction_mode='bidirectional',
        initial_goal_prior_sigma_cm=0.5,
    ).score(emissions, centers)

    assert disabled.log_likelihood == pytest.approx(base.log_likelihood)
    assert disabled.diagnostics['goal_state_space_initial_goal_prior'] == 'disabled'
    assert initial.log_likelihood > base.log_likelihood
    assert initial.diagnostics['goal_state_space_initial_goal_prior'] == 'provided'
    assert initial.diagnostics['goal_state_space_initial_goal_prior_weight'] == 1.0
    assert initial.diagnostics['goal_state_space_initial_goal_prior_max_factor'] > 1.0


def test_goal_state_space_terminal_goal_prior_rewards_endpoint_at_goal():
    centers = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    log_likelihood = np.log(
        np.array(
            [
                [0.70, 0.20, 0.08, 0.02],
                [0.10, 0.20, 0.60, 0.10],
                [0.01, 0.01, 0.01, 0.97],
            ]
        )
    )
    emissions = LogEmissionTensor(
        log_likelihood=log_likelihood,
        spike_counts=np.zeros((3, 1), dtype=int),
        times=np.array([0.0, 1.0, 2.0]),
        dt=1.0,
        cell_ids=np.array([1]),
        n_spikes=0,
    )
    base = GoalStateSpaceReplayModel(
        candidate_goals=np.array([[3.0, 0.0]]),
        transition_sigma_cm_sqrt_s=1.0,
        drift_speed_cm_s=1.0,
        max_step_sigma=10.0,
    ).score(emissions, centers)
    disabled = GoalStateSpaceReplayModel(
        candidate_goals=np.array([[3.0, 0.0]]),
        transition_sigma_cm_sqrt_s=1.0,
        drift_speed_cm_s=1.0,
        max_step_sigma=10.0,
        terminal_goal_prior_sigma_cm=0.3,
        terminal_goal_prior_weight=0.0,
    ).score(emissions, centers)
    terminal = GoalStateSpaceReplayModel(
        candidate_goals=np.array([[3.0, 0.0]]),
        transition_sigma_cm_sqrt_s=1.0,
        drift_speed_cm_s=1.0,
        max_step_sigma=10.0,
        terminal_goal_prior_sigma_cm=0.3,
    ).score(emissions, centers)

    assert disabled.log_likelihood == pytest.approx(base.log_likelihood)
    assert disabled.diagnostics['goal_state_space_terminal_goal_prior'] == 'disabled'
    assert terminal.log_likelihood > base.log_likelihood
    assert terminal.diagnostics['goal_state_space_terminal_goal_prior'] == 'provided'
    assert terminal.diagnostics['goal_state_space_terminal_goal_prior_weight'] == 1.0
    assert terminal.diagnostics['goal_state_space_terminal_goal_prior_max_factor'] > 1.0


def test_goal_state_space_rejects_invalid_goal_prior():
    centers = np.array([[0.0, 0.0], [1.0, 0.0]])
    emissions = LogEmissionTensor(
        log_likelihood=np.log(np.full((1, 2), 0.5)),
        spike_counts=np.zeros((1, 1), dtype=int),
        times=np.array([0.0]),
        dt=1.0,
        cell_ids=np.array([1]),
        n_spikes=0,
    )
    model = GoalStateSpaceReplayModel(
        candidate_goals=np.array([[0.0, 0.0], [1.0, 0.0]]),
        goal_prior_weights=np.array([0.0, 0.0]),
    )

    with pytest.raises(ValueError, match='goal_prior_weights'):
        model.score(emissions, centers)


def test_goal_state_space_rejects_invalid_initial_position_prior():
    centers = np.array([[0.0, 0.0], [1.0, 0.0]])
    emissions = LogEmissionTensor(
        log_likelihood=np.log(np.full((1, 2), 0.5)),
        spike_counts=np.zeros((1, 1), dtype=int),
        times=np.array([0.0]),
        dt=1.0,
        cell_ids=np.array([1]),
        n_spikes=0,
    )
    model = GoalStateSpaceReplayModel(
        candidate_goals=np.array([[0.0, 0.0], [1.0, 0.0]]),
        initial_position_prior_weights=np.array([0.0, 0.0]),
    )

    with pytest.raises(ValueError, match='initial_position_prior_weights'):
        model.score(emissions, centers)


def test_goal_state_space_rejects_invalid_initial_position_prior_direction_mode():
    centers = np.array([[0.0, 0.0], [1.0, 0.0]])
    emissions = LogEmissionTensor(
        log_likelihood=np.log(np.full((1, 2), 0.5)),
        spike_counts=np.zeros((1, 1), dtype=int),
        times=np.array([0.0]),
        dt=1.0,
        cell_ids=np.array([1]),
        n_spikes=0,
    )
    model = GoalStateSpaceReplayModel(
        candidate_goals=np.array([[0.0, 0.0]]),
        initial_position_prior_direction_mode='sideways',
    )

    with pytest.raises(ValueError, match='initial_position_prior_direction_mode'):
        model.score(emissions, centers)


def test_goal_state_space_rejects_invalid_reverse_terminal_position_prior():
    centers = np.array([[0.0, 0.0], [1.0, 0.0]])
    emissions = LogEmissionTensor(
        log_likelihood=np.log(np.full((1, 2), 0.5)),
        spike_counts=np.zeros((1, 1), dtype=int),
        times=np.array([0.0]),
        dt=1.0,
        cell_ids=np.array([1]),
        n_spikes=0,
    )
    model = GoalStateSpaceReplayModel(
        candidate_goals=np.array([[0.0, 0.0]]),
        reverse_terminal_position_prior_weights=np.array([0.0, 0.0]),
    )

    with pytest.raises(ValueError, match='reverse_terminal_position_prior_weights'):
        model.score(emissions, centers)


def test_goal_state_space_rejects_invalid_reset_probability():
    centers = np.array([[0.0, 0.0], [1.0, 0.0]])
    emissions = LogEmissionTensor(
        log_likelihood=np.log(np.full((1, 2), 0.5)),
        spike_counts=np.zeros((1, 1), dtype=int),
        times=np.array([0.0]),
        dt=1.0,
        cell_ids=np.array([1]),
        n_spikes=0,
    )
    model = GoalStateSpaceReplayModel(
        candidate_goals=np.array([[0.0, 0.0]]),
        reset_probability=1.0,
    )

    with pytest.raises(ValueError, match='reset_probability'):
        model.score(emissions, centers)


def test_goal_state_space_rejects_invalid_lateral_sigma_scale():
    centers = np.array([[0.0, 0.0], [1.0, 0.0]])
    emissions = LogEmissionTensor(
        log_likelihood=np.log(np.full((1, 2), 0.5)),
        spike_counts=np.zeros((1, 1), dtype=int),
        times=np.array([0.0]),
        dt=1.0,
        cell_ids=np.array([1]),
        n_spikes=0,
    )
    model = GoalStateSpaceReplayModel(
        candidate_goals=np.array([[0.0, 0.0]]),
        lateral_sigma_scale=0.0,
    )

    with pytest.raises(ValueError, match='lateral_sigma_scale'):
        model.score(emissions, centers)


def test_goal_state_space_rejects_invalid_diffusion_mixture_weight():
    centers = np.array([[0.0, 0.0], [1.0, 0.0]])
    emissions = LogEmissionTensor(
        log_likelihood=np.log(np.full((1, 2), 0.5)),
        spike_counts=np.zeros((1, 1), dtype=int),
        times=np.array([0.0]),
        dt=1.0,
        cell_ids=np.array([1]),
        n_spikes=0,
    )
    model = GoalStateSpaceReplayModel(
        candidate_goals=np.array([[0.0, 0.0]]),
        diffusion_mixture_weight=1.1,
    )

    with pytest.raises(ValueError, match='diffusion_mixture_weight'):
        model.score(emissions, centers)


def test_goal_state_space_rejects_invalid_reset_initial_position_prior_weight():
    centers = np.array([[0.0, 0.0], [1.0, 0.0]])
    emissions = LogEmissionTensor(
        log_likelihood=np.log(np.full((1, 2), 0.5)),
        spike_counts=np.zeros((1, 1), dtype=int),
        times=np.array([0.0]),
        dt=1.0,
        cell_ids=np.array([1]),
        n_spikes=0,
    )
    model = GoalStateSpaceReplayModel(
        candidate_goals=np.array([[0.0, 0.0]]),
        reset_initial_position_prior_weight=1.1,
    )

    with pytest.raises(ValueError, match='reset_initial_position_prior_weight'):
        model.score(emissions, centers)


def test_goal_state_space_rejects_invalid_component_switch_probability():
    centers = np.array([[0.0, 0.0], [1.0, 0.0]])
    emissions = LogEmissionTensor(
        log_likelihood=np.log(np.full((1, 2), 0.5)),
        spike_counts=np.zeros((1, 1), dtype=int),
        times=np.array([0.0]),
        dt=1.0,
        cell_ids=np.array([1]),
        n_spikes=0,
    )
    model = GoalStateSpaceReplayModel(
        candidate_goals=np.array([[0.0, 0.0]]),
        component_switch_probability=1.1,
    )

    with pytest.raises(ValueError, match='component_switch_probability'):
        model.score(emissions, centers)


def test_goal_state_space_rejects_invalid_direction_mode():
    centers = np.array([[0.0, 0.0], [1.0, 0.0]])
    emissions = LogEmissionTensor(
        log_likelihood=np.log(np.full((1, 2), 0.5)),
        spike_counts=np.zeros((1, 1), dtype=int),
        times=np.array([0.0]),
        dt=1.0,
        cell_ids=np.array([1]),
        n_spikes=0,
    )
    model = GoalStateSpaceReplayModel(
        candidate_goals=np.array([[0.0, 0.0]]),
        direction_mode='sideways',
    )

    with pytest.raises(ValueError, match='direction_mode'):
        model.score(emissions, centers)


def test_goal_state_space_rejects_invalid_terminal_goal_prior_sigma():
    centers = np.array([[0.0, 0.0], [1.0, 0.0]])
    emissions = LogEmissionTensor(
        log_likelihood=np.log(np.full((1, 2), 0.5)),
        spike_counts=np.zeros((1, 1), dtype=int),
        times=np.array([0.0]),
        dt=1.0,
        cell_ids=np.array([1]),
        n_spikes=0,
    )
    model = GoalStateSpaceReplayModel(
        candidate_goals=np.array([[0.0, 0.0]]),
        terminal_goal_prior_sigma_cm=-1.0,
    )

    with pytest.raises(ValueError, match='terminal_goal_prior_sigma_cm'):
        model.score(emissions, centers)


def test_goal_state_space_rejects_invalid_terminal_goal_prior_weight():
    centers = np.array([[0.0, 0.0], [1.0, 0.0]])
    emissions = LogEmissionTensor(
        log_likelihood=np.log(np.full((1, 2), 0.5)),
        spike_counts=np.zeros((1, 1), dtype=int),
        times=np.array([0.0]),
        dt=1.0,
        cell_ids=np.array([1]),
        n_spikes=0,
    )
    model = GoalStateSpaceReplayModel(
        candidate_goals=np.array([[0.0, 0.0]]),
        terminal_goal_prior_weight=1.1,
    )

    with pytest.raises(ValueError, match='terminal_goal_prior_weight'):
        model.score(emissions, centers)


def test_goal_state_space_rejects_invalid_initial_goal_prior_sigma():
    centers = np.array([[0.0, 0.0], [1.0, 0.0]])
    emissions = LogEmissionTensor(
        log_likelihood=np.log(np.full((1, 2), 0.5)),
        spike_counts=np.zeros((1, 1), dtype=int),
        times=np.array([0.0]),
        dt=1.0,
        cell_ids=np.array([1]),
        n_spikes=0,
    )
    model = GoalStateSpaceReplayModel(
        candidate_goals=np.array([[0.0, 0.0]]),
        initial_goal_prior_sigma_cm=-1.0,
    )

    with pytest.raises(ValueError, match='initial_goal_prior_sigma_cm'):
        model.score(emissions, centers)


def test_goal_state_space_rejects_invalid_initial_goal_prior_weight():
    centers = np.array([[0.0, 0.0], [1.0, 0.0]])
    emissions = LogEmissionTensor(
        log_likelihood=np.log(np.full((1, 2), 0.5)),
        spike_counts=np.zeros((1, 1), dtype=int),
        times=np.array([0.0]),
        dt=1.0,
        cell_ids=np.array([1]),
        n_spikes=0,
    )
    model = GoalStateSpaceReplayModel(
        candidate_goals=np.array([[0.0, 0.0]]),
        initial_goal_prior_weight=1.1,
    )

    with pytest.raises(ValueError, match='initial_goal_prior_weight'):
        model.score(emissions, centers)


def test_goal_state_space_rejects_invalid_toward_direction_prior_weight():
    centers = np.array([[0.0, 0.0], [1.0, 0.0]])
    emissions = LogEmissionTensor(
        log_likelihood=np.log(np.full((1, 2), 0.5)),
        spike_counts=np.zeros((1, 1), dtype=int),
        times=np.array([0.0]),
        dt=1.0,
        cell_ids=np.array([1]),
        n_spikes=0,
    )
    model = GoalStateSpaceReplayModel(
        candidate_goals=np.array([[0.0, 0.0]]),
        direction_mode='bidirectional',
        toward_direction_prior_weight=1.1,
    )

    with pytest.raises(ValueError, match='toward_direction_prior_weight'):
        model.score(emissions, centers)


def test_goal_state_space_rejects_invalid_reverse_terminal_position_prior_weight():
    centers = np.array([[0.0, 0.0], [1.0, 0.0]])
    emissions = LogEmissionTensor(
        log_likelihood=np.log(np.full((1, 2), 0.5)),
        spike_counts=np.zeros((1, 1), dtype=int),
        times=np.array([0.0]),
        dt=1.0,
        cell_ids=np.array([1]),
        n_spikes=0,
    )
    model = GoalStateSpaceReplayModel(
        candidate_goals=np.array([[0.0, 0.0]]),
        reverse_terminal_position_prior_weight=1.1,
    )

    with pytest.raises(ValueError, match='reverse_terminal_position_prior_weight'):
        model.score(emissions, centers)


def test_benchmark_registry_includes_goal_state_space_models():
    config = BenchmarkConfig(
        models=(
            'sorted-spike-state-space-goal',
            'sorted-spike-state-space-goal-bidirectional',
            'sorted-spike-state-space-goal-forward-biased',
            'sorted-spike-state-space-goal-forward-biased-switching',
            'sorted-spike-state-space-goal-reverse-biased',
            'state-space-goal',
            'state-space-goal-bidirectional',
            'state-space-goal-forward-biased',
            'state-space-goal-forward-biased-switching',
            'state-space-goal-reverse-biased',
        ),
        goal_state_space_drift_speed_cm_s=123.0,
        goal_state_space_lateral_sigma_scale=0.4,
        goal_state_space_diffusion_mixture_weight=0.25,
        goal_state_space_reset_probability=0.05,
        goal_state_space_reset_initial_position_prior_weight=0.6,
        goal_state_space_component_switch_probability=0.07,
        goal_state_space_initial_position_prior_direction_mode='toward',
        goal_state_space_terminal_prior_sigma_cm=11.0,
        goal_state_space_terminal_goal_prior_weight=0.7,
        goal_state_space_initial_goal_prior_sigma_cm=13.0,
        goal_state_space_initial_goal_prior_weight=0.9,
        goal_state_space_toward_direction_prior_weight=0.8,
        goal_state_space_reverse_terminal_position_prior_weight=0.6,
    )

    models = _build_models(config, session=None)

    assert set(models) == {
        'sorted-spike-state-space-goal',
        'sorted-spike-state-space-goal-bidirectional',
        'sorted-spike-state-space-goal-forward-biased',
        'sorted-spike-state-space-goal-forward-biased-switching',
        'sorted-spike-state-space-goal-reverse-biased',
        'state-space-goal',
        'state-space-goal-bidirectional',
        'state-space-goal-forward-biased',
        'state-space-goal-forward-biased-switching',
        'state-space-goal-reverse-biased',
    }
    assert isinstance(models['sorted-spike-state-space-goal'], GoalStateSpaceReplayModel)
    assert models['sorted-spike-state-space-goal'].drift_speed_cm_s == 123.0
    assert models['sorted-spike-state-space-goal'].lateral_sigma_scale == 0.4
    assert models['sorted-spike-state-space-goal'].diffusion_mixture_weight == 0.25
    assert models['sorted-spike-state-space-goal'].reset_probability == 0.05
    assert models['sorted-spike-state-space-goal'].reset_initial_position_prior_weight == 0.6
    assert models['sorted-spike-state-space-goal'].component_switch_probability == 0.07
    assert models['sorted-spike-state-space-goal'].initial_position_prior_direction_mode == 'toward'
    assert models['sorted-spike-state-space-goal'].terminal_goal_prior_sigma_cm == 11.0
    assert models['sorted-spike-state-space-goal'].terminal_goal_prior_weight == 0.7
    assert models['sorted-spike-state-space-goal'].initial_goal_prior_sigma_cm == 13.0
    assert models['sorted-spike-state-space-goal'].initial_goal_prior_weight == 0.9
    assert models['sorted-spike-state-space-goal'].toward_direction_prior_weight == 0.8
    assert models['sorted-spike-state-space-goal'].reverse_terminal_position_prior_weight == 0.6
    assert models['sorted-spike-state-space-goal'].direction_mode == 'toward'
    assert models['sorted-spike-state-space-goal-bidirectional'].direction_mode == 'bidirectional'
    assert models['sorted-spike-state-space-goal-forward-biased'].direction_mode == 'bidirectional'
    assert models['sorted-spike-state-space-goal-forward-biased'].toward_direction_prior_weight == 0.9
    assert models['sorted-spike-state-space-goal-forward-biased-switching'].direction_mode == 'bidirectional'
    assert models['sorted-spike-state-space-goal-forward-biased-switching'].toward_direction_prior_weight == 0.9
    assert models['sorted-spike-state-space-goal-forward-biased-switching'].component_switch_probability == 0.03
    assert models['sorted-spike-state-space-goal-reverse-biased'].direction_mode == 'bidirectional'
    assert models['sorted-spike-state-space-goal-reverse-biased'].toward_direction_prior_weight == 0.1
    assert models['state-space-goal'].name == 'state-space-goal'
    assert models['state-space-goal-bidirectional'].name == 'state-space-goal-bidirectional'
    assert models['state-space-goal-forward-biased'].name == 'state-space-goal-forward-biased'
    assert models['state-space-goal-forward-biased-switching'].name == 'state-space-goal-forward-biased-switching'
    assert models['state-space-goal-reverse-biased'].name == 'state-space-goal-reverse-biased'


def test_goal_state_space_cli_kwargs_roundtrip():
    args = Namespace(
        goal_state_space_transition_sigma_cm_sqrt_s=77.0,
        goal_state_space_drift_speed_cm_s=321.0,
        goal_state_space_max_step_sigma=5.5,
    )

    assert _goal_state_space_kwargs(args) == {
        'goal_state_space_transition_sigma_cm_sqrt_s': 77.0,
        'goal_state_space_drift_speed_cm_s': 321.0,
        'goal_state_space_max_step_sigma': 5.5,
    }


def test_goal_state_space_ground_truth_reuses_score_table_metadata():
    scores = pd.DataFrame(
        {
            'goal_state_space_transition_sigma_cm_sqrt_s': [70.0, 70.0],
            'goal_state_space_drift_speed_cm_s': [250.0, 250.0],
            'goal_state_space_max_step_sigma': [3.5, 3.5],
        }
    )

    assert _goal_state_space_kwargs_for_scores(
        scores,
        transition_sigma_cm_sqrt_s=85.0,
        drift_speed_cm_s=400.0,
        max_step_sigma=4.0,
    ) == {
        'goal_state_space_transition_sigma_cm_sqrt_s': 70.0,
        'goal_state_space_drift_speed_cm_s': 250.0,
        'goal_state_space_max_step_sigma': 3.5,
    }

    assert _goal_state_space_kwargs_for_scores(
        pd.DataFrame(),
        transition_sigma_cm_sqrt_s=85.0,
        drift_speed_cm_s=400.0,
        max_step_sigma=4.0,
    ) == {
        'goal_state_space_transition_sigma_cm_sqrt_s': 85.0,
        'goal_state_space_drift_speed_cm_s': 400.0,
        'goal_state_space_max_step_sigma': 4.0,
    }
