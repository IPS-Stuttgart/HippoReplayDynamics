import itertools

import numpy as np
import pytest
from scipy.special import logsumexp

from hipporeplayimm.benchmarks import BenchmarkConfig, _build_models
from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.goal_state_space import GoalStateSpaceReplayModel, _goal_transition_matrix
from hipporeplayimm.well_route_state_space import WellRouteStateSpaceReplayModel


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
    assert score.diagnostics['goal_state_space_most_likely_goal_x'] == 3.0
    assert score.diagnostics['goal_state_space_most_likely_goal_probability'] > 0.5
    assert score.diagnostics['goal_state_space_evidence_support'] == 'exact_full_grid'


def test_goal_state_space_uses_transition_durations_from_times():
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
        times=np.array([0.0, 1.0, 3.0]),
        dt=1.0,
        cell_ids=np.array([1]),
        n_spikes=0,
    )
    goal = np.array([3.0, 0.0])
    model = GoalStateSpaceReplayModel(
        candidate_goals=goal[None, :],
        transition_sigma_cm_sqrt_s=1.0,
        drift_speed_cm_s=1.0,
        max_step_sigma=10.0,
    )

    score = model.score(emissions, centers)

    duration_aware_transitions = (
        _goal_transition_matrix(
            centers,
            goal,
            drift_step_cm=1.0,
            sigma_cm=1.0,
            max_step_sigma=10.0,
        ),
        _goal_transition_matrix(
            centers,
            goal,
            drift_step_cm=2.0,
            sigma_cm=np.sqrt(2.0),
            max_step_sigma=10.0,
        ),
    )
    constant_dt_transitions = (
        _goal_transition_matrix(
            centers,
            goal,
            drift_step_cm=1.0,
            sigma_cm=1.0,
            max_step_sigma=10.0,
        ),
        _goal_transition_matrix(
            centers,
            goal,
            drift_step_cm=1.0,
            sigma_cm=1.0,
            max_step_sigma=10.0,
        ),
    )

    duration_aware_logp = _bruteforce_single_goal_log_evidence(
        log_likelihood,
        duration_aware_transitions,
    )
    constant_dt_logp = _bruteforce_single_goal_log_evidence(
        log_likelihood,
        constant_dt_transitions,
    )

    assert np.allclose(score.log_likelihood, duration_aware_logp)
    assert not np.allclose(score.log_likelihood, constant_dt_logp)
    assert score.diagnostics['goal_state_space_transition_durations'] == '1,2'
    assert score.diagnostics['goal_state_space_drift_step_cm_per_step'] == '1,2'


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


def test_goal_transition_matrix_normalizes_when_nearest_weight_underflows():
    centers = np.array([[0.0, 0.0], [1_000_000.0, 0.0]])

    transition = _goal_transition_matrix(
        centers,
        np.array([1_000_000.0, 0.0]),
        drift_step_cm=500_000.0,
        sigma_cm=1e-6,
        max_step_sigma=1.0,
    ).toarray()

    assert np.all(np.isfinite(transition))
    assert np.all(transition >= 0.0)
    assert np.allclose(transition.sum(axis=0), 1.0)


def test_goal_transition_matrix_rejects_nonfinite_bin_centers():
    centers = np.array([[0.0, 0.0], [np.nan, 0.0]])

    with pytest.raises(ValueError, match='bin_centers must be finite'):
        _goal_transition_matrix(
            centers,
            np.array([0.0, 0.0]),
            drift_step_cm=0.0,
            sigma_cm=1.0,
            max_step_sigma=4.0,
        )


def test_well_route_default_routes_support_one_dimensional_bin_centers():
    centers = np.array([[0.0], [1.0], [2.0], [3.0]])
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

    score = WellRouteStateSpaceReplayModel(
        transition_sigma_cm_sqrt_s=1.0,
        drift_speed_cm_s=1.0,
        max_step_sigma=10.0,
        max_default_points=3,
    ).score(emissions, centers)

    assert np.isfinite(score.log_likelihood)
    assert score.trajectory_log_posterior.shape == emissions.log_likelihood.shape
    assert np.allclose(logsumexp(score.terminal_log_posterior), 0.0)
    assert score.diagnostics['route_state_space_candidate_routes'] == 6
    assert score.diagnostics['decoded_endpoint_y'] == 0.0


def test_benchmark_registry_includes_goal_state_space_models():
    config = BenchmarkConfig(
        models=('sorted-spike-state-space-goal', 'state-space-goal'),
        goal_state_space_drift_speed_cm_s=123.0,
    )

    models = _build_models(config, session=None)

    assert set(models) == {'sorted-spike-state-space-goal', 'state-space-goal'}
    assert isinstance(models['sorted-spike-state-space-goal'], GoalStateSpaceReplayModel)
    assert models['sorted-spike-state-space-goal'].drift_speed_cm_s == 123.0
    assert models['state-space-goal'].name == 'state-space-goal'


def _bruteforce_single_goal_log_evidence(
    log_likelihood: np.ndarray,
    transitions,
) -> float:
    brute_terms = []
    n_time, n_bins = log_likelihood.shape
    for path in itertools.product(range(n_bins), repeat=n_time):
        logp = -np.log(n_bins) + log_likelihood[0, path[0]]
        for time_index in range(1, n_time):
            transition = transitions[time_index - 1]
            logp += np.log(transition[path[time_index], path[time_index - 1]])
            logp += log_likelihood[time_index, path[time_index]]
        brute_terms.append(logp)
    return float(logsumexp(brute_terms))
