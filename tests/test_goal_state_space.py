import itertools
from types import SimpleNamespace

import numpy as np
from scipy.special import logsumexp

from hipporeplayimm.benchmarks import BenchmarkConfig, _build_models
from hipporeplayimm.cli import _goal_state_space_kwargs
from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.goal_state_space import GoalStateSpaceReplayModel


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


def test_goal_state_space_cli_kwargs_reach_benchmark_model():
    args = SimpleNamespace(
        goal_state_space_transition_sigma_cm_sqrt_s=42.0,
        goal_state_space_drift_speed_cm_s=321.0,
        goal_state_space_max_step_sigma=7.0,
    )

    kwargs = _goal_state_space_kwargs(args)
    config = BenchmarkConfig(models=('state-space-goal',), **kwargs)
    model = _build_models(config, session=None)['state-space-goal']

    assert kwargs == {
        'goal_state_space_transition_sigma_cm_sqrt_s': 42.0,
        'goal_state_space_drift_speed_cm_s': 321.0,
        'goal_state_space_max_step_sigma': 7.0,
    }
    assert model.transition_sigma_cm_sqrt_s == 42.0
    assert model.drift_speed_cm_s == 321.0
    assert model.max_step_sigma == 7.0
