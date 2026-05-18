import numpy as np
from scipy.special import logsumexp

from hipporeplayimm.benchmarks import BenchmarkConfig, _build_models
from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.goal_state_space import GoalStateSpaceReplayModel


def _emissions(log_likelihood):
    values = np.asarray(log_likelihood, dtype=float)
    return LogEmissionTensor(
        log_likelihood=values,
        spike_counts=np.zeros((values.shape[0], 1), dtype=int),
        times=np.arange(values.shape[0], dtype=float),
        dt=1.0,
        cell_ids=np.array([1]),
        n_spikes=0,
    )


def test_goal_state_space_prefers_right_goal_on_rightward_event():
    centers = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    emissions = _emissions(
        np.log(
            [
                [0.70, 0.20, 0.08, 0.02],
                [0.10, 0.20, 0.60, 0.10],
                [0.02, 0.08, 0.20, 0.70],
            ]
        )
    )
    score = GoalStateSpaceReplayModel(
        candidate_goals=np.array([[0.0, 0.0], [3.0, 0.0]]),
        transition_sigma_cm_sqrt_s=1.0,
        drift_speed_cm_s=1.0,
        max_step_sigma=10.0,
    ).score(emissions, centers)

    assert np.isfinite(score.log_likelihood)
    assert score.trajectory_log_posterior.shape == emissions.log_likelihood.shape
    assert np.allclose(logsumexp(score.terminal_log_posterior), 0.0)
    assert score.diagnostics["goal_state_space_most_likely_goal_x"] == 3.0
    assert score.diagnostics["goal_state_space_most_likely_goal_probability"] > 0.5
    assert score.diagnostics["goal_state_space_evidence_support"] == "exact_full_grid"


def test_benchmark_registry_includes_goal_state_space_models():
    config = BenchmarkConfig(
        models=("sorted-spike-state-space-goal", "state-space-goal"),
        goal_state_space_drift_speed_cm_s=123.0,
    )

    models = _build_models(config, session=None)

    assert set(models) == {"sorted-spike-state-space-goal", "state-space-goal"}
    assert isinstance(models["sorted-spike-state-space-goal"], GoalStateSpaceReplayModel)
    assert models["sorted-spike-state-space-goal"].drift_speed_cm_s == 123.0
    assert models["state-space-goal"].name == "state-space-goal"
