import numpy as np

from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.goal_state_space import GoalStateSpaceReplayModel


def _emissions(probabilities: np.ndarray) -> LogEmissionTensor:
    values = np.asarray(probabilities, dtype=float)
    return LogEmissionTensor(
        log_likelihood=np.log(values),
        spike_counts=np.zeros((values.shape[0], 1), dtype=int),
        times=np.arange(values.shape[0], dtype=float),
        dt=1.0,
        cell_ids=np.array([1]),
        n_spikes=0,
    )


def _model(candidate_goals: np.ndarray | None) -> GoalStateSpaceReplayModel:
    return GoalStateSpaceReplayModel(
        candidate_goals=candidate_goals,
        transition_sigma_cm_sqrt_s=1.0,
        drift_speed_cm_s=1.0,
        max_step_sigma=10.0,
    )


def test_duplicate_explicit_goals_do_not_change_goal_prior_or_evidence():
    centers = np.array([[0.0], [1.0], [2.0]])
    emissions = _emissions(
        np.array(
            [
                [0.80, 0.15, 0.05],
                [0.05, 0.15, 0.80],
            ]
        )
    )

    unique = _model(np.array([[0.0], [2.0]])).score(emissions, centers)
    duplicated = _model(np.array([[0.0], [2.0], [2.0]])).score(emissions, centers)

    assert duplicated.diagnostics['goal_state_space_candidate_goals'] == 2
    assert np.allclose(duplicated.log_likelihood, unique.log_likelihood)
    assert np.allclose(duplicated.terminal_log_posterior, unique.terminal_log_posterior)
    assert np.allclose(duplicated.trajectory_log_posterior, unique.trajectory_log_posterior)


def test_default_goal_selection_uses_unique_spatial_coordinates():
    centers = np.array([[0.0], [1.0], [1.0]])
    emissions = _emissions(
        np.array(
            [
                [0.70, 0.20, 0.10],
                [0.10, 0.30, 0.60],
            ]
        )
    )

    default = _model(None).score(emissions, centers)
    explicit_unique = _model(np.array([[0.0], [1.0]])).score(emissions, centers)

    assert default.diagnostics['goal_state_space_candidate_goals'] == 2
    assert np.allclose(default.log_likelihood, explicit_unique.log_likelihood)
    assert np.allclose(default.terminal_log_posterior, explicit_unique.terminal_log_posterior)
    assert np.allclose(default.trajectory_log_posterior, explicit_unique.trajectory_log_posterior)
