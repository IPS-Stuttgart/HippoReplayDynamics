import numpy as np
from scipy.special import logsumexp

from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.goal_state_space import GoalStateSpaceReplayModel, _goal_transition_matrix


def _simple_emissions() -> LogEmissionTensor:
    return LogEmissionTensor(
        log_likelihood=np.log(
            np.array(
                [
                    [0.70, 0.20, 0.08, 0.02],
                    [0.10, 0.20, 0.60, 0.10],
                    [0.02, 0.08, 0.20, 0.70],
                ]
            )
        ),
        spike_counts=np.zeros((3, 1), dtype=int),
        times=np.array([0.0, 1.0, 2.0]),
        dt=1.0,
        cell_ids=np.array([1]),
        n_spikes=0,
    )


def test_goal_state_space_accepts_vector_bin_centers_for_1d_tracks():
    emissions = _simple_emissions()
    centers = np.array([0.0, 1.0, 2.0, 3.0])

    score = GoalStateSpaceReplayModel(
        candidate_goals=np.array([0.0, 3.0]),
        transition_sigma_cm_sqrt_s=1.0,
        drift_speed_cm_s=1.0,
        max_step_sigma=10.0,
    ).score(emissions, centers)

    assert np.isfinite(score.log_likelihood)
    assert score.trajectory_log_posterior.shape == emissions.log_likelihood.shape
    assert np.allclose(logsumexp(score.terminal_log_posterior), 0.0)
    assert score.diagnostics["goal_state_space_candidate_goals"] == 2
    assert score.diagnostics["goal_state_space_most_likely_goal_x"] == 3.0
    assert score.diagnostics["goal_state_space_most_likely_goal_y"] == 0.0


def test_goal_transition_matrix_accepts_vector_bin_centers_for_1d_tracks():
    transition = _goal_transition_matrix(
        np.array([0.0, 1.0, 2.0]),
        2.0,
        drift_step_cm=1.0,
        sigma_cm=1.0,
        max_step_sigma=10.0,
    ).toarray()

    assert transition.shape == (3, 3)
    assert np.all(np.isfinite(transition))
    assert np.all(transition >= 0.0)
    assert np.allclose(transition.sum(axis=0), 1.0)
