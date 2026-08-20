from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.goal_state_space import GoalStateSpaceReplayModel, _goal_transition_matrix


def _minimal_emissions() -> LogEmissionTensor:
    return LogEmissionTensor(
        log_likelihood=np.log(np.array([[0.6, 0.4], [0.4, 0.6]], dtype=float)),
        spike_counts=np.zeros((2, 1), dtype=int),
        times=np.array([0.0, 0.02], dtype=float),
        dt=0.02,
        cell_ids=np.array([1], dtype=int),
        n_spikes=0,
    )


@pytest.mark.parametrize(
    "bin_centers",
    [
        np.array([[False], [True]], dtype=bool),
        np.array([["0.0"], ["1.0"]]),
        np.array([[0.0 + 1.0j], [1.0 + 0.0j]]),
    ],
)
def test_goal_model_rejects_lossy_bin_center_coercions(bin_centers: np.ndarray) -> None:
    model = GoalStateSpaceReplayModel(
        candidate_goals=np.array([[0.0], [1.0]], dtype=float),
    )

    with pytest.raises(ValueError, match="bin_centers"):
        model.score(_minimal_emissions(), bin_centers)


@pytest.mark.parametrize(
    "candidate_goals",
    [
        np.array([[False], [True]], dtype=bool),
        np.array([["0.0"], ["1.0"]]),
        np.array([[0.0 + 1.0j], [1.0 + 0.0j]]),
    ],
)
def test_goal_model_rejects_lossy_candidate_goal_coercions(candidate_goals: np.ndarray) -> None:
    model = GoalStateSpaceReplayModel(candidate_goals=candidate_goals)

    with pytest.raises(ValueError, match="candidate_goals"):
        model.score(
            _minimal_emissions(),
            np.array([[0.0], [1.0]], dtype=float),
        )


@pytest.mark.parametrize(
    "goal",
    [
        np.array([True], dtype=bool),
        np.array(["1.0"]),
        np.array([1.0 + 2.0j]),
    ],
)
def test_goal_transition_rejects_lossy_goal_vector_coercions(goal: np.ndarray) -> None:
    with pytest.raises(ValueError, match="goal"):
        _goal_transition_matrix(
            np.array([[0.0], [1.0]], dtype=float),
            goal,
            drift_step_cm=0.5,
            sigma_cm=1.0,
            max_step_sigma=4.0,
        )


def test_goal_model_keeps_numeric_integer_coordinates_valid() -> None:
    model = GoalStateSpaceReplayModel(candidate_goals=np.array([[0], [1]], dtype=int))

    score = model.score(_minimal_emissions(), np.array([[0], [1]], dtype=int))

    assert np.isfinite(score.log_likelihood)
