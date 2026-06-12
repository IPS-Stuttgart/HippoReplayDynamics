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
    ("kwargs", "message"),
    [
        ({"transition_sigma_cm_sqrt_s": float("nan")}, "transition_sigma_cm_sqrt_s"),
        ({"drift_speed_cm_s": float("nan")}, "drift_speed_cm_s"),
        ({"max_step_sigma": float("inf")}, "max_step_sigma"),
    ],
)
def test_goal_state_space_model_rejects_nonfinite_parameters(kwargs, message):
    model = GoalStateSpaceReplayModel(candidate_goals=np.array([[1.0, 0.0]], dtype=float), **kwargs)

    with pytest.raises(ValueError, match=message):
        model.score(_minimal_emissions(), np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"sigma_cm": float("nan")}, "sigma_cm"),
        ({"drift_step_cm": float("nan")}, "drift_step_cm"),
        ({"max_step_sigma": float("inf")}, "max_step_sigma"),
    ],
)
def test_goal_transition_matrix_rejects_nonfinite_parameters(kwargs, message):
    params = {"sigma_cm": 1.0, "drift_step_cm": 0.5, "max_step_sigma": 10.0}
    params.update(kwargs)

    with pytest.raises(ValueError, match=message):
        _goal_transition_matrix(
            np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float),
            np.array([1.0, 0.0], dtype=float),
            **params,
        )
