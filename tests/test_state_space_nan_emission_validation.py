import numpy as np
import pytest

from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.goal_state_space import GoalStateSpaceReplayModel
from hipporeplayimm.well_route_state_space import WellRouteStateSpaceReplayModel


def _nan_emissions() -> LogEmissionTensor:
    return LogEmissionTensor(
        log_likelihood=np.array([[0.0, np.nan], [0.0, 0.0]], dtype=float),
        spike_counts=np.empty((2, 0), dtype=int),
        times=np.array([0.0, 0.02], dtype=float),
        dt=0.02,
        cell_ids=np.empty(0, dtype=int),
        n_spikes=0,
    )


def _bin_centers() -> np.ndarray:
    return np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)


def test_goal_state_space_rejects_nan_emission_values() -> None:
    model = GoalStateSpaceReplayModel(candidate_goals=np.array([[1.0, 0.0]], dtype=float))

    with pytest.raises(ValueError, match="NaN"):
        model.score(_nan_emissions(), _bin_centers())


def test_route_state_space_rejects_nan_emission_values() -> None:
    routes = np.array([[[0.0, 0.0], [1.0, 0.0]]], dtype=float)
    model = WellRouteStateSpaceReplayModel(candidate_routes=routes)

    with pytest.raises(ValueError, match="NaN"):
        model.score(_nan_emissions(), _bin_centers())
