from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.well_route_state_space import WellRouteStateSpaceReplayModel, routes_from_wells


def _emissions() -> LogEmissionTensor:
    return LogEmissionTensor(
        log_likelihood=np.zeros((2, 2), dtype=float),
        spike_counts=np.empty((2, 0), dtype=int),
        times=np.array([0.0, 0.02], dtype=float),
        dt=0.02,
        cell_ids=np.empty(0, dtype=int),
        n_spikes=0,
    )


def _centers() -> np.ndarray:
    return np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)


def _routes() -> np.ndarray:
    return np.array([[[0.0, 0.0], [1.0, 0.0]]], dtype=float)


@pytest.mark.parametrize(
    "bin_centers",
    [
        np.array([[False, False], [True, False]]),
        np.array([["0.0", "0.0"], ["1.0", "0.0"]]),
        np.array([[0.0 + 1.0j, 0.0], [1.0, 0.0]]),
    ],
)
def test_route_state_space_rejects_lossy_bin_center_coercions(bin_centers: np.ndarray) -> None:
    model = WellRouteStateSpaceReplayModel(candidate_routes=_routes())

    with pytest.raises(ValueError, match="bin_centers.*numeric real coordinates"):
        model.score(_emissions(), bin_centers)


@pytest.mark.parametrize(
    "candidate_routes",
    [
        np.array([[[False, False], [True, False]]]),
        np.array([[["0.0", "0.0"], ["1.0", "0.0"]]]),
        np.array([[[0.0 + 1.0j, 0.0], [1.0, 0.0]]]),
    ],
)
def test_route_state_space_rejects_lossy_candidate_route_coercions(candidate_routes: np.ndarray) -> None:
    model = WellRouteStateSpaceReplayModel(candidate_routes=candidate_routes)

    with pytest.raises(ValueError, match="candidate_routes.*numeric real coordinates"):
        model.score(_emissions(), _centers())


@pytest.mark.parametrize(
    "well_locations",
    [
        np.array([[False, False], [True, False]]),
        np.array([["0.0", "0.0"], ["1.0", "0.0"]]),
        np.array([[0.0 + 1.0j, 0.0], [1.0, 0.0]]),
    ],
)
def test_routes_from_wells_rejects_lossy_coordinate_coercions(well_locations: np.ndarray) -> None:
    with pytest.raises(ValueError, match="well_locations.*numeric real coordinates"):
        routes_from_wells(well_locations)
