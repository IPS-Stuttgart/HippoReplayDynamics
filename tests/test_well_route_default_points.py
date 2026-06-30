from __future__ import annotations

import numpy as np

from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.well_route_state_space import WellRouteStateSpaceReplayModel, _farthest_point_subset


def test_default_route_farthest_subset_deduplicates_bin_centers() -> None:
    points = np.array([[0.0], [0.0], [1.0], [1.0], [2.0]])

    subset = _farthest_point_subset(points, max_points=5)

    assert subset.shape == (3, 1)
    assert np.allclose(subset[:, 0], [0.0, 1.0, 2.0])


def test_default_route_candidate_count_uses_unique_centers() -> None:
    centers = np.array([[0.0], [0.0], [1.0], [1.0], [2.0]])
    emissions = LogEmissionTensor(
        log_likelihood=np.log(np.full((1, centers.shape[0]), 1.0 / centers.shape[0])),
        spike_counts=np.zeros((1, 1), dtype=int),
        times=np.array([0.0]),
        dt=1.0,
        cell_ids=np.array([1]),
        n_spikes=0,
    )

    score = WellRouteStateSpaceReplayModel(max_default_points=10).score(emissions, centers)

    assert score.diagnostics["route_state_space_candidate_routes"] == 6
