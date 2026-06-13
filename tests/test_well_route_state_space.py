from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.well_route_state_space import WellRouteStateSpaceReplayModel


def _route_emissions() -> LogEmissionTensor:
    return LogEmissionTensor(
        log_likelihood=np.zeros((2, 2), dtype=float),
        spike_counts=np.empty((2, 0), dtype=int),
        times=np.array([0.0, 0.02], dtype=float),
        dt=0.02,
        cell_ids=np.empty(0, dtype=int),
        n_spikes=0,
    )


def _empty_route_emissions() -> LogEmissionTensor:
    return LogEmissionTensor(
        log_likelihood=np.empty((0, 2), dtype=float),
        spike_counts=np.empty((0, 0), dtype=int),
        times=np.empty(0, dtype=float),
        dt=0.02,
        cell_ids=np.empty(0, dtype=int),
        n_spikes=0,
    )


def _candidate_routes() -> np.ndarray:
    return np.array([[[0.0, 0.0], [1.0, 0.0]]], dtype=float)


def test_route_state_space_rejects_empty_emissions_with_clear_error() -> None:
    model = WellRouteStateSpaceReplayModel(candidate_routes=_candidate_routes())

    with pytest.raises(ValueError, match="at least one time bin"):
        model.score(
            _empty_route_emissions(),
            np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float),
        )


def test_route_state_space_rejects_mismatched_emission_grid() -> None:
    model = WellRouteStateSpaceReplayModel(candidate_routes=_candidate_routes())

    with pytest.raises(ValueError, match="emissions.n_bins"):
        model.score(
            _route_emissions(),
            np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=float),
        )


def test_route_state_space_rejects_one_dimensional_bin_centers() -> None:
    model = WellRouteStateSpaceReplayModel(candidate_routes=_candidate_routes())

    with pytest.raises(ValueError, match="bin_centers"):
        model.score(_route_emissions(), np.array([0.0, 1.0], dtype=float))


def test_route_state_space_rejects_invalid_dynamic_parameters() -> None:
    model = WellRouteStateSpaceReplayModel(
        candidate_routes=_candidate_routes(),
        transition_sigma_cm_sqrt_s=0.0,
    )

    with pytest.raises(ValueError, match="transition_sigma_cm_sqrt_s"):
        model.score(
            _route_emissions(),
            np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float),
        )
