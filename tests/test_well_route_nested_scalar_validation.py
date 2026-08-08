from __future__ import annotations

import warnings

import numpy as np
import pytest

from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.well_route_state_space import WellRouteStateSpaceReplayModel, routes_from_wells


def _object_scalar(value: object) -> np.ndarray:
    wrapped = np.empty((), dtype=object)
    wrapped[()] = value
    return wrapped


def _nested_object_scalar(value: object) -> np.ndarray:
    return _object_scalar(_object_scalar(value))


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
    ("parameter", "bad_value"),
    [
        ("transition_sigma_cm_sqrt_s", _nested_object_scalar(True)),
        ("drift_speed_cm_s", _nested_object_scalar(np.bool_(False))),
        ("max_step_sigma", _nested_object_scalar(np.complex128(4.0 + 2.0j))),
    ],
)
def test_route_state_space_rejects_nested_lossy_dynamic_scalars(
    parameter: str,
    bad_value: object,
) -> None:
    model = WellRouteStateSpaceReplayModel(candidate_routes=_routes(), **{parameter: bad_value})

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match=parameter):
            model.score(_emissions(), _centers())


def test_route_state_space_accepts_nested_real_dynamic_scalar() -> None:
    model = WellRouteStateSpaceReplayModel(
        candidate_routes=_routes(),
        max_step_sigma=_nested_object_scalar(4.0),
    )

    score = model.score(_emissions(), _centers())

    assert np.isfinite(score.log_likelihood)
    assert score.diagnostics["route_state_space_max_step_sigma"] == pytest.approx(4.0)


def _object_coordinate_array(value: object) -> np.ndarray:
    coordinates = np.empty((2, 2), dtype=object)
    coordinates[:] = 0.0
    coordinates[0, 0] = _nested_object_scalar(value)
    coordinates[1, 0] = 1.0
    return coordinates


def test_route_state_space_rejects_nested_complex_bin_center() -> None:
    model = WellRouteStateSpaceReplayModel(candidate_routes=_routes())

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="bin_centers.*numeric real coordinates"):
            model.score(_emissions(), _object_coordinate_array(np.complex128(1.0 + 2.0j)))


def test_routes_from_wells_rejects_nested_complex_coordinate() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="well_locations.*numeric real coordinates"):
            routes_from_wells(_object_coordinate_array(np.complex128(1.0 + 2.0j)))


def test_route_state_space_rejects_nested_non_scalar_coordinate() -> None:
    model = WellRouteStateSpaceReplayModel(candidate_routes=_routes())
    malformed = _object_coordinate_array(np.array([0.0]))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="bin_centers.*numeric real coordinates"):
            model.score(_emissions(), malformed)
