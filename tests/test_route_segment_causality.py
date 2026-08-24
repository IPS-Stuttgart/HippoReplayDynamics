from __future__ import annotations

import numpy as np

from scripts.build_replay_behavior_route_primitives import (
    segment_well_to_well_routes,
)


def test_completed_route_smoothing_cannot_use_later_position_samples() -> None:
    times = np.linspace(0.0, 4.0, 401)
    x = np.piecewise(
        times,
        [times < 1.0, (times >= 1.0) & (times <= 2.0), times > 2.0],
        [0.0, lambda value: 80.0 * (value - 1.0), 80.0],
    )
    position = np.column_stack((times, x, np.zeros_like(times)))
    changed = position.copy()
    changed[times > 2.0, 1] = 10_000.0
    wells = np.asarray(
        [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 1.0]],
        dtype=float,
    )
    kwargs = {
        "session_id": "RatX/OpenX",
        "run_times": np.asarray([[0.0, 4.0]], dtype=float),
        "well_sequence": wells,
        "minimum_route_samples": 5,
        "speed_threshold_cm_s": 1.0,
        "minimum_departure_s": 0.02,
        "arrival_radius_cm": 3.0,
        "minimum_arrival_dwell_s": 0.02,
        "destination_window_s": 0.10,
        "minimum_route_displacement_cm": 10.0,
        "median_window_s": 0.167,
        "gaussian_sigma_s": 0.100,
    }

    routes, points = segment_well_to_well_routes(position=position, **kwargs)
    changed_routes, changed_points = segment_well_to_well_routes(
        position=changed,
        **kwargs,
    )

    assert len(routes) >= 1
    np.testing.assert_allclose(
        routes.select_dtypes(include=[np.number]).to_numpy(),
        changed_routes.select_dtypes(include=[np.number]).to_numpy(),
        atol=0.0,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        points[["time_s", "x_cm", "y_cm"]].to_numpy(),
        changed_points[["time_s", "x_cm", "y_cm"]].to_numpy(),
        atol=0.0,
        rtol=0.0,
    )
