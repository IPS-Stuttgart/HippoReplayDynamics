from __future__ import annotations

import numpy as np
import pandas as pd

from hipporeplayimm.replay_spatial_export import (
    BehaviorFieldConfig,
    SPATIAL_CANDIDATE_NAMES,
    build_pre_replay_candidate_fields,
)


def _route_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    route_specs = [
        ("r0", 0, 1, 0.0, 1.0, 1.0),
        ("r1", 0, 2, 2.0, 3.0, -1.0),
        ("r2", 0, 1, 4.0, 5.0, 1.0),
        ("r3", 0, 2, 6.0, 7.0, -1.0),
        ("future", 0, 2, 20.0, 21.0, 20.0),
    ]
    routes = pd.DataFrame(
        [
            {
                "route_id": route_id,
                "origin_well_id": origin,
                "destination_well_id": destination,
                "movement_start_time_s": start,
                "movement_end_time_s": end,
                "interval_end_time_s": end,
            }
            for route_id, origin, destination, start, end, _ in route_specs
        ]
    )
    point_rows = []
    for route_id, _, _, start, end, terminal_y in route_specs:
        for point_index, fraction in enumerate(np.linspace(0.0, 1.0, 21)):
            if route_id in {"r2", "r3"} and fraction <= 0.5:
                y = 0.0
            else:
                y = terminal_y * fraction
            point_rows.append(
                {
                    "route_id": route_id,
                    "point_index": point_index,
                    "time_s": start + fraction * (end - start),
                    "x_cm": 10.0 * fraction,
                    "y_cm": y,
                    "arc_fraction": fraction,
                }
            )
    return routes, pd.DataFrame(point_rows)


def _coordinates() -> np.ndarray:
    x, y = np.meshgrid(
        np.linspace(0.0, 10.0, 9),
        np.linspace(-3.0, 3.0, 7),
    )
    return np.column_stack((x.ravel(), y.ravel()))


def test_behavior_fields_use_signed_filtering_to_smoothing_revision() -> None:
    routes, points = _route_tables()
    result = build_pre_replay_candidate_fields(
        routes,
        points,
        _coordinates(),
        event_start_s=10.0,
        current_location_xy=np.array([5.0, 0.0]),
        current_location_time_s=9.9,
        config=BehaviorFieldConfig(
            observation_sigma_cm=8.0,
            route_kernel_sigma_cm=1.0,
        ),
    )

    revision_index = SPATIAL_CANDIDATE_NAMES.index("smoothing_revision")
    assert result.revision_snippet_count >= 2
    assert result.revision_total_weight > 0.0
    assert result.available[revision_index]
    assert np.any(result.fields[revision_index] > 0.0)
    assert np.any(result.fields[revision_index] < 0.0)
    assert all(
        np.isclose(filtered.sum(), 1.0)
        for filtered in result.filtered_probabilities
    )
    assert all(
        np.isclose(smoothed.sum(), 1.0)
        for smoothed in result.smoothed_probabilities
    )
    assert np.all(result.snippet_end_s <= 10.0)


def test_future_routes_cannot_change_pre_replay_fields() -> None:
    routes, points = _route_tables()
    baseline = build_pre_replay_candidate_fields(
        routes[~routes["route_id"].eq("future")],
        points[~points["route_id"].eq("future")],
        _coordinates(),
        event_start_s=10.0,
        current_location_xy=np.array([5.0, 0.0]),
        current_location_time_s=9.9,
    )
    with_future = build_pre_replay_candidate_fields(
        routes,
        points,
        _coordinates(),
        event_start_s=10.0,
        current_location_xy=np.array([5.0, 0.0]),
        current_location_time_s=9.9,
    )

    np.testing.assert_allclose(with_future.fields, baseline.fields)
    np.testing.assert_array_equal(with_future.available, baseline.available)
    np.testing.assert_allclose(with_future.available_s, baseline.available_s)
    assert with_future.history_cutoff_s == baseline.history_cutoff_s


def test_route_is_unavailable_until_its_fill_interval_ends() -> None:
    routes, points = _route_tables()
    baseline = build_pre_replay_candidate_fields(
        routes,
        points,
        _coordinates(),
        event_start_s=10.0,
        current_location_xy=np.array([5.0, 0.0]),
        current_location_time_s=9.9,
    )
    straddling_route = pd.DataFrame(
        [
            {
                "route_id": "straddling",
                "origin_well_id": 0,
                "destination_well_id": 1,
                "movement_start_time_s": 8.0,
                "movement_end_time_s": 9.0,
                "interval_end_time_s": 11.0,
            }
        ]
    )
    straddling_points = pd.DataFrame(
        [
            {
                "route_id": "straddling",
                "point_index": point_index,
                "time_s": 8.0 + fraction,
                "x_cm": 10.0 * fraction,
                "y_cm": 1000.0 * fraction,
                "arc_fraction": fraction,
            }
            for point_index, fraction in enumerate(np.linspace(0.0, 1.0, 21))
        ]
    )
    with_straddling = build_pre_replay_candidate_fields(
        pd.concat([routes, straddling_route], ignore_index=True),
        pd.concat([points, straddling_points], ignore_index=True),
        _coordinates(),
        event_start_s=10.0,
        current_location_xy=np.array([5.0, 0.0]),
        current_location_time_s=9.9,
    )
    changed_points = straddling_points.copy()
    changed_points["y_cm"] *= -10_000.0
    with_changed_future = build_pre_replay_candidate_fields(
        pd.concat([routes, straddling_route], ignore_index=True),
        pd.concat([points, changed_points], ignore_index=True),
        _coordinates(),
        event_start_s=10.0,
        current_location_xy=np.array([5.0, 0.0]),
        current_location_time_s=9.9,
    )

    np.testing.assert_allclose(with_straddling.fields, baseline.fields)
    np.testing.assert_allclose(with_changed_future.fields, baseline.fields)
    np.testing.assert_array_equal(with_straddling.available, baseline.available)
    np.testing.assert_allclose(with_straddling.available_s, baseline.available_s)
    assert with_straddling.history_cutoff_s == baseline.history_cutoff_s


def test_behavior_field_hyperparameter_digest_is_stable() -> None:
    baseline = BehaviorFieldConfig()
    same = BehaviorFieldConfig()
    changed = BehaviorFieldConfig(observation_sigma_cm=12.0)

    assert baseline.sha256() == same.sha256()
    assert baseline.sha256() != changed.sha256()
