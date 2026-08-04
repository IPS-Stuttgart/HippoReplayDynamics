from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.test_replay_commitment_composition_boundary_null import (
    circular_dwell_signature,
    circular_switch_count,
    evaluate_mode_segmentation,
    shifted_mode_timeline,
)


def test_circular_shift_preserves_mode_counts_switches_and_dwells() -> None:
    modes = np.array([1, 1, 1, 0, 0, 2, 2, 1, 1])

    shifted = shifted_mode_timeline(modes, 4)

    np.testing.assert_array_equal(
        np.bincount(shifted, minlength=3),
        np.bincount(modes, minlength=3),
    )
    assert circular_switch_count(shifted) == circular_switch_count(modes)
    assert circular_dwell_signature(shifted) == circular_dwell_signature(modes)


def _path(points: list[tuple[float, float]]) -> np.ndarray:
    return np.asarray(points, dtype=float)


def test_mode_segmentation_uses_directed_well_pair_as_route_identity() -> None:
    event_bins = pd.DataFrame(
        {
            "time_bin": np.arange(8),
            "imm_posterior_mean_x_cm": [0, 5, 10, 10, 0, 0, 0, 0],
            "imm_posterior_mean_y_cm": [0, 0, 0, 0, 0, 5, 10, 10],
        }
    )
    modes = np.array([1, 1, 1, 0, 1, 1, 1, 0])
    primitive_paths = {
        "horizontal": _path([(0, 0), (10, 0)]),
        "vertical": _path([(0, 0), (0, 10)]),
    }
    route_paths = {
        "route_h": _path([(0, 0), (10, 0)]),
        "route_v": _path([(0, 0), (0, 10)]),
    }
    routes = pd.DataFrame(
        [
            {"route_id": "route_h", "origin_well_id": 1, "destination_well_id": 2},
            {"route_id": "route_v", "origin_well_id": 1, "destination_well_id": 3},
        ]
    ).set_index("route_id", drop=False)

    result = evaluate_mode_segmentation(
        event_bins,
        modes,
        primitive_ids=list(primitive_paths),
        primitive_paths=primitive_paths,
        route_ids=list(route_paths),
        route_paths=route_paths,
        routes_by_id=routes,
        minimum_bout_bins=3,
        minimum_bout_path_cm=5.0,
    )

    assert result["composition_evaluable"]
    assert result["eligible_continuous_bout_count"] == 2
    assert result["distinct_route_classes"] == 2
    assert result["route_identity_changes"] == 1
    assert result["switch_alignment"] == 1.0


def test_route_identity_does_not_change_between_repeated_route_traversals() -> None:
    event_bins = pd.DataFrame(
        {
            "time_bin": np.arange(8),
            "imm_posterior_mean_x_cm": [0, 5, 10, 10, 1, 6, 11, 11],
            "imm_posterior_mean_y_cm": [0, 0, 0, 0, 1, 1, 1, 1],
        }
    )
    modes = np.array([1, 1, 1, 0, 1, 1, 1, 0])
    primitives = {
        "p1": _path([(0, 0), (10, 0)]),
        "p2": _path([(1, 1), (11, 1)]),
    }
    routes = {
        "traversal_1": _path([(0, 0), (10, 0)]),
        "traversal_2": _path([(1, 1), (11, 1)]),
    }
    route_meta = pd.DataFrame(
        [
            {"route_id": route_id, "origin_well_id": 1, "destination_well_id": 2}
            for route_id in routes
        ]
    ).set_index("route_id", drop=False)

    result = evaluate_mode_segmentation(
        event_bins,
        modes,
        primitive_ids=list(primitives),
        primitive_paths=primitives,
        route_ids=list(routes),
        route_paths=routes,
        routes_by_id=route_meta,
        minimum_bout_bins=3,
        minimum_bout_path_cm=5.0,
    )

    assert result["eligible_continuous_bout_count"] == 2
    assert result["distinct_route_classes"] == 1
    assert result["route_identity_changes"] == 0
    assert result["switch_alignment"] == 0.0
