from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from scripts.build_replay_behavior_route_primitives import (
    build_cross_validated_primitives,
    build_gate_summary,
    extract_fixed_length_subpaths,
    map_events_to_route_libraries,
    segment_well_to_well_routes,
    smooth_position_trace,
)


def _linear_position() -> np.ndarray:
    time_s = np.arange(0.0, 30.1, 0.1)
    return np.column_stack([time_s, time_s, np.zeros_like(time_s), np.ones_like(time_s)])


def test_smoothing_suppresses_single_frame_tracking_jump() -> None:
    time_s = np.arange(20, dtype=float) / 10.0
    x = time_s.copy()
    x[10] = 100.0
    raw = np.column_stack([time_s, x, np.zeros_like(x)])

    smoothed = smooth_position_trace(raw, median_window_s=0.3, gaussian_sigma_s=0.1)

    assert smoothed.shape == (20, 4)
    assert smoothed[10, 1] < 3.0
    assert np.all(np.isfinite(smoothed))


def test_well_fill_intervals_get_directed_route_labels_and_folds() -> None:
    wells = np.array([[0.0, 1], [10.0, 2], [20.0, 3], [30.0, 4]], dtype=float)
    routes, points = segment_well_to_well_routes(
        session_id="Rat1/Open1",
        position=_linear_position(),
        run_times=np.array([[0.0, 30.0]]),
        well_sequence=wells,
        n_folds=2,
        speed_threshold_cm_s=0.1,
        arrival_radius_cm=1.0,
        minimum_route_displacement_cm=5.0,
        minimum_route_samples=5,
        route_resample_points=11,
    )

    assert routes["route_index"].tolist() == [1, 2]
    assert routes["origin_well_id"].tolist() == [1, 2]
    assert routes["destination_well_id"].tolist() == [2, 3]
    assert routes["cv_fold"].tolist() == [1, 0]
    assert routes["interval_start_time_s"].tolist() == [10.0, 20.0]
    assert points.groupby("route_id").size().tolist() == [11, 11]


def test_cross_validated_primitives_exclude_all_subpaths_from_heldout_route() -> None:
    wells = np.array([[0.0, 1], [10.0, 2], [20.0, 3], [30.0, 4]], dtype=float)
    routes, points = segment_well_to_well_routes(
        session_id="Rat1/Open1",
        position=_linear_position(),
        run_times=np.array([[0.0, 30.0]]),
        well_sequence=wells,
        n_folds=2,
        speed_threshold_cm_s=0.1,
        arrival_radius_cm=1.0,
        minimum_route_displacement_cm=5.0,
        minimum_route_samples=5,
        route_resample_points=11,
    )
    candidates, paths = extract_fixed_length_subpaths(
        routes,
        points,
        primitive_length_cm=5.0,
        primitive_stride_cm=5.0,
        primitive_resample_points=5,
    )
    primitives, primitive_points = build_cross_validated_primitives(
        candidates,
        paths,
        n_folds=2,
        cluster_threshold_cm=2.0,
        minimum_cluster_routes=1,
    )

    assert not primitives.empty
    assert not primitive_points.empty
    for row in routes.itertuples(index=False):
        library = primitives[primitives["excluded_cv_fold"].eq(row.cv_fold)]
        source_ids = "|".join(library["source_route_ids"].astype(str))
        assert row.route_id not in source_ids


@dataclass(frozen=True)
class _Ripple:
    peak: float


class _Session:
    def __init__(self, peak: float) -> None:
        self._peak = peak

    def ripple(self, _: int) -> _Ripple:
        return _Ripple(self._peak)


def test_event_maps_to_next_movement_and_never_uses_its_route_in_library() -> None:
    routes = pd.DataFrame(
        [
            {
                "session": "Rat1/Open1",
                "rat": "Rat1",
                "route_id": "r1",
                "route_index": 1,
                "cv_fold": 1,
                "interval_start_time_s": 10.0,
                "interval_end_time_s": 20.0,
                "movement_start_time_s": 11.0,
                "movement_end_time_s": 17.0,
                "origin_well_id": 1,
                "destination_well_id": 2,
                "duration_s": 10.0,
                "movement_duration_s": 6.0,
            },
            {
                "session": "Rat1/Open1",
                "rat": "Rat1",
                "route_id": "r2",
                "route_index": 2,
                "cv_fold": 0,
                "interval_start_time_s": 20.0,
                "interval_end_time_s": 30.0,
                "movement_start_time_s": 21.0,
                "movement_end_time_s": 27.0,
                "origin_well_id": 2,
                "destination_well_id": 3,
                "duration_s": 10.0,
                "movement_duration_s": 6.0,
            },
        ]
    )
    route_points = pd.DataFrame(
        [
            {"route_id": "r2", "time_s": 21.0, "x_cm": 0.0, "y_cm": 0.0},
            {"route_id": "r2", "time_s": 27.0, "x_cm": 10.0, "y_cm": 0.0},
        ]
    )
    primitives = pd.DataFrame(
        [
            {
                "session": "Rat1/Open1",
                "excluded_cv_fold": 0,
                "primitive_id": "p1",
                "source_route_ids": "r1",
            }
        ]
    )
    frozen = pd.DataFrame([{"session": "Rat1/Open1", "event_index": 0}])

    mapped = map_events_to_route_libraries(
        frozen,
        routes,
        primitives,
        route_points,
        session_loader=lambda _: _Session(18.0),
    )

    row = mapped.iloc[0]
    assert row["event_route_relation"] == "next_movement"
    assert row["enclosing_route_id"] == "r2"
    assert bool(row["route_library_eligible"])
    assert not bool(row["enclosing_route_leaked_into_library"])


def test_gate_is_nonvacuous_and_detects_route_leakage() -> None:
    frozen = pd.DataFrame(
        [
            {"session": "Rat1/Open1", "event_index": 0},
            {"session": "Rat2/Open1", "event_index": 0},
            {"session": "Rat3/Open1", "event_index": 0},
            {"session": "Rat4/Open1", "event_index": 0},
        ]
    )
    routes = pd.DataFrame(
        [
            {"session": f"Rat{rat}/Open1", "cv_fold": fold}
            for rat in range(1, 5)
            for fold in range(2)
        ]
    )
    primitives = pd.DataFrame([{"session": "Rat1/Open1"}])
    eligibility = frozen.copy()
    eligibility["route_library_eligible"] = True
    eligibility["candidate_primitives"] = 3
    eligibility["enclosing_route_leaked_into_library"] = False
    eligibility.loc[0, "enclosing_route_leaked_into_library"] = True

    gates = build_gate_summary(
        frozen,
        routes,
        primitives,
        eligibility,
        minimum_routes_per_session=2,
        minimum_candidate_primitives=3,
    ).set_index("gate")

    assert not bool(gates.loc["no_enclosing_route_leakage", "passed"])
    assert not bool(gates.loc["overall", "passed"])
