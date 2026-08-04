from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.compute_replay_commitment_composition_metrics import (
    _choice_statistics,
    _route_suffixes_near_start,
    best_path_fit,
    build_gate_summary,
    path_fit_distance_cm,
    previous_reward_arrival_time,
    shortest_transition_surprise,
)


def test_path_fit_is_direction_sensitive_in_physical_space() -> None:
    forward = np.array([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]])
    reverse = forward[::-1]
    offset = forward + np.array([0.0, 10.0])

    assert path_fit_distance_cm(forward, forward) == 0.0
    assert path_fit_distance_cm(forward, reverse) > 5.0
    assert path_fit_distance_cm(forward, offset) == 10.0


def test_best_path_fit_returns_nearest_behavior_template() -> None:
    query = np.array([[0.0, 0.0], [10.0, 0.0]])
    paths = {
        "near": np.array([[0.0, 1.0], [10.0, 1.0]]),
        "far": np.array([[0.0, 20.0], [10.0, 20.0]]),
    }

    path_id, distance = best_path_fit(query, ["far", "near"], paths)

    assert path_id == "near"
    assert distance == 1.0


def test_alternative_suffixes_match_current_location_and_different_goal() -> None:
    routes = pd.DataFrame(
        [
            {"route_id": "same_goal", "destination_well_id": 2},
            {"route_id": "alt_near", "destination_well_id": 3},
            {"route_id": "alt_far", "destination_well_id": 4},
        ]
    ).set_index("route_id", drop=False)
    paths = {
        "same_goal": np.array([[0.0, 0.0], [10.0, 0.0]]),
        "alt_near": np.array([[1.0, 0.0], [1.0, 10.0]]),
        "alt_far": np.array([[50.0, 0.0], [60.0, 0.0]]),
    }

    selected = _route_suffixes_near_start(
        actual_start=np.array([0.0, 0.0]),
        actual_destination_well=2,
        training_route_ids=list(paths),
        routes_by_id=routes,
        route_paths=paths,
        maximum_start_distance_cm=5.0,
        minimum_alternatives=1,
        fallback_alternatives=10,
    )

    assert [row[0] for row in selected] == ["alt_near"]


def test_choice_entropy_and_route_probability_use_training_fold_only() -> None:
    routes = pd.DataFrame(
        [
            {"session": "Rat1/Open1", "cv_fold": 0, "origin_well_id": 1, "destination_well_id": 2},
            {"session": "Rat1/Open1", "cv_fold": 1, "origin_well_id": 1, "destination_well_id": 2},
            {"session": "Rat1/Open1", "cv_fold": 2, "origin_well_id": 1, "destination_well_id": 3},
        ]
    )

    probability, entropy, count, total = _choice_statistics(
        routes,
        session="Rat1/Open1",
        excluded_fold=0,
        origin_well=1,
        destination_well=2,
    )

    assert probability == 0.5
    assert entropy == np.log(2.0)
    assert count == 1
    assert total == 2


def test_shortest_transition_surprise_accumulates_negative_log_probability() -> None:
    graph = {
        (0, 0): [((1, 0), 0.2), ((0, 1), 2.0)],
        (1, 0): [((2, 0), 0.3)],
        (0, 1): [((2, 0), 2.0)],
    }

    assert shortest_transition_surprise(graph, (0, 0), (2, 0)) == 0.5
    assert np.isnan(shortest_transition_surprise(graph, (2, 0), (0, 0)))


def test_elapsed_reward_control_uses_previous_arrival_not_future_route_start() -> None:
    routes = pd.DataFrame(
        [
            {
                "session": "Rat1/Open1",
                "movement_end_time_s": 90.0,
                "interval_start_time_s": 80.0,
            },
            {
                "session": "Rat1/Open1",
                "movement_end_time_s": 130.0,
                "interval_start_time_s": 110.0,
            },
        ]
    )

    arrival = previous_reward_arrival_time(
        routes,
        session="Rat1/Open1",
        event_peak_s=100.0,
    )

    assert arrival == 90.0
    assert 100.0 - arrival == 10.0


def test_metric_gate_fails_when_outcome_cohorts_are_empty() -> None:
    frozen = pd.DataFrame([{"session": "Rat1/Open1", "event_index": 1}])
    events = pd.DataFrame(
        [
            {
                "session": "Rat1/Open1",
                "rat": "Rat1",
                "composition_evaluable": False,
                "future_commitment_index": np.nan,
            }
        ]
    )

    gates = build_gate_summary(events, frozen).set_index("gate")

    assert not bool(gates.loc["composition_cohort_present", "passed"])
    assert not bool(gates.loc["commitment_cohort_present", "passed"])
    assert not bool(gates.loc["overall", "passed"])


def test_metric_gate_requires_decoder_error_for_each_session() -> None:
    frozen = pd.DataFrame(
        [
            {"session": "Rat1/Open1", "event_index": 1},
            {"session": "Rat2/Open1", "event_index": 2},
        ]
    )
    events = pd.DataFrame(
        [
            {
                "session": "Rat1/Open1",
                "rat": "Rat1",
                "composition_evaluable": True,
                "future_commitment_index": 1.0,
                "run_decoder_error_cm": 10.0,
            },
            {
                "session": "Rat2/Open1",
                "rat": "Rat2",
                "composition_evaluable": True,
                "future_commitment_index": 1.0,
                "run_decoder_error_cm": np.nan,
            },
        ]
    )

    gates = build_gate_summary(events, frozen).set_index("gate")

    assert not bool(
        gates.loc["run_decoder_error_available_for_all_sessions", "passed"]
    )
