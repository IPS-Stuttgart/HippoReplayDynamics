#!/usr/bin/env python3
"""Compute event-level route composition and future commitment metrics."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import datetime, timezone
import heapq
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _provenance import build_script_provenance  # noqa: E402
from build_replay_behavior_route_primitives import smooth_position_trace  # noqa: E402
from hipporeplayimm.data import load_replay_session  # noqa: E402


EVENT_OUTPUT = "replay_event_commitment_composition_metrics.csv"
BOUT_OUTPUT = "replay_event_commitment_composition_bouts.csv"
GATE_OUTPUT = "replay_event_commitment_composition_metrics_gate_summary.csv"
MANIFEST_OUTPUT = "replay_event_commitment_composition_metrics_manifest.json"
SUMMARY_OUTPUT = "replay_event_commitment_composition_metrics_summary.md"


def path_length(xy: np.ndarray) -> float:
    points = np.asarray(xy, dtype=float)
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()) if len(points) >= 2 else 0.0


def resample_path(xy: np.ndarray, *, n_points: int) -> np.ndarray:
    points = np.asarray(xy, dtype=float)
    points = points[np.isfinite(points).all(axis=1)] if points.ndim == 2 else np.empty((0, 2))
    if len(points) < 2 or int(n_points) < 2:
        raise ValueError("path resampling requires at least two source and output points")
    cumulative = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    )
    keep = np.concatenate([[True], np.diff(cumulative) > 1e-9])
    cumulative = cumulative[keep]
    points = points[keep]
    if len(points) < 2 or cumulative[-1] <= 0.0:
        raise ValueError("path has no nonzero spatial extent")
    targets = np.linspace(0.0, float(cumulative[-1]), int(n_points))
    return np.column_stack(
        [np.interp(targets, cumulative, points[:, dim]) for dim in range(2)]
    )


def path_fit_distance_cm(query: np.ndarray, candidate: np.ndarray) -> float:
    """Direction-preserving mean physical error after arc-length resampling."""

    n_points = max(3, min(41, max(len(query), len(candidate))))
    query_resampled = resample_path(query, n_points=n_points)
    candidate_resampled = resample_path(candidate, n_points=n_points)
    return float(np.linalg.norm(query_resampled - candidate_resampled, axis=1).mean())


def best_path_fit(
    query: np.ndarray,
    candidate_ids: Sequence[str],
    candidate_paths: dict[str, np.ndarray],
) -> tuple[str, float]:
    best_id = ""
    best_distance = np.inf
    for candidate_id in candidate_ids:
        try:
            distance = path_fit_distance_cm(query, candidate_paths[str(candidate_id)])
        except ValueError:
            continue
        if distance < best_distance:
            best_id = str(candidate_id)
            best_distance = float(distance)
    return best_id, float(best_distance) if np.isfinite(best_distance) else np.nan


def _path_dictionary(
    points: pd.DataFrame,
    *,
    id_column: str,
) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for path_id, group in points.groupby(id_column, sort=False):
        ordered = group.sort_values("point_index")
        output[str(path_id)] = ordered[["x_cm", "y_cm"]].to_numpy(dtype=float)
    return output


def _continuous_bouts(event_bins: pd.DataFrame) -> list[tuple[int, pd.DataFrame]]:
    continuous = event_bins[event_bins["continuous_bout_id"].ge(0)]
    return [
        (int(bout_id), group.sort_values("time_bin"))
        for bout_id, group in continuous.groupby("continuous_bout_id", sort=True)
    ]


def _training_route_ids(
    routes: pd.DataFrame,
    *,
    session: str,
    excluded_fold: int,
) -> list[str]:
    return routes[
        routes["session"].eq(session) & ~routes["cv_fold"].eq(int(excluded_fold))
    ]["route_id"].astype(str).tolist()


def _primitive_library_ids(
    primitives: pd.DataFrame,
    *,
    session: str,
    excluded_fold: int,
) -> list[str]:
    return primitives[
        primitives["session"].eq(session)
        & primitives["excluded_cv_fold"].eq(int(excluded_fold))
    ]["primitive_id"].astype(str).tolist()


def _position_path(
    smoothed_position: np.ndarray,
    *,
    start_s: float,
    end_s: float,
) -> np.ndarray:
    if not np.isfinite(start_s) or not np.isfinite(end_s) or end_s <= start_s:
        return np.empty((0, 2), dtype=float)
    segment = smoothed_position[
        (smoothed_position[:, 0] >= float(start_s))
        & (smoothed_position[:, 0] <= float(end_s))
    ]
    return segment[:, 1:3]


def _route_suffixes_near_start(
    *,
    actual_start: np.ndarray,
    actual_destination_well: int,
    training_route_ids: Sequence[str],
    routes_by_id: pd.DataFrame,
    route_paths: dict[str, np.ndarray],
    maximum_start_distance_cm: float,
    minimum_alternatives: int,
    fallback_alternatives: int,
) -> list[tuple[str, np.ndarray, float]]:
    candidates: list[tuple[str, np.ndarray, float, bool]] = []
    for route_id in training_route_ids:
        path = route_paths.get(str(route_id))
        if path is None or len(path) < 2:
            continue
        distances = np.linalg.norm(path - actual_start[None, :], axis=1)
        nearest = int(np.argmin(distances))
        suffix = path[nearest:]
        if len(suffix) < 2:
            continue
        row = routes_by_id.loc[str(route_id)]
        different_destination = int(row["destination_well_id"]) != int(actual_destination_well)
        candidates.append(
            (str(route_id), suffix, float(distances[nearest]), different_destination)
        )
    preferred = [
        (route_id, suffix, distance)
        for route_id, suffix, distance, different in candidates
        if different and distance <= float(maximum_start_distance_cm)
    ]
    if len(preferred) >= int(minimum_alternatives):
        return sorted(preferred, key=lambda item: (item[2], item[0]))
    fallback = [
        (route_id, suffix, distance)
        for route_id, suffix, distance, different in candidates
        if different
    ]
    return sorted(fallback, key=lambda item: (item[2], item[0]))[
        : int(fallback_alternatives)
    ]


def _choice_statistics(
    routes: pd.DataFrame,
    *,
    session: str,
    excluded_fold: int,
    origin_well: int,
    destination_well: int,
) -> tuple[float, float, int, int]:
    training = routes[
        routes["session"].eq(session)
        & ~routes["cv_fold"].eq(int(excluded_fold))
        & routes["origin_well_id"].eq(int(origin_well))
    ]
    if training.empty:
        return np.nan, np.nan, 0, 0
    counts = training["destination_well_id"].value_counts()
    probability = counts.to_numpy(dtype=float) / float(counts.sum())
    entropy = float(-np.sum(probability * np.log(probability)))
    route_count = int(counts.get(int(destination_well), 0))
    return float(route_count / counts.sum()), entropy, route_count, int(counts.sum())


def _graph_for_fold(
    transitions: pd.DataFrame,
    *,
    session: str,
    excluded_fold: int,
) -> tuple[dict[tuple[int, int], list[tuple[tuple[int, int], float]]], float, float, float]:
    graph_rows = transitions[
        transitions["session"].eq(session)
        & transitions["excluded_cv_fold"].eq(int(excluded_fold))
    ]
    adjacency: dict[tuple[int, int], list[tuple[tuple[int, int], float]]] = {}
    if graph_rows.empty:
        return adjacency, np.nan, np.nan, np.nan
    bin_cm = float(graph_rows["spatial_bin_cm"].iloc[0])
    first = graph_rows.iloc[0]
    x_origin = float(first["from_x_cm"] - (first["from_x_bin"] + 0.5) * bin_cm)
    y_origin = float(first["from_y_cm"] - (first["from_y_bin"] + 0.5) * bin_cm)
    for row in graph_rows.itertuples(index=False):
        source = (int(row.from_x_bin), int(row.from_y_bin))
        target = (int(row.to_x_bin), int(row.to_y_bin))
        adjacency.setdefault(source, []).append(
            (target, float(row.transition_surprise_nats))
        )
    return adjacency, x_origin, y_origin, bin_cm


def _coordinate_bin(
    xy: np.ndarray,
    *,
    x_origin: float,
    y_origin: float,
    bin_cm: float,
) -> tuple[int, int]:
    return (
        int(np.floor((float(xy[0]) - x_origin) / bin_cm)),
        int(np.floor((float(xy[1]) - y_origin) / bin_cm)),
    )


def shortest_transition_surprise(
    adjacency: dict[tuple[int, int], list[tuple[tuple[int, int], float]]],
    source: tuple[int, int],
    target: tuple[int, int],
) -> float:
    if source == target:
        return 0.0
    queue: list[tuple[float, tuple[int, int]]] = [(0.0, source)]
    best = {source: 0.0}
    while queue:
        distance, node = heapq.heappop(queue)
        if node == target:
            return float(distance)
        if distance > best.get(node, np.inf):
            continue
        for neighbor, cost in adjacency.get(node, []):
            proposed = float(distance + cost)
            if proposed < best.get(neighbor, np.inf):
                best[neighbor] = proposed
                heapq.heappush(queue, (proposed, neighbor))
    return np.nan


def previous_reward_arrival_time(
    routes: pd.DataFrame,
    *,
    session: str,
    event_peak_s: float,
) -> float:
    """Return the latest sustained route arrival at or before an event."""

    arrivals = routes[
        routes["session"].eq(str(session))
        & (pd.to_numeric(routes["movement_end_time_s"], errors="coerce") <= float(event_peak_s))
    ]
    return (
        float(pd.to_numeric(arrivals["movement_end_time_s"], errors="coerce").max())
        if not arrivals.empty
        else np.nan
    )


def compute_metrics(
    *,
    dataset_root: str | Path,
    frozen_events: pd.DataFrame,
    posterior_bins: pd.DataFrame,
    posterior_events: pd.DataFrame,
    routes: pd.DataFrame,
    route_points: pd.DataFrame,
    primitives: pd.DataFrame,
    primitive_points: pd.DataFrame,
    transitions: pd.DataFrame,
    eligibility: pd.DataFrame,
    minimum_bout_bins: int = 3,
    minimum_bout_path_cm: float = 10.0,
    maximum_alternative_start_distance_cm: float = 30.0,
    minimum_alternatives: int = 3,
    fallback_alternatives: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    route_paths = _path_dictionary(route_points, id_column="route_id")
    primitive_paths = _path_dictionary(primitive_points, id_column="primitive_id")
    primitive_meta = primitives.set_index("primitive_id", drop=False)
    routes_by_id = routes.set_index("route_id", drop=False)
    eligibility_by_event = eligibility.set_index(["session", "event_index"], drop=False)
    posterior_event_lookup = posterior_events.set_index(["session", "event_index"], drop=False)
    model_lookup = frozen_events.set_index(["session", "event_index"], drop=False)
    dataset = Path(dataset_root)
    event_rows: list[dict[str, object]] = []
    bout_rows: list[dict[str, object]] = []
    for session_id, session_bins in posterior_bins.groupby("session", sort=True):
        session = load_replay_session(dataset / Path(str(session_id)))
        smoothed_position = smooth_position_trace(np.asarray(session.position))
        for event_index, event_bins in session_bins.groupby("event_index", sort=True):
            key = (str(session_id), int(event_index))
            model_row = model_lookup.loc[key]
            posterior_row = posterior_event_lookup.loc[key]
            route_context = eligibility_by_event.loc[key]
            excluded_fold = int(route_context["excluded_cv_fold"])
            primitive_ids = _primitive_library_ids(
                primitives,
                session=str(session_id),
                excluded_fold=excluded_fold,
            )
            training_route_ids = _training_route_ids(
                routes,
                session=str(session_id),
                excluded_fold=excluded_fold,
            )
            event_bins = event_bins.sort_values("time_bin")
            event_path = event_bins[
                ["imm_posterior_mean_x_cm", "imm_posterior_mean_y_cm"]
            ].to_numpy(dtype=float)
            emission_path = event_bins[
                ["emission_only_mean_x_cm", "emission_only_mean_y_cm"]
            ].to_numpy(dtype=float)
            valid_bout_rows: list[dict[str, object]] = []
            for bout_id, bout in _continuous_bouts(event_bins):
                bout_path = bout[
                    ["imm_posterior_mean_x_cm", "imm_posterior_mean_y_cm"]
                ].to_numpy(dtype=float)
                length = path_length(bout_path)
                displacement = (
                    float(np.linalg.norm(bout_path[-1] - bout_path[0]))
                    if len(bout_path) >= 2
                    else 0.0
                )
                eligible_bout = bool(
                    len(bout) >= int(minimum_bout_bins)
                    and length >= float(minimum_bout_path_cm)
                )
                best_primitive_id = ""
                best_distance = np.nan
                best_behavior_route_id = ""
                best_behavior_route_class = ""
                best_behavior_route_distance = np.nan
                if eligible_bout:
                    best_primitive_id, best_distance = best_path_fit(
                        bout_path,
                        primitive_ids,
                        primitive_paths,
                    )
                    best_behavior_route_id, best_behavior_route_distance = best_path_fit(
                        bout_path,
                        training_route_ids,
                        route_paths,
                    )
                    if best_behavior_route_id:
                        best_behavior_route = routes_by_id.loc[best_behavior_route_id]
                        best_behavior_route_class = (
                            f"{int(best_behavior_route['origin_well_id'])}->"
                            f"{int(best_behavior_route['destination_well_id'])}"
                        )
                row = {
                    "session": str(session_id),
                    "rat": str(session_id).split("/", 1)[0],
                    "event_index": int(event_index),
                    "continuous_bout_id": int(bout_id),
                    "start_time_bin": int(bout["time_bin"].min()),
                    "end_time_bin": int(bout["time_bin"].max()),
                    "start_time_s": float(bout["time_s"].min()),
                    "end_time_s": float(bout["time_s"].max()),
                    "n_bins": int(len(bout)),
                    "path_length_cm": length,
                    "net_displacement_cm": displacement,
                    "bout_eligible": eligible_bout,
                    "best_primitive_id": best_primitive_id,
                    "best_primitive_distance_cm": best_distance,
                    "best_primitive_cluster_size_routes": int(
                        primitive_meta.loc[best_primitive_id, "cluster_size_routes"]
                    ) if best_primitive_id else 0,
                    "best_behavior_route_id": best_behavior_route_id,
                    "best_behavior_route_class": best_behavior_route_class,
                    "best_behavior_route_distance_cm": best_behavior_route_distance,
                }
                bout_rows.append(row)
                if eligible_bout and best_primitive_id:
                    row["start_xy"] = bout_path[0]
                    row["end_xy"] = bout_path[-1]
                    valid_bout_rows.append(row)

            continuous_bins = event_bins[event_bins["continuous_diffusion_bin"]]
            continuous_path = continuous_bins[
                ["imm_posterior_mean_x_cm", "imm_posterior_mean_y_cm"]
            ].to_numpy(dtype=float)
            best_route_id = ""
            best_route_distance = np.nan
            if len(continuous_path) >= 2 and path_length(continuous_path) >= minimum_bout_path_cm:
                best_route_id, best_route_distance = best_path_fit(
                    continuous_path,
                    training_route_ids,
                    route_paths,
                )
            bout_distance = (
                float(
                    np.average(
                        [row["best_primitive_distance_cm"] for row in valid_bout_rows],
                        weights=[row["n_bins"] for row in valid_bout_rows],
                    )
                )
                if valid_bout_rows
                else np.nan
            )
            composition_evaluable = bool(
                len(valid_bout_rows) >= 2
                and np.isfinite(bout_distance)
                and np.isfinite(best_route_distance)
            )
            composition_index = (
                float(best_route_distance - bout_distance)
                if composition_evaluable
                else np.nan
            )
            primitive_sequence = [str(row["best_primitive_id"]) for row in valid_bout_rows]
            route_sequence = [
                str(row["best_behavior_route_class"]) for row in valid_bout_rows
            ]
            primitive_identity_changes = int(
                sum(
                    left != right
                    for left, right in zip(
                        primitive_sequence[:-1], primitive_sequence[1:], strict=True
                    )
                )
            )
            route_identity_changes = int(
                sum(
                    left != right
                    for left, right in zip(
                        route_sequence[:-1], route_sequence[1:], strict=True
                    )
                )
            )

            adjacency, x_origin, y_origin, graph_bin_cm = _graph_for_fold(
                transitions,
                session=str(session_id),
                excluded_fold=excluded_fold,
            )
            surprise_values: list[float] = []
            if adjacency:
                for left, right in zip(valid_bout_rows[:-1], valid_bout_rows[1:], strict=True):
                    source = _coordinate_bin(
                        np.asarray(left["end_xy"]),
                        x_origin=x_origin,
                        y_origin=y_origin,
                        bin_cm=graph_bin_cm,
                    )
                    target = _coordinate_bin(
                        np.asarray(right["start_xy"]),
                        x_origin=x_origin,
                        y_origin=y_origin,
                        bin_cm=graph_bin_cm,
                    )
                    value = shortest_transition_surprise(adjacency, source, target)
                    if np.isfinite(value):
                        surprise_values.append(float(value))

            event_peak = float(route_context["event_peak_s"])
            movement_start = float(route_context["route_movement_start_time_s"])
            movement_end = float(route_context["route_movement_end_time_s"])
            previous_reward_arrival = previous_reward_arrival_time(
                routes,
                session=str(session_id),
                event_peak_s=event_peak,
            )
            future_start = max(event_peak, movement_start)
            actual_future = _position_path(
                smoothed_position,
                start_s=future_start,
                end_s=movement_end,
            )
            actual_past = _position_path(
                smoothed_position,
                start_s=movement_start,
                end_s=min(event_peak, movement_end),
            )
            actual_future_distance = np.nan
            actual_future_emission_distance = np.nan
            alternative_distances: list[float] = []
            alternative_emission_distances: list[float] = []
            alternatives: list[tuple[str, np.ndarray, float]] = []
            future_informative = bool(
                len(actual_future) >= 2 and path_length(actual_future) >= minimum_bout_path_cm
            )
            if future_informative:
                actual_future_distance = path_fit_distance_cm(event_path, actual_future)
                actual_future_emission_distance = path_fit_distance_cm(
                    emission_path,
                    actual_future,
                )
                alternatives = _route_suffixes_near_start(
                    actual_start=actual_future[0],
                    actual_destination_well=int(route_context["destination_well_id"]),
                    training_route_ids=training_route_ids,
                    routes_by_id=routes_by_id,
                    route_paths=route_paths,
                    maximum_start_distance_cm=float(maximum_alternative_start_distance_cm),
                    minimum_alternatives=int(minimum_alternatives),
                    fallback_alternatives=int(fallback_alternatives),
                )
                for _, alternative, _ in alternatives:
                    alternative_distances.append(path_fit_distance_cm(event_path, alternative))
                    alternative_emission_distances.append(
                        path_fit_distance_cm(emission_path, alternative)
                    )
            future_commitment = (
                float(np.median(alternative_distances) - actual_future_distance)
                if len(alternative_distances) >= int(minimum_alternatives)
                and np.isfinite(actual_future_distance)
                else np.nan
            )
            emission_future_commitment = (
                float(
                    np.median(alternative_emission_distances)
                    - actual_future_emission_distance
                )
                if len(alternative_emission_distances) >= int(minimum_alternatives)
                and np.isfinite(actual_future_emission_distance)
                else np.nan
            )
            past_overlap_distance = (
                path_fit_distance_cm(event_path, actual_past)
                if len(actual_past) >= 2 and path_length(actual_past) >= minimum_bout_path_cm
                else np.nan
            )
            current_x = float(np.interp(event_peak, smoothed_position[:, 0], smoothed_position[:, 1]))
            current_y = float(np.interp(event_peak, smoothed_position[:, 0], smoothed_position[:, 2]))
            actual_goal = actual_future[-1] if len(actual_future) else np.array([np.nan, np.nan])
            next_goal_error = float(np.linalg.norm(event_path[-1] - actual_goal)) if len(actual_future) else np.nan
            route_probability, choice_entropy, route_frequency, outgoing_routes = _choice_statistics(
                routes,
                session=str(session_id),
                excluded_fold=excluded_fold,
                origin_well=int(route_context["origin_well_id"]),
                destination_well=int(route_context["destination_well_id"]),
            )
            trajectory_family_logz = max(
                float(model_row["logZ_diffusion"]),
                float(model_row["logZ_first_order_imm"]),
                float(model_row["logZ_momentum_exact_sparse"]),
            )
            event_rows.append(
                {
                    "session": str(session_id),
                    "rat": str(session_id).split("/", 1)[0],
                    "event_index": int(event_index),
                    "analysis_role": model_row["analysis_role"],
                    "clean_imm": bool(model_row["clean_imm"]),
                    "raw_momentum_win": bool(model_row["raw_momentum_win"]),
                    "confident_momentum_win": bool(model_row["confident_momentum_win"]),
                    "delta_momentum_minus_imm": float(model_row["delta_momentum_minus_imm"]),
                    "delta_imm_minus_fragmented": float(model_row["delta_imm_minus_fragmented"]),
                    "trajectory_minus_stationary_log_evidence": float(
                        trajectory_family_logz - model_row["logZ_stationary"]
                    ),
                    "composition_index": composition_index,
                    "composition_index_cm": composition_index,
                    "composition_evaluable": composition_evaluable,
                    "continuous_bout_count": int(posterior_row["continuous_bout_count"]),
                    "eligible_continuous_bout_count": int(len(valid_bout_rows)),
                    "mean_bout_best_primitive_distance_cm": bout_distance,
                    "best_single_route_distance_cm": best_route_distance,
                    "best_single_route_id": best_route_id,
                    "distinct_best_primitives": int(len(set(primitive_sequence))),
                    "primitive_identity_changes": primitive_identity_changes,
                    "distinct_best_route_classes": int(len(set(route_sequence))),
                    "route_identity_changes": route_identity_changes,
                    "switch_alignment": float(
                        route_identity_changes / max(len(route_sequence) - 1, 1)
                    )
                    if len(route_sequence) >= 2
                    else np.nan,
                    "transition_surprise": float(np.mean(surprise_values)) if surprise_values else np.nan,
                    "transition_surprise_nats": float(np.mean(surprise_values)) if surprise_values else np.nan,
                    "transition_surprise_pairs": int(len(surprise_values)),
                    "future_commitment_index": future_commitment,
                    "future_commitment_index_cm": future_commitment,
                    "emission_only_future_commitment_index_cm": emission_future_commitment,
                    "actual_future_path_distance_cm": actual_future_distance,
                    "matched_alternative_path_distance_median_cm": float(np.median(alternative_distances))
                    if alternative_distances else np.nan,
                    "matched_alternative_routes": int(len(alternative_distances)),
                    "future_path_informative": future_informative,
                    "next_goal_error": next_goal_error,
                    "next_goal_error_cm": next_goal_error,
                    "past_path_overlap": -past_overlap_distance if np.isfinite(past_overlap_distance) else np.nan,
                    "past_path_fit_distance_cm": past_overlap_distance,
                    "future_path_overlap": -actual_future_distance if np.isfinite(actual_future_distance) else np.nan,
                    "choice_entropy_before_event": choice_entropy,
                    "route_behavior_probability": route_probability,
                    "route_frequency": route_frequency,
                    "outgoing_behavior_routes": outgoing_routes,
                    "time_to_departure": float(movement_start - event_peak),
                    "time_to_departure_s": float(movement_start - event_peak),
                    "previous_reward_arrival_time_s": previous_reward_arrival,
                    "elapsed_time_since_reward_s": float(event_peak - previous_reward_arrival)
                    if np.isfinite(previous_reward_arrival)
                    else np.nan,
                    "current_animal_x_cm": current_x,
                    "current_animal_y_cm": current_y,
                    "next_well_x_cm": float(actual_goal[0]),
                    "next_well_y_cm": float(actual_goal[1]),
                    "current_well": int(route_context["origin_well_id"]),
                    "next_well": int(route_context["destination_well_id"]),
                    "event_route_relation": route_context["event_route_relation"],
                    "event_peak_s": event_peak,
                    "event_duration_ms": float(event_bins["bin_duration_s"].sum() * 1000.0),
                    "n_spikes": int(posterior_row["n_spikes"]),
                    "active_cell_count": int(posterior_row["n_active_cells"]),
                    "posterior_entropy": float(event_bins["imm_posterior_spatial_entropy"].mean()),
                    "posterior_path_length_cm": path_length(event_path),
                    "posterior_net_displacement_cm": float(
                        np.linalg.norm(event_path[-1] - event_path[0])
                    ),
                    "run_decoder_error_cm": np.nan,
                    "excluded_cv_fold": excluded_fold,
                    "candidate_primitives": int(route_context["candidate_primitives"]),
                    "candidate_training_routes": int(len(training_route_ids)),
                }
            )
    return pd.DataFrame(event_rows), pd.DataFrame(bout_rows)


def build_gate_summary(events: pd.DataFrame, frozen_events: pd.DataFrame) -> pd.DataFrame:
    expected = int(len(frozen_events))
    observed = int(len(events))
    composition_count = int(events.get("composition_evaluable", pd.Series(dtype=bool)).sum())
    commitment_count = int(events.get("future_commitment_index", pd.Series(dtype=float)).notna().sum())
    rats_composition = int(
        events.loc[events.get("composition_evaluable", False), "rat"].nunique()
    ) if not events.empty else 0
    rats_commitment = int(
        events.loc[events.get("future_commitment_index", pd.Series(dtype=float)).notna(), "rat"].nunique()
    ) if not events.empty else 0
    decoder = pd.to_numeric(
        events.get("run_decoder_error_cm", pd.Series(index=events.index, dtype=float)),
        errors="coerce",
    )
    decoder_sessions = int(events.loc[decoder.notna(), "session"].nunique()) if not events.empty else 0
    expected_sessions = int(events["session"].nunique()) if not events.empty else 0
    gates = [
        ("frozen_events_present", expected > 0, expected, ">0"),
        ("all_frozen_events_joined", observed == expected and expected > 0, observed, expected),
        ("composition_cohort_present", composition_count > 0, composition_count, ">0"),
        ("composition_spans_all_rats", rats_composition == 4, rats_composition, 4),
        ("commitment_cohort_present", commitment_count > 0, commitment_count, ">0"),
        ("commitment_spans_all_rats", rats_commitment == 4, rats_commitment, 4),
        (
            "stationary_and_fragmented_phases_excluded_from_composition",
            True,
            "MAP mode 1 only",
            "MAP mode 1 only",
        ),
        ("behavior_templates_cross_validated", True, True, True),
        ("model_axis_not_used_to_define_behavior_templates", True, True, True),
        (
            "run_decoder_error_available_for_all_sessions",
            expected_sessions > 0 and decoder_sessions == expected_sessions,
            decoder_sessions,
            expected_sessions,
        ),
    ]
    rows = [
        {"gate": gate, "passed": bool(passed), "value": value, "required": required}
        for gate, passed, value, required in gates
    ]
    rows.append(
        {
            "gate": "overall",
            "passed": bool(all(row["passed"] for row in rows)),
            "value": int(sum(bool(row["passed"]) for row in rows)),
            "required": len(rows),
        }
    )
    return pd.DataFrame(rows)


def run_analysis(
    *,
    dataset_root: str | Path,
    frozen_events_csv: str | Path,
    route_dir: str | Path,
    posterior_dir: str | Path,
    output_dir: str | Path,
    run_decoder_summary_csv: str | Path | None = None,
    minimum_bout_bins: int = 3,
    minimum_bout_path_cm: float = 10.0,
    maximum_alternative_start_distance_cm: float = 30.0,
    minimum_alternatives: int = 3,
    fallback_alternatives: int = 10,
) -> dict[str, Path]:
    route_root = Path(route_dir)
    posterior_root = Path(posterior_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frozen_events = pd.read_csv(frozen_events_csv)
    inputs = {
        "routes": route_root / "replay_behavior_route_segments.csv",
        "route_points": route_root / "replay_behavior_route_segment_points.csv",
        "primitives": route_root / "replay_behavior_route_primitives.csv",
        "primitive_points": route_root / "replay_behavior_route_primitive_points.csv",
        "transitions": route_root / "replay_behavior_transition_graph.csv",
        "eligibility": route_root / "replay_event_route_library_eligibility.csv",
        "posterior_bins": posterior_root / "replay_commitment_composition_posterior_bins.csv",
        "posterior_events": posterior_root / "replay_commitment_composition_posterior_event_summary.csv",
    }
    loaded = {name: pd.read_csv(path) for name, path in inputs.items()}
    events, bouts = compute_metrics(
        dataset_root=dataset_root,
        frozen_events=frozen_events,
        posterior_bins=loaded["posterior_bins"],
        posterior_events=loaded["posterior_events"],
        routes=loaded["routes"],
        route_points=loaded["route_points"],
        primitives=loaded["primitives"],
        primitive_points=loaded["primitive_points"],
        transitions=loaded["transitions"],
        eligibility=loaded["eligibility"],
        minimum_bout_bins=int(minimum_bout_bins),
        minimum_bout_path_cm=float(minimum_bout_path_cm),
        maximum_alternative_start_distance_cm=float(
            maximum_alternative_start_distance_cm
        ),
        minimum_alternatives=int(minimum_alternatives),
        fallback_alternatives=int(fallback_alternatives),
    )
    if run_decoder_summary_csv is not None:
        decoder = pd.read_csv(run_decoder_summary_csv)
        value_column = next(
            (
                column
                for column in (
                    "median_posterior_mean_error_cm",
                    "posterior_mean_error_cm_median",
                )
                if column in decoder
            ),
            None,
        )
        if "session" not in decoder or value_column is None:
            raise ValueError(
                "RUN decoder summary requires session and a median posterior-error column"
            )
        decoder_lookup = (
            decoder[["session", value_column]]
            .drop_duplicates("session")
            .set_index("session")[value_column]
        )
        events["run_decoder_error_cm"] = events["session"].map(decoder_lookup)
    gates = build_gate_summary(events, frozen_events)
    paths: dict[str, Path] = {}
    for name, frame in {
        EVENT_OUTPUT: events,
        BOUT_OUTPUT: bouts,
        GATE_OUTPUT: gates,
    }.items():
        path = output / name
        frame.to_csv(path, index=False)
        paths[name] = path
    provenance_inputs: dict[str, str | Path] = {
        "dataset_root": dataset_root,
        "frozen_events_csv": frozen_events_csv,
        **inputs,
    }
    if run_decoder_summary_csv is not None:
        provenance_inputs["run_decoder_summary_csv"] = run_decoder_summary_csv
    provenance = build_script_provenance(
        input_paths=provenance_inputs,
        cwd=ROOT,
    )
    manifest = {
        "analysis": "replay_commitment_composition_event_metrics",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "events": int(len(events)),
        "composition_evaluable_events": int(events["composition_evaluable"].sum()),
        "commitment_evaluable_events": int(events["future_commitment_index"].notna().sum()),
        "parameters": {
            "minimum_bout_bins": int(minimum_bout_bins),
            "minimum_bout_path_cm": float(minimum_bout_path_cm),
            "maximum_alternative_start_distance_cm": float(
                maximum_alternative_start_distance_cm
            ),
            "minimum_alternatives": int(minimum_alternatives),
            "fallback_alternatives": int(fallback_alternatives),
        },
        "definitions": {
            "composition_index": "best_single_behavior_route_distance_cm - bin_weighted_mean_best_bout_primitive_distance_cm",
            "future_commitment_index": "median_start_matched_alternative_distance_cm - actual_future_route_distance_cm",
            "positive_direction": "positive means more composition or more actual-future commitment",
            "composition_phase_filter": "MAP first-order IMM mode 1 (continuous diffusion) only; modes 0 and 2 excluded",
        },
        "outputs": {name: str(path) for name, path in paths.items()},
        "provenance": provenance,
    }
    manifest_path = output / MANIFEST_OUTPUT
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths[MANIFEST_OUTPUT] = manifest_path
    overall = bool(gates.loc[gates["gate"].eq("overall"), "passed"].iloc[0])
    summary = [
        "# Replay commitment/composition event metrics",
        "",
        f"- Frozen events joined: {len(events)}/{len(frozen_events)}",
        f"- Composition-evaluable events: {int(events['composition_evaluable'].sum())}",
        f"- Future-commitment-evaluable events: {int(events['future_commitment_index'].notna().sum())}",
        f"- Overall readiness gate: {'PASS' if overall else 'FAIL'}",
        "",
        "No primary hypothesis test is performed here. The table freezes independently defined outcomes before regression against the predeclared continuous momentum-minus-IMM evidence axis.",
    ]
    summary_path = output / SUMMARY_OUTPUT
    summary_path.write_text("\n".join(summary) + "\n", encoding="utf-8")
    paths[SUMMARY_OUTPUT] = summary_path
    return paths


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--frozen-events", required=True)
    parser.add_argument("--route-dir", required=True)
    parser.add_argument("--posterior-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-decoder-summary")
    parser.add_argument("--minimum-bout-bins", type=int, default=3)
    parser.add_argument("--minimum-bout-path-cm", type=float, default=10.0)
    parser.add_argument("--maximum-alternative-start-distance-cm", type=float, default=30.0)
    parser.add_argument("--minimum-alternatives", type=int, default=3)
    parser.add_argument("--fallback-alternatives", type=int, default=10)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_analysis(
        dataset_root=args.dataset_root,
        frozen_events_csv=args.frozen_events,
        route_dir=args.route_dir,
        posterior_dir=args.posterior_dir,
        output_dir=args.output_dir,
        run_decoder_summary_csv=args.run_decoder_summary,
        minimum_bout_bins=args.minimum_bout_bins,
        minimum_bout_path_cm=args.minimum_bout_path_cm,
        maximum_alternative_start_distance_cm=args.maximum_alternative_start_distance_cm,
        minimum_alternatives=args.minimum_alternatives,
        fallback_alternatives=args.fallback_alternatives,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
