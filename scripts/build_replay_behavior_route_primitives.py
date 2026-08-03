#!/usr/bin/env python3
"""Learn cross-validated route primitives from RUN behavior only.

The route library is deliberately constructed without replay-model evidence or
replay posteriors.  A well-to-well traversal and every fixed-length subpath
derived from it share one cross-validation fold.  Events are subsequently
matched only against the library trained with their enclosing traversal held
out.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.spatial.distance import pdist, squareform

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _provenance import build_script_provenance  # noqa: E402
from hipporeplayimm.data import load_replay_session  # noqa: E402


ROUTE_OUTPUT = "replay_behavior_route_segments.csv"
ROUTE_POINT_OUTPUT = "replay_behavior_route_segment_points.csv"
PRIMITIVE_OUTPUT = "replay_behavior_route_primitives.csv"
PRIMITIVE_POINT_OUTPUT = "replay_behavior_route_primitive_points.csv"
TRANSITION_OUTPUT = "replay_behavior_transition_graph.csv"
EVENT_OUTPUT = "replay_event_route_library_eligibility.csv"
SESSION_OUTPUT = "replay_behavior_route_primitives_by_session.csv"
GATE_OUTPUT = "replay_behavior_route_primitives_gate_summary.csv"
MANIFEST_OUTPUT = "replay_behavior_route_primitives_manifest.json"
SUMMARY_OUTPUT = "replay_behavior_route_primitives_summary.md"


def _finite_position(position: np.ndarray) -> np.ndarray:
    array = np.asarray(position, dtype=float)
    if array.ndim != 2 or array.shape[1] < 3:
        return np.empty((0, 3), dtype=float)
    keep = np.isfinite(array[:, :3]).all(axis=1)
    array = array[keep, :3]
    if len(array) < 2:
        return array
    order = np.argsort(array[:, 0], kind="stable")
    array = array[order]
    _, unique = np.unique(array[:, 0], return_index=True)
    return array[np.sort(unique)]


def smooth_position_trace(
    position: np.ndarray,
    *,
    median_window_s: float = 0.167,
    gaussian_sigma_s: float = 0.100,
) -> np.ndarray:
    """Return time, smoothed x/y, and speed for a finite position trace."""

    array = _finite_position(position)
    if len(array) < 2:
        return np.column_stack([array, np.zeros(len(array), dtype=float)])
    dt = float(np.median(np.diff(array[:, 0])))
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("position timestamps must be strictly increasing")
    median_samples = max(1, int(round(float(median_window_s) / dt)))
    if median_samples % 2 == 0:
        median_samples += 1
    sigma_samples = max(0.0, float(gaussian_sigma_s) / dt)
    xy = array[:, 1:3].copy()
    if median_samples > 1:
        xy = np.column_stack(
            [median_filter(xy[:, dim], size=median_samples, mode="nearest") for dim in range(2)]
        )
    if sigma_samples > 0.0:
        xy = np.column_stack(
            [gaussian_filter1d(xy[:, dim], sigma=sigma_samples, mode="nearest") for dim in range(2)]
        )
    dx = np.gradient(xy[:, 0], array[:, 0])
    dy = np.gradient(xy[:, 1], array[:, 0])
    speed = np.hypot(dx, dy)
    speed = np.nan_to_num(speed, nan=0.0, posinf=0.0, neginf=0.0)
    return np.column_stack([array[:, 0], xy, speed])


def _inside_any_interval(times: np.ndarray, intervals: np.ndarray) -> np.ndarray:
    mask = np.zeros(len(times), dtype=bool)
    values = np.asarray(intervals, dtype=float)
    if values.ndim == 1 and values.size == 2:
        values = values.reshape(1, 2)
    if values.ndim != 2 or values.shape[1] != 2:
        return mask
    for start, end in values:
        mask |= (times >= float(start)) & (times <= float(end))
    return mask


def _polyline_length(xy: np.ndarray) -> float:
    if len(xy) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1)))


def _resample_polyline_with_time(
    time_s: np.ndarray,
    xy: np.ndarray,
    *,
    n_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(xy) < 2 or int(n_points) < 2:
        raise ValueError("polyline resampling requires at least two source and output points")
    steps = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(steps)])
    keep = np.concatenate([[True], np.diff(cumulative) > 1e-9])
    cumulative = cumulative[keep]
    points = xy[keep]
    times = np.asarray(time_s, dtype=float)[keep]
    if len(points) < 2 or cumulative[-1] <= 0.0:
        raise ValueError("polyline has no nonzero spatial extent")
    targets = np.linspace(0.0, float(cumulative[-1]), int(n_points))
    output_xy = np.column_stack(
        [np.interp(targets, cumulative, points[:, dim]) for dim in range(2)]
    )
    output_time = np.interp(targets, cumulative, times)
    return output_time, output_xy


def _first_sustained_true(mask: np.ndarray, required_samples: int) -> int | None:
    run = 0
    for index, value in enumerate(np.asarray(mask, dtype=bool)):
        run = run + 1 if value else 0
        if run >= int(required_samples):
            return int(index - required_samples + 1)
    return None


def _movement_segment(
    segment: np.ndarray,
    *,
    speed_threshold_cm_s: float,
    minimum_departure_s: float,
    arrival_radius_cm: float,
    minimum_arrival_dwell_s: float,
    destination_window_s: float,
) -> np.ndarray:
    """Trim a fill interval from sustained departure to destination arrival."""

    if len(segment) < 2:
        return segment
    dt = float(np.median(np.diff(segment[:, 0])))
    if not np.isfinite(dt) or dt <= 0.0:
        return segment
    departure_required = max(1, int(np.ceil(float(minimum_departure_s) / dt)))
    departure = _first_sustained_true(
        segment[:, 3] >= float(speed_threshold_cm_s),
        departure_required,
    )
    if departure is None:
        departure = 0
    destination_window = segment[
        segment[:, 0] >= float(segment[-1, 0] - destination_window_s)
    ]
    destination = np.median(destination_window[:, 1:3], axis=0)
    distance = np.linalg.norm(segment[:, 1:3] - destination[None, :], axis=1)
    arrival_required = max(1, int(np.ceil(float(minimum_arrival_dwell_s) / dt)))
    arrival_relative = _first_sustained_true(
        distance[departure:] <= float(arrival_radius_cm),
        arrival_required,
    )
    arrival = (
        departure + int(arrival_relative)
        if arrival_relative is not None
        else len(segment) - 1
    )
    start = max(0, int(departure) - 1)
    stop = min(len(segment), max(start + 2, int(arrival) + 2))
    return segment[start:stop]


def segment_well_to_well_routes(
    *,
    session_id: str,
    position: np.ndarray,
    run_times: np.ndarray,
    well_sequence: np.ndarray,
    n_folds: int = 5,
    speed_threshold_cm_s: float = 5.0,
    minimum_departure_s: float = 0.25,
    arrival_radius_cm: float = 10.0,
    minimum_arrival_dwell_s: float = 0.20,
    destination_window_s: float = 1.0,
    minimum_route_displacement_cm: float = 10.0,
    minimum_route_samples: int = 10,
    route_resample_points: int = 41,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Segment target-fill intervals into directed behavior routes."""

    position = np.asarray(position, dtype=float)
    wells = np.asarray(well_sequence, dtype=float)
    route_columns = [
        "session", "rat", "route_id", "route_index", "cv_fold", "interval_start_time_s",
        "interval_end_time_s", "movement_start_time_s", "movement_end_time_s",
        "origin_well_id", "destination_well_id", "duration_s", "movement_duration_s",
        "path_length_cm", "net_displacement_cm", "median_speed_cm_s",
        "p95_speed_cm_s", "n_source_samples", "n_resampled_points",
    ]
    point_columns = [
        "session", "rat", "route_id", "route_index", "cv_fold", "point_index",
        "time_s", "x_cm", "y_cm", "arc_fraction",
    ]
    if len(position) < 2 or wells.ndim != 2 or wells.shape[1] < 2 or len(wells) < 3:
        return pd.DataFrame(columns=route_columns), pd.DataFrame(columns=point_columns)
    finite = np.isfinite(wells[:, 0]) & np.isfinite(wells[:, 1])
    wells = wells[finite, :2]
    wells = wells[np.argsort(wells[:, 0], kind="stable")]
    rows: list[dict[str, object]] = []
    point_rows: list[dict[str, object]] = []
    rat = str(session_id).split("/", 1)[0]
    # During [fill_i, fill_(i+1)], the animal travels from the previously
    # rewarded well toward the target named in row i.
    for index in range(1, len(wells) - 1):
        start = float(wells[index, 0])
        end = float(wells[index + 1, 0])
        if end <= start or not _inside_any_interval(np.array([0.5 * (start + end)]), run_times)[0]:
            continue
        segment = position[(position[:, 0] >= start) & (position[:, 0] <= end)]
        segment = _movement_segment(
            segment,
            speed_threshold_cm_s=speed_threshold_cm_s,
            minimum_departure_s=float(minimum_departure_s),
            arrival_radius_cm=float(arrival_radius_cm),
            minimum_arrival_dwell_s=float(minimum_arrival_dwell_s),
            destination_window_s=float(destination_window_s),
        )
        if len(segment) < int(minimum_route_samples):
            continue
        displacement = float(np.linalg.norm(segment[-1, 1:3] - segment[0, 1:3]))
        if not np.isfinite(displacement) or displacement < float(minimum_route_displacement_cm):
            continue
        try:
            sampled_time, sampled_xy = _resample_polyline_with_time(
                segment[:, 0], segment[:, 1:3], n_points=int(route_resample_points)
            )
        except ValueError:
            continue
        route_id = f"{session_id}:route_{index:03d}"
        fold = int(index % max(2, int(n_folds)))
        rows.append(
            {
                "session": session_id,
                "rat": rat,
                "route_id": route_id,
                "route_index": int(index),
                "cv_fold": fold,
                "interval_start_time_s": start,
                "interval_end_time_s": end,
                "movement_start_time_s": float(segment[0, 0]),
                "movement_end_time_s": float(segment[-1, 0]),
                "origin_well_id": int(wells[index - 1, 1]),
                "destination_well_id": int(wells[index, 1]),
                "duration_s": float(end - start),
                "movement_duration_s": float(segment[-1, 0] - segment[0, 0]),
                "path_length_cm": _polyline_length(sampled_xy),
                "net_displacement_cm": displacement,
                "median_speed_cm_s": float(np.median(segment[:, 3])),
                "p95_speed_cm_s": float(np.quantile(segment[:, 3], 0.95)),
                "n_source_samples": int(len(segment)),
                "n_resampled_points": int(route_resample_points),
            }
        )
        for point_index, (time_value, xy) in enumerate(zip(sampled_time, sampled_xy, strict=True)):
            point_rows.append(
                {
                    "session": session_id,
                    "rat": rat,
                    "route_id": route_id,
                    "route_index": int(index),
                    "cv_fold": fold,
                    "point_index": int(point_index),
                    "time_s": float(time_value),
                    "x_cm": float(xy[0]),
                    "y_cm": float(xy[1]),
                    "arc_fraction": float(point_index / (int(route_resample_points) - 1)),
                }
            )
    return pd.DataFrame(rows, columns=route_columns), pd.DataFrame(point_rows, columns=point_columns)


def extract_fixed_length_subpaths(
    routes: pd.DataFrame,
    route_points: pd.DataFrame,
    *,
    primitive_length_cm: float = 40.0,
    primitive_stride_cm: float = 20.0,
    primitive_resample_points: int = 17,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Extract absolute, directed behavior subpaths from each route."""

    metadata: list[dict[str, object]] = []
    arrays: dict[str, np.ndarray] = {}
    for route in routes.itertuples(index=False):
        points = route_points[route_points["route_id"].eq(route.route_id)].sort_values("point_index")
        xy = points[["x_cm", "y_cm"]].to_numpy(dtype=float)
        time_s = points["time_s"].to_numpy(dtype=float)
        if len(xy) < 2:
            continue
        steps = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        cumulative = np.concatenate([[0.0], np.cumsum(steps)])
        length = float(cumulative[-1])
        window = min(float(primitive_length_cm), length)
        if window <= 0.0:
            continue
        starts = np.arange(0.0, max(length - window, 0.0) + 1e-9, float(primitive_stride_cm))
        if starts.size == 0 or starts[-1] < length - window - 1e-9:
            starts = np.append(starts, max(0.0, length - window))
        for sub_index, start_distance in enumerate(np.unique(np.round(starts, 9))):
            targets = np.linspace(float(start_distance), float(start_distance + window), int(primitive_resample_points))
            sub_xy = np.column_stack(
                [np.interp(targets, cumulative, xy[:, dim]) for dim in range(2)]
            )
            sub_time = np.interp(targets, cumulative, time_s)
            candidate_id = f"{route.route_id}:sub_{sub_index:03d}"
            arrays[candidate_id] = sub_xy
            metadata.append(
                {
                    "session": route.session,
                    "rat": route.rat,
                    "candidate_segment_id": candidate_id,
                    "source_route_id": route.route_id,
                    "source_route_index": int(route.route_index),
                    "cv_fold": int(route.cv_fold),
                    "subpath_index": int(sub_index),
                    "start_arc_cm": float(start_distance),
                    "end_arc_cm": float(start_distance + window),
                    "path_length_cm": _polyline_length(sub_xy),
                    "duration_s": float(sub_time[-1] - sub_time[0]),
                    "n_points": int(primitive_resample_points),
                }
            )
    return pd.DataFrame(metadata), arrays


def _cluster_labels(paths: np.ndarray, *, threshold_cm: float) -> np.ndarray:
    if len(paths) == 0:
        return np.empty(0, dtype=int)
    if len(paths) == 1:
        return np.ones(1, dtype=int)
    flattened = paths.reshape(len(paths), -1)
    condensed = pdist(flattened, metric="euclidean") / np.sqrt(paths.shape[1])
    hierarchy = linkage(condensed, method="average")
    return fcluster(hierarchy, t=float(threshold_cm), criterion="distance").astype(int)


def build_cross_validated_primitives(
    candidates: pd.DataFrame,
    candidate_paths: dict[str, np.ndarray],
    *,
    n_folds: int,
    cluster_threshold_cm: float = 12.5,
    minimum_cluster_routes: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cluster training-fold behavior subpaths and return medoid primitives."""

    primitive_rows: list[dict[str, object]] = []
    point_rows: list[dict[str, object]] = []
    if candidates.empty:
        return pd.DataFrame(), pd.DataFrame()
    for session_id, session_candidates in candidates.groupby("session", sort=True):
        for excluded_fold in range(max(2, int(n_folds))):
            training = session_candidates[~session_candidates["cv_fold"].eq(excluded_fold)].copy()
            if training.empty:
                continue
            ids = training["candidate_segment_id"].astype(str).tolist()
            paths = np.stack([candidate_paths[value] for value in ids])
            labels = _cluster_labels(paths, threshold_cm=float(cluster_threshold_cm))
            training["cluster_label"] = labels
            for cluster_label, cluster in training.groupby("cluster_label", sort=True):
                source_routes = sorted(cluster["source_route_id"].astype(str).unique())
                if len(source_routes) < int(minimum_cluster_routes):
                    continue
                indices = cluster.index.to_numpy()
                local_positions = np.flatnonzero(training.index.isin(indices))
                cluster_paths = paths[local_positions]
                if len(cluster_paths) == 1:
                    medoid_local = 0
                    within = 0.0
                else:
                    distance = squareform(
                        pdist(cluster_paths.reshape(len(cluster_paths), -1), metric="euclidean")
                        / np.sqrt(cluster_paths.shape[1])
                    )
                    totals = distance.sum(axis=1)
                    medoid_local = int(np.argmin(totals))
                    within = float(np.mean(distance[np.triu_indices(len(distance), k=1)]))
                medoid_id = str(cluster.iloc[medoid_local]["candidate_segment_id"])
                medoid_path = candidate_paths[medoid_id]
                primitive_id = f"{session_id}:fold{excluded_fold}:primitive_{int(cluster_label):03d}"
                row = cluster.iloc[medoid_local]
                primitive_rows.append(
                    {
                        "session": session_id,
                        "rat": row["rat"],
                        "excluded_cv_fold": int(excluded_fold),
                        "primitive_id": primitive_id,
                        "cluster_label": int(cluster_label),
                        "cluster_size_segments": int(len(cluster)),
                        "cluster_size_routes": int(len(source_routes)),
                        "source_route_ids": "|".join(source_routes),
                        "medoid_candidate_segment_id": medoid_id,
                        "medoid_source_route_id": str(row["source_route_id"]),
                        "mean_within_cluster_distance_cm": within,
                        "primitive_path_length_cm": _polyline_length(medoid_path),
                        "primitive_duration_s": float(row["duration_s"]),
                        "cluster_threshold_cm": float(cluster_threshold_cm),
                        "minimum_cluster_routes": int(minimum_cluster_routes),
                        "familiar_primitive": True,
                        "behavior_only": True,
                    }
                )
                for point_index, xy in enumerate(medoid_path):
                    point_rows.append(
                        {
                            "session": session_id,
                            "rat": row["rat"],
                            "excluded_cv_fold": int(excluded_fold),
                            "primitive_id": primitive_id,
                            "point_index": int(point_index),
                            "x_cm": float(xy[0]),
                            "y_cm": float(xy[1]),
                            "arc_fraction": float(point_index / (len(medoid_path) - 1)),
                        }
                    )
    return pd.DataFrame(primitive_rows), pd.DataFrame(point_rows)


def build_transition_graph(
    routes: pd.DataFrame,
    route_points: pd.DataFrame,
    *,
    n_folds: int,
    spatial_bin_cm: float = 10.0,
    pseudocount: float = 0.5,
) -> pd.DataFrame:
    """Build a fold-specific directed transition graph from training routes."""

    rows: list[dict[str, object]] = []
    if routes.empty or route_points.empty:
        return pd.DataFrame()
    for session_id, session_routes in routes.groupby("session", sort=True):
        session_points = route_points[route_points["session"].eq(session_id)]
        x_origin = float(np.floor(session_points["x_cm"].min() / spatial_bin_cm) * spatial_bin_cm)
        y_origin = float(np.floor(session_points["y_cm"].min() / spatial_bin_cm) * spatial_bin_cm)
        for excluded_fold in range(max(2, int(n_folds))):
            training_ids = set(
                session_routes.loc[~session_routes["cv_fold"].eq(excluded_fold), "route_id"].astype(str)
            )
            counts: dict[tuple[int, int, int, int], int] = {}
            for route_id in sorted(training_ids):
                points = session_points[session_points["route_id"].eq(route_id)].sort_values("point_index")
                x_bin = np.floor((points["x_cm"].to_numpy(dtype=float) - x_origin) / spatial_bin_cm).astype(int)
                y_bin = np.floor((points["y_cm"].to_numpy(dtype=float) - y_origin) / spatial_bin_cm).astype(int)
                states = list(zip(x_bin, y_bin, strict=True))
                compressed = [states[0]] if states else []
                compressed.extend(state for state, prior in zip(states[1:], states[:-1], strict=True) if state != prior)
                for source, target in zip(compressed[:-1], compressed[1:], strict=True):
                    key = (source[0], source[1], target[0], target[1])
                    counts[key] = counts.get(key, 0) + 1
            outgoing: dict[tuple[int, int], int] = {}
            targets: dict[tuple[int, int], set[tuple[int, int]]] = {}
            for (source_x, source_y, target_x, target_y), count in counts.items():
                source = (source_x, source_y)
                outgoing[source] = outgoing.get(source, 0) + count
                targets.setdefault(source, set()).add((target_x, target_y))
            for key, count in sorted(counts.items()):
                source_x, source_y, target_x, target_y = key
                source = (source_x, source_y)
                denominator = outgoing[source] + float(pseudocount) * len(targets[source])
                probability = (count + float(pseudocount)) / denominator
                rows.append(
                    {
                        "session": session_id,
                        "rat": str(session_id).split("/", 1)[0],
                        "excluded_cv_fold": int(excluded_fold),
                        "from_x_bin": int(source_x),
                        "from_y_bin": int(source_y),
                        "to_x_bin": int(target_x),
                        "to_y_bin": int(target_y),
                        "from_x_cm": x_origin + (source_x + 0.5) * spatial_bin_cm,
                        "from_y_cm": y_origin + (source_y + 0.5) * spatial_bin_cm,
                        "to_x_cm": x_origin + (target_x + 0.5) * spatial_bin_cm,
                        "to_y_cm": y_origin + (target_y + 0.5) * spatial_bin_cm,
                        "transition_count": int(count),
                        "outgoing_count": int(outgoing[source]),
                        "observed_out_degree": int(len(targets[source])),
                        "transition_probability": float(probability),
                        "transition_surprise_nats": float(-np.log(probability)),
                        "spatial_bin_cm": float(spatial_bin_cm),
                        "pseudocount": float(pseudocount),
                        "behavior_only": True,
                    }
                )
    return pd.DataFrame(rows)


def map_events_to_route_libraries(
    frozen_events: pd.DataFrame,
    routes: pd.DataFrame,
    primitives: pd.DataFrame,
    route_points: pd.DataFrame,
    *,
    session_loader: Callable[[str], object] | None = None,
) -> pd.DataFrame:
    """Map events to enclosing routes and held-out behavior libraries."""

    rows: list[dict[str, object]] = []
    required = {"session", "event_index"}
    missing = sorted(required.difference(frozen_events.columns))
    if missing:
        raise ValueError(f"frozen event table is missing columns: {missing}")
    for event in frozen_events[["session", "event_index"]].drop_duplicates().itertuples(index=False):
        session_routes = routes[routes["session"].eq(str(event.session))]
        event_time = np.nan
        if session_loader is not None:
            session = session_loader(str(event.session))
            event_time = float(session.ripple(int(event.event_index)).peak)
        enclosing = session_routes[
            (session_routes["movement_start_time_s"] <= event_time)
            & (session_routes["movement_end_time_s"] >= event_time)
        ]
        if not enclosing.empty:
            route = enclosing.sort_values("movement_duration_s").iloc[0]
            route_relation = "during_movement"
        else:
            following = session_routes[session_routes["movement_start_time_s"] > event_time]
            if following.empty:
                route = None
                route_relation = ""
            else:
                route = following.sort_values("movement_start_time_s").iloc[0]
                route_relation = "next_movement"
        if route is None:
            rows.append(
                {
                    "session": str(event.session),
                    "rat": str(event.session).split("/", 1)[0],
                    "event_index": int(event.event_index),
                    "event_peak_s": event_time,
                    "route_library_eligible": False,
                    "exclusion_reason": "no_current_or_future_behavior_route",
                }
            )
            continue
        fold = int(route["cv_fold"])
        library = primitives[
            primitives["session"].eq(str(event.session))
            & primitives["excluded_cv_fold"].eq(fold)
        ]
        source_ids = set()
        for value in library.get("source_route_ids", pd.Series(dtype=str)).dropna().astype(str):
            source_ids.update(item for item in value.split("|") if item)
        future = route_points[
            route_points["route_id"].eq(route["route_id"])
            & (route_points["time_s"] >= event_time)
        ]
        if route_relation == "next_movement":
            future = route_points[route_points["route_id"].eq(route["route_id"])]
        leakage = str(route["route_id"]) in source_ids
        rows.append(
            {
                "session": str(event.session),
                "rat": str(event.session).split("/", 1)[0],
                "event_index": int(event.event_index),
                "event_peak_s": event_time,
                "enclosing_route_id": str(route["route_id"]),
                "enclosing_route_index": int(route["route_index"]),
                "event_route_relation": route_relation,
                "excluded_cv_fold": fold,
                "origin_well_id": int(route["origin_well_id"]),
                "destination_well_id": int(route["destination_well_id"]),
                "route_interval_start_time_s": float(route["interval_start_time_s"]),
                "route_interval_end_time_s": float(route["interval_end_time_s"]),
                "route_movement_start_time_s": float(route["movement_start_time_s"]),
                "route_movement_end_time_s": float(route["movement_end_time_s"]),
                "time_to_route_movement_start_s": float(route["movement_start_time_s"] - event_time),
                "time_to_route_movement_end_s": float(route["movement_end_time_s"] - event_time),
                "future_route_points": int(len(future)),
                "candidate_primitives": int(len(library)),
                "candidate_source_routes": int(len(source_ids)),
                "enclosing_route_leaked_into_library": leakage,
                "route_library_eligible": bool(len(library) > 0 and not leakage),
                "exclusion_reason": "" if len(library) > 0 and not leakage else "incomplete_or_leaky_route_library",
                "behavior_only_library": True,
            }
        )
    return pd.DataFrame(rows)


def summarize_sessions(
    routes: pd.DataFrame,
    primitives: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    sessions = sorted(set(routes.get("session", pd.Series(dtype=str)).astype(str)) | set(events["session"].astype(str)))
    for session_id in sessions:
        route_group = routes[routes["session"].eq(session_id)]
        primitive_group = primitives[primitives["session"].eq(session_id)]
        event_group = events[events["session"].eq(session_id)]
        rows.append(
            {
                "session": session_id,
                "rat": session_id.split("/", 1)[0],
                "behavior_routes": int(len(route_group)),
                "route_folds_present": int(route_group["cv_fold"].nunique()) if not route_group.empty else 0,
                "cross_validated_primitives": int(len(primitive_group)),
                "median_primitives_per_excluded_fold": float(
                    primitive_group.groupby("excluded_cv_fold").size().median()
                ) if not primitive_group.empty else np.nan,
                "frozen_events": int(len(event_group)),
                "eligible_events": int(event_group.get("route_library_eligible", pd.Series(dtype=bool)).sum()),
                "median_candidate_primitives_per_event": float(event_group["candidate_primitives"].median())
                if "candidate_primitives" in event_group and not event_group.empty else np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_gate_summary(
    frozen_events: pd.DataFrame,
    routes: pd.DataFrame,
    primitives: pd.DataFrame,
    event_eligibility: pd.DataFrame,
    *,
    minimum_routes_per_session: int = 20,
    minimum_candidate_primitives: int = 3,
) -> pd.DataFrame:
    sessions = int(frozen_events["session"].nunique())
    rats = int(frozen_events["session"].astype(str).str.split("/").str[0].nunique())
    route_counts = routes.groupby("session").size() if not routes.empty else pd.Series(dtype=int)
    event_count = int(len(frozen_events))
    eligible = int(event_eligibility.get("route_library_eligible", pd.Series(dtype=bool)).sum())
    no_leakage = bool(
        len(event_eligibility) == event_count
        and not event_eligibility.get("enclosing_route_leaked_into_library", pd.Series([True])).fillna(True).any()
    )
    candidate_ready = bool(
        len(event_eligibility) == event_count
        and (event_eligibility.get("candidate_primitives", pd.Series(dtype=float)).fillna(0) >= int(minimum_candidate_primitives)).all()
    )
    gates = [
        ("frozen_events_present", event_count > 0, event_count, ">0"),
        ("all_sessions_have_routes", len(route_counts) == sessions, len(route_counts), sessions),
        (
            "minimum_routes_per_session",
            bool(len(route_counts) == sessions and (route_counts >= int(minimum_routes_per_session)).all()),
            int(route_counts.min()) if len(route_counts) else 0,
            int(minimum_routes_per_session),
        ),
        ("all_four_rats_retained", rats == 4, rats, 4),
        ("behavior_primitives_present", len(primitives) > 0, int(len(primitives)), ">0"),
        ("all_events_mapped", len(event_eligibility) == event_count, int(len(event_eligibility)), event_count),
        ("all_events_route_library_eligible", eligible == event_count, eligible, event_count),
        ("minimum_candidate_primitives_per_event", candidate_ready, bool(candidate_ready), True),
        ("no_enclosing_route_leakage", no_leakage, no_leakage, True),
        ("behavior_library_uses_no_replay_outcomes", True, True, True),
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
    output_dir: str | Path,
    n_folds: int = 5,
    median_window_s: float = 0.167,
    gaussian_sigma_s: float = 0.100,
    movement_speed_cm_s: float = 5.0,
    minimum_departure_s: float = 0.25,
    arrival_radius_cm: float = 10.0,
    minimum_arrival_dwell_s: float = 0.20,
    destination_window_s: float = 1.0,
    minimum_route_displacement_cm: float = 10.0,
    primitive_length_cm: float = 40.0,
    primitive_stride_cm: float = 20.0,
    cluster_threshold_cm: float = 12.5,
    minimum_cluster_routes: int = 2,
    spatial_bin_cm: float = 10.0,
    transition_pseudocount: float = 0.5,
) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frozen_events = pd.read_csv(frozen_events_csv)
    dataset = Path(dataset_root)

    all_routes: list[pd.DataFrame] = []
    all_route_points: list[pd.DataFrame] = []
    sessions: dict[str, object] = {}
    for session_id in sorted(frozen_events["session"].astype(str).unique()):
        session = load_replay_session(dataset / Path(session_id))
        sessions[session_id] = session
        smoothed = smooth_position_trace(
            np.asarray(session.position),
            median_window_s=float(median_window_s),
            gaussian_sigma_s=float(gaussian_sigma_s),
        )
        routes, points = segment_well_to_well_routes(
            session_id=session_id,
            position=smoothed,
            run_times=np.asarray(session.run_times),
            well_sequence=np.asarray(session.well_sequence),
            n_folds=int(n_folds),
            speed_threshold_cm_s=float(movement_speed_cm_s),
            minimum_departure_s=float(minimum_departure_s),
            arrival_radius_cm=float(arrival_radius_cm),
            minimum_arrival_dwell_s=float(minimum_arrival_dwell_s),
            destination_window_s=float(destination_window_s),
            minimum_route_displacement_cm=float(minimum_route_displacement_cm),
        )
        all_routes.append(routes)
        all_route_points.append(points)
    routes = pd.concat(all_routes, ignore_index=True) if all_routes else pd.DataFrame()
    route_points = pd.concat(all_route_points, ignore_index=True) if all_route_points else pd.DataFrame()
    candidate_meta, candidate_paths = extract_fixed_length_subpaths(
        routes,
        route_points,
        primitive_length_cm=float(primitive_length_cm),
        primitive_stride_cm=float(primitive_stride_cm),
    )
    primitives, primitive_points = build_cross_validated_primitives(
        candidate_meta,
        candidate_paths,
        n_folds=int(n_folds),
        cluster_threshold_cm=float(cluster_threshold_cm),
        minimum_cluster_routes=int(minimum_cluster_routes),
    )
    transitions = build_transition_graph(
        routes,
        route_points,
        n_folds=int(n_folds),
        spatial_bin_cm=float(spatial_bin_cm),
        pseudocount=float(transition_pseudocount),
    )

    def session_loader(session_id: str) -> object:
        return sessions[session_id]

    event_eligibility = map_events_to_route_libraries(
        frozen_events,
        routes,
        primitives,
        route_points,
        session_loader=session_loader,
    )
    by_session = summarize_sessions(routes, primitives, event_eligibility)
    gates = build_gate_summary(frozen_events, routes, primitives, event_eligibility)

    frames = {
        ROUTE_OUTPUT: routes,
        ROUTE_POINT_OUTPUT: route_points,
        PRIMITIVE_OUTPUT: primitives,
        PRIMITIVE_POINT_OUTPUT: primitive_points,
        TRANSITION_OUTPUT: transitions,
        EVENT_OUTPUT: event_eligibility,
        SESSION_OUTPUT: by_session,
        GATE_OUTPUT: gates,
    }
    paths: dict[str, Path] = {}
    for name, frame in frames.items():
        path = output / name
        frame.to_csv(path, index=False)
        paths[name] = path

    provenance = build_script_provenance(
        input_paths={
            "dataset_root": dataset_root,
            "frozen_events_csv": frozen_events_csv,
        },
        cwd=ROOT,
    )
    manifest = {
        "analysis": "replay_behavior_route_primitives",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "non_outcome_behavior_only": True,
        "cross_validation_unit": "well_to_well_route",
        "parameters": {
            "n_folds": int(n_folds),
            "median_window_s": float(median_window_s),
            "gaussian_sigma_s": float(gaussian_sigma_s),
            "movement_speed_cm_s": float(movement_speed_cm_s),
            "minimum_departure_s": float(minimum_departure_s),
            "arrival_radius_cm": float(arrival_radius_cm),
            "minimum_arrival_dwell_s": float(minimum_arrival_dwell_s),
            "destination_window_s": float(destination_window_s),
            "minimum_route_displacement_cm": float(minimum_route_displacement_cm),
            "primitive_length_cm": float(primitive_length_cm),
            "primitive_stride_cm": float(primitive_stride_cm),
            "cluster_threshold_cm": float(cluster_threshold_cm),
            "minimum_cluster_routes": int(minimum_cluster_routes),
            "spatial_bin_cm": float(spatial_bin_cm),
            "transition_pseudocount": float(transition_pseudocount),
        },
        "counts": {
            "routes": int(len(routes)),
            "candidate_subpaths": int(len(candidate_meta)),
            "cross_validated_primitives": int(len(primitives)),
            "events": int(len(event_eligibility)),
            "eligible_events": int(event_eligibility.get("route_library_eligible", pd.Series(dtype=bool)).sum()),
        },
        "outputs": {name: str(path) for name, path in paths.items()},
        "provenance": provenance,
    }
    manifest_path = output / MANIFEST_OUTPUT
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths[MANIFEST_OUTPUT] = manifest_path

    overall = bool(gates.loc[gates["gate"].eq("overall"), "passed"].iloc[0])
    summary_lines = [
        "# Replay behavior route-primitives audit",
        "",
        "This is a behavior-only, non-outcome artifact. Replay-model scores and posteriors were not used to learn routes or primitives.",
        "",
        f"- Behavior routes: {len(routes)}",
        f"- Candidate fixed-length subpaths: {len(candidate_meta)}",
        f"- Cross-validated primitive rows: {len(primitives)}",
        f"- Frozen events mapped: {len(event_eligibility)}",
        f"- Route-library eligible events: {int(event_eligibility.get('route_library_eligible', pd.Series(dtype=bool)).sum())}",
        f"- Overall gate: {'PASS' if overall else 'FAIL'}",
        "",
        "Each event uses the primitive library for which its enclosing behavior route was excluded. Full route outcomes remain separate and are used only by later commitment tests.",
    ]
    summary_path = output / SUMMARY_OUTPUT
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    paths[SUMMARY_OUTPUT] = summary_path
    return paths


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--frozen-events", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--median-window-s", type=float, default=0.167)
    parser.add_argument("--gaussian-sigma-s", type=float, default=0.100)
    parser.add_argument("--movement-speed-cm-s", type=float, default=5.0)
    parser.add_argument("--minimum-departure-s", type=float, default=0.25)
    parser.add_argument("--arrival-radius-cm", type=float, default=10.0)
    parser.add_argument("--minimum-arrival-dwell-s", type=float, default=0.20)
    parser.add_argument("--destination-window-s", type=float, default=1.0)
    parser.add_argument("--minimum-route-displacement-cm", type=float, default=10.0)
    parser.add_argument("--primitive-length-cm", type=float, default=40.0)
    parser.add_argument("--primitive-stride-cm", type=float, default=20.0)
    parser.add_argument("--cluster-threshold-cm", type=float, default=12.5)
    parser.add_argument("--minimum-cluster-routes", type=int, default=2)
    parser.add_argument("--spatial-bin-cm", type=float, default=10.0)
    parser.add_argument("--transition-pseudocount", type=float, default=0.5)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_analysis(
        dataset_root=args.dataset_root,
        frozen_events_csv=args.frozen_events,
        output_dir=args.output_dir,
        n_folds=args.n_folds,
        median_window_s=args.median_window_s,
        gaussian_sigma_s=args.gaussian_sigma_s,
        movement_speed_cm_s=args.movement_speed_cm_s,
        minimum_departure_s=args.minimum_departure_s,
        arrival_radius_cm=args.arrival_radius_cm,
        minimum_arrival_dwell_s=args.minimum_arrival_dwell_s,
        destination_window_s=args.destination_window_s,
        minimum_route_displacement_cm=args.minimum_route_displacement_cm,
        primitive_length_cm=args.primitive_length_cm,
        primitive_stride_cm=args.primitive_stride_cm,
        cluster_threshold_cm=args.cluster_threshold_cm,
        minimum_cluster_routes=args.minimum_cluster_routes,
        spatial_bin_cm=args.spatial_bin_cm,
        transition_pseudocount=args.transition_pseudocount,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
