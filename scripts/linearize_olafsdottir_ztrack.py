#!/usr/bin/env python3
"""Linearize Olafsdottir Z-track position samples onto a 1D centerline."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from hipporeplayimm.olafsdottir2016 import read_axona_pos


def load_centerline(path: str | Path) -> np.ndarray:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    points = payload.get("points_cm", payload)
    arr = np.asarray(points, dtype=float)
    return validate_centerline(arr)


def validate_centerline(centerline: np.ndarray) -> np.ndarray:
    arr = np.asarray(centerline, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] < 2:
        raise ValueError("centerline must be an N x 2 array with at least two points")
    if not np.all(np.isfinite(arr)):
        raise ValueError("centerline points must be finite")
    segment_lengths = np.linalg.norm(np.diff(arr, axis=0), axis=1)
    if np.any(segment_lengths <= 0.0):
        raise ValueError("centerline contains repeated adjacent points")
    return arr


def smooth_positions(xy: np.ndarray, valid: np.ndarray, *, window_samples: int) -> np.ndarray:
    """Interpolate invalid samples and apply a short moving-average smoother."""

    xy = np.asarray(xy, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("xy must be N x 2")
    if valid.shape != (xy.shape[0],):
        raise ValueError("valid mask must contain one value per position sample")
    if xy.shape[0] == 0:
        return xy.copy()
    if not np.any(valid):
        return xy.copy()

    sample_index = np.arange(xy.shape[0], dtype=float)
    filled = xy.copy()
    for dim in range(2):
        values = xy[:, dim]
        filled[:, dim] = np.interp(sample_index, sample_index[valid], values[valid])
    window = max(int(window_samples), 1)
    if window <= 1:
        return filled
    kernel = np.ones(window, dtype=float) / float(window)
    return np.column_stack(
        [
            np.convolve(filled[:, dim], kernel, mode="same")
            for dim in range(2)
        ]
    )


def infer_centerline_from_positions(
    xy: np.ndarray,
    valid: np.ndarray,
    *,
    bin_size_cm: float = 4.0,
    simplify_step_cm: float = 4.0,
) -> np.ndarray:
    """Infer a conservative centerline as the diameter path of occupied bins."""

    xy = np.asarray(xy, dtype=float)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(xy).all(axis=1)
    if valid.sum() < 2:
        raise ValueError("at least two valid samples are needed to infer a centerline")
    bin_size = float(bin_size_cm)
    if not np.isfinite(bin_size) or bin_size <= 0.0:
        raise ValueError("bin_size_cm must be finite and positive")

    origin = np.nanmin(xy[valid], axis=0)
    occupied = np.floor((xy[valid] - origin) / bin_size).astype(int)
    cells = np.unique(occupied, axis=0)
    keep = _largest_component_mask(cells)
    cells = cells[keep]
    if cells.shape[0] < 2:
        raise ValueError("occupied track component is too small to infer a centerline")
    graph = _neighbor_graph(cells, bin_size)
    endpoint_a, _ = _farthest_node(graph, 0)
    endpoint_b, _ = _farthest_node(graph, endpoint_a)
    path_indices = _shortest_path(graph, endpoint_a, endpoint_b)
    points = origin + (cells[np.asarray(path_indices, dtype=int)] + 0.5) * bin_size
    return simplify_centerline(points, min_step_cm=simplify_step_cm)


def simplify_centerline(points: np.ndarray, *, min_step_cm: float) -> np.ndarray:
    arr = validate_centerline(points)
    min_step = max(float(min_step_cm), 0.0)
    if min_step <= 0.0 or arr.shape[0] <= 2:
        return arr
    kept = [arr[0]]
    distance_since_keep = 0.0
    for prev, point in zip(arr[:-1], arr[1:]):
        distance_since_keep += float(np.linalg.norm(point - prev))
        if distance_since_keep >= min_step:
            kept.append(point)
            distance_since_keep = 0.0
    if not np.allclose(kept[-1], arr[-1]):
        kept.append(arr[-1])
    return validate_centerline(np.asarray(kept, dtype=float))


def project_points_to_centerline(
    xy: np.ndarray,
    valid: np.ndarray,
    centerline: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project 2D points onto a polyline and return linear coordinate/error."""

    xy = np.asarray(xy, dtype=float)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(xy).all(axis=1)
    center = validate_centerline(centerline)
    starts = center[:-1]
    ends = center[1:]
    vectors = ends - starts
    lengths = np.linalg.norm(vectors, axis=1)
    cumulative = np.r_[0.0, np.cumsum(lengths)]
    linear = np.full(xy.shape[0], np.nan, dtype=float)
    error = np.full(xy.shape[0], np.nan, dtype=float)
    if not np.any(valid):
        return linear, error

    valid_indices = np.flatnonzero(valid)
    for chunk in np.array_split(valid_indices, max(1, math.ceil(valid_indices.shape[0] / 5000))):
        points = xy[chunk]
        best_error = np.full(points.shape[0], np.inf, dtype=float)
        best_linear = np.full(points.shape[0], np.nan, dtype=float)
        for segment_index, (start, vector, length) in enumerate(zip(starts, vectors, lengths)):
            rel = points - start
            t = np.clip((rel @ vector) / (length * length), 0.0, 1.0)
            projected = start + t[:, None] * vector
            distances = np.linalg.norm(points - projected, axis=1)
            improve = distances < best_error
            best_error[improve] = distances[improve]
            best_linear[improve] = cumulative[segment_index] + t[improve] * length
        linear[chunk] = best_linear
        error[chunk] = best_error
    return linear, error


def speed_from_linear_position(times_s: np.ndarray, linear_cm: np.ndarray, valid: np.ndarray) -> np.ndarray:
    times = np.asarray(times_s, dtype=float)
    linear = np.asarray(linear_cm, dtype=float)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(linear) & np.isfinite(times)
    speed = np.full(linear.shape, np.nan, dtype=float)
    if valid.sum() < 2:
        return speed
    filled = linear.copy()
    idx = np.arange(linear.shape[0], dtype=float)
    filled[~valid] = np.interp(idx[~valid], idx[valid], linear[valid])
    dt = np.gradient(times)
    with np.errstate(divide="ignore", invalid="ignore"):
        speed_values = np.abs(np.gradient(filled) / dt)
    speed[valid] = speed_values[valid]
    return speed


def occupancy_by_linear_bin(
    linear_cm: np.ndarray,
    times_s: np.ndarray,
    valid: np.ndarray,
    *,
    bin_size_cm: float,
    track_length_cm: float,
) -> pd.DataFrame:
    bin_size = float(bin_size_cm)
    edges = np.arange(0.0, float(track_length_cm) + bin_size, bin_size)
    if edges.shape[0] < 2:
        edges = np.array([0.0, bin_size], dtype=float)
    dt = _sample_durations(times_s)
    keep = np.asarray(valid, dtype=bool) & np.isfinite(linear_cm)
    bins = np.searchsorted(edges, linear_cm[keep], side="right") - 1
    bins = np.clip(bins, 0, edges.shape[0] - 2)
    occupancy = np.zeros(edges.shape[0] - 1, dtype=float)
    np.add.at(occupancy, bins, dt[keep])
    return pd.DataFrame(
        {
            "metric": "occupancy_by_linear_bin",
            "bin_start_cm": edges[:-1],
            "bin_end_cm": edges[1:],
            "value": occupancy,
            "unit": "s",
        }
    )


def linearize_pos_file(
    pos_path: str | Path,
    output_dir: str | Path,
    *,
    centerline_path: str | Path | None = None,
    infer_bin_size_cm: float = 4.0,
    simplify_step_cm: float = 4.0,
    smoothing_window_samples: int = 5,
    occupancy_bin_size_cm: float = 5.0,
) -> dict[str, float | str]:
    position = read_axona_pos(pos_path)
    xy_raw = np.column_stack([position.x_cm, position.y_cm])
    xy = smooth_positions(xy_raw, position.valid, window_samples=smoothing_window_samples)
    if centerline_path is None:
        centerline = infer_centerline_from_positions(
            xy,
            position.valid,
            bin_size_cm=infer_bin_size_cm,
            simplify_step_cm=simplify_step_cm,
        )
        geometry_source = "inferred_occupied_bin_diameter"
    else:
        centerline = load_centerline(centerline_path)
        geometry_source = "configured_centerline"
    linear, projection_error = project_points_to_centerline(xy, position.valid, centerline)
    valid_projected = position.valid & np.isfinite(linear) & np.isfinite(projection_error)
    speed = speed_from_linear_position(position.times_s, linear, valid_projected)
    segment_lengths = np.linalg.norm(np.diff(centerline, axis=0), axis=1)
    cumulative = np.r_[0.0, np.cumsum(segment_lengths)]
    track_length = float(cumulative[-1])

    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "time_s": position.times_s,
            "x_cm": position.x_cm,
            "y_cm": position.y_cm,
            "linear_position_cm": linear,
            "speed_cm_s": speed,
            "valid_position": valid_projected,
        }
    ).to_csv(outdir / "linearized_position.csv", index=False)

    finite_errors = projection_error[valid_projected]
    finite_times = position.times_s[np.isfinite(position.times_s)]
    position_start_time_s = float(finite_times[0]) if finite_times.size else np.nan
    position_end_time_s = float(finite_times[-1]) if finite_times.size else np.nan
    session_duration_s = (
        float(position_end_time_s - position_start_time_s)
        if np.isfinite(position_start_time_s) and np.isfinite(position_end_time_s)
        else np.nan
    )
    diagnostics = [
        {
            "metric": "fraction_valid_position",
            "value": float(np.mean(valid_projected)) if valid_projected.size else 0.0,
            "unit": "fraction",
            "bin_start_cm": np.nan,
            "bin_end_cm": np.nan,
        },
        {
            "metric": "median_projection_error_cm",
            "value": float(np.nanmedian(finite_errors)) if finite_errors.size else np.nan,
            "unit": "cm",
            "bin_start_cm": np.nan,
            "bin_end_cm": np.nan,
        },
        {
            "metric": "max_projection_error_cm",
            "value": float(np.nanmax(finite_errors)) if finite_errors.size else np.nan,
            "unit": "cm",
            "bin_start_cm": np.nan,
            "bin_end_cm": np.nan,
        },
        {
            "metric": "track_length_cm",
            "value": track_length,
            "unit": "cm",
            "bin_start_cm": np.nan,
            "bin_end_cm": np.nan,
        },
        {
            "metric": "position_start_time_s",
            "value": position_start_time_s,
            "unit": "s",
            "bin_start_cm": np.nan,
            "bin_end_cm": np.nan,
        },
        {
            "metric": "position_end_time_s",
            "value": position_end_time_s,
            "unit": "s",
            "bin_start_cm": np.nan,
            "bin_end_cm": np.nan,
        },
        {
            "metric": "session_duration_s",
            "value": session_duration_s,
            "unit": "s",
            "bin_start_cm": np.nan,
            "bin_end_cm": np.nan,
        },
    ]
    diag_df = pd.concat(
        [
            pd.DataFrame(diagnostics),
            occupancy_by_linear_bin(
                linear,
                position.times_s,
                valid_projected,
                bin_size_cm=occupancy_bin_size_cm,
                track_length_cm=track_length,
            ),
        ],
        ignore_index=True,
    )
    diag_df.to_csv(outdir / "linearization_diagnostics.csv", index=False)

    geometry = {
        "source": geometry_source,
        "pos_path": str(pos_path),
        "centerline_points_cm": centerline.tolist(),
        "centerline_distance_cm": cumulative.tolist(),
        "track_length_cm": track_length,
        "parameters": {
            "infer_bin_size_cm": infer_bin_size_cm,
            "simplify_step_cm": simplify_step_cm,
            "smoothing_window_samples": smoothing_window_samples,
            "occupancy_bin_size_cm": occupancy_bin_size_cm,
        },
        "diagnostics": {
            row["metric"]: row["value"]
            for row in diagnostics
        },
        "acceptance_guidance": {
            "fraction_valid_position": "> 0.90 during track running",
            "projection_error": "inspect median/max relative to track width",
            "occupancy": "inspect occupancy_by_linear_bin rows for clear coverage",
        },
    }
    (outdir / "track_geometry.json").write_text(
        json.dumps(geometry, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "linearized_position": str(outdir / "linearized_position.csv"),
        "track_geometry": str(outdir / "track_geometry.json"),
        "linearization_diagnostics": str(outdir / "linearization_diagnostics.csv"),
        "fraction_valid_position": float(np.mean(valid_projected)) if valid_projected.size else 0.0,
        "median_projection_error_cm": float(np.nanmedian(finite_errors)) if finite_errors.size else np.nan,
        "track_length_cm": track_length,
        "position_start_time_s": position_start_time_s,
        "position_end_time_s": position_end_time_s,
        "session_duration_s": session_duration_s,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pos", required=True, type=Path, help="Input Axona .pos file.")
    parser.add_argument("--output", required=True, type=Path, help="Output directory.")
    parser.add_argument(
        "--centerline-json",
        type=Path,
        default=None,
        help="Optional JSON centerline as {'points_cm': [[x,y], ...]}; omitted means infer from occupancy.",
    )
    parser.add_argument("--infer-bin-size-cm", type=float, default=4.0)
    parser.add_argument("--simplify-step-cm", type=float, default=4.0)
    parser.add_argument("--smoothing-window-samples", type=int, default=5)
    parser.add_argument("--occupancy-bin-size-cm", type=float, default=5.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    summary = linearize_pos_file(
        args.pos,
        args.output,
        centerline_path=args.centerline_json,
        infer_bin_size_cm=args.infer_bin_size_cm,
        simplify_step_cm=args.simplify_step_cm,
        smoothing_window_samples=args.smoothing_window_samples,
        occupancy_bin_size_cm=args.occupancy_bin_size_cm,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _largest_component_mask(cells: np.ndarray) -> np.ndarray:
    graph = _neighbor_graph(cells, bin_size_cm=1.0)
    visited = np.zeros(cells.shape[0], dtype=bool)
    best: list[int] = []
    for start in range(cells.shape[0]):
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        component: list[int] = []
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbor, _weight in graph[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)
        if len(component) > len(best):
            best = component
    keep = np.zeros(cells.shape[0], dtype=bool)
    keep[best] = True
    return keep


def _neighbor_graph(cells: np.ndarray, bin_size_cm: float) -> list[list[tuple[int, float]]]:
    by_coord = {tuple(map(int, coord)): i for i, coord in enumerate(cells)}
    graph: list[list[tuple[int, float]]] = [[] for _ in range(cells.shape[0])]
    for i, (x, y) in enumerate(cells):
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                j = by_coord.get((int(x + dx), int(y + dy)))
                if j is not None:
                    graph[i].append((j, math.hypot(dx, dy) * float(bin_size_cm)))
    return graph


def _farthest_node(graph: list[list[tuple[int, float]]], start: int) -> tuple[int, np.ndarray]:
    distances, _previous = _dijkstra(graph, start)
    finite = np.isfinite(distances)
    if not finite.any():
        return start, distances
    indices = np.flatnonzero(finite)
    return int(indices[np.argmax(distances[finite])]), distances


def _shortest_path(graph: list[list[tuple[int, float]]], start: int, end: int) -> list[int]:
    _distances, previous = _dijkstra(graph, start)
    path = [int(end)]
    while path[-1] != int(start):
        prior = previous[path[-1]]
        if prior < 0:
            raise ValueError("occupied-bin graph endpoints are disconnected")
        path.append(int(prior))
    return list(reversed(path))


def _dijkstra(graph: list[list[tuple[int, float]]], start: int) -> tuple[np.ndarray, np.ndarray]:
    import heapq

    distances = np.full(len(graph), np.inf, dtype=float)
    previous = np.full(len(graph), -1, dtype=int)
    distances[start] = 0.0
    heap: list[tuple[float, int]] = [(0.0, int(start))]
    while heap:
        dist, node = heapq.heappop(heap)
        if dist > distances[node]:
            continue
        for neighbor, weight in graph[node]:
            candidate = dist + weight
            if candidate < distances[neighbor]:
                distances[neighbor] = candidate
                previous[neighbor] = node
                heapq.heappush(heap, (candidate, neighbor))
    return distances, previous


def _sample_durations(times_s: np.ndarray) -> np.ndarray:
    times = np.asarray(times_s, dtype=float)
    if times.shape[0] == 0:
        return np.empty(0, dtype=float)
    if times.shape[0] == 1:
        return np.ones(1, dtype=float)
    diffs = np.diff(times)
    finite_positive = diffs[np.isfinite(diffs) & (diffs > 0.0)]
    fallback = float(np.median(finite_positive)) if finite_positive.size else 1.0 / 50.0
    durations = np.empty(times.shape[0], dtype=float)
    durations[:-1] = np.where(np.isfinite(diffs) & (diffs > 0.0), diffs, fallback)
    durations[-1] = fallback
    return np.minimum(durations, 1.0)


if __name__ == "__main__":
    raise SystemExit(main())
