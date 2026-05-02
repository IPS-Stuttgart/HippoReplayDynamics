#!/usr/bin/env python3
"""Track a batch of Pfeiffer/Foster replay events and models."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from hipporeplayimm.benchmarks import BenchmarkConfig, _build_models
from hipporeplayimm.data import load_replay_session
from hipporeplayimm.encoding import EmissionConfig, build_emissions, fit_place_field_encoding

from track_event import _session_path, _trajectory_from_prefix_scores, _validate_session_files


def parse_event_ids(spec: str) -> list[int]:
    """Parse event ids from comma-separated values and inclusive ranges."""
    event_ids: list[int] = []
    for part in spec.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            start_str, end_str = token.split("-", maxsplit=1)
            start = int(start_str)
            end = int(end_str)
            if end < start:
                raise ValueError(f"Invalid descending event range: {token}")
            event_ids.extend(range(start, end + 1))
        else:
            event_ids.append(int(token))
    if not event_ids:
        raise ValueError("At least one event id is required")
    return sorted(dict.fromkeys(event_ids))


def _safe_name(value: str) -> str:
    return value.replace("/", "_").replace("\\", "_")


def _path_length(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return 0.0
    dx = np.diff(x)
    dy = np.diff(y)
    return float(np.sum(np.sqrt(dx * dx + dy * dy)))


def _summary_row(
    *,
    session: str,
    event_id: int,
    model_name: str,
    trajectory: pd.DataFrame,
    runtime_s: float,
    csv_path: Path,
    npz_path: Path,
) -> dict[str, object]:
    mean_x = trajectory["posterior_mean_x"].to_numpy(dtype=float)
    mean_y = trajectory["posterior_mean_y"].to_numpy(dtype=float)
    map_x = trajectory["map_x"].to_numpy(dtype=float)
    map_y = trajectory["map_y"].to_numpy(dtype=float)
    duration = 0.0
    if len(trajectory) > 1:
        duration = float(trajectory["time_s"].iloc[-1] - trajectory["time_s"].iloc[0])

    return {
        "status": "success",
        "session": session,
        "event_id": int(event_id),
        "model": model_name,
        "n_time_bins": int(len(trajectory)),
        "duration_s": duration,
        "total_spikes": int(trajectory["spikes_in_bin"].sum()),
        "mean_posterior_entropy": float(trajectory["posterior_entropy"].mean()),
        "median_posterior_entropy": float(trajectory["posterior_entropy"].median()),
        "min_posterior_entropy": float(trajectory["posterior_entropy"].min()),
        "max_posterior_entropy": float(trajectory["posterior_entropy"].max()),
        "mean_map_probability": float(trajectory["map_probability"].mean()),
        "max_map_probability": float(trajectory["map_probability"].max()),
        "posterior_mean_path_length": _path_length(mean_x, mean_y),
        "map_path_length": _path_length(map_x, map_y),
        "posterior_mean_start_x": float(mean_x[0]),
        "posterior_mean_start_y": float(mean_y[0]),
        "posterior_mean_end_x": float(mean_x[-1]),
        "posterior_mean_end_y": float(mean_y[-1]),
        "map_start_x": float(map_x[0]),
        "map_start_y": float(map_y[0]),
        "map_end_x": float(map_x[-1]),
        "map_end_y": float(map_y[-1]),
        "runtime_s": runtime_s,
        "trajectory_csv": str(csv_path),
        "posterior_npz": str(npz_path),
        "error": "",
    }


def _failure_row(
    *,
    session: str,
    event_id: int,
    model_name: str,
    runtime_s: float,
    error: Exception,
) -> dict[str, object]:
    return {
        "status": "failure",
        "session": session,
        "event_id": int(event_id),
        "model": model_name,
        "n_time_bins": 0,
        "duration_s": 0.0,
        "total_spikes": 0,
        "mean_posterior_entropy": np.nan,
        "median_posterior_entropy": np.nan,
        "min_posterior_entropy": np.nan,
        "max_posterior_entropy": np.nan,
        "mean_map_probability": np.nan,
        "max_map_probability": np.nan,
        "posterior_mean_path_length": np.nan,
        "map_path_length": np.nan,
        "posterior_mean_start_x": np.nan,
        "posterior_mean_start_y": np.nan,
        "posterior_mean_end_x": np.nan,
        "posterior_mean_end_y": np.nan,
        "map_start_x": np.nan,
        "map_start_y": np.nan,
        "map_end_x": np.nan,
        "map_end_y": np.nan,
        "runtime_s": runtime_s,
        "trajectory_csv": "",
        "posterior_npz": "",
        "error": f"{type(error).__name__}: {error}",
    }


def run_batch(args: argparse.Namespace) -> Path:
    event_ids = parse_event_ids(args.events)
    session_path = _session_path(args.dataset_root, args.session)
    if not session_path.is_dir():
        raise FileNotFoundError(f"Requested session directory does not exist: {session_path}")
    _validate_session_files(session_path)

    output_dir = Path(args.output)
    trajectory_dir = output_dir / "trajectories"
    posterior_dir = output_dir / "posteriors"
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    posterior_dir.mkdir(parents=True, exist_ok=True)

    session = load_replay_session(session_path)
    encoding = fit_place_field_encoding(session)
    config = BenchmarkConfig(
        candidate_top_k=args.candidate_top_k,
        pyrecest_particles=args.pyrecest_particles,
        models=tuple(args.models),
    )
    models = _build_models(config, session=session)

    rows: list[dict[str, object]] = []
    safe_session = _safe_name(args.session)

    for event_id in event_ids:
        if event_id < 0 or event_id >= session.ripple_count:
            raise IndexError(
                f"Event id {event_id} is outside available range 0..{session.ripple_count - 1}"
            )
        emissions = build_emissions(
            session,
            encoding,
            int(event_id),
            EmissionConfig(time_bin_s=args.time_bin_s),
        )
        for model_name in args.models:
            model = models[model_name]
            start_time = time.perf_counter()
            try:
                trajectory, log_posteriors = _trajectory_from_prefix_scores(
                    model,
                    emissions,
                    encoding.bin_centers,
                )
                runtime_s = time.perf_counter() - start_time
                safe_model = _safe_name(model_name)
                stem = f"{safe_session}_event{int(event_id):04d}_{safe_model}"
                csv_path = trajectory_dir / f"{stem}_trajectory.csv"
                npz_path = posterior_dir / f"{stem}_posterior.npz"
                trajectory.to_csv(csv_path, index=False)
                np.savez_compressed(
                    npz_path,
                    log_posteriors=log_posteriors,
                    times=emissions.times,
                    bin_centers=encoding.bin_centers,
                    x_edges=encoding.x_edges,
                    y_edges=encoding.y_edges,
                    grid_shape=np.asarray(encoding.grid_shape, dtype=int),
                    cell_ids=encoding.cell_ids,
                    spike_counts=emissions.spike_counts,
                )
                rows.append(
                    _summary_row(
                        session=args.session,
                        event_id=event_id,
                        model_name=model_name,
                        trajectory=trajectory,
                        runtime_s=runtime_s,
                        csv_path=csv_path,
                        npz_path=npz_path,
                    )
                )
                print(f"Tracked {args.session} event {event_id} with {model_name}")
            except Exception as exc:
                runtime_s = time.perf_counter() - start_time
                rows.append(
                    _failure_row(
                        session=args.session,
                        event_id=event_id,
                        model_name=model_name,
                        runtime_s=runtime_s,
                        error=exc,
                    )
                )
                print(
                    f"Failed {args.session} event {event_id} with {model_name}: {exc}",
                    flush=True,
                )
                if not args.continue_on_error:
                    raise

    summary_path = output_dir / "batch_summary.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    print(f"Wrote batch summary: {summary_path}")
    return summary_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Track a batch of replay events and models.")
    parser.add_argument("--dataset-root", required=True, help="Path to DataSetFromPfeifferFoster.")
    parser.add_argument("--session", required=True, help="Session ID, e.g. Rat1/Open1.")
    parser.add_argument("--events", default="0-25", help="Comma-separated event IDs and ranges, e.g. 0-25,30.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["stationary", "diffusion", "momentum", "imm"],
        choices=(
            "random",
            "stationary",
            "diffusion",
            "momentum",
            "imm",
            "pyrecest-goal-particle",
            "pyrecest-goal-particle-imm",
        ),
    )
    parser.add_argument("--candidate-top-k", default=64, type=int)
    parser.add_argument("--pyrecest-particles", default=512, type=int)
    parser.add_argument("--time-bin-s", default=0.02, type=float)
    parser.add_argument("--output", default="results/track-batch")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep processing remaining event/model pairs if one pair fails.",
    )
    run_batch(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
