#!/usr/bin/env python3
"""Export time-resolved replay trajectories for one Pfeiffer/Foster ripple event."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from hipporeplayimm.benchmarks import BenchmarkConfig, _build_models
from hipporeplayimm.data import load_replay_session
from hipporeplayimm.encoding import EmissionConfig, LogEmissionTensor, build_emissions, fit_place_field_encoding


_REQUIRED_SESSION_FILES = (
    "Position_Data.mat",
    "Ripple_Events.mat",
    "Spike_Data.mat",
    "Epochs.mat",
)


def _session_path(dataset_root: str | Path, session_id: str) -> Path:
    """Resolve a Rat/Open session ID to one session directory."""
    parts = session_id.replace("\\", "/").split("/")
    if len(parts) != 2 or not all(parts):
        raise ValueError("session must have the form 'RatN/OpenM', for example 'Rat1/Open1'")
    return Path(dataset_root) / parts[0] / parts[1]


def _validate_session_files(session_path: Path) -> None:
    """Fail early with a clear message if the requested session is incomplete."""
    missing = [name for name in _REQUIRED_SESSION_FILES if not (session_path / name).exists()]
    if missing:
        joined = ", ".join(missing)
        raise FileNotFoundError(f"Requested session {session_path} is missing required file(s): {joined}")


def _prefix_emissions(emissions: LogEmissionTensor, stop: int) -> LogEmissionTensor:
    """Return emissions restricted to time bins [0, stop)."""
    return LogEmissionTensor(
        log_likelihood=emissions.log_likelihood[:stop],
        spike_counts=emissions.spike_counts[:stop],
        times=emissions.times[:stop],
        dt=emissions.dt,
        cell_ids=emissions.cell_ids,
        n_spikes=int(emissions.spike_counts[:stop].sum()),
    )


def _trajectory_from_prefix_scores(model: object, emissions: LogEmissionTensor, bin_centers: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    """Track by repeatedly scoring prefixes and saving each terminal posterior.

    This intentionally reuses the repository's existing model implementations,
    so the workflow tracks with exactly the same dynamics used by event scoring.
    Ripple events are short, so the quadratic prefix computation is acceptable
    for manually dispatched exploratory runs.
    """
    rows: list[dict[str, float | int | str]] = []
    log_posteriors = np.empty((emissions.n_time, bin_centers.shape[0]), dtype=float)

    for time_index in range(emissions.n_time):
        score = model.score(_prefix_emissions(emissions, time_index + 1), bin_centers)
        if score.terminal_log_posterior is None:
            raise RuntimeError(f"Model {score.model_name} did not return a terminal posterior.")
        log_posterior = np.asarray(score.terminal_log_posterior, dtype=float)
        log_posteriors[time_index] = log_posterior
        posterior = np.exp(log_posterior)
        posterior /= max(float(posterior.sum()), np.finfo(float).tiny)
        mean_xy = posterior @ bin_centers
        map_bin = int(np.argmax(log_posterior))

        row: dict[str, float | int | str] = {
            "time_index": time_index,
            "time_s": float(emissions.times[time_index]),
            "posterior_mean_x": float(mean_xy[0]),
            "posterior_mean_y": float(mean_xy[1]),
            "map_x": float(bin_centers[map_bin, 0]),
            "map_y": float(bin_centers[map_bin, 1]),
            "map_bin": map_bin,
            "map_probability": float(posterior[map_bin]),
            "posterior_entropy": float(-np.sum(posterior * log_posterior)),
            "spikes_in_bin": int(emissions.spike_counts[time_index].sum()),
            "prefix_log_likelihood": float(score.log_likelihood),
        }
        for key, value in score.diagnostics.items():
            if key.startswith("pyrecest_") or key.startswith("decoded_"):
                row[f"diagnostic_{key}"] = value
        rows.append(row)

    return pd.DataFrame(rows), log_posteriors


def run_tracking(args: argparse.Namespace) -> None:
    session_path = _session_path(args.dataset_root, args.session)
    if not session_path.is_dir():
        raise FileNotFoundError(f"Requested session directory does not exist: {session_path}")
    _validate_session_files(session_path)

    session = load_replay_session(session_path)
    encoding = fit_place_field_encoding(session)
    emissions = build_emissions(session, encoding, int(args.event_id), EmissionConfig(time_bin_s=args.time_bin_s))
    config = BenchmarkConfig(
        candidate_top_k=args.candidate_top_k,
        pyrecest_particles=args.pyrecest_particles,
        models=(args.model,),
    )
    model = _build_models(config, session=session)[args.model]
    trajectory, log_posteriors = _trajectory_from_prefix_scores(model, emissions, encoding.bin_centers)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_session = args.session.replace("/", "_").replace("\\", "_")
    safe_model = args.model.replace("/", "_").replace("\\", "_")
    stem = f"{safe_session}_event{int(args.event_id):04d}_{safe_model}"

    csv_path = output_dir / f"{stem}_trajectory.csv"
    npz_path = output_dir / f"{stem}_posterior.npz"
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
    print(f"Wrote trajectory CSV: {csv_path}")
    print(f"Wrote posterior NPZ: {npz_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Track one Pfeiffer/Foster replay event.")
    parser.add_argument("--dataset-root", required=True, help="Path to DataSetFromPfeifferFoster.")
    parser.add_argument("--session", required=True, help="Session ID, e.g. Rat1/Open1.")
    parser.add_argument("--event-id", required=True, type=int, help="Ripple event index.")
    parser.add_argument(
        "--model",
        default="imm",
        choices=("random", "stationary", "diffusion", "momentum", "imm", "pyrecest-goal-particle", "pyrecest-goal-particle-imm"),
    )
    parser.add_argument("--candidate-top-k", default=64, type=int)
    parser.add_argument("--pyrecest-particles", default=512, type=int)
    parser.add_argument("--time-bin-s", default=0.02, type=float)
    parser.add_argument("--output", default="results/tracks")
    run_tracking(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
