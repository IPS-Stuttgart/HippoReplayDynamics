#!/usr/bin/env python3
"""Export time-resolved replay trajectories for one Pfeiffer/Foster ripple event."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from hipporeplayimm.benchmarks import BenchmarkConfig, _build_models
from hipporeplayimm.data import load_replay_session
from hipporeplayimm.duration_dynamics import DurationFloat, attach_duration_metadata
from hipporeplayimm.encoding import EmissionConfig, LogEmissionTensor, build_emissions, fit_place_field_encoding
from hipporeplayimm.models import CandidateKinematicModel, _advance_pair_log_alpha, _init_pair_log_alpha, _mode_transition_matrix

_REQUIRED_SESSION_FILES = ("Position_Data.mat", "Ripple_Events.mat", "Spike_Data.mat", "Epochs.mat")
_IMM_MODES = ("stationary", "diffusion", "momentum", "jump")
_TRACK_MODEL_CHOICES = (
    "random", "stationary", "diffusion", "momentum", "imm",
    "sorted-spike-state-space-stationary", "sorted-spike-state-space-diffusion", "sorted-spike-state-space-fragmented", "sorted-spike-state-space-jump",
    "sorted-spike-state-space-momentum", "sorted-spike-state-space-momentum-exact-sparse", "sorted-spike-state-space-trajectory-imm-exact-sparse",
    "sorted-spike-state-space-trajectory-imm-anchored-exact-sparse", "sorted-spike-state-space-trajectory-imm-low-leak-exact-sparse",
    "sorted-spike-state-space-trajectory-imm-persistent-exact-sparse", "sorted-spike-state-space-displacement-momentum",
    "sorted-spike-state-space-velocity-momentum", "sorted-spike-state-space-displacement-imm", "sorted-spike-state-space-imm",
    "sorted-spike-state-space-first-order-imm", "sorted-spike-state-space-goal", "state-space-stationary", "state-space-diffusion", "state-space-fragmented",
    "state-space-jump", "state-space-momentum", "state-space-momentum-exact-sparse", "state-space-trajectory-imm-exact-sparse",
    "state-space-displacement-momentum", "state-space-velocity-momentum", "state-space-displacement-imm", "state-space-imm",
    "state-space-first-order-imm", "state-space-goal", "pyrecest-goal-particle", "pyrecest-goal-particle-imm",
)


def _session_path(dataset_root: str | Path, session_id: str) -> Path:
    parts = session_id.replace("\\", "/").split("/")
    if len(parts) != 2 or not all(parts):
        raise ValueError("session must have the form 'RatN/OpenM', for example 'Rat1/Open1'")
    return Path(dataset_root) / parts[0] / parts[1]


def _validate_session_files(session_path: Path) -> None:
    missing = [name for name in _REQUIRED_SESSION_FILES if not (session_path / name).exists()]
    if missing:
        raise FileNotFoundError(f"Requested session {session_path} is missing required file(s): {', '.join(missing)}")


def _prefix_emissions(emissions: LogEmissionTensor, stop: int) -> LogEmissionTensor:
    bin_durations = getattr(emissions, "bin_durations", None)
    prefix = LogEmissionTensor(
        log_likelihood=emissions.log_likelihood[:stop],
        spike_counts=emissions.spike_counts[:stop],
        times=emissions.times[:stop],
        dt=float(getattr(emissions.dt, "base", emissions.dt)),
        cell_ids=emissions.cell_ids,
        n_spikes=int(emissions.spike_counts[:stop].sum()),
        bin_durations=None if bin_durations is None else np.asarray(bin_durations, dtype=float)[:stop],
    )
    transition_durations = getattr(emissions, "transition_durations", None)
    if transition_durations is None:
        transition_durations = getattr(emissions.dt, "transition_durations", None)
    if transition_durations is not None:
        prefix.transition_durations = np.asarray(transition_durations, dtype=float)[: max(stop - 1, 0)]
        attach_duration_metadata(prefix)
        prefix.dt = DurationFloat(prefix.dt, prefix.transition_durations)
    return prefix


def _mode_probability_row(modes: tuple[str, ...], probabilities: np.ndarray) -> dict[str, float | str]:
    probabilities = np.asarray(probabilities, dtype=float)
    total = float(probabilities.sum())
    if total <= 0.0:
        raise ValueError("mode probabilities must have positive total mass")
    probabilities = probabilities / total
    row: dict[str, float | str] = {f"mode_{mode}_probability": float(probability) for mode, probability in zip(modes, probabilities, strict=True)}
    row["most_likely_mode"] = modes[int(np.argmax(probabilities))]
    return row


def _imm_mode_probabilities_for_prefix(model: object, emissions: LogEmissionTensor, bin_centers: np.ndarray) -> dict[str, float | str]:
    if not isinstance(model, CandidateKinematicModel) or model.mode != "imm":
        return {}
    if emissions.n_time <= 1:
        return _mode_probability_row(_IMM_MODES, np.full(len(_IMM_MODES), 1.0 / len(_IMM_MODES)))

    candidates = model.candidate_indices(emissions)
    first, second = candidates[0], candidates[1]
    transition_modes = _mode_transition_matrix(len(_IMM_MODES), model.mode_stickiness)
    log_alpha = np.stack(
        [
            _init_pair_log_alpha(
                emissions,
                first,
                second,
                bin_centers,
                mode=mode,
                stationary_sigma_cm=model.stationary_sigma_cm,
                diffusion_sigma_cm=model.diffusion_sigma_cm,
                momentum_sigma_cm=model.momentum_sigma_cm,
            )
            for mode in _IMM_MODES
        ],
        axis=0,
    ) - np.log(len(_IMM_MODES))
    prev_prev, prev = first, second
    for time_index in range(2, emissions.n_time):
        curr = candidates[time_index]
        next_alpha = []
        for dst_mode_index, dst_mode in enumerate(_IMM_MODES):
            mixed_prev = logsumexp(log_alpha + np.log(transition_modes[:, dst_mode_index])[:, None, None], axis=0)
            next_alpha.append(
                _advance_pair_log_alpha(
                    mixed_prev,
                    prev_prev,
                    prev,
                    curr,
                    emissions.log_likelihood[time_index, curr],
                    bin_centers,
                    mode=dst_mode,
                    stationary_sigma_cm=model.stationary_sigma_cm,
                    diffusion_sigma_cm=model.diffusion_sigma_cm,
                    momentum_sigma_cm=model.momentum_sigma_cm,
                    velocity_decay=model.velocity_decay,
                )
            )
        log_alpha = np.stack(next_alpha, axis=0)
        prev_prev, prev = prev, curr

    log_mode_mass = logsumexp(log_alpha, axis=(1, 2))
    return _mode_probability_row(_IMM_MODES, np.exp(log_mode_mass - logsumexp(log_mode_mass)))


def _copy_score_diagnostics(row: dict[str, float | int | str], diagnostics: dict[str, float | int | str]) -> None:
    for key, value in diagnostics.items():
        if key.startswith("mode_") or key == "most_likely_mode":
            row[key] = value
        elif key.startswith("pyrecest_mode_") or key == "pyrecest_most_likely_mode":
            row[key] = value
        elif key.startswith("pyrecest_") or key.startswith("decoded_"):
            row[f"diagnostic_{key}"] = value


def _posterior_entropy(log_weights: np.ndarray) -> float:
    values = np.asarray(log_weights, dtype=float)
    normalizer = logsumexp(values)
    if not np.isfinite(normalizer):
        raise ValueError("posterior log weights must contain finite probability mass")
    normalized = values - normalizer
    posterior = np.exp(normalized)
    with np.errstate(invalid="ignore"):
        entropy_terms = np.where(posterior > 0.0, posterior * normalized, 0.0)
    return float(-np.sum(entropy_terms))


def _coordinate_or_zero(values: np.ndarray, axis: int) -> float:
    values = np.asarray(values, dtype=float)
    return float(values[axis]) if values.shape[0] > axis else 0.0


def _bin_coordinate_or_zero(bin_centers: np.ndarray, bin_index: int, axis: int) -> float:
    centers = np.asarray(bin_centers, dtype=float)
    return float(centers[bin_index, axis]) if centers.shape[1] > axis else 0.0


def _trajectory_row(
    time_index: int,
    log_posterior: np.ndarray,
    emissions: LogEmissionTensor,
    bin_centers: np.ndarray,
    likelihood_name: str,
    likelihood: float,
) -> dict[str, float | int | str]:
    normalized = np.asarray(log_posterior, dtype=float) - logsumexp(log_posterior)
    posterior = np.exp(normalized)
    mean_xy = posterior @ bin_centers
    map_bin = int(np.argmax(normalized))
    return {
        "time_index": time_index,
        "time_s": float(emissions.times[time_index]),
        "posterior_mean_x": _coordinate_or_zero(mean_xy, 0),
        "posterior_mean_y": _coordinate_or_zero(mean_xy, 1),
        "map_x": _bin_coordinate_or_zero(bin_centers, map_bin, 0),
        "map_y": _bin_coordinate_or_zero(bin_centers, map_bin, 1),
        "map_bin": map_bin,
        "map_probability": float(posterior[map_bin]),
        "posterior_entropy": _posterior_entropy(normalized),
        "spikes_in_bin": int(emissions.spike_counts[time_index].sum()),
        likelihood_name: float(likelihood),
    }


def _trajectory_rows_from_log_posteriors(
    *,
    log_posteriors: np.ndarray,
    emissions: LogEmissionTensor,
    bin_centers: np.ndarray,
    score,
    likelihood_column: str,
) -> pd.DataFrame:
    rows = []
    for time_index, log_posterior in enumerate(log_posteriors):
        row = _trajectory_row(time_index, log_posterior, emissions, bin_centers, likelihood_column, score.log_likelihood)
        _copy_score_diagnostics(row, score.diagnostics)
        rows.append(row)
    return pd.DataFrame(rows)


def _trajectory_from_prefix_scores(model: object, emissions: LogEmissionTensor, bin_centers: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    full_score = model.score(emissions, bin_centers)
    if full_score.trajectory_log_posterior is not None:
        log_posteriors = np.asarray(full_score.trajectory_log_posterior, dtype=float)
        trajectory = _trajectory_rows_from_log_posteriors(
            log_posteriors=log_posteriors,
            emissions=emissions,
            bin_centers=bin_centers,
            score=full_score,
            likelihood_column="event_log_likelihood",
        )
        if isinstance(model, CandidateKinematicModel) and model.mode == "imm":
            for time_index in range(emissions.n_time):
                mode_row = _imm_mode_probabilities_for_prefix(
                    model, _prefix_emissions(emissions, time_index + 1), bin_centers
                )
                for key, value in mode_row.items():
                    trajectory.loc[time_index, key] = value
        return trajectory, log_posteriors

    rows: list[dict[str, float | int | str]] = []
    log_posteriors = np.empty((emissions.n_time, bin_centers.shape[0]), dtype=float)
    for time_index in range(emissions.n_time):
        prefix = _prefix_emissions(emissions, time_index + 1)
        score = model.score(prefix, bin_centers)
        if score.terminal_log_posterior is None:
            raise RuntimeError(f"Model {score.model_name} did not return a terminal posterior.")
        log_posterior = np.asarray(score.terminal_log_posterior, dtype=float)
        log_posteriors[time_index] = log_posterior
        row = _trajectory_row(time_index, log_posterior, emissions, bin_centers, "prefix_log_likelihood", score.log_likelihood)
        _copy_score_diagnostics(row, score.diagnostics)
        row.update(_imm_mode_probabilities_for_prefix(model, prefix, bin_centers))
        rows.append(row)
    return pd.DataFrame(rows), log_posteriors


def run_tracking(args: argparse.Namespace) -> None:
    session_path = _session_path(args.dataset_root, args.session)
    if not session_path.is_dir():
        raise FileNotFoundError(f"Requested session directory does not exist: {session_path}")
    _validate_session_files(session_path)
    session = load_replay_session(session_path)
    encoding = fit_place_field_encoding(session)
    emissions = build_emissions(session, encoding, int(args.event_id), EmissionConfig(time_bin_s=args.time_bin_s, spike_rate_scale=args.spike_rate_scale))
    config = BenchmarkConfig(candidate_top_k=args.candidate_top_k, pyrecest_particles=args.pyrecest_particles, models=(args.model,))
    model = _build_models(config, session=session)[args.model]
    trajectory, log_posteriors = _trajectory_from_prefix_scores(model, emissions, encoding.bin_centers)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_session = args.session.replace("/", "_").replace("\\", "_")
    safe_model = str(getattr(model, "name", args.model)).replace("/", "_").replace("\\", "_")
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
        trajectory_log_posteriors=log_posteriors,
    )
    print(f"Wrote trajectory CSV: {csv_path}")
    print(f"Wrote posterior NPZ: {npz_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Track one Pfeiffer/Foster replay event.")
    parser.add_argument("--dataset-root", required=True, help="Path to DataSetFromPfeifferFoster.")
    parser.add_argument("--session", required=True, help="Session ID, e.g. Rat1/Open1.")
    parser.add_argument("--event-id", required=True, type=int, help="Ripple event index.")
    parser.add_argument("--model", default="imm", choices=_TRACK_MODEL_CHOICES)
    parser.add_argument("--candidate-top-k", default=64, type=int)
    parser.add_argument("--pyrecest-particles", default=512, type=int)
    parser.add_argument("--time-bin-s", default=0.02, type=float)
    parser.add_argument("--spike-rate-scale", default=1.0, type=float)
    parser.add_argument("--output", default="results/tracks")
    run_tracking(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
