#!/usr/bin/env python3
"""Plot representative stationary-best and fragmented-best Tanni events.

The exact-model subset contains no confidently nontrajectory events at the
calibrated 5.5 log-evidence margin.  This diagnostic therefore selects the
strongest stationary-best and fragmented-best *ambiguous* rows, with at most
one event per animal in each group, and labels that boundary explicitly.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd
from scipy.special import logsumexp

from hipporeplayimm.data import RippleEvent
from hipporeplayimm.encoding import EmissionConfig, build_emissions
from hipporeplayimm.sorted_spike_state_space import SortedSpikeStateSpaceReplayModel
from hipporeplayimm.state_space import StateSpaceDecoderConfig
from hipporeplayimm.tanni2022 import posterior_from_log_likelihood, read_tanni_position

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_tanni2022_wall_distance_replay import fit_decoder_encoding, make_replay_session  # noqa: E402


MODELS = ("stationary", "diffusion", "fragmented", "first-order-imm")
MODEL_LABELS = {
    "stationary": "Stationary",
    "diffusion": "Diffusion",
    "fragmented": "Fragmented",
    "first-order-imm": "First-order IMM",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--examples-per-group", type=int, default=3)
    parser.add_argument("--model-bin-size-cm", type=float, default=16.0)
    parser.add_argument("--decode-bin-s", type=float, default=0.020)
    parser.add_argument("--claim-margin", type=float, default=5.5)
    return parser.parse_args()


def select_examples(decisions: pd.DataFrame, examples_per_group: int) -> pd.DataFrame:
    """Select strongest ambiguous nontrajectory-best rows across animals."""

    selected = []
    for group, best_model in (("stationary_best_ambiguous", "stationary"), ("fragmented_best_ambiguous", "fragmented")):
        candidates = decisions.loc[decisions["best_model"].eq(best_model) & ~decisions["ordered_trajectory_confident"].astype(bool)].copy()
        candidates = candidates.sort_values(
            ["delta_ordered_minus_static_or_fragmented", "best_minus_runner_up_log_evidence"],
            ascending=[True, False],
        )
        distinct_animals = candidates.groupby("animal", sort=False, as_index=False).head(1)
        chosen = distinct_animals.head(max(int(examples_per_group), 0)).copy()
        chosen["diagnostic_group"] = group
        chosen["selection_rank"] = np.arange(1, chosen.shape[0] + 1)
        selected.append(chosen)
    return pd.concat(selected, ignore_index=True) if selected else pd.DataFrame()


def _posterior(score) -> np.ndarray:
    values = np.asarray(score.trajectory_log_posterior, dtype=float)
    values = values - logsumexp(values, axis=1, keepdims=True)
    return np.exp(values)


def _path_segments(path: np.ndarray) -> np.ndarray:
    if path.shape[0] < 2:
        return np.empty((0, 2, 2), dtype=float)
    return np.stack((path[:-1], path[1:]), axis=1)


def _active_cell_order(encoding, counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    active = np.flatnonzero(counts.sum(axis=0) > 0)
    if active.size == 0:
        return active, np.empty((0, 2), dtype=float)
    peaks = encoding.bin_centers[np.argmax(encoding.rates_hz[active], axis=1)]
    order = np.lexsort((peaks[:, 1], peaks[:, 0]))
    return active[order], peaks[order]


def _plot_raster(ax, emissions, encoding) -> None:
    active, peaks = _active_cell_order(encoding, emissions.spike_counts)
    for row, cell_index in enumerate(active):
        bins = np.repeat(np.arange(emissions.n_time), emissions.spike_counts[:, cell_index])
        if bins.size:
            times_ms = (emissions.times[bins] - emissions.times[0]) * 1000.0
            ax.scatter(times_ms, np.full(times_ms.shape, row), s=13, color="#222222", marker="|")
    ax.set_xlim(-10.0, (emissions.times[-1] - emissions.times[0]) * 1000.0 + 10.0)
    ax.set_ylim(-1.0, max(active.size, 1))
    ax.set_xlabel("Time from first bin center (ms)")
    ax.set_ylabel("Active cells\n(sorted by field peak x, then y)")
    ax.set_title(f"Spike raster: {int(emissions.n_spikes)} spikes, {active.size} active cells")
    if peaks.size:
        ax.text(
            0.99,
            0.97,
            f"field peaks span x={np.ptp(peaks[:, 0]):.0f} cm, y={np.ptp(peaks[:, 1]):.0f} cm",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            color="#555555",
        )


def _plot_evidence(ax, event: pd.Series) -> None:
    values = np.asarray([float(event[model]) for model in MODELS])
    relative = values - np.max(values)
    colors = ["#345995", "#5f8f73", "#bd7b2d", "#b6323b"]
    ax.barh(np.arange(len(MODELS)), relative, color=colors)
    ax.set_yticks(np.arange(len(MODELS)), [MODEL_LABELS[model] for model in MODELS])
    ax.invert_yaxis()
    ax.axvline(-5.5, color="#777777", linestyle="--", linewidth=1)
    ax.set_xlabel("log evidence relative to best")
    ax.set_title("Exact-core evidence")
    for index, value in enumerate(relative):
        ax.text(value - 0.08 if value < -0.2 else value + 0.04, index, f"{value:+.2f}", va="center", ha="right" if value < -0.2 else "left", fontsize=8)


def _plot_path(ax, posterior: np.ndarray, independent: np.ndarray, encoding) -> None:
    valid = encoding.occupancy_s >= 0.05
    ax.scatter(
        encoding.bin_centers[valid, 0],
        encoding.bin_centers[valid, 1],
        c=np.log1p(encoding.occupancy_s[valid]),
        cmap="Greys",
        s=8,
        alpha=0.25,
    )
    model_path = posterior @ encoding.bin_centers
    emission_path = independent @ encoding.bin_centers
    segments = _path_segments(model_path)
    if segments.size:
        collection = LineCollection(segments, cmap="viridis", linewidth=3)
        collection.set_array(np.arange(segments.shape[0]))
        ax.add_collection(collection)
    ax.plot(emission_path[:, 0], emission_path[:, 1], color="#777777", linestyle="--", linewidth=1.2, label="emission-only mean")
    ax.scatter(model_path[0, 0], model_path[0, 1], marker="o", s=45, facecolor="white", edgecolor="#222222", zorder=4, label="start")
    ax.scatter(model_path[-1, 0], model_path[-1, 1], marker="X", s=55, color="#222222", zorder=4, label="end")
    ax.set_xlim(float(encoding.x_edges[0]), float(encoding.x_edges[-1]))
    ax.set_ylim(float(encoding.y_edges[0]), float(encoding.y_edges[-1]))
    ax.set_aspect("equal")
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")
    ax.set_title("Winning-model posterior mean path")
    ax.legend(loc="upper right", fontsize=7, frameon=False)


def _plot_snapshots(fig, axes, posterior: np.ndarray, encoding) -> None:
    maximum = max(float(np.nanmax(posterior)), np.finfo(float).eps)
    for time_index, ax in enumerate(axes):
        grid = posterior[time_index].reshape(encoding.grid_shape)
        ax.pcolormesh(encoding.x_edges, encoding.y_edges, grid.T, cmap="magma", vmin=0.0, vmax=maximum, shading="flat")
        mean = posterior[time_index] @ encoding.bin_centers
        map_position = encoding.bin_centers[int(np.argmax(posterior[time_index]))]
        ax.scatter(mean[0], mean[1], s=18, facecolor="white", edgecolor="#222222", linewidth=0.7)
        ax.scatter(map_position[0], map_position[1], s=16, marker="x", color="#55d6be", linewidth=1.0)
        ax.set_aspect("equal")
        ax.set_title(f"{time_index * 20}-{(time_index + 1) * 20} ms", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.text(0.5, 0.415, "Winning-model posterior snapshots (white dot: mean; cyan x: MAP)", ha="center", fontsize=11)


def plot_event(event: pd.Series, emissions, encoding, score, output_path: Path, claim_margin: float) -> dict[str, object]:
    posterior = _posterior(score)
    independent = posterior_from_log_likelihood(emissions.log_likelihood)
    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    outer = fig.add_gridspec(3, 1, height_ratios=[1.2, 1.0, 1.0])
    top = outer[0].subgridspec(1, 3, width_ratios=[1.7, 1.0, 1.2])
    _plot_raster(fig.add_subplot(top[0]), emissions, encoding)
    _plot_path(fig.add_subplot(top[1]), posterior, independent, encoding)
    _plot_evidence(fig.add_subplot(top[2]), event)
    snapshot_axes = []
    for row in range(2):
        strip = outer[row + 1].subgridspec(1, 5)
        snapshot_axes.extend(fig.add_subplot(strip[column]) for column in range(5))
    _plot_snapshots(fig, snapshot_axes[: emissions.n_time], posterior, encoding)
    group_label = "stationary-best" if event["best_model"] == "stationary" else "fragmented-best"
    fig.suptitle(
        f"{event['animal']} {event['session']} event {int(event['event_index'])}: {group_label}, ambiguous at {claim_margin:g}\n"
        f"ordered minus static/fragmented = {float(event['delta_ordered_minus_static_or_fragmented']):+.2f}; "
        f"best minus runner-up = {float(event['best_minus_runner_up_log_evidence']):+.2f}; "
        f"wall quartile Q{int(event['wall_quartile'])}",
        fontsize=14,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    model_path = posterior @ encoding.bin_centers
    return {
        "animal": event["animal"],
        "session": event["session"],
        "event_index": int(event["event_index"]),
        "diagnostic_group": event["diagnostic_group"],
        "selection_rank": int(event["selection_rank"]),
        "best_model": event["best_model"],
        "ordered_trajectory_confident": bool(event["ordered_trajectory_confident"]),
        "delta_ordered_minus_static_or_fragmented": float(event["delta_ordered_minus_static_or_fragmented"]),
        "best_minus_runner_up_log_evidence": float(event["best_minus_runner_up_log_evidence"]),
        "n_spikes": int(emissions.n_spikes),
        "n_active_cells": int(np.count_nonzero(emissions.spike_counts.sum(axis=0))),
        "winning_model_path_length_cm": float(np.linalg.norm(np.diff(model_path, axis=0), axis=1).sum()),
        "winning_model_net_displacement_cm": float(np.linalg.norm(model_path[-1] - model_path[0])),
        "figure_path": str(output_path),
    }


def make_overview(manifest: pd.DataFrame, output_path: Path) -> None:
    images = [plt.imread(path) for path in manifest["figure_path"]]
    if not images:
        return
    fig, axes = plt.subplots(len(images), 1, figsize=(14, 6 * len(images)), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, image, row in zip(axes, images, manifest.itertuples(index=False), strict=True):
        ax.imshow(image)
        ax.axis("off")
        ax.set_title(f"{row.diagnostic_group}: {row.animal} event {row.event_index}", fontsize=11)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    evidence_dir = args.evidence_dir.resolve()
    model_dir = args.model_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    decisions = pd.read_csv(model_dir / "tanni2022_wall_balanced_model_decisions.csv")
    selected = select_examples(decisions, args.examples_per_group)
    ripple_events = pd.read_csv(evidence_dir / "tanni2022_ripple_candidates.csv")
    selected = selected.merge(
        ripple_events[["animal", "session", "event_index", "window_start_time_s", "window_end_time_s", "peak_time_s", "peak_ripple_z"]],
        on=["animal", "session", "event_index"],
        how="left",
        validate="one_to_one",
    )
    session_manifest = pd.read_csv(evidence_dir / "tanni2022_session_manifest.csv")
    rows = []
    for session_row in session_manifest.itertuples(index=False):
        events = selected.loc[(selected["animal"] == session_row.animal) & (selected["session"] == session_row.session)]
        if events.empty:
            continue
        position = read_tanni_position(Path(session_row.nwb_path))
        session = make_replay_session(Path(session_row.nwb_path), position)
        encoding, _ = fit_decoder_encoding(
            session,
            position,
            bin_size_cm=args.model_bin_size_cm,
            smoothing_sigma_bins=1.5,
            running_speed_cm_s=10.0,
            min_running_spikes=30,
            max_mean_rate_hz=4.0,
            min_peak_rate_hz=2.0,
            min_split_half_stability=0.25,
        )
        selected_session = replace(session, excitatory_neurons=encoding.cell_ids)
        for _, event in events.iterrows():
            ripple = RippleEvent(
                start=float(event["window_start_time_s"]),
                end=float(event["window_end_time_s"]),
                peak=float(event["peak_time_s"]),
                raw_power=float(event["peak_ripple_z"]),
                z_power_session=float(event["peak_ripple_z"]),
                z_power_epoch=float(event["peak_ripple_z"]),
            )
            emissions = build_emissions(selected_session, encoding, ripple, EmissionConfig(time_bin_s=args.decode_bin_s))
            scorer = SortedSpikeStateSpaceReplayModel(
                mode=str(event["best_model"]),
                config=StateSpaceDecoderConfig(mode=str(event["best_model"]), valid_occupancy_threshold_s=0.05),
                name=str(event["best_model"]),
            )
            score = scorer.score(emissions, encoding.bin_centers, occupancy_s=encoding.occupancy_s, return_trajectory=True)
            slug = f"{event['diagnostic_group']}_{event['animal']}_event_{int(event['event_index'])}"
            rows.append(plot_event(event, emissions, encoding, score, output_dir / f"{slug}.png", args.claim_margin))
    output_manifest = pd.DataFrame(rows)
    output_manifest.to_csv(output_dir / "tanni2022_nontrajectory_event_plot_manifest.csv", index=False)
    make_overview(output_manifest, output_dir / "tanni2022_nontrajectory_event_overview.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
