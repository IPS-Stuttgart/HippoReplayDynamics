#!/usr/bin/env python3
"""Paired cell-subsampling of every cached candidate, without a dynamics prior.

This intervention measures sensitivity of the readout, not the ground-truth
prevalence of continuous replay. Real events have no known latent trajectory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hipporeplayimm.replay_coverage import continuity_metrics, decode_independent
from scripts._provenance import build_script_provenance, file_sha256

STRATA = ["dataset", "likelihood", "estimator", "bin_filter", "cell_fraction"]
EVENT_KEY = ["dataset", "animal", "session", "event_index", "likelihood", "estimator", "bin_filter"]


def decoding_support(occupied, grid, bounds):
    """Do not assign posterior mass to a grid center outside a known arena."""
    valid = np.asarray(occupied, dtype=bool).copy()
    bounds = np.asarray(bounds, dtype=float)
    if bounds.shape != (2, grid.shape[1]) or valid.shape != (len(grid),):
        raise ValueError("invalid occupancy/grid/arena shapes")
    if np.isfinite(bounds).all():
        if np.any(bounds[1] <= bounds[0]):
            raise ValueError("invalid arena bounds")
        valid &= ((grid >= bounds[0]) & (grid <= bounds[1])).all(axis=1)
    elif not np.isnan(bounds).all():
        raise ValueError("arena bounds must be fully known or explicitly missing")
    if valid.sum() < 2:
        raise ValueError("insufficient reachable occupied states")
    return valid


def population_subsets(cell_ids, fractions, replicates, seed, session_key):
    """One full reference; nested recording subsets, stable to session ordering."""
    ids = np.asarray(cell_ids, dtype=int)
    fractions = np.asarray(fractions, dtype=float)
    if len(ids) < 2 or replicates < 1 or not np.isfinite(fractions).all() or np.any((fractions <= 0) | (fractions >= 1)):
        raise ValueError("need >=2 units, replicates>=1, and fractions strictly between 0 and 1")
    if len(set(fractions)) != len(fractions) or len(set(ids)) != len(ids):
        raise ValueError("duplicate fractions or cell IDs")
    token = int.from_bytes(hashlib.sha256(session_key.encode()).digest()[:8], "little")
    yield 1.0, 0, np.arange(len(ids))
    for rep in range(replicates):
        rng = np.random.default_rng(np.random.SeedSequence([seed, token, rep]))
        order = rng.permutation(len(ids))
        for fraction in sorted(fractions, reverse=True):
            size = max(1, int(np.floor(len(ids) * fraction)))
            yield float(fraction), rep, np.sort(order[:size])


def event_windows(base_counts, durations, width=4, base_s=.005):
    """20 ms windows every 5 ms; never span an event edge or partial bin."""
    counts = np.asarray(base_counts)
    durations = np.asarray(durations, dtype=float)
    if counts.ndim != 2 or durations.shape != (len(counts),) or width < 1:
        raise ValueError("invalid count/duration shape or window width")
    full = np.isclose(durations, base_s, atol=1e-9, rtol=0)
    partial = np.flatnonzero(~full)
    if len(partial) and (len(partial) > 1 or partial[0] != len(full) - 1 or not 0 < durations[-1] < base_s):
        raise ValueError("partial or irregular bin before the final bin")
    n = int(full.sum())
    if n < width:
        return np.empty((0, counts.shape[1]), dtype=np.int64)
    cumulative = np.vstack([np.zeros((1, counts.shape[1]), dtype=np.int64), counts[:n].cumsum(axis=0, dtype=np.int64)])
    return cumulative[width:] - cumulative[:-width]


def decode_compact(counts, rates, grid, likelihood, chunk_size=512):
    keys = ["map", "posterior_mean", "posterior_rms_cm", "posterior_entropy_nats"]
    parts = {key: [] for key in keys}
    for start in range(0, len(counts), chunk_size):
        decoded = decode_independent(counts[start:start + chunk_size], rates, grid, .020, likelihood=likelihood)
        for key in keys:
            parts[key].append(decoded[key])
    result = {}
    for key, value in parts.items():
        shape = (0, grid.shape[1]) if key in {"map", "posterior_mean"} else (0,)
        result[key] = np.concatenate(value) if value else np.empty(shape)
    return result


def observed_metrics(path, counts, rms, entropy, apply_support):
    """Continuity uses overlap; speed uses independent non-overlapping windows."""
    supported = (counts.sum(axis=1) >= 3) & (np.count_nonzero(counts, axis=1) >= 2)
    valid = supported if apply_support else np.ones(len(path), dtype=bool)
    continuous = continuity_metrics(path, valid_bins=valid)
    overlap_jumps = np.linalg.norm(np.diff(path, axis=0), axis=1)
    overlap_pairs = valid[:-1] & valid[1:]
    indices = np.arange(0, len(path), 4)
    measured_path, measured_valid = path[indices], valid[indices]
    steps = np.linalg.norm(np.diff(measured_path, axis=0), axis=1) / .020
    adjacent = measured_valid[:-1] & measured_valid[1:]
    core = (indices >= continuous["continuous_start"]) & (indices < continuous["continuous_end_exclusive"])
    core_adjacent = adjacent & core[:-1] & core[1:]
    return {
        **continuous, "overlapping_frames": len(path),
        "supported_frame_fraction": float(supported.mean()) if len(supported) else np.nan,
        "large_jump_fraction": float(np.mean(overlap_jumps[overlap_pairs] >= 20)) if overlap_pairs.any() else np.nan,
        "nonoverlapping_valid_steps": int(adjacent.sum()),
        "nonoverlapping_event_median_speed_cm_s": float(np.median(steps[adjacent])) if adjacent.any() else np.nan,
        "selected_core_nonoverlapping_steps": int(core_adjacent.sum()) if continuous["continuity_pass"] else 0,
        "selected_core_median_speed_cm_s": float(np.median(steps[core_adjacent])) if continuous["continuity_pass"] and core_adjacent.any() else np.nan,
        "median_posterior_rms_cm": float(np.median(rms[valid])) if valid.any() else np.nan,
        "median_posterior_entropy_nats": float(np.median(entropy[valid])) if valid.any() else np.nan,
    }


def pair_to_full(events):
    full = events[events.cell_fraction.eq(1.0)]
    if full.duplicated(EVENT_KEY).any():
        raise ValueError("duplicate full-population references")
    columns = ["continuity_pass", "nonoverlapping_event_median_speed_cm_s", "median_posterior_rms_cm"]
    ref = full[EVENT_KEY + columns].rename(columns={name: f"full_{name}" for name in columns})
    paired = events.merge(ref, on=EVENT_KEY, how="left", validate="many_to_one", indicator=True)
    if len(events) == 0 or not paired["_merge"].eq("both").all():
        raise ValueError("missing full-population candidate references")
    paired["continuity_pass_difference"] = paired.continuity_pass.astype(int) - paired.full_continuity_pass.astype(int)
    paired["lost_full_continuity"] = paired.full_continuity_pass & ~paired.continuity_pass
    paired["gained_continuity"] = ~paired.full_continuity_pass & paired.continuity_pass
    return paired.drop(columns="_merge")


def aggregate(events, bootstraps, seed):
    paired = pair_to_full(events)
    paired["speed_measurable"] = np.isfinite(paired.nonoverlapping_event_median_speed_cm_s)
    paired["selected_speed_measurable"] = np.isfinite(paired.selected_core_median_speed_cm_s)
    population = paired.groupby(STRATA + ["animal", "session", "population_replicate"], as_index=False).agg(
        events=("event_index", "size"), retained_cells=("retained_cells", "first"),
        continuity_fraction=("continuity_pass", "mean"),
        full_continuity_fraction=("full_continuity_pass", "mean"),
        continuity_pass_difference=("continuity_pass_difference", "mean"),
        lost_full_continuity_fraction=("lost_full_continuity", "mean"),
        gained_continuity_fraction=("gained_continuity", "mean"),
        mean_large_jump_fraction=("large_jump_fraction", "mean"),
        median_event_speed_cm_s=("nonoverlapping_event_median_speed_cm_s", "median"),
        median_selected_core_speed_cm_s=("selected_core_median_speed_cm_s", "median"),
        median_posterior_rms_cm=("median_posterior_rms_cm", "median"),
        measurable_event_fraction=("speed_measurable", "mean"),
        measurable_selected_core_fraction=("selected_speed_measurable", "mean"),
    )
    values = ["continuity_fraction", "continuity_pass_difference", "lost_full_continuity_fraction", "gained_continuity_fraction", "mean_large_jump_fraction", "median_event_speed_cm_s", "median_selected_core_speed_cm_s", "median_posterior_rms_cm", "measurable_event_fraction", "measurable_selected_core_fraction"]
    session = population.groupby(STRATA + ["animal", "session"], as_index=False)[values].mean()
    animals = session.groupby(STRATA + ["animal"], as_index=False)[values].mean()
    rng = np.random.default_rng(seed)
    rows = []
    for key, local in animals.groupby(STRATA, sort=True):
        for metric in values:
            finite = local[metric].dropna().to_numpy(float)
            lo, hi = np.nan, np.nan
            if len(finite) >= 2:
                lo, hi = np.quantile(rng.choice(finite, (bootstraps, len(finite)), replace=True).mean(axis=1), [.025, .975])
            rows.append({**dict(zip(STRATA, key, strict=True)), "metric": metric, "animals": len(local), "finite_animals": len(finite), "equal_animal_mean": float(finite.mean()) if len(finite) else np.nan, "animal_bootstrap_ci95_low": lo, "animal_bootstrap_ci95_high": hi, "uncertainty_scope": "animal_cluster_bootstrap_after_averaging_recording_subsets_and_sessions"})
    return population, session, animals, pd.DataFrame(rows)


def plot_summary(summary, path):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for column, dataset in enumerate(["pfeiffer_foster", "tanni2022"]):
        for row, (metric, estimator, ylabel) in enumerate([
            ("continuity_fraction", "map", "Geometric continuity pass fraction"),
            ("median_event_speed_cm_s", "posterior_mean", "Posterior-mean step speed (cm/s)"),
        ]):
            ax = axes[row, column]
            local = summary[summary.dataset.eq(dataset) & summary.metric.eq(metric) & summary.estimator.eq(estimator) & summary.bin_filter.eq("at_least_2cells_3spikes")]
            for likelihood, label, color in [("poisson", "Poisson", "#087e8b"), ("conditional_multinomial", "Count-conditioned", "#c0504d")]:
                subset = local[local.likelihood.eq(likelihood)].sort_values("cell_fraction")
                ax.plot(subset.cell_fraction, subset.equal_animal_mean, "o-", color=color, label=label)
                ax.fill_between(subset.cell_fraction, subset.animal_bootstrap_ci95_low, subset.animal_bootstrap_ci95_high, color=color, alpha=.16)
            ax.set(xlabel="Fraction of RUN-QC units retained", ylabel=ylabel)
            ax.set_title("Pfeiffer/Foster" if dataset == "pfeiffer_foster" else "Tanni: all arena sizes")
            ax.spines[["top", "right"]].set_visible(False)
            if row == 0:
                ax.set_ylim(0, 1)
                ax.legend(frameon=False, fontsize=9)
    fig.suptitle("Same candidate windows, fewer recorded units\nReadout sensitivity, not a ground-truth replay or speed test", fontsize=12)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run(args):
    root = args.input_dir.resolve()
    source_gates = pd.read_csv(root / "coverage_input_gate_summary.csv")
    if source_gates.loc[source_gates.gate.eq("overall_input_readiness"), "passed"].tolist() != [True]:
        raise ValueError("input readiness must pass")
    sessions = pd.read_csv(root / "coverage_input_sessions.csv")
    if args.sessions:
        sessions = sessions[sessions.apply(lambda row: f"{row.dataset}:{row.animal}:{row.session}" in args.sessions, axis=1)]
        if len(sessions) != len(set(args.sessions)):
            raise ValueError("unknown/duplicate requested sessions")
    if sessions.empty or not sessions.status.eq("cached").all() or (sessions.candidates <= 0).any():
        raise ValueError("no complete input sessions")
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    inputs = {"input_manifest": root / "coverage_input_manifest.json", "input_sessions": root / "coverage_input_sessions.csv", "input_candidates": root / "coverage_input_candidates.csv", "script": Path(__file__), "decoder": ROOT / "src/hipporeplayimm/replay_coverage.py"}
    provenance = build_script_provenance(input_paths=inputs, cwd=ROOT)
    initial_hashes = {key: file_sha256(value) for key, value in inputs.items()}
    frames, population_specs, cache_hashes = [], [], {}
    for record in sessions.itertuples(index=False):
        if file_sha256(record.artifact_path) != record.artifact_sha256:
            raise ValueError("recording cache hash mismatch")
        cache_hashes[record.artifact_path] = record.artifact_sha256
        with np.load(record.artifact_path, allow_pickle=False) as cache:
            mask = cache["unit_qc_mask"].astype(bool)
            ids = cache["cell_ids"][mask]
            occupied = decoding_support(cache["valid_spatial_bins"], cache["bin_centers_cm"], cache["arena_bounds_cm"])
            removed = int(np.count_nonzero(cache["valid_spatial_bins"] & ~occupied))
            rates = cache["rates_hz"][mask][:, occupied]
            grid = cache["bin_centers_cm"][occupied]
            base = cache["candidate_base_counts"][:, mask]
            durations, offsets = cache["candidate_base_durations_s"], cache["candidate_offsets"]
            event_ids = cache["candidate_event_indices"]
            event_durations = cache["candidate_end_s"] - cache["candidate_start_s"]
            window_chunks = [event_windows(base[a:b], durations[a:b]) for a, b in pairwise(offsets)]
            window_offsets = np.r_[0, np.cumsum([len(c) for c in window_chunks])]
            all_counts = np.concatenate(window_chunks)
            session_key = f"{record.dataset}:{record.animal}:{record.session}"
            rows = []
            for fraction, rep, subset in population_subsets(ids, args.fractions, args.replicates, args.seed, session_key):
                counts = all_counts[:, subset]
                event_spikes = [int(base[a:b, subset].sum()) for a, b in pairwise(offsets)]
                event_active = [int(np.count_nonzero(base[a:b, subset].sum(axis=0))) for a, b in pairwise(offsets)]
                population_specs.append({"session_key": session_key, "cell_fraction": fraction, "population_replicate": rep, "cell_ids": ids[subset].tolist(), "spatial_states": int(occupied.sum()), "occupied_centers_outside_known_arena_removed": removed})
                print(f"{session_key} fraction={fraction} rep={rep} cells={len(subset)} frames={len(counts)}", flush=True)
                for likelihood in ["poisson", "conditional_multinomial"]:
                    decoded = decode_compact(counts, rates[subset], grid, likelihood)
                    for event_row, (event, a, b) in enumerate(zip(event_ids, window_offsets[:-1], window_offsets[1:], strict=True)):
                        for estimator in ["map", "posterior_mean"]:
                            for support in [False, True]:
                                rows.append({
                                    "dataset": record.dataset, "animal": record.animal, "session": record.session,
                                    "event_index": int(event), "cell_fraction": fraction, "population_replicate": rep,
                                    "arena_area_m2": getattr(record, "arena_area_m2", np.nan),
                                    "retained_cells": len(subset), "full_qc_cells": len(ids),
                                    "retained_event_spikes": event_spikes[event_row],
                                    "retained_event_active_units": event_active[event_row],
                                    "duration_s": float(event_durations[event_row]),
                                    "likelihood": likelihood, "estimator": estimator,
                                    "bin_filter": "at_least_2cells_3spikes" if support else "unfiltered",
                                    **observed_metrics(decoded[estimator][a:b], counts[a:b], decoded["posterior_rms_cm"][a:b], decoded["posterior_entropy_nats"][a:b], support),
                                })
            frame = pd.DataFrame(rows)
            frames.append(frame)
            frame.to_csv(out / f"events__{record.dataset}__{record.animal}__{record.session.split('/')[-1]}.csv", index=False)
    events = pd.concat(frames, ignore_index=True)
    population, session, animals, summary = aggregate(events, args.bootstraps, args.seed + 1)
    conditions = 1 + len(args.fractions) * args.replicates
    expected = int(sessions.candidates.sum()) * conditions * 8
    unchanged = all(file_sha256(value) == initial_hashes[key] for key, value in inputs.items()) and all(file_sha256(path) == digest for path, digest in cache_hashes.items())
    gates = [
        ("every_candidate_condition_present", len(events) == expected and not events.duplicated(EVENT_KEY + ["cell_fraction", "population_replicate"]).any()),
        ("every_session_represented", events[["dataset", "animal", "session"]].drop_duplicates().shape[0] == len(sessions)),
        ("paired_full_references_present", len(pair_to_full(events)) == len(events)),
        ("unchanged_inputs_and_code", unchanged),
    ]
    gates.append(("overall_technical", all(bool(value) for _, value in gates)))
    for filename, table in [("event_metrics", events), ("population_summary", population), ("session_summary", session), ("animal_summary", animals), ("paired_summary", summary)]:
        table.to_csv(out / f"coverage_subsampling_{filename}.csv", index=False)
    pd.DataFrame([{"gate": name, "passed": bool(value)} for name, value in gates]).to_csv(out / "coverage_subsampling_gate_summary.csv", index=False)
    plot_summary(summary, out / "coverage_subsampling.png")
    manifest = {
        **provenance, "created_at_utc": datetime.now(UTC).isoformat(),
        "input_sha256_at_start": initial_hashes, "cache_sha256": cache_hashes,
        "fractions": args.fractions, "replicates": args.replicates, "seed": args.seed,
        "population_subsets": population_specs, "sessions": len(sessions),
        "unique_candidates": int(sessions.candidates.sum()), "metric_rows": len(events),
        "selection_window_ms": 20, "selection_stride_ms": 5,
        "speed_window_ms": 20, "speed_stride_ms": 20,
        "uniform_spatial_prior": True, "temporal_prior": None,
        "partial_boundary_bin_policy": "exclude_centers_outside_known_arena_occupancy_mask_fixed_across_cell_subsets",
        "claim_boundary": "recording sensitivity, not replay ground truth, calibration, or speed uniformity",
    }
    (out / "coverage_subsampling_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out / "coverage_subsampling_report.md").write_text(
        "# Paired Recording-Subsampling Audit\n\n"
        f"{int(sessions.candidates.sum())} distinct candidates in {len(sessions)} sessions; {len(events)} repeated metric rows.\n\n"
        "All source candidates remain in every population condition. Subsets are nested per session/replicate and selected without replay scores.\n\n"
        "The 20 ms flat-prior decoder is independent across windows. Geometric continuity uses 5 ms strides; speed uses 20 ms non-overlapping windows. Unsupported bins are never bridged.\n\n"
        "Native Tanni wall coordinates exclude occupied grid centers outside the arena; this fixed spatial mask does not change with cell count. PF has no verified wall mask at this stage. Boundary-centroid/grid-alignment sensitivity remains required.\n\n"
        "Paired differences use each candidate's full-population reference. Recording replicates are averaged within session, sessions within animal, then animals equally. Bootstrap intervals resample animals; few animals limit precision.\n\n"
        "A continuity gain under subsampling is NOT evidence of additional true replay. Lower posterior-mean speed can arise from increased uncertainty. Stationary/discontinuous simulations and independent recovery calibration are still required.\n"
    )
    if not all(value for _, value in gates):
        raise RuntimeError("subsampling technical gates failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sessions", nargs="*")
    parser.add_argument("--fractions", nargs="+", type=float, default=[.25, .5, .75])
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--bootstraps", type=int, default=5000)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
