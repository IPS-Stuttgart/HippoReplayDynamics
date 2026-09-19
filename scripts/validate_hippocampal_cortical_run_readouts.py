#!/usr/bin/env python3
"""Blocked RUN-only spatial readout check; no replay or context classification."""
from __future__ import annotations

import argparse
import hashlib
import json
import socket
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import itertools

from _provenance import build_script_provenance
from preflight_hippocampal_cortical_archive import POSITION, safe_path, sha256, validate_spikes, write_json

PARAMETERS = {"bin_s": 0.25, "min_native_speed": 2.5, "min_position_frames": 5, "gap_s": 1.0,
                  "spatial_bins": 25, "smoothing_sigma_bins": 1.0, "prior_seconds": 0.25,
                  "folds": 5, "guard_s": 1.0, "min_training_spikes": 20, "shuffles": 99, "seed": 20260920}
COHORTS = ("CA1_pyramidal", "CA3_pyramidal", "RSC_author_non_narrow", "RSC_export_pyramidal")


def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256(json.dumps(parts).encode()).digest()[:8], "little")


def unit_mask(area, types, cohort):
    area, types = np.asarray(area), np.asarray(types)
    if cohort == "RSC_author_non_narrow":
        return (area == "RSC") & (types != "Narrow Interneuron")
    if cohort not in COHORTS:
        raise ValueError("unknown cohort")
    return (area == cohort.split("_")[0]) & (types == "Pyramidal Cell")


def clock_segments(clock, gap_s=1.0):
    clock = np.asarray(clock, float)
    if clock.ndim != 1 or len(clock) < 2 or not np.isfinite(clock).all() or (np.diff(clock) <= 0).any():
        raise ValueError("invalid position timestamps")
    cuts = np.r_[0, np.flatnonzero(np.diff(clock) > gap_s) + 1, len(clock)]
    return [(int(a), int(b)) for a, b in itertools.pairwise(cuts)]


def prepare_run(clock, xy, speed, times_by_unit):
    p = PARAMETERS
    clock, xy, speed = np.asarray(clock), np.asarray(xy), np.asarray(speed)
    if xy.shape != (len(clock), 2) or speed.shape != clock.shape or len(clock_segments(clock)) != 1:
        raise ValueError("position segment shape/gap mismatch")
    finite = np.isfinite(xy).all(axis=1) & np.isfinite(speed)
    if finite.sum() < 100:
        raise ValueError("too few finite behavior samples")
    center = xy[finite].mean(axis=0)
    _, singular, axes = np.linalg.svd(xy[finite] - center, full_matrices=False)
    if not singular[0] > 0:
        raise ValueError("no spatial extent")
    axis = axes[0]
    if axis[np.argmax(np.abs(axis))] < 0:
        axis = -axis
    projected = (xy - center) @ axis
    lo, hi = np.quantile(projected[finite], [0.01, 0.99])
    if hi <= lo:
        raise ValueError("no robust spatial extent")
    normalized = (projected - lo) / (hi - lo)
    n = int(np.floor((clock[-1] - clock[0]) / p["bin_s"]))
    if n < 250:
        raise ValueError("recorded segment too short")
    edges = clock[0] + np.arange(n + 1) * p["bin_s"]
    frame_bin = np.searchsorted(edges, clock, side="right") - 1
    valid = finite & (frame_bin >= 0) & (frame_bin < n)
    frames = np.bincount(frame_bin[valid], minlength=n)
    position = np.bincount(frame_bin[valid], weights=normalized[valid], minlength=n) / np.maximum(frames, 1)
    velocity = np.bincount(frame_bin[valid], weights=speed[valid], minlength=n) / np.maximum(frames, 1)
    eligible = (frames >= p["min_position_frames"]) & (velocity > p["min_native_speed"])
    counts = np.column_stack([np.diff(np.searchsorted(np.asarray(t), edges, side="left")) for t in times_by_unit])
    geometry = {"axis": axis.tolist(), "center": center.tolist(), "projected_q01": float(lo), "projected_q99": float(hi),
                    "axis_variance_fraction": float(singular[0]**2 / (singular**2).sum()),
                    "coordinate": "fraction_of_segment_behavior_q01_to_q99_extent_not_cm",
                    "geometry_uses_behavior_only": True}
    return edges[:-1] + p["bin_s"] / 2, position, counts, eligible, geometry


def fold_masks(n, eligible, fold):
    p = PARAMETERS
    if fold not in range(p["folds"]):
        raise ValueError("invalid fold")
    cuts = np.linspace(0, n, p["folds"] + 1).astype(int)
    a, b = cuts[fold:fold + 2]
    guard = int(np.ceil(p["guard_s"] / p["bin_s"]))
    train = np.asarray(eligible, bool).copy()
    train[max(0, a - guard):min(n, b + guard)] = False
    test = np.zeros(n, bool)
    test[a:b] = np.asarray(eligible, bool)[a:b]
    return train, test


def fit_composition(counts, position, train):
    p = PARAMETERS
    counts, position, train = np.asarray(counts), np.asarray(position), np.asarray(train, bool)
    if counts.ndim != 2 or len(counts) != len(position) or train.shape != position.shape or not np.isfinite(counts).all() or (counts < 0).any():
        raise ValueError("invalid encoding arrays")
    if train.sum() < 200:
        raise ValueError("fewer than 200 eligible training bins")
    use = counts[train].sum(axis=0) >= p["min_training_spikes"]
    if use.sum() < 2:
        raise ValueError("fewer than two training-active units")
    bins = np.clip(np.floor(position[train] * p["spatial_bins"]).astype(int), 0, p["spatial_bins"] - 1)
    occupancy = np.bincount(bins, minlength=p["spatial_bins"]) * p["bin_s"]
    sums = np.zeros((p["spatial_bins"], use.sum()))
    np.add.at(sums, bins, counts[train][:, use])
    totals = counts[train][:, use].sum(axis=0) / (train.sum() * p["bin_s"])
    numer = gaussian_filter1d(sums, p["smoothing_sigma_bins"], axis=0, mode="reflect") + p["prior_seconds"] * totals
    denom = gaussian_filter1d(occupancy, p["smoothing_sigma_bins"], mode="reflect") + p["prior_seconds"]
    rates = np.maximum(numer / denom[:, None], 1e-8)
    composition = rates / rates.sum(axis=1, keepdims=True)
    return composition, use, occupancy


def decode(counts, composition):
    # Conditioning on the population count removes a total-rate-only readout.
    logp = np.asarray(counts) @ np.log(composition).T
    logp -= logsumexp(logp, axis=1, keepdims=True)
    return np.exp(logp)


def evaluate(counts, position, train, test, key):
    if np.asarray(test).sum() < 40:
        raise ValueError("fewer than 40 eligible test bins")
    composition, use, occupancy = fit_composition(counts, position, train)
    test_counts = counts[test][:, use]
    posterior = decode(test_counts, composition)
    centers = (np.arange(PARAMETERS["spatial_bins"]) + 0.5) / PARAMETERS["spatial_bins"]
    mean = posterior @ centers
    maximum = centers[np.argmax(posterior, axis=1)]
    truth = position[test]
    errors = np.abs(mean - truth)
    rng = np.random.default_rng(stable_seed(PARAMETERS["seed"], *key))
    # Shift whole population vectors among eligible held-out RUN bins only.
    # This is a compressed-RUN shift, not a claim of a physical-time null.
    guard = max(1, int(0.1 * len(truth)))
    shifts = rng.integers(guard, len(truth) - guard + 1, size=PARAMETERS["shuffles"])
    null_errors = np.array([np.mean(np.abs(np.roll(mean, int(s)) - truth)) for s in shifts])
    metrics = {"n_training_bins": int(train.sum()), "n_test_bins": int(test.sum()), "n_encoding_units": int(use.sum()),
                   "training_occupancy_nonzero_bins": int((occupancy > 0).sum()),
                   "mean_absolute_error_fraction": float(errors.mean()), "median_absolute_error_fraction": float(np.median(errors)),
                   "map_median_absolute_error_fraction": float(np.median(np.abs(maximum - truth))),
                   "training_median_baseline_mae": float(np.mean(np.abs(np.median(position[train]) - truth))),
                   "null_median_mae_fraction": float(np.median(null_errors)),
                   "gain_over_null_mae_fraction": float(np.median(null_errors) - errors.mean()),
                   "empirical_shift_p": float((1 + (null_errors <= errors.mean()).sum()) / (1 + len(null_errors))),
                   "nonzero_spike_test_fraction": float((test_counts.sum(axis=1) > 0).mean()),
                   "test_spikes": int(test_counts.sum()), "test_active_units": int((test_counts.sum(axis=0) > 0).sum())}
    audit = {"composition": composition, "use": use, "occupancy": occupancy, "train_indices": np.flatnonzero(train),
                 "test_indices": np.flatnonzero(test), "true_position": truth, "posterior_mean": mean, "posterior_map": maximum,
                 "null_mae": null_errors, "shifts": shifts, "test_counts": test_counts}
    return metrics, audit


def write_summaries(rows, destination):
    measures = ["median_absolute_error_fraction", "gain_over_null_mae_fraction", "n_encoding_units", "nonzero_spike_test_fraction"]
    keys = ["asset_id", "animal", "segment", "cohort"]
    table = pd.DataFrame(rows).reindex(columns=["status", *keys, *measures])
    for column in measures:
        table[column] = pd.to_numeric(table[column], errors="coerce")
    ok = table[table.status == "scored"]
    grouped = ok.groupby(keys)
    segment = grouped[measures].median()
    segment["n_scored_folds"] = grouped.size()
    segment.reset_index().to_csv(destination / "segment_summary.csv", index=False)
    grouped = ok.groupby(["animal", "asset_id", "cohort"])
    per_file = grouped[measures].median()
    per_file["n_scored_folds"] = grouped.size()
    per_file = per_file.reset_index()
    per_file.to_csv(destination / "file_summary.csv", index=False)
    grouped = per_file.groupby(["animal", "cohort"])
    by_animal = grouped[measures].median()
    by_animal["n_files_with_scores"] = grouped.size()
    by_animal.reset_index().to_csv(destination / "by_animal_summary.csv", index=False)
    return table


def run(args):
    source = pd.read_csv(args.inventory_csv)
    if len(source) != 15 or source.asset_id.nunique() != 15 or not (source.status == "inspected").all():
        raise ValueError("complete verified 15-file inventory required")
    provenance = build_script_provenance(input_paths={"inventory": args.inventory_csv}, cwd=ROOT)
    if provenance["git_dirty"] is not False or len(provenance["code_commit"]) != 40:
        raise ValueError("clean committed producer required")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.output_dir / "manifest.json", {**provenance, "hostname": socket.gethostname(),
        "created_at_utc": datetime.now(UTC).isoformat(), "parameters": PARAMETERS, "cohorts": COHORTS,
        "scope": "RUN_spatial_readout_feasibility_only_no_context_labels_no_POST_scoring",
        "author_code_commit": "ff579297beb12a3bbe86dde12b7dc39857315d54",
        "context_readiness_remains_unresolved": True})
    rows, sessions, units = [], [], []
    for entry in source.itertuples(index=False):
        started = time.monotonic()
        record = {"asset_id": entry.asset_id, "animal": entry.animal, "path": entry.path, "status": "failed", "failure_reason": ""}
        try:
            path = safe_path(args.dataset_root, entry.path)
            if sha256(path) != entry.raw_sha256:
                raise ValueError("raw SHA256 changed since preflight")
            with h5py.File(path, "r") as f:
                clock, xy = f[f"{POSITION}/timestamps"][()], f[f"{POSITION}/data"][()]
                speed = f["processing/behavior/Speed/data"][()]
                if f["processing/behavior/Speed/data"].attrs["unit"] != "cm/s" or f["processing/behavior/Speed/data"].attrs.get("conversion", 1) != 1:
                    raise ValueError("unexpected native speed convention")
                if not np.array_equal(clock, f["processing/behavior/Speed/timestamps"][()]):
                    raise ValueError("speed clock mismatch")
                ids, stops, spikes = f["units/id"][()], f["units/spike_times_index"][()], f["units/spike_times"][()]
                validate_spikes(ids, stops, spikes)
                by_unit = [spikes[a:b] for a, b in zip(np.r_[0, stops[:-1]], stops, strict=True)]
                area, types = f["units/cell_area"].asstr()[()], f["units/cell_type"].asstr()[()]
            segments = clock_segments(clock)
            for segment, (a, b) in enumerate(segments):
                centers, position, counts, eligible, geometry = prepare_run(clock[a:b], xy[a:b], speed[a:b], by_unit)
                dest = args.output_dir / entry.asset_id / f"recorded_segment_{segment}"
                dest.mkdir(parents=True)
                write_json(dest / "geometry.json", {**geometry, "start_s": float(clock[a]), "end_s": float(clock[b-1]),
                    "segment_identity": "clock_contiguous_observed_position_not_verified_maze", "eligible_bins": int(eligible.sum())})
                if geometry["axis_variance_fraction"] < 0.90:
                    raise ValueError("single-axis spatial representation inadequate")
                for cohort in COHORTS:
                    mask = unit_mask(area, types, cohort)
                    units.append({"asset_id": entry.asset_id, "animal": entry.animal, "segment": segment, "cohort": cohort, "n_source_units": int(mask.sum())})
                    for fold in range(PARAMETERS["folds"]):
                        row = {"asset_id": entry.asset_id, "animal": entry.animal, "segment": segment, "cohort": cohort, "fold": fold,
                                   "status": "failed", "failure_reason": ""}
                        try:
                            train, test = fold_masks(len(centers), eligible, fold)
                            metrics, audit = evaluate(counts[:, mask], position, train, test, (entry.asset_id, segment, cohort, fold))
                            np.savez_compressed(dest / f"{cohort}_fold{fold}.npz", **audit, unit_ids=ids[mask], test_time_s=centers[test])
                            row.update(metrics, status="scored")
                        except ValueError as exc:
                            row["failure_reason"] = str(exc)
                        rows.append(row)
            if sha256(path) != entry.raw_sha256:
                raise ValueError("raw input changed during validation")
            record.update(status="processed", n_position_segments=len(segments))
        except (OSError, KeyError, ValueError, TypeError, RuntimeError) as exc:
            record["failure_reason"] = f"{type(exc).__name__}: {exc}"
        record["runtime_s"] = time.monotonic() - started
        sessions.append(record)
        pd.DataFrame(rows).to_csv(args.output_dir / "fold_metrics.csv", index=False)
        pd.DataFrame(sessions).to_csv(args.output_dir / "session_status.csv", index=False)
        pd.DataFrame(units).to_csv(args.output_dir / "unit_cohort_counts.csv", index=False)
        print(json.dumps(record), flush=True)
    table = write_summaries(rows, args.output_dir)
    complete = len(sessions) == 15 and all(x["status"] == "processed" for x in sessions) and len(rows) == 360 and (table.status == "scored").all()
    write_json(args.output_dir / "terminal_status.json", {"returncode": 0 if complete else 2,
        "completed_at_utc": datetime.now(UTC).isoformat(), "all_expected_folds_scored": bool(complete),
        "context_decoder_ready": False, "biological_claim": "not_tested",
        "output_sha256": {str(p.relative_to(args.output_dir)): sha256(p) for p in sorted(args.output_dir.rglob("*")) if p.is_file()}})
    return 0 if complete else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory-csv", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
