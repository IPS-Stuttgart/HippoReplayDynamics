#!/usr/bin/env python3
"""Test calibration failure induced by MUA selection with unchanged tuning."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"src")]

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from hipporeplayimm.regional_content_frontier import (
    block_histograms,
    bootstrap_compatibility,
    calls_from_bf,
    discrimination,
    regional_log_bf,
)
from hipporeplayimm.regional_content_mua_null import (
    detector_endpoints,
    gain_profile,
    load_detector,
    poisson_train,
    select_indices,
    stable_seed,
    window_means,
)
from scripts._provenance import build_script_provenance, file_sha256
from scripts.audit_edge_support_content import load_npz, recount


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def detect(detector, spikes, duration, cells):
    return detector.detect_high_mua_in_interval(spikes, np.array([0., duration]), np.zeros(2), 0., duration,
        bin_s=.001, gaussian_sd_s=.010, z_threshold=3., maximum_speed_cm_s=5.,
        minimum_duration_s=.050, maximum_duration_s=2., minimum_active_cells=max(1, int(np.ceil(.1*cells))))


def draw_states(region, weights, n, rng, balanced=False):
    z = np.repeat([0, 1], n//2) if balanced else rng.binomial(1, .30, n)
    if len(z) != n:
        raise ValueError("balanced sample size must be even")
    states = np.empty(n, int)
    for label in (0, 1):
        support = np.flatnonzero(region == bool(label))
        w = weights[support]
        if w.sum() == 0:
            raise ValueError("no observed RUN calibration support for a class")
        states[z == label] = rng.choice(support, (z == label).sum(), p=w/w.sum())
    return z, states


def run_session(row, args, detector, inputs):
    target = args.output_dir/row.session.replace("/", "_")
    target.mkdir()
    previous_manifest = json.loads((args.frontier_root/"manifest.json").read_text())
    enc_path = Path(previous_manifest["input_file_paths"][row.session+":encoder"])
    if file_sha256(enc_path) != previous_manifest["input_file_sha256"][row.session+":encoder"]:
        raise ValueError("frozen encoder changed")
    data = load_npz(enc_path)
    original = Path(row.folder)/"run_calibration_audit.npz"
    recorded = json.loads((Path(row.folder)/"outputs.json").read_text())
    if file_sha256(original) != recorded[original.name]:
        raise ValueError("RUN calibration changed")
    prior = load_npz(original)
    meta_path = enc_path.parent/"encoding_manifest.json"
    metadata = json.loads(meta_path.read_text())
    if not metadata["training_only"] or metadata["holdout_spikes_used_for_rate_or_unit_selection"]:
        raise ValueError("training-only encoder required")
    for name, path in {"encoder": enc_path, "encoder_metadata": meta_path, "calibration": original}.items():
        inputs[row.session+":"+name] = path
    ids, keep, valid = data["cell_ids"], data["unit_qc_mask"], data["valid_spatial_bins"]
    rates_all = np.maximum(data["rates_hz"][:, valid], 1e-4)
    rates, region, grid = rates_all[keep], prior["region"], prior["grid"]
    np.testing.assert_array_equal(rates, prior["rates"])
    np.testing.assert_array_equal(ids[keep], prior["cell_ids"])
    cal = prior["calibration"]
    nearest = cKDTree(grid).query(prior["run_windows"][cal, 2:4])[1]
    weights = np.bincount(nearest, minlength=len(grid)).astype(float)
    observed_calls = calls_from_bf(prior["native_run"][cal], prior["run_counts"][cal].sum(axis=1))
    run_blocks = block_histograms(observed_calls, prior["run_bouts"][cal], prior["run_windows"][cal, 4].astype(int))
    rng = np.random.default_rng(stable_seed(args.seed, row.session, "calibration"))
    calibration_truth, calibration_states = draw_states(region, weights, 2*args.calibration_per_class, rng, True)
    calibration_counts = rng.poisson(.02*rates[:, calibration_states].T)
    calibration_bf = regional_log_bf(calibration_counts, rates, region)
    calibration_calls = calls_from_bf(calibration_bf, calibration_counts.sum(axis=1))
    generator_blocks = block_histograms(calibration_calls, np.arange(len(calibration_calls)), calibration_truth)
    np.savez_compressed(target/"calibration.npz", rates_all=rates_all, ids=ids, keep=keep, region=region, grid=grid,
        weights=weights, run_blocks=run_blocks, generator_blocks=generator_blocks,
        counts=calibration_counts, states=calibration_states, truth=calibration_truth, log_bf=calibration_bf)
    results, interval_rows, events_rows, detection_rows, distribution_rows = [], [], [], [], []
    for replicate in range(args.replicates):
        _, states = draw_states(region, weights, args.duration_s//2,
            np.random.default_rng(stable_seed(args.seed, row.session, "states", replicate)))
        for peak in (1, 3, 6):
            folder = target/f"rep{replicate}_peak{peak}"
            folder.mkdir()
            _edges, gain = gain_profile(peak)
            spikes = poisson_train(rates_all, ids, states, gain,
                np.random.default_rng(stable_seed(args.seed, row.session, "spikes", replicate, peak)))
            detected = detect(detector, spikes, args.duration_s, len(ids))
            endpoints, endpoint_ids, crossing = detector_endpoints(detected)
            selected = select_indices(endpoint_ids, args.max_windows, args.seed, row.session, replicate, peak, "endpoint")
            endpoints, endpoint_ids = endpoints[selected], endpoint_ids[selected]
            fixed = np.column_stack((np.arange(len(states))*2+1.08, np.arange(len(states))*2+1.10))
            fixed_ids = select_indices(np.arange(len(states)), args.max_windows, args.seed, row.session, replicate, "fixed")
            fixed = fixed[fixed_ids]
            a_counts = recount(spikes, ids[keep], endpoints[:, 0], endpoints[:, 1])
            fixed_counts = recount(spikes, ids[keep], fixed[:, 0], fixed[:, 1])
            expected, exposures, locations = window_means(endpoints, states, rates, gain)
            independent = np.random.default_rng(stable_seed(args.seed, row.session, "independent", replicate, peak)).poisson(expected)
            _, fixed_exposure, fixed_locations = window_means(fixed, states, rates, gain)
            arrays = {"spikes": spikes, "states": states, "gain": gain, "ids": ids, "keep": keep,
                          "selected_windows": endpoints, "selected_event_ids": endpoint_ids, "selected_counts": a_counts,
                          "independent_counts": independent, "expected_counts": expected, "exposures": exposures,
                          "selected_locations": locations, "fixed_windows": fixed, "fixed_counts": fixed_counts,
                          "fixed_epoch_ids": fixed_ids, "fixed_exposures": fixed_exposure, "fixed_locations": fixed_locations}
            np.savez_compressed(folder/"spike_train.npz", **arrays)
            pd.DataFrame(detected).to_csv(folder/"detector_events.csv", index=False)
            meta = {"animal": row.animal, "session": row.session, "replicate": replicate, "peak_gain": peak}
            detection_rows.append(dict(**meta, detector_events=len(detected), state_boundary_excluded=crossing,
                selected_endpoints=len(endpoints), fixed_windows=len(fixed), detector_cells=len(ids),
                decoding_cells=int(keep.sum()), total_generated_spikes=len(spikes),
                detector_events_per_minute=len(detected)/(args.duration_s/60),
                generated_home_prevalence=float(region[states].mean()),
                selected_home_prevalence=float(region[locations].mean()) if len(locations) else np.nan,
                status="complete" if len(endpoints) else "no_selected_events"))
            for cohort, windows, n, exposure, loc, keys in (
                ("fixed", fixed, fixed_counts, fixed_exposure, fixed_locations, fixed_ids),
                ("selected", endpoints, a_counts, exposures, locations, endpoint_ids),
                ("independent_same_window", endpoints, independent, exposures, locations, endpoint_ids)):
                if not len(n):
                    continue
                z = region[loc].astype(int)
                for readout, e in (("frozen_poisson", .02), ("oracle_gain", exposure)):
                    bf = regional_log_bf(n, rates, region, e)
                    calls = calls_from_bf(bf, n.sum(axis=1))
                    event_meta = dict(**meta, cohort=cohort, readout=readout)
                    event_frame = pd.DataFrame(dict(event_id=keys, start_s=windows[:, 0], end_s=windows[:, 1],
                        true_home=z, latent_state=loc, exposure_s=exposure, n_spikes=n.sum(axis=1),
                        n_active_cells=np.count_nonzero(n, axis=1), log_bf=bf, category=calls, **event_meta))
                    events_rows.append(event_frame)
                    for label in (0, 1):
                        for category in (0, 1, 2):
                            distribution_rows.append(dict(**event_meta, true_home=label, category=category,
                                count=int(((z == label) & (calls == category)).sum()), denominator=int((z == label).sum())))
                    target_blocks = block_histograms(calls, np.floor(windows[:, 0]/30).astype(int))
                    for calibration, cal_blocks in (("observed_RUN", run_blocks), ("generator_unselected", generator_blocks)):
                        fit, intervals, _ = bootstrap_compatibility(cal_blocks, target_blocks,
                            np.random.default_rng(stable_seed(args.seed, row.session, replicate, peak, cohort, readout, calibration)), args.bootstrap)
                        model_meta = dict(**event_meta, calibration=calibration)
                        results.append(dict(**model_meta, windows=len(n), true_prevalence=float(z.mean()),
                            mean_spikes=float(n.sum(axis=1).mean()), silent_fraction=float((n.sum(axis=1) == 0).mean()),
                            **fit, prevalence_bias=fit["prevalence"]-float(z.mean()), **discrimination(bf, z)))
                        interval_rows.extend(dict(**model_meta, true_prevalence=float(z.mean()),
                            **interval, contains_truth=bool(interval["status"] == "feasible" and interval["lower"] <= z.mean() <= interval["upper"])) for interval in intervals)
            write_json(folder/"outputs.json", {p.name: file_sha256(p) for p in folder.iterdir() if p.is_file()})
            print(row.session, "rep", replicate, "peak", peak, "detected", len(detected), "retained", len(endpoints), flush=True)
    for name, values in (("calibration_fits", results), ("compatibility", interval_rows),
                         ("detection", detection_rows), ("call_distributions", distribution_rows)):
        pd.DataFrame(values).to_csv(target/(name+".csv"), index=False)
    pd.concat(events_rows, ignore_index=True).to_csv(target/"event_readouts.csv.gz", index=False)
    write_json(target/"outputs.json", {p.name: file_sha256(p) for p in target.iterdir() if p.is_file()})
    return {"animal": row.animal, "session": row.session, "status": "complete", "folder": str(target),
                "detector_cells": len(ids), "decoding_cells": int(keep.sum())}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frontier-root", type=Path, default=Path("/mnt/seagate10tb/florianpfaff/regional-content-frontier-pf-20260915"))
    p.add_argument("--detector-script", type=Path, default=Path("/home/florianpfaff/HippoReplayIMM-replay-geometry-hypotheses/scripts/select_pfeiffer_foster_speed_candidates.py"))
    p.add_argument("--detector-manifest", type=Path, default=Path("/home/florianpfaff/HippoReplayIMM-replay-geometry-hypotheses/results/pfeiffer-foster-speed-candidates-full-v1/pfeiffer_speed_candidate_manifest.json"))
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=2026091517)
    p.add_argument("--duration-s", type=int, default=600)
    p.add_argument("--replicates", type=int, default=4)
    p.add_argument("--max-windows", type=int, default=200)
    p.add_argument("--calibration-per-class", type=int, default=5000)
    p.add_argument("--bootstrap", type=int, default=200)
    p.add_argument("--session-limit", type=int)
    args = p.parse_args()
    if args.duration_s < 60 or args.duration_s % 2 or min(args.replicates, args.max_windows, args.calibration_per_class, args.bootstrap) < 1:
        p.error("positive sizes; even duration >=60 s")
    frozen = json.loads(args.detector_manifest.read_text())["parameters"]
    for key, value in {"mua_bin_ms": 1., "mua_gaussian_sd_ms": 10., "mua_z_threshold": 3., "maximum_speed_cm_s": 5.,
                           "minimum_duration_ms": 50., "maximum_duration_ms": 2000., "minimum_active_cell_fraction": .1}.items():
        if frozen[key] != value:
            raise ValueError("unexpected frozen detector settings")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    inputs = {"frontier_manifest": args.frontier_root/"manifest.json", "sessions": args.frontier_root/"sessions.csv",
        "detector_script": args.detector_script, "detector_manifest": args.detector_manifest, "script": Path(__file__),
        "null_core": ROOT/"src/hipporeplayimm/regional_content_mua_null.py",
        "readout_core": ROOT/"src/hipporeplayimm/regional_content_frontier.py",
        "bounds_core": ROOT/"src/hipporeplayimm/regional_content_bounds.py",
        "protocol": ROOT/"docs/regional_content_mua_selection_null_protocol.md"}
    manifest = build_script_provenance(input_paths=inputs, cwd=ROOT)
    manifest.update(status="running", started_at_utc=datetime.now(UTC).isoformat(),
        parameters={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        tuning_changes=False, latent_state_changes_within_epoch=False, no_real_replay_truth=True)
    write_json(args.output_dir/"manifest.json", manifest)
    sessions = pd.read_csv(inputs["sessions"])
    if not sessions.status.eq("complete").all() or sessions.session.duplicated().any():
        raise ValueError("invalid input cohort")
    if args.session_limit:
        sessions = sessions.iloc[:args.session_limit]
    detector = load_detector(args.detector_script)
    rows = []
    for row in sessions.itertuples():
        rows.append(run_session(row, args, detector, inputs))
        pd.DataFrame(rows).to_csv(args.output_dir/"sessions.csv", index=False)
    for name in ("calibration_fits", "compatibility", "detection", "call_distributions"):
        pd.concat([pd.read_csv(Path(row["folder"])/(name+".csv")) for row in rows], ignore_index=True).to_csv(args.output_dir/(name+".csv"), index=False)
    final = build_script_provenance(input_paths=inputs, cwd=ROOT)
    for key, value in manifest["input_file_sha256"].items():
        if final["input_file_sha256"][key] != value:
            raise ValueError("input changed: "+key)
    manifest.update(final, status="complete", finished_at_utc=datetime.now(UTC).isoformat())
    write_json(args.output_dir/"manifest.json", manifest)


if __name__ == "__main__":
    main()
