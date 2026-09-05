#!/usr/bin/env python3
"""Paired full/half-cell decoding of frozen detector and window cohorts."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from hipporeplayimm.replay_coverage_data import count_candidate_bins, index_spike_times
from hipporeplayimm.replay_coverage_detector_sensitivity import (
    IDENTITY,
    paired_metrics,
    session_summary,
    summarize_animals,
)

from scripts._provenance import build_script_provenance, file_sha256
from scripts.analyze_replay_coverage_subsampling import (
    decode_compact,
    decoding_support,
    event_windows,
    population_subsets,
)

METRIC_KEY = ["window_uid", "cell_fraction", "population_replicate", "likelihood", "estimator", "bin_filter"]


def input_arrays(record, windows):
    if file_sha256(record["source_cache_path"]) != record["source_cache_sha256"]:
        raise ValueError("source map/spike cache changed")
    with np.load(record["source_cache_path"], allow_pickle=False) as a:
        mask = a["unit_qc_mask"].astype(bool)
        ids, spikes = a["cell_ids"][mask], a["spikes"]
        support = decoding_support(a["valid_spatial_bins"], a["bin_centers_cm"], a["arena_bounds_cm"])
        rates, grid = a["rates_hz"][mask][:, support], a["bin_centers_cm"][support]
    indexed = index_spike_times(spikes, ids)
    chunks, starts, ends, offsets = [], [], [], [0]
    for row in windows.itertuples(index=False):
        edges, counts = count_candidate_bins(indexed, ids, row.start_s, row.end_s, .005)
        local = event_windows(counts, np.diff(edges))
        chunks.append(local)
        starts.append(edges[:len(local)])
        ends.append(edges[4:4 + len(local)])
        offsets.append(offsets[-1] + len(local))
    return {"cell_ids": ids, "rates_hz": rates, "grid_cm": grid,
        "frame_counts": np.concatenate(chunks) if chunks else np.empty((0, len(ids)), np.int64),
        "frame_start_s": np.concatenate(starts) if starts else np.empty(0),
        "frame_end_s": np.concatenate(ends) if ends else np.empty(0),
        "frame_offsets": np.asarray(offsets, np.int64), "window_uids": windows.window_uid.to_numpy(str)}


def metric_rows(arrays, decoded, full_decoded, windows, spec):
    subset = np.asarray(spec["indices"], int)
    rows = []
    for i, w in enumerate(windows.itertuples(index=False)):
        a, b = arrays["frame_offsets"][i:i + 2]
        counts, full_counts = arrays["frame_counts"][a:b][:, subset], arrays["frame_counts"][a:b]
        for estimator in ["map", "posterior_mean"]:
            for support in [False, True]:
                rows.append({"window_uid": w.window_uid, "cell_fraction": spec["cell_fraction"],
                    "population_replicate": spec["population_replicate"], "likelihood": spec["likelihood"],
                    "estimator": estimator, "bin_filter": "at_least_2cells_3spikes" if support else "unfiltered",
                    "retained_cells": len(subset),
                    **paired_metrics(decoded[estimator][a:b], counts, decoded["posterior_rms_cm"][a:b],
                        decoded["posterior_entropy_nats"][a:b], full_decoded[estimator][a:b], full_counts, support)})
    return rows


def process_session(record, output, seed, replicates):
    started = time.monotonic()
    identity = {k: record[k] for k in IDENTITY}
    label = "__".join(identity[k].replace("/", "_") for k in IDENTITY)
    if file_sha256(record["windows_path"]) != record["windows_sha256"]:
        raise ValueError("frozen window definitions changed")
    source = pd.read_csv(record["windows_path"], float_precision="round_trip")
    windows = source[source.eligible].sort_values("window_uid").reset_index(drop=True)
    arrays = input_arrays(record, windows)
    path = output / f"input__{label}.npz"
    np.savez_compressed(path, **arrays)
    specs, frames, decoded_paths = [], [], []
    full = {}
    for fraction, rep, subset in population_subsets(arrays["cell_ids"], [.5], replicates, seed, ":".join(identity.values())):
        spec = {"cell_fraction": fraction, "population_replicate": rep, "indices": subset.tolist(), "cell_ids": arrays["cell_ids"][subset].tolist()}
        specs.append(spec)
        for likelihood in ["poisson", "conditional_multinomial"]:
            decoded = decode_compact(arrays["frame_counts"][:, subset], arrays["rates_hz"][subset], arrays["grid_cm"], likelihood)
            if fraction == 1.0:
                full[likelihood] = decoded
            score_path = output / f"decoded__{label}__{fraction:g}__{rep}__{likelihood}.npz"
            np.savez_compressed(score_path, **decoded)
            decoded_paths.append({**spec, "likelihood": likelihood, "path": str(score_path), "sha256": file_sha256(score_path)})
            frames.extend(metric_rows(arrays, decoded, full[likelihood], windows, {**spec, "likelihood": likelihood}))
    empty_names = list(paired_metrics(np.empty((0, 2)), np.empty((0, 1)), [], [], np.empty((0, 2)), np.empty((0, 1)), False))
    metrics = pd.DataFrame(frames, columns=METRIC_KEY + ["retained_cells"] + empty_names)
    if metrics.duplicated(METRIC_KEY).any() or len(metrics) != len(windows) * (1 + replicates) * 8:
        raise ValueError("missing or duplicate scoring conditions")
    metrics_path = output / f"metrics__{label}.csv.gz"
    metrics.to_csv(metrics_path, index=False)
    population = session_summary(metrics, source, specs)
    population_path = output / f"population__{label}.csv"
    population.to_csv(population_path, index=False)
    metadata = {**identity, "status": "complete", "source_windows_path": record["windows_path"], "source_windows_sha256": record["windows_sha256"],
        "source_cache_path": record["source_cache_path"], "source_cache_sha256": record["source_cache_sha256"],
        "eligible_windows": len(windows), "metric_rows": len(metrics), "decoding_frames": len(arrays["frame_counts"]),
        "input_arrays_path": str(path), "input_arrays_sha256": file_sha256(path), "population_specs": specs,
        "decoded_paths": decoded_paths, "metrics_path": str(metrics_path), "metrics_sha256": file_sha256(metrics_path),
        "population_path": str(population_path), "population_sha256": file_sha256(population_path), "runtime_s": time.monotonic() - started}
    (output / f"session__{label}.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({k: metadata[k] for k in IDENTITY + ["eligible_windows", "metric_rows", "runtime_s"]}), flush=True)
    return metadata


def run(args):
    manifest_path = args.input_dir / "coverage_event_definition_manifest.json"
    audit_path = args.input_dir / "coverage_event_definition_reconstruction_audit.json"
    parent, audit = json.loads(manifest_path.read_text()), json.loads(audit_path.read_text())
    if parent["status"] not in {"complete", "complete_with_ripple_unavailable"} or audit["status"] != "pass" or audit["input_file_sha256"]["preparation_manifest"] != file_sha256(manifest_path):
        raise ValueError("linked successful preparation and audit required")
    records = parent["results"]
    if args.sessions:
        requested = set(args.sessions)
        records = [r for r in records if ":".join(r[k] for k in IDENTITY) in requested]
        if len(records) != len(requested):
            raise ValueError("missing or duplicate requested sessions")
    if not records or any(r["status"] != "complete" for r in records):
        raise ValueError("no complete sessions")
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "inputs").mkdir()
    inputs = {"event_manifest": manifest_path, "event_audit": audit_path,
        "script": Path(__file__), "library": ROOT / "src/hipporeplayimm/replay_coverage_detector_sensitivity.py",
        "decoder": ROOT / "src/hipporeplayimm/replay_coverage.py", "counting_library": ROOT / "src/hipporeplayimm/replay_coverage_data.py",
        "decoding_helpers": ROOT / "scripts/analyze_replay_coverage_subsampling.py", "provenance_library": ROOT / "scripts/_provenance.py",
        "protocol": ROOT / "docs/replay_coverage_detector_decoding_protocol.md"}
    meta = build_script_provenance(input_paths=inputs, cwd=ROOT)
    for name, path in inputs.items():
        shutil.copy2(path, out / "inputs" / f"{name}{Path(path).suffix}")
    meta.update(status="running", created_at_utc=datetime.now(UTC).isoformat(), seed=args.seed, replicates=args.replicates,
        workers=args.workers, bootstraps=args.bootstraps, requested_sessions=args.sessions,
        versions={name: version(name) for name in ["numpy", "pandas", "scipy"]},
        likelihoods=["poisson", "conditional_multinomial"], temporal_prior=None,
        decoder_window_s=.020, decoder_stride_s=.005, speed_stride_s=.020,
        speed_gap_policy="all_intermediate_frames_required_in_each_population; paired_speed_uses_common_steps",
        claim_boundary="paired recording sensitivity of geometric screening and speed readouts; not latent replay truth or uniformity")
    target = out / "coverage_detector_sensitivity_manifest.json"
    target.write_text(json.dumps(meta, indent=2) + "\n")
    results = []
    failures = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_session, r, out, args.seed, args.replicates): r for r in records}
        for f in as_completed(futures):
            try:
                results.append(f.result())
            except Exception as exc:  # noqa: BLE001 -- collect worker errors, then fail the whole run.
                failures.append({**{k: futures[f][k] for k in IDENTITY}, "error": f"{type(exc).__name__}: {exc}"})
    if failures:
        meta.update(status="failed", results=results, failures=failures)
        target.write_text(json.dumps(meta, indent=2) + "\n")
        raise RuntimeError(f"{len(failures)} session workers failed; see terminal manifest")
    results.sort(key=lambda r: tuple(r[k] for k in IDENTITY))
    population = pd.concat([pd.read_csv(r["population_path"]) for r in results], ignore_index=True)
    session, animal, summary = summarize_animals(population, args.bootstraps, args.seed + 1)
    for name, frame in [("population_summary", population), ("session_summary", session), ("animal_summary", animal), ("summary", summary)]:
        frame.to_csv(out / f"coverage_detector_sensitivity_{name}.csv", index=False)
    unchanged = all(file_sha256(path) == meta["input_file_sha256"][name] for name, path in inputs.items())
    unchanged &= all(file_sha256(r["source_cache_path"]) == r["source_cache_sha256"] and file_sha256(r["source_windows_path"]) == r["source_windows_sha256"] for r in results)
    n = sum(r["eligible_windows"] for r in results)
    gates = {"nonempty_eligible_cohort": n > 0, "all_source_sessions_processed": len(results) == len(records),
        "required_condition_rows_complete": n > 0 and sum(r["metric_rows"] for r in results) == n * (1 + args.replicates) * 8,
        "explicit_zero_and_unavailable_denominators": len(session[IDENTITY].drop_duplicates()) == len(records),
        "source_code_and_inputs_unchanged": bool(unchanged)}
    gates["overall_technical"] = all(gates.values())
    pd.DataFrame([{"gate": k, "passed": bool(v)} for k, v in gates.items()]).to_csv(out / "coverage_detector_sensitivity_gate_summary.csv", index=False)
    meta.update(status="complete" if gates["overall_technical"] else "failed", sessions=len(results), eligible_windows=n,
        metric_rows=sum(r["metric_rows"] for r in results), results=results,
        output_sha256={p.name: file_sha256(p) for p in out.glob("coverage_detector_sensitivity_*.csv")})
    target.write_text(json.dumps(meta, indent=2) + "\n")
    if not gates["overall_technical"]:
        raise RuntimeError("technical gates failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--bootstraps", type=int, default=5000)
    parser.add_argument("--sessions", nargs="*")
    args = parser.parse_args()
    if min(args.workers, args.replicates, args.bootstraps) < 1:
        parser.error("workers, replicates and bootstraps must be positive")
    run(args)
