#!/usr/bin/env python3
"""Apply the transferred PF-style two-shuffle criterion to fixed candidate cores."""

from __future__ import annotations

import argparse
import hashlib
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
from hipporeplayimm.replay_coverage_shuffle_baseline import CRITERIA, FAMILIES, shuffle_test

from scripts._provenance import build_script_provenance, file_sha256
from scripts.analyze_replay_coverage_subsampling import decoding_support, event_windows, population_subsets

IDENTITY = ["dataset", "animal", "session"]
OBSERVATIONS = ["original_order", "order_randomized"]
CONDITION = ["detector", "observation", "bin_filter", "min_frames", "alpha"]


def stable_seed(seed, key):
    return int.from_bytes(hashlib.sha256(f"{seed}:{key}".encode()).digest()[:8], "little")


def order_permutation(durations, seed):
    durations = np.asarray(durations, float)
    permutation = np.arange(len(durations))
    complete = np.flatnonzero(np.isclose(durations, .005, atol=1e-9, rtol=0))
    permutation[complete] = np.random.default_rng(seed).permutation(complete)
    return permutation


def input_arrays(record, windows, seed):
    if file_sha256(record["source_cache_path"]) != record["source_cache_sha256"]:
        raise ValueError("source map/spike cache changed")
    with np.load(record["source_cache_path"], allow_pickle=False) as a:
        mask = a["unit_qc_mask"].astype(bool)
        ids, spikes = a["cell_ids"][mask], a["spikes"]
        support = decoding_support(a["valid_spatial_bins"], a["bin_centers_cm"], a["arena_bounds_cm"])
        rates, grid = a["rates_hz"][mask], a["bin_centers_cm"]
        x = (a["x_edges_cm"][:-1] + a["x_edges_cm"][1:]) / 2
        y = (a["y_edges_cm"][:-1] + a["y_edges_cm"][1:]) / 2
        xx, yy = np.meshgrid(x, y, indexing="ij")
        if not np.allclose(grid, np.column_stack([xx.ravel(), yy.ravel()])):
            raise ValueError("rectangular map flattening does not match x/y shuffle operator")
    indexed = index_spike_times(spikes, ids)
    base, starts, durations, permutation, frames = [], [], [], [], []
    base_offsets, frame_offsets = [0], [0]
    for w in windows.itertuples(index=False):
        edges, counts = count_candidate_bins(indexed, ids, w.start_s, w.end_s, .005)
        duration = np.diff(edges)
        perm = order_permutation(duration, stable_seed(seed, w.window_uid))
        base.append(counts)
        starts.append(edges[:-1])
        durations.append(duration)
        permutation.append(perm + base_offsets[-1])
        base_offsets.append(base_offsets[-1] + len(counts))
        for observed in [counts, counts[perm]]:
            decoded_counts = event_windows(observed, duration)
            frames.append(decoded_counts)
            frame_offsets.append(frame_offsets[-1] + len(decoded_counts))
    return {"cell_ids": ids, "rates_hz": rates, "support": support, "grid_cm": grid[support], "grid_shape": np.array([len(x), len(y)]),
        "base_counts": np.concatenate(base) if base else np.empty((0, len(ids)), np.int32),
        "base_starts_s": np.concatenate(starts) if starts else np.empty(0),
        "base_durations_s": np.concatenate(durations) if durations else np.empty(0),
        "base_offsets": np.array(base_offsets), "permutation": np.concatenate(permutation) if permutation else np.empty(0, int),
        "frame_counts": np.concatenate(frames) if frames else np.empty((0, len(ids)), np.int64),
        "frame_offsets": np.array(frame_offsets), "window_uids": windows.window_uid.to_numpy(str)}


def decision_rows(result, windows, spec, shuffles):
    rows = []
    index = {int(k): j for j, k in enumerate(result["tested_observations"])}
    successes = result["null_pass"].sum(axis=1)
    for i, w in enumerate(windows.itertuples(index=False)):
        for obs, name in enumerate(OBSERVATIONS):
            oi = 2 * i + obs
            for ci, (filtered, minimum) in enumerate(CRITERIA):
                passed = bool(result["geometric_pass"][oi, ci])
                p = result["p_values"][oi, ci]
                counts = successes[:, index[oi], ci] if passed else [np.nan, np.nan]
                rows.append({**{k: getattr(w, k) for k in IDENTITY}, "window_uid": w.window_uid,
                    "detector": w.detector, "observation": name, "bin_filter": "at_least_2cells_3spikes" if filtered else "edge_only",
                    "min_frames": minimum, "cell_fraction": spec["cell_fraction"], "population_replicate": spec["population_replicate"],
                    "retained_cells": len(spec["cell_ids"]), "geometric_pass": passed,
                    "test_status": "tested" if passed else "geometric_fail_not_tested", "n_shuffles": shuffles if passed else 0,
                    "cell_identity_successes": counts[0], "xy_roll_successes": counts[1],
                    "cell_identity_p": p[0], "xy_roll_p": p[1],
                    **{f"accepted_alpha_{alpha:g}": passed and bool(np.all(p < alpha)) for alpha in [.01, .02, .05]}})
    return rows


def population_summary(metrics, record, specs):
    identity = {k: record[k] for k in IDENTITY}
    ripple = "native_ripple_table" if record["dataset"] == "pfeiffer_foster" else "lfp_ripple_detected"
    rows = []
    for spec in specs:
        for detector in ["source_high_mua", ripple]:
            for obs in OBSERVATIONS:
                for filtered, minimum in CRITERIA:
                    label = "at_least_2cells_3spikes" if filtered else "edge_only"
                    subset = metrics[(metrics.cell_fraction == spec["cell_fraction"]) & (metrics.population_replicate == spec["population_replicate"])
                        & (metrics.detector == detector) & (metrics.observation == obs) & (metrics.bin_filter == label) & (metrics.min_frames == minimum)]
                    for alpha in [.01, .02, .05]:
                        rows.append({**identity, "detector": detector, "observation": obs, "bin_filter": label, "min_frames": minimum,
                            "alpha": alpha, "cell_fraction": spec["cell_fraction"], "population_replicate": spec["population_replicate"],
                            "detector_available": detector == "source_high_mua" or record["ripple_status"] == "available",
                            "eligible_events": len(subset), "geometric_events": int(subset.geometric_pass.sum()),
                            "accepted_events": int(subset[f"accepted_alpha_{alpha:g}"].sum()),
                            "geometric_fraction": subset.geometric_pass.mean(), "accepted_fraction": subset[f"accepted_alpha_{alpha:g}"].mean()})
    return pd.DataFrame(rows)


def summarize(population, bootstraps=5000, seed=20260906):
    keys = IDENTITY + CONDITION
    session = population.groupby(keys + ["cell_fraction"], dropna=False).agg(
        eligible_events=("eligible_events", "first"), detector_available=("detector_available", "first"),
        replicates=("population_replicate", "nunique"), geometric_fraction=("geometric_fraction", "mean"), accepted_fraction=("accepted_fraction", "mean")).reset_index()
    full = session[session.cell_fraction == 1].drop(columns=["cell_fraction", "replicates"])
    half = session[session.cell_fraction == .5].drop(columns=["cell_fraction", "replicates", "eligible_events", "detector_available"])
    paired = full.merge(half, on=keys, suffixes=("_full", "_half"), validate="one_to_one")
    if len(paired) != len(full) or len(paired) != len(half):
        raise ValueError("missing population denominator")
    for metric in ["geometric_fraction", "accepted_fraction"]:
        paired[f"{metric}_delta"] = paired[f"{metric}_half"] - paired[f"{metric}_full"]
    values = [f"{metric}_{suffix}" for metric in ["geometric_fraction", "accepted_fraction"] for suffix in ["full", "half", "delta"]]
    animal = paired.groupby(["dataset", "animal"] + CONDITION, dropna=False)[values].mean().reset_index()
    rows = []
    rng = np.random.default_rng(seed)
    for keys, group in animal.groupby(["dataset"] + CONDITION, dropna=False, sort=True):
        key = dict(zip(["dataset"] + CONDITION, keys, strict=True))
        for metric in values:
            measured = group[metric].dropna().to_numpy()
            means = measured[rng.integers(0, len(measured), (bootstraps, len(measured)))].mean(axis=1) if len(measured) else np.full(bootstraps, np.nan)
            low, high = np.quantile(means, [.025, .975]) if len(measured) else [np.nan, np.nan]
            rows.append({**key, "metric": metric, "animals_measurable": len(measured), "animals_total": len(group),
                "equal_animal_mean": np.mean(measured) if len(measured) else np.nan, "ci_low": low, "ci_high": high,
                "animals_negative": int((measured < 0).sum()), "animals_positive": int((measured > 0).sum())})
    return paired, animal, pd.DataFrame(rows)


def process_session(record, output, seed, shuffles, batch_size):
    started = time.monotonic()
    identity = {k: record[k] for k in IDENTITY}
    label = "__".join(str(identity[k]).replace("/", "_") for k in IDENTITY)
    if file_sha256(record["windows_path"]) != record["windows_sha256"]:
        raise ValueError("source windows changed")
    source = pd.read_csv(record["windows_path"], float_precision="round_trip")
    windows = source[source.eligible & source.window_variant.eq("detected_core")].sort_values("window_uid").reset_index(drop=True)
    arrays = input_arrays(record, windows, seed)
    array_path = output / f"input__{label}.npz"
    np.savez_compressed(array_path, **arrays)
    specs, rows, files = [], [], []
    for fraction, rep, subset in population_subsets(arrays["cell_ids"], [.5], 3, seed, ":".join(identity.values())):
        spec = {"cell_fraction": fraction, "population_replicate": rep, "indices": subset.tolist(), "cell_ids": arrays["cell_ids"][subset].tolist()}
        specs.append(spec)
        bank_seed = stable_seed(seed, f"{label}:{fraction}:{rep}:maps")
        result = shuffle_test(arrays["frame_counts"][:, subset], arrays["frame_offsets"], arrays["grid_cm"], arrays["rates_hz"][subset],
            arrays["support"], arrays["grid_shape"], shuffles, bank_seed, batch_size)
        archive = {k: v for k, v in result.items() if k not in {"banks", "sample_paths"}}
        for family in FAMILIES:
            archive[f"bank__{family}"] = result["banks"][family]
            archive[f"sample_paths__{family}"] = result["sample_paths"][family]
        path = output / f"shuffles__{label}__{fraction:g}__{rep}.npz"
        np.savez_compressed(path, **archive)
        files.append({**spec, "bank_seed": bank_seed, "path": str(path), "sha256": file_sha256(path)})
        rows.extend(decision_rows(result, windows, spec, shuffles))
        print(json.dumps({**identity, "population": [fraction, rep], "eligible_events": len(windows),
            "observations_tested": len(result["tested_observations"]), "runtime_s": time.monotonic() - started}), flush=True)
    columns = IDENTITY + ["window_uid", "detector", "observation", "bin_filter", "min_frames", "cell_fraction", "population_replicate", "retained_cells",
        "geometric_pass", "test_status", "n_shuffles", "cell_identity_successes", "xy_roll_successes", "cell_identity_p", "xy_roll_p"] + [f"accepted_alpha_{a:g}" for a in [.01, .02, .05]]
    metrics = pd.DataFrame(rows, columns=columns)
    if len(metrics) != len(windows) * 32 or metrics.duplicated(["window_uid", "observation", "bin_filter", "min_frames", "cell_fraction", "population_replicate"]).any():
        raise ValueError("duplicate or missing event condition rows")
    metrics_path, population_path = output / f"decisions__{label}.csv.gz", output / f"population__{label}.csv"
    metrics.to_csv(metrics_path, index=False)
    population_summary(metrics, record, specs).to_csv(population_path, index=False)
    meta = {**identity, "status": "complete", "source": record, "eligible_events": len(windows), "decision_rows": len(metrics),
        "input_arrays_path": str(array_path), "input_arrays_sha256": file_sha256(array_path), "population_specs": specs, "shuffle_files": files,
        "metrics_path": str(metrics_path), "metrics_sha256": file_sha256(metrics_path), "population_path": str(population_path),
        "population_sha256": file_sha256(population_path), "runtime_s": time.monotonic() - started}
    (output / f"session__{label}.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def run(args):
    source = args.input_dir / "coverage_event_definition_manifest.json"
    audit_path = args.input_dir / "coverage_event_definition_reconstruction_audit.json"
    parent, audit = json.loads(source.read_text()), json.loads(audit_path.read_text())
    if parent["status"] not in {"complete", "complete_with_ripple_unavailable"} or audit["status"] != "pass" or audit["input_file_sha256"]["preparation_manifest"] != file_sha256(source):
        raise ValueError("successful linked preparation and audit required")
    records = parent["results"]
    if args.sessions:
        records = [r for r in records if ":".join(r[k] for k in IDENTITY) in args.sessions]
        if len(records) != len(set(args.sessions)):
            raise ValueError("missing or duplicate requested sessions")
    if not records or any(r["status"] != "complete" for r in records):
        raise ValueError("complete input sessions required")
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "inputs").mkdir()
    inputs = {"event_manifest": source, "event_audit": audit_path, "script": Path(__file__),
        "library": ROOT / "src/hipporeplayimm/replay_coverage_shuffle_baseline.py",
        "counting": ROOT / "src/hipporeplayimm/replay_coverage_data.py", "helpers": ROOT / "scripts/analyze_replay_coverage_subsampling.py",
        "provenance": ROOT / "scripts/_provenance.py", "protocol": ROOT / "docs/replay_coverage_shuffle_baseline_protocol.md"}
    if getattr(args, "subsampling_manifest", None):
        inputs["frozen_subsets"] = args.subsampling_manifest
    meta = build_script_provenance(input_paths=inputs, cwd=ROOT)
    for key, path in inputs.items():
        shutil.copy2(path, out / "inputs" / f"{key}{path.suffix}")
    meta.update(status="running", created_at_utc=datetime.now(UTC).isoformat(), shuffles_per_family=args.shuffles, seed=args.seed,
        batch_size=args.batch_size, workers=args.workers, bootstraps=args.bootstraps, requested_sessions=args.sessions,
        benchmark_scope="published_budget" if args.shuffles == 5000 else "technical_smoke",
        families=FAMILIES, versions={name: version(name) for name in ["numpy", "pandas", "scipy"]},
        claim_boundary="PF-style transferred geometric and two-shuffle screen; 8cm common maps, not exact author reproduction or real replay truth")
    manifest = out / "coverage_shuffle_baseline_manifest.json"
    manifest.write_text(json.dumps(meta, indent=2) + "\n")
    results, failures = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_session, r, out, args.seed, args.shuffles, args.batch_size): r for r in records}
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as exc:  # noqa: BLE001 -- preserve worker failures and complete remaining workers.
                failures.append({**{k: futures[future][k] for k in IDENTITY}, "error": f"{type(exc).__name__}: {exc}"})
    meta.update(results=sorted(results, key=lambda r: tuple(r[k] for k in IDENTITY)), failures=failures)
    if failures:
        meta["status"] = "failed"
        manifest.write_text(json.dumps(meta, indent=2) + "\n")
        raise RuntimeError(f"{len(failures)} workers failed; see manifest")
    population = pd.concat([pd.read_csv(r["population_path"]) for r in meta["results"]], ignore_index=True)
    session, animal, summary = summarize(population, args.bootstraps, args.seed + 1)
    for name, frame in [("population_summary", population), ("session_summary", session), ("animal_summary", animal), ("summary", summary)]:
        frame.to_csv(out / f"coverage_shuffle_baseline_{name}.csv", index=False)
    n = sum(r["eligible_events"] for r in results)
    subsets_match = False
    if "frozen_subsets" in inputs:
        frozen = json.loads(inputs["frozen_subsets"].read_text())["population_subsets"]
        lookup = {(p["session_key"], p["cell_fraction"], p["population_replicate"]): p["cell_ids"] for p in frozen}
        subsets_match = all(spec["cell_ids"] == lookup.get((":".join(r[k] for k in IDENTITY), spec["cell_fraction"], spec["population_replicate"]))
            for r in results for spec in r["population_specs"])
    unchanged = all(file_sha256(path) == meta["input_file_sha256"][key] for key, path in inputs.items())
    unchanged &= all(file_sha256(r["source"]["source_cache_path"]) == r["source"]["source_cache_sha256"] and file_sha256(r["source"]["windows_path"]) == r["source"]["windows_sha256"] for r in results)
    gates = {"nonempty_cohort": n > 0, "all_sessions_processed": len(results) == len(records),
        "all_condition_rows": n > 0 and sum(r["decision_rows"] for r in results) == n * 32,
        "source_code_and_inputs_unchanged": bool(unchanged)}
    gates["overall_technical"] = all(gates.values())
    gates["published_shuffle_budget"] = args.shuffles == 5000
    gates["matches_frozen_subsets"] = bool(subsets_match)
    gates["all_preparation_sessions"] = len(records) == len(parent["results"])
    gates["full_benchmark_complete"] = all(gates.values())
    pd.DataFrame([{"gate": k, "passed": bool(v)} for k, v in gates.items()]).to_csv(out / "coverage_shuffle_baseline_gate_summary.csv", index=False)
    meta.update(status="complete" if gates["overall_technical"] else "failed", eligible_events=n, sessions=len(results),
        output_sha256={p.name: file_sha256(p) for p in out.glob("coverage_shuffle_baseline_*.csv")})
    manifest.write_text(json.dumps(meta, indent=2) + "\n")
    if not gates["overall_technical"]:
        raise RuntimeError("technical gates failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shuffles", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--bootstraps", type=int, default=5000)
    parser.add_argument("--sessions", nargs="*")
    parser.add_argument("--subsampling-manifest", type=Path)
    options = parser.parse_args()
    if min(options.shuffles, options.batch_size, options.workers, options.bootstraps) < 1:
        parser.error("counts and batch sizes must be positive")
    run(options)
