#!/usr/bin/env python3
"""Known-path clock recovery and flat-prior decoding, simulations only."""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np
import pandas as pd

from hipporeplayimm.lagged_neural_prediction import full_count_bins
from hipporeplayimm.literal_replay_clock import (
    DT,
    PATHS,
    REPEATS,
    SpatialRates,
    bin_average,
    clock_coordinates,
    coarse_grid,
    decode,
    multinomial_samples,
    normalize,
    path_log_score,
    recovery_credit,
    rng,
    sample_geometry,
    time_samples,
)
from scripts._provenance import build_script_provenance, file_sha256

IDS = ["dataset", "animal", "session"]


def record_task(item, source, output, n_paths=PATHS, repeats=REPEATS):
    tag = item["tag"]
    with np.load(source / f"{tag}_cache.npz") as z:
        rates, centers = z["rates"], z["centers"]
        profiles = {}
        for name in sorted(k for k in z.files if k.startswith("counts_")):
            event = int(name.split("_")[1])
            counts, _ = full_count_bins(z[name], z[f"edges_{event}"])
            if len(counts) >= 3:
                profiles[event] = counts.sum(axis=1).astype(np.int32)
    if not profiles:
        raise ValueError("native profiles missing")
    spatial = SpatialRates(centers, rates)
    grids = {8: (rates, centers), 16: coarse_grid(rates, centers)}
    gains = np.exp(0.35 * rng(tag, "gains").standard_normal(len(rates)) - 0.35**2 / 2)
    chosen = rng(tag, "profiles").choice(sorted(profiles), n_paths, replace=True)
    rows, path_rows = [], []
    for path_id, event in enumerate(chosen):
        totals = profiles[int(event)]
        n = len(totals)
        points, path_rates, attempts = sample_geometry(spatial, rng(tag, path_id, "geometry"), curved=bool(path_id % 2))
        s, c, clocks = clock_coordinates(points, path_rates)
        bins = {model: bin_average(clock, path_rates, n) for model, clock in clocks.items()}
        means = {model: bin_average(clock, points, n) for model, clock in clocks.items()}
        edges = {model: np.interp(np.linspace(0, 1, n + 1), clock, s) for model, clock in clocks.items()}
        mid = {model: time_samples(clock, points, (np.arange(n) + 0.5) / n) for model, clock in clocks.items()}
        metadata = {
            **{k: item[k] for k in IDS},
            "path_id": path_id,
            "template_event": int(event),
            "n_bins": n,
            "n_spikes": int(totals.sum()),
            "duration_s": DT * n,
            "path_length_cm": s[-1],
            "shape": "curved" if path_id % 2 else "straight",
            "sampling_attempts": attempts,
            "clock_separation_rms_cm": float(np.sqrt(np.mean(np.sum((mid["physical"] - mid["neural"]) ** 2, axis=1)))),
            "code_arc_length": c[-1],
            "code_density_cv": float(np.std(np.diff(c) / np.diff(s)) / np.mean(np.diff(c) / np.diff(s))),
        }
        path_rows.append(metadata)
        arrays = {"points": points, "path_rates": path_rates, "totals": totals, "gains": gains, "s": s, "c": c}
        for model in clocks:
            arrays[f"clock_{model}"] = clocks[model]
            arrays[f"bin_rates_{model}"] = bins[model]
            arrays[f"position_mean_{model}"] = means[model]
        for condition in ("exact", "gain_drift"):
            for truth in clocks:
                emitted = bins[truth] * (1 if condition == "exact" else gains)
                p_truth = normalize(emitted)
                expected = {model: float(np.sum(totals[:, None] * p_truth * np.log(normalize(bins[model])))) for model in clocks}
                for repeat in range(repeats):
                    counts = multinomial_samples(emitted, totals, rng(tag, path_id, condition, truth, repeat, "counts"))
                    arrays[f"counts_{condition}_{truth}_{repeat}"] = counts
                    scores = {model: path_log_score(counts, bins[model]) for model in clocks}
                    delta = scores["neural"] - scores["physical"]
                    for grid, (decoder_rates, decoder_centers) in grids.items():
                        mean, maximum, rms = decode(counts, decoder_rates, decoder_centers)
                        mean_steps = np.linalg.norm(np.diff(mean, axis=0), axis=1)
                        row = metadata | {
                            "condition": condition,
                            "generator": truth,
                            "repeat": repeat,
                            "decoder_grid_cm": grid,
                            "log_score_physical": scores["physical"],
                            "log_score_neural": scores["neural"],
                            "delta_neural_minus_physical": delta,
                            "expected_delta_neural_minus_physical": expected["neural"] - expected["physical"],
                            "oracle_correct": recovery_credit(delta, truth),
                            "posterior_mean_error_cm": float(np.mean(np.linalg.norm(mean - means[truth], axis=1))),
                            "map_error_cm": float(np.mean(np.linalg.norm(maximum - means[truth], axis=1))),
                            "posterior_rms_cm": float(np.mean(rms)),
                            "decoded_mean_step_speed_cm_s": float(np.mean(mean_steps) / DT),
                            "decoded_map_step_speed_cm_s": float(np.mean(np.linalg.norm(np.diff(maximum, axis=0), axis=1)) / DT),
                            "true_bin_mean_step_speed_cm_s": float(np.mean(np.linalg.norm(np.diff(means[truth], axis=0), axis=1)) / DT),
                            "true_arc_speed_cm_s": float(s[-1] / (n * DT)),
                            "true_arc_speed_cv": float(np.std(np.diff(edges[truth])) / np.mean(np.diff(edges[truth]))),
                            "hard_continuous_step_fraction": float(np.mean(mean_steps < 20)),
                            "zero_spike_bins": int(np.sum(totals == 0)),
                        }
                        rows.append(row)
        np.savez_compressed(output / f"{tag}_p{path_id:03d}.npz", **arrays)
    pd.DataFrame(rows).to_csv(output / f"{tag}_scores.csv.gz", index=False)
    pd.DataFrame(path_rows).to_csv(output / f"{tag}_paths.csv", index=False)
    return {**item, "n_paths": n_paths, "n_rows": len(rows), "n_observations": n_paths * repeats * 4}


def summarize(table):
    primary = table[table.decoder_grid_cm.eq(8)].copy()
    keys = IDS + ["path_id", "condition", "generator", "repeat"]
    if primary.empty or primary.duplicated(keys).any():
        raise ValueError("nonempty unique observation rows required")
    fields = ["oracle_correct", "delta_neural_minus_physical", "posterior_mean_error_cm", "decoded_mean_step_speed_cm_s", "true_arc_speed_cm_s", "clock_separation_rms_cm"]
    paths = primary.groupby(IDS + ["condition", "path_id"], as_index=False)[fields].mean()
    records = paths.groupby(IDS + ["condition"], as_index=False)[fields].mean()
    animals = records.groupby(["dataset", "animal", "condition"], as_index=False)[fields].mean()
    summary = animals.groupby(["dataset", "condition"], as_index=False)[fields].mean()
    summary = summary.merge(
        animals.groupby(["dataset", "condition"], as_index=False).agg(n_animals=("animal", "nunique"), minimum_animal_accuracy=("oracle_correct", "min")), validate="one_to_one"
    )
    summary["oracle_practical_pass"] = (summary.oracle_correct >= 0.80) & (summary.minimum_animal_accuracy > 0.50)
    return paths, records, animals, summary


def run(args):
    source = args.source_dir
    mp = source / "conditional_2d_manifest.json"
    parent = json.loads(mp.read_text())
    items = sorted(parent["completed"], key=lambda x: (x["dataset"], x["animal"], x["session"]))
    for item in items:
        item["tag"] = "__".join(str(item[k]).replace("/", "_") for k in IDS)
        name = item["tag"] + "_cache.npz"
        if file_sha256(source / name) != parent["output_sha256"][name]:
            raise ValueError("source hash mismatch")
    if parent["status"] != "complete" or len(items) != 33 or len({x["animal"] for x in items}) != 9:
        raise ValueError("all 33 complete recordings required")
    provenance = build_script_provenance(input_paths={"source_manifest": mp, "protocol": ROOT / "docs/literal_replay_clock_protocol.md"}, cwd=ROOT)
    if provenance["git_dirty"] or provenance["code_commit"] == "unavailable":
        raise ValueError("clean committed producer required")
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    manifest = provenance | {
        "status": "running",
        "source_dir": str(source),
        "completed": [],
        "real_events_rescored": False,
        "biological_mechanism_established": False,
        "oracle_knows_path": True,
        "paths_per_recording": PATHS,
        "repeats": REPEATS,
    }
    path = out / "literal_clock_manifest.json"
    started = time.monotonic()
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(record_task, item, source, out) for item in items]
            for future in as_completed(futures):
                result = future.result()
                manifest["completed"].append(result)
                path.write_text(json.dumps(manifest, indent=2) + "\n")
                print(json.dumps(result), flush=True)
        table = pd.concat([pd.read_csv(out / f"{item['tag']}_scores.csv.gz") for item in items], ignore_index=True)
        for name, frame in zip(("paths", "recordings", "animals", "summary"), summarize(table), strict=True):
            frame.to_csv(out / f"literal_clock_{name}.csv", index=False)
        manifest.update(status="complete", n_rows=len(table), n_observations=sum(x["n_observations"] for x in manifest["completed"]), runtime_s=time.monotonic() - started)
        manifest["output_sha256"] = {p.name: file_sha256(p) for p in sorted(out.iterdir()) if p != path}
    except BaseException as error:
        manifest.update(status="failed", error=repr(error))
        raise
    finally:
        path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    run(parser.parse_args())
