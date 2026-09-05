#!/usr/bin/env python3
"""RatInABox field-factorial and same-observation decoder-resolution benchmark."""

from __future__ import annotations

import argparse
import importlib.metadata
import inspect
import itertools
import json
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from hipporeplayimm.replay_coverage_data import array_sha256
from hipporeplayimm.replay_coverage_geometry import (
    CORE,
    FACTORS,
    KEY,
    centered_grid,
    decode_chunked,
    decoder_settings,
    field_population,
    generate_observations,
    make_paths,
    measurement_metrics,
    resolution_truth,
    window_counts,
)
from hipporeplayimm.replay_coverage_recovery import gradient_from_moments
from scripts._provenance import build_script_provenance, file_sha256

FIELDS = list(itertools.product([3.7, 8.75], [1., 1.4], [20., 30., 40.]))


def build_batch(args, population_seed, field_id):
    area, aspect, sigma = FIELDS[field_id]
    centers = np.random.default_rng([args.seed, population_seed, 0]).uniform(-.5, .5, (128, 2))
    evaluate, physical, bounds = field_population(area, aspect, sigma, centers, args.peak_hz, args.floor_hz)
    paths = make_paths(args.seed, population_seed, args.paths, args.duration_ms)
    observations = generate_observations(paths, evaluate, args.seed, population_seed, field_id, count_rate_hz=args.count_rate_hz)
    return paths, observations, evaluate, physical, bounds


def observation_manifest(paths, observations, population_seed, field_id):
    rows = []
    for trial, obs in zip(paths, observations, strict=True):
        path = trial["path"]
        for (family, cells), values in obs.items():
            rows.append({"population_seed": population_seed, "field_id": field_id,
                "path_id": trial["path_id"], "truth_kind": trial["truth_kind"], "gradient": trial["gradient"],
                "observation": family, "n_cells": cells,
                "path_sha256": array_sha256(path["midpoints_cm"]), "speed_sha256": array_sha256(path["speed_cm_s"]),
                "counts_sha256": array_sha256(values), "total_schedule_sha256": array_sha256(values.sum(axis=1)),
                "spikes": int(values.sum()), "active_cells": int(np.count_nonzero(values.sum(axis=0))),
                "duration_ms": len(values)})
    return pd.DataFrame(rows)


def score_batch(args, population_seed, field_id, paths, observations, evaluate, bounds):
    area, aspect, sigma = FIELDS[field_id]
    records, geometry = [], []
    for cells in [64, 128]:
        settings = [(8, 20, 5)] if args.baseline_only else decoder_settings(cells, sigma, aspect)
        for domain in ["common_core", "full_arena"]:
            for spacing, window, stride in settings:
                grid = centered_grid(CORE if domain == "common_core" else bounds, spacing)
                rates = evaluate(grid)[:, :cells].T
                truth = [resolution_truth(t["path"], window, stride) for t in paths]
                # Coverage is explicitly a synthetic field-distance summary, not a decoder QC threshold.
                geometry.append({"population_seed": population_seed, "field_id": field_id, "n_cells": cells,
                    "support_domain": domain, "grid_cm": spacing, "window_ms": window, "stride_ms": stride,
                    "n_states": len(grid), "grid_sha256": array_sha256(grid), "rates_sha256": array_sha256(rates),
                    "population_rate_hz_median": float(np.median(rates.sum(axis=0))),
                    "fraction_states_no_field_above_half_peak": float(np.mean(rates.max(axis=0) < (args.peak_hz + args.floor_hz) / 2))})
                for family, likelihood in [("native_poisson", "poisson"), ("common_count_schedule", "conditional_multinomial")]:
                    chunks = [window_counts(obs[family, cells], window, stride) for obs in observations]
                    offsets = np.r_[0, np.cumsum([len(v) for v in chunks])]
                    decoded = decode_chunked(np.concatenate(chunks), rates, grid, window, likelihood)
                    rules = ["literal_20cm_10frames"]
                    if stride != 5:
                        rules.append("time_scaled_4000cm_s_45ms")
                    for trial, counts, target, a, b, obs in zip(paths, chunks, truth, offsets[:-1], offsets[1:], observations, strict=True):
                        identity = {"population_seed": population_seed, "field_id": field_id,
                            "area_m2": area, "aspect": aspect, "sigma_cm": sigma, "n_cells": cells,
                            "support_domain": domain, "grid_cm": spacing, "window_ms": window, "stride_ms": stride,
                            "observation": family, "likelihood": likelihood,
                            "path_id": trial["path_id"], "truth_kind": trial["truth_kind"], "gradient": trial["gradient"],
                            "fine_counts_sha256": array_sha256(obs[family, cells]), "window_counts_sha256": array_sha256(counts),
                            "spikes": int(obs[family, cells].sum()), "duration_ms": args.duration_ms}
                        for estimator, support, rule in itertools.product(["map", "posterior_mean"], [False, True], rules):
                            records.append({**identity, "estimator": estimator,
                                "bin_filter": "at_least_2cells_3spikes" if support else "unfiltered", "continuity_rule": rule,
                                **measurement_metrics(decoded[estimator][a:b], counts, target, window, stride, rule, support,
                                    decoded["posterior_rms_cm"][a:b], decoded["posterior_entropy_nats"][a:b])})
    frame = pd.DataFrame(records)
    if frame.empty or frame.duplicated(KEY).any():
        raise AssertionError("empty or duplicate metrics")
    return frame, pd.DataFrame(geometry)


def summarize_batch(events):
    rows, gradients = [], []
    for key, frame in events.groupby(FACTORS + ["truth_kind", "gradient"], sort=True):
        identity = dict(zip(FACTORS + ["truth_kind", "gradient"], key, strict=True))
        eligible = frame.truth_geometric_eligible & frame.truth_kind.eq("continuous")
        row = {**identity, "events": len(frame), "truth_eligible_events": int(eligible.sum()),
               "accepted_events": int(frame.continuity_pass.sum()), "acceptance_fraction": float(frame.continuity_pass.mean()),
               "eligible_recovery_fraction": float(frame.loc[eligible, "continuity_pass"].mean()) if eligible.any() else np.nan}
        for metric in ["spikes", "valid_bins", "all_steps", "selected_steps", "large_jump_fraction",
                       "median_position_error_cm", "median_posterior_rms_cm", "median_posterior_entropy_nats",
                       "all_median_speed_cm_s", "selected_median_speed_cm_s"]:
            row[metric] = float(frame[metric].mean()) if frame[metric].notna().any() else np.nan
        rows.append(row)
        if identity["truth_kind"] == "continuous":
            for selection, axis, readout in itertools.product(["all", "selected"], ["true_coordinate", "decoded_coordinate"], ["decoded", "true_arclength", "true_chord"]):
                result = gradient_from_moments(frame, f"{selection}__{axis}__{readout}", 1000.)
                gradients.append({**identity, "selection": selection, "coordinate": axis, "readout": readout, **result})
    return pd.DataFrame(rows), pd.DataFrame(gradients)


def inputs():
    paths = {"protocol": ROOT / "docs/replay_coverage_geometry_protocol.md", "scorer": Path(__file__)}
    for name in ["replay_coverage_geometry", "replay_coverage", "replay_coverage_counterfactual", "replay_coverage_recovery", "replay_coverage_data", "replay_coverage_validation", "encoding"]:
        paths[name] = ROOT / f"src/hipporeplayimm/{name}.py"
    paths["provenance"] = ROOT / "scripts/_provenance.py"
    from ratinabox.Environment import Environment
    from ratinabox.Neurons import PlaceCells
    paths["ratinabox_environment"] = Path(inspect.getfile(Environment))
    paths["ratinabox_neurons"] = Path(inspect.getfile(PlaceCells))
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--populations", type=int, default=8)
    parser.add_argument("--paths", type=int, default=24)
    parser.add_argument("--duration-ms", type=int, default=200)
    parser.add_argument("--peak-hz", type=float, default=20.)
    parser.add_argument("--floor-hz", type=float, default=.02)
    parser.add_argument("--count-rate-hz", type=float, default=300.)
    parser.add_argument("--field-ids", nargs="+", type=int, default=list(range(12)))
    parser.add_argument("--baseline-only", action="store_true", help="smoke only; omit full resolution factorial")
    args = parser.parse_args()
    if args.populations < 1 or args.paths < 1 or len(set(args.field_ids)) != len(args.field_ids) or any(i not in range(12) for i in args.field_ids):
        parser.error("invalid population/path/field configuration")
    if args.output_dir.exists():
        parser.error("output directory already exists; never overwrite a frozen run")
    args.output_dir.mkdir(parents=True)
    start = time.monotonic()
    provenance = build_script_provenance(input_paths=inputs())
    snapshot = args.output_dir / "inputs"
    snapshot.mkdir()
    for name, path in inputs().items():
        shutil.copy2(path, snapshot / f"{name}{path.suffix}")
    manifest = {**provenance, "created_at_utc": datetime.now(UTC).isoformat(),
                "parameters": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                "versions": {name: importlib.metadata.version(name) for name in ["numpy", "scipy", "pandas", "ratinabox"]},
                "claim_boundary": "synthetic_information_and_resolution_experiment_not_biological_replication",
                "status": "running"}
    manifest_path = args.output_dir / "geometry_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    summaries, gradients, observations, geometries, batches = [], [], [], [], []
    for population in range(args.populations):
        for field_id in args.field_ids:
            tick = time.monotonic()
            paths, obs, evaluate, physical, bounds = build_batch(args, population, field_id)
            events, geometry = score_batch(args, population, field_id, paths, obs, evaluate, bounds)
            summary, gradient = summarize_batch(events)
            path = args.output_dir / f"events_p{population:02d}_f{field_id:02d}.csv.gz"
            events.to_csv(path, index=False, compression="gzip")
            summaries.append(summary)
            gradients.append(gradient)
            observations.append(observation_manifest(paths, obs, population, field_id))
            geometries.append(geometry)
            batches.append({"population_seed": population, "field_id": field_id, "file": path.name,
                "sha256": file_sha256(path), "rows": len(events), "centers_sha256": array_sha256(physical),
                "runtime_s": time.monotonic() - tick})
            print(json.dumps(batches[-1]), flush=True)
    for name, frames in [("population_summary", summaries), ("population_gradients", gradients),
                         ("observations", observations), ("decoder_grids", geometries)]:
        pd.concat(frames, ignore_index=True).to_csv(args.output_dir / f"geometry_{name}.csv", index=False)
    pd.DataFrame(batches).to_csv(args.output_dir / "geometry_batches.csv", index=False)
    unchanged = provenance["input_file_sha256"] == {name: file_sha256(path) for name, path in inputs().items()}
    gates = [{"gate": "all_batches_complete", "pass": len(batches) == args.populations * len(args.field_ids)},
             {"gate": "inputs_unchanged", "pass": unchanged}]
    gates.append({"gate": "overall_technical", "pass": all(g["pass"] for g in gates)})
    pd.DataFrame(gates).to_csv(args.output_dir / "geometry_gate_summary.csv", index=False)
    manifest.update(status="complete" if all(g["pass"] for g in gates) else "technical_fail",
        runtime_s=time.monotonic() - start, batches=len(batches), metric_rows=sum(b["rows"] for b in batches),
        output_sha256={p.name: file_sha256(p) for p in args.output_dir.iterdir() if p.is_file() and p != manifest_path})
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    if manifest["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
