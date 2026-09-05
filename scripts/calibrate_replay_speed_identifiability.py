#!/usr/bin/env python3
"""Fit/calibrate synthetic speed intervals on A; test new A/B observations."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from hipporeplayimm.replay_coverage import continuity_metrics, decode_independent
from hipporeplayimm.replay_coverage_data import CoverageInputConfig, array_sha256, fit_coverage_population
from hipporeplayimm.replay_coverage_geometry import window_counts
from hipporeplayimm.replay_coverage_map_mismatch import mismatch_observations
from hipporeplayimm.replay_coverage_recovery import map_interpolator, simulate_path, spatial_covariate
from hipporeplayimm.replay_coverage_validation import training_session
from hipporeplayimm.replay_speed_identifiability import (
    bootstrap_slope,
    conformal_radius,
    event_slope,
    fit_inverse,
    interval_decision,
    inverse_intervals,
)
from scripts._provenance import build_script_provenance, file_sha256
from scripts.simulate_replay_coverage_recovery import seed_parts
from scripts.validate_replay_coverage_run_decoder import session_from_cache

IDENTITY = ["dataset", "animal", "session"]
READOUT = ["estimator", "bin_filter", "selection"]
PANEL_KEY = IDENTITY + ["phase", "draw_id", "generator", "observation"] + READOUT
TEST_CONDITIONS = [(a, b) for a in ["A", "B"] for b in ["poisson", "shared_gain"]]


def draw_schedule(seed, session_key, fit_draws=40, calibration_draws=99, test_draws=100, fixed_draws=20):
    if min(fit_draws, calibration_draws, test_draws, fixed_draws) < 1:
        raise ValueError("positive draw counts required")
    rows = []
    for phase, count in [("fit", fit_draws), ("calibration", calibration_draws), ("test", test_draws)]:
        for i in range(count):
            draw_seed = int(np.random.SeedSequence(seed_parts(seed, f"{session_key}:{phase}", i, 1)).generate_state(1, dtype=np.uint64)[0])
            rows.append({"phase": phase, "draw_id": i, "stratum": "uniform", "gradient":
                         float(np.random.default_rng(draw_seed).uniform(-.75, .75)), "draw_seed": hex(draw_seed)})
    for index, gradient in enumerate([-.5, -.25, 0., .25, .5]):
        for i in range(fixed_draws):
            draw = test_draws + index*fixed_draws+i
            draw_seed = int(np.random.SeedSequence(seed_parts(seed, f"{session_key}:test", draw, 1)).generate_state(1, dtype=np.uint64)[0])
            rows.append({"phase": "test", "draw_id": draw, "stratum": f"fixed_{gradient:+.2f}",
                         "gradient": gradient, "draw_seed": hex(draw_seed)})
    return pd.DataFrame(rows)


def training_generator(full, model, config):
    lo, hi = full.run_times[:, 0].min(), full.run_times[:, 1].max()
    data = training_session(full, (lo+hi)/2, hi, 1., config.maximum_position_gap_s)
    arrays, _, _ = fit_coverage_population(data, config)
    cells = np.array([np.flatnonzero(arrays["cell_ids"] == c).item() for c in model["cell_ids"]])
    rates = arrays["rates_hz"][cells]
    np.testing.assert_array_equal(arrays["x_edges_cm"], model["x_edges_cm"])
    np.testing.assert_array_equal(arrays["y_edges_cm"], model["y_edges_cm"])
    np.testing.assert_array_equal(rates[:, model["valid_spatial_bins"]], model["rates_hz"])
    return rates


def decode_panel(fine_chunks, model, seed, bootstrap):
    chunks = [window_counts(fine, 20, 5) for fine in fine_chunks]
    offsets = np.r_[0, np.cumsum([len(c) for c in chunks])]
    counts = np.concatenate(chunks)
    decoded = decode_independent(counts, model["rates_hz"]*3., model["grid_cm"], .02)
    del decoded["posterior"]
    output = []
    for estimator in ["map", "posterior_mean"]:
        for support in [False, True]:
            moments = {"all": [], "selected": []}
            valid_windows, total_windows, continuous_events = 0, 0, 0
            for lo, hi, count in zip(offsets[:-1], offsets[1:], chunks, strict=True):
                points = decoded[estimator][lo:hi]
                valid = ((count.sum(axis=1) >= 3) & ((count > 0).sum(axis=1) >= 2)) if support else np.ones(len(count), bool)
                valid_windows += int(valid.sum())
                total_windows += len(valid)
                core = continuity_metrics(points, valid_bins=valid)
                continuous_events += int(core["continuity_pass"])
                idx = np.arange(0, len(points), 4)
                keep = np.array([valid[a:b+1].all() for a, b in pairwise(idx)], bool)
                selected = keep & (idx[:-1] >= core["continuous_start"]) & (idx[1:] < core["continuous_end_exclusive"]) & core["continuity_pass"]
                speed = np.linalg.norm(np.diff(points[idx], axis=0), axis=1) / .02 / 1000.
                q = spatial_covariate((points[idx[:-1]]+points[idx[1:]])/2, model["domain_cm"])
                for name, mask in [("all", keep), ("selected", selected)]:
                    x, y = q[mask], speed[mask]
                    moments[name].append([float(v.mean()) for v in [x, y, x*x, x*y]] if len(x) else [np.nan]*4)
            for selection, values in moments.items():
                key = f"{estimator}:{support}:{selection}"
                if bootstrap:
                    statistic, lower, upper = bootstrap_slope(values, seed_parts(seed, key, stream=9))
                else:
                    statistic, lower, upper = event_slope(values), np.nan, np.nan
                output.append({"estimator": estimator, "bin_filter": "at_least_2cells_3spikes" if support else "unfiltered",
                    "selection": selection, "statistic": statistic, "naive_lower": lower, "naive_upper": upper,
                    "contributing_events": int(np.isfinite(values).all(axis=1).sum()), "source_events": len(chunks),
                    "continuous_events": continuous_events, "valid_windows": valid_windows, "total_windows": total_windows})
    return pd.DataFrame(output)


def simulate_panel(model, profiles, draw):
    seed = int(draw["draw_seed"], 16)
    evaluators = {"A": map_interpolator(model["training_full_rates_hz"], model["x_edges_cm"], model["y_edges_cm"]),
                  "B": map_interpolator(model["generator_rates_hz"], model["generator_x_edges_cm"], model["generator_y_edges_cm"])}
    conditions = TEST_CONDITIONS if draw["phase"] == "test" else [("A", "poisson")]
    chunks = {key: [] for key in conditions}
    path_digest = hashlib.sha256()
    for profile in profiles.itertuples(index=False):
        event = int(profile.source_event_index)
        path = simulate_path(int(profile.n_base_bins), model["domain_cm"], "continuous", float(draw["gradient"]),
                             1000., seed_parts(seed, "path", event, 2))
        path_digest.update(array_sha256(path["midpoints_cm"]).encode())
        for generator in sorted({g for g, _ in conditions}):
            observations, _ = mismatch_observations(evaluators[generator](path["midpoints_cm"]),
                                                    seed_parts(seed, generator, event, 3), 3.)
            for name, counts in observations.items():
                if (generator, name) in chunks:
                    chunks[generator, name].append(counts)
    frames = []
    for (generator, observation), values in chunks.items():
        decoder_seed = int(np.random.SeedSequence(seed_parts(seed, f"{generator}:{observation}", stream=4)).generate_state(1, dtype=np.uint64)[0])
        frame = decode_panel(values, model, decoder_seed, draw["phase"] == "test")
        frame["generator"], frame["observation"] = generator, observation
        frame["path_sha256"] = path_digest.hexdigest()
        frame["counts_sha256"] = hashlib.sha256("".join(array_sha256(c) for c in values).encode()).hexdigest()
        frame["spikes"] = sum(int(c.sum()) for c in values)
        for key, value in draw.items():
            frame[key] = value
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def evaluate_panels(panels):
    decisions, fitted = [], []
    for key, data in panels.groupby(READOUT, sort=True):
        training = data[data.phase.eq("fit")]
        calibration = data[data.phase.eq("calibration")]
        if not training.generator.eq("A").all() or not calibration.generator.eq("A").all() or not training.observation.eq("poisson").all() or not calibration.observation.eq("poisson").all():
            raise ValueError("test generator leaked into inverse fitting")
        model = fit_inverse(training.statistic, training.gradient)
        radius = conformal_radius(model, calibration.statistic, calibration.gradient)
        fitted.append({**dict(zip(READOUT, key, strict=True)), **model, "calibration_radius": radius,
                       "n_calibration": len(calibration), "finite_calibration": int(np.isfinite(calibration.statistic).sum())})
        for row in data[data.phase.eq("test")].to_dict("records"):
            intervals = {"raw_bootstrap": (row["statistic"], row["naive_lower"], row["naive_upper"]),
                         **inverse_intervals(model, radius, row["statistic"])}
            for method, (point, lower, upper) in intervals.items():
                record = {**row, "method": method, "estimate": point, "lower": lower, "upper": upper}
                for bound in [.10, .25, .50]:
                    stats = interval_decision(lower, upper, row["gradient"], bound)
                    if bound == .25:
                        record.update(stats)
                    record[f"equivalence_{bound:.2f}"] = stats["equivalence_claim"]
                    record[f"inside_{bound:.2f}"] = stats["truth_inside_equivalence"]
                decisions.append(record)
    return pd.DataFrame(decisions), pd.DataFrame(fitted)


def validate_panel_contract(panels, schedule, identity):
    expected = []
    for draw in schedule.to_dict("records"):
        conditions = TEST_CONDITIONS if draw["phase"] == "test" else [("A", "poisson")]
        for generator, observation in conditions:
            for estimator in ["map", "posterior_mean"]:
                for support in ["unfiltered", "at_least_2cells_3spikes"]:
                    for selection in ["all", "selected"]:
                        expected.append({**identity, **draw, "generator": generator, "observation": observation,
                                         "estimator": estimator, "bin_filter": support, "selection": selection})
    expected = pd.DataFrame(expected)
    if panels.duplicated(PANEL_KEY).any() or len(panels) != len(expected):
        raise ValueError("missing or duplicate panel keys")
    joined = expected.merge(panels[PANEL_KEY+["gradient", "draw_seed", "stratum"]], on=PANEL_KEY,
                            how="outer", validate="one_to_one", indicator=True, suffixes=("_expected", "_actual"))
    if not joined._merge.eq("both").all():
        raise ValueError("unexpected panel keys")
    for column in ["gradient", "draw_seed", "stratum"]:
        if not joined[f"{column}_expected"].eq(joined[f"{column}_actual"]).all():
            raise ValueError(f"panel schedule mismatch: {column}")


def summarize_decisions(frame):
    rows = []
    for key, part in frame.groupby(IDENTITY+READOUT+["generator", "observation", "stratum", "method"], sort=True):
        row = dict(zip(IDENTITY+READOUT+["generator", "observation", "stratum", "method"], key, strict=True))
        finite = part[part.finite_interval]
        row.update(panels=len(part), finite_panels=len(finite), coverage=float(part.covered.mean()),
            finite_fraction=float(part.finite_interval.mean()), finite_coverage=float(finite.covered.mean()) if len(finite) else np.nan,
            median_finite_width=float(finite.interval_width.median()) if len(finite) else np.nan,
            nonzero_fraction=float(part.nonzero_claim.mean()), equivalence_fraction=float(part.equivalence_claim.mean()))
        for bound in [.10, .25, .50]:
            name = f"{bound:.2f}"
            inside = part[part[f"inside_{name}"]]
            outside = part[~part[f"inside_{name}"]]
            row[f"inside_panels_{name}"] = len(inside)
            row[f"outside_panels_{name}"] = len(outside)
            row[f"true_equivalence_fraction_{name}"] = float(inside[f"equivalence_{name}"].mean()) if len(inside) else np.nan
            row[f"false_equivalence_fraction_{name}"] = float(outside[f"equivalence_{name}"].mean()) if len(outside) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def score_session(payload):
    record, args, config, sessions, units, profiles = payload
    out = args.output_dir.resolve()
    start = time.monotonic()
    identity = {k: record[k] for k in IDENTITY}
    key = ":".join(identity.values())
    label = key.replace(":", "__").replace("/", "_")
    def choose(frame):
        return np.logical_and.reduce([frame[k].eq(v) for k, v in identity.items()])
    source = next(sessions[choose(sessions)].itertuples(index=False))
    if file_sha256(source.artifact_path) != source.artifact_sha256 or file_sha256(record["model_path"]) != record["model_sha256"]:
        raise ValueError("frozen model/cache changed")
    input_hashes = {source.artifact_path: source.artifact_sha256, record["model_path"]: record["model_sha256"]}
    full = session_from_cache(source, units[choose(units)])
    with np.load(record["model_path"], allow_pickle=False) as f:
        model = dict(f)
    model["training_full_rates_hz"] = training_generator(full, model, config)
    np.savez_compressed(out/f"model__{label}.npz", **model)
    local = profiles[choose(profiles)].sort_values("source_event_index")
    if args.max_profiles:
        local = local.head(args.max_profiles)
    schedule = draw_schedule(args.seed, key, args.fit_draws, args.calibration_draws, args.test_draws, args.fixed_draws)
    schedule.to_csv(out/f"schedule__{label}.csv", index=False)
    frames = []
    print(f"{label}: {len(local)} profiles, {len(schedule)} draws", flush=True)
    for index, draw in enumerate(schedule.to_dict("records")):
        panel = simulate_panel(model, local, draw)
        for k, v in identity.items():
            panel[k] = v
        frames.append(panel)
        if (index+1) % 20 == 0:
            print(f"{label}: {index+1}/{len(schedule)} draws", flush=True)
    panels = pd.concat(frames, ignore_index=True)
    validate_panel_contract(panels, schedule, identity)
    decisions, fitted = evaluate_panels(panels)
    for name, frame in [("panels", panels), ("decisions", decisions), ("fits", fitted)]:
        frame.to_csv(out/f"{name}__{label}.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    batch = {**identity, "label": label, "status": "scored", "profiles": len(local), "draws": len(schedule),
             "panel_rows": len(panels), "decision_rows": len(decisions), "training_cells": len(model["cell_ids"]),
             "runtime_s": time.monotonic()-start}
    return batch, summarize_decisions(decisions), input_hashes


def run(args):
    start = time.monotonic()
    started_at = datetime.now(UTC).isoformat()
    reference, out = args.reference_dir.resolve(), args.output_dir.resolve()
    files = {"reference_manifest": reference/"coverage_map_mismatch_manifest.json", "reference_batches": reference/"coverage_map_mismatch_batches.csv",
        "script": Path(__file__), "protocol": ROOT/"docs/replay_speed_identifiability_protocol.md"}
    parent = json.loads(files["reference_manifest"].read_text())
    if parent["status"] != "complete":
        raise ValueError("complete reference benchmark required")
    files.update({k: Path(parent["input_file_paths"][k]) for k in ["sessions", "source_units", "profiles"]})
    for name in ["replay_speed_identifiability", "replay_coverage", "replay_coverage_data", "replay_coverage_geometry",
                 "replay_coverage_map_mismatch", "replay_coverage_recovery", "replay_coverage_validation", "encoding"]:
        files[name] = ROOT/"src/hipporeplayimm"/f"{name}.py"
    files.update({name: ROOT/"scripts"/f"{name}.py" for name in ["_provenance", "simulate_replay_coverage_recovery", "validate_replay_coverage_run_decoder"]})
    provenance = build_script_provenance(input_paths=files, cwd=ROOT)
    if provenance["git_dirty"] and not args.allow_dirty:
        raise ValueError("commit before production, or mark technical smoke --allow-dirty")
    out.mkdir(parents=True, exist_ok=False)
    (out/"inputs").mkdir()
    for name, path in files.items():
        shutil.copy2(path, out/"inputs"/f"{name}{path.suffix}")
    config = CoverageInputConfig(**parent["encoding_settings"])
    planned = pd.read_csv(files["reference_batches"])
    planned = planned[planned.direction.eq(0)]
    if args.sessions:
        planned = planned[planned.apply(lambda r: f"{r.dataset}:{r.animal}:{r.session}" in args.sessions, axis=1)]
        if len(planned) != len(set(args.sessions)):
            raise ValueError("session selection mismatch")
    if planned.empty or planned.duplicated(IDENTITY).any() or not planned.status.eq("scored").all():
        raise ValueError("complete available first-half maps required")
    profiles = pd.read_csv(files["profiles"], usecols=IDENTITY+["source_event_index", "n_base_bins"]).drop_duplicates()
    if profiles.duplicated(IDENTITY+["source_event_index"]).any():
        raise ValueError("inconsistent source duration profiles")
    sessions = pd.read_csv(files["sessions"])
    units = pd.read_csv(files["source_units"], dtype={"source_cell_type_allowed": "boolean"})
    batches, summaries, input_hashes = [], [], {}
    tasks = [(row, args, config, sessions, units, profiles) for row in planned.to_dict("records")]
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for future in as_completed([pool.submit(score_session, task) for task in tasks]):
            batch, summary, hashes = future.result()
            batches.append(batch)
            summaries.append(summary)
            input_hashes.update(hashes)
            pd.DataFrame(batches).sort_values(IDENTITY).to_csv(out/"speed_identifiability_batches.csv", index=False)
    pd.concat(summaries, ignore_index=True).sort_values(IDENTITY+READOUT+["generator", "observation", "stratum", "method"]).to_csv(out/"speed_identifiability_session_summary.csv", index=False)
    unchanged = all(file_sha256(path) == provenance["input_file_sha256"][key] for key, path in files.items()) and all(file_sha256(path) == sha for path, sha in input_hashes.items())
    expected_panels = (args.fit_draws+args.calibration_draws)*8+(args.test_draws+5*args.fixed_draws)*32
    expected_decisions = (args.test_draws+5*args.fixed_draws)*32*3
    gates = {"sessions_complete": len(batches) == len(planned) and len(batches) > 0,
             "panel_rows_complete": all(b["panel_rows"] == expected_panels for b in batches),
             "decision_rows_complete": all(b["decision_rows"] == expected_decisions for b in batches), "inputs_unchanged": unchanged}
    gates["overall_technical"] = all(gates.values())
    pd.DataFrame([{"gate": k, "passed": v} for k, v in gates.items()]).to_csv(out/"speed_identifiability_gate_summary.csv", index=False)
    manifest = {**provenance, "parameters": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "model_cache_sha256": input_hashes, "runtime_s": time.monotonic()-start,
        "started_at_utc": started_at, "created_at_utc": datetime.now(UTC).isoformat(),
        "runtime": {"python": sys.version, "libraries": {name: importlib.metadata.version(name) for name in ["numpy", "scipy", "pandas"]},
                    "thread_limits": {name: os.environ.get(name) for name in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"]}},
        "status": "complete" if gates["overall_technical"] else "failed", "sessions": len(batches),
        "claim_boundary": "surrogate calibration/transfer only; no biological equivalence authorized",
        "output_sha256": {str(p.relative_to(out)): file_sha256(p) for p in out.rglob("*") if p.is_file() and "inputs" not in p.relative_to(out).parts}}
    (out/"speed_identifiability_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    if not gates["overall_technical"]:
        raise RuntimeError("technical gates failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--fit-draws", type=int, default=40)
    parser.add_argument("--calibration-draws", type=int, default=99)
    parser.add_argument("--test-draws", type=int, default=100)
    parser.add_argument("--fixed-draws", type=int, default=20)
    parser.add_argument("--sessions", nargs="*")
    parser.add_argument("--max-profiles", type=int, help="technical smoke only")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("positive worker count required")
    if min(args.fit_draws, args.calibration_draws, args.test_draws, args.fixed_draws) < 1:
        parser.error("positive draw counts required")
    if args.max_profiles is not None and args.max_profiles < 5:
        parser.error("at least five profiles required")
    run(args)


if __name__ == "__main__":
    main()
