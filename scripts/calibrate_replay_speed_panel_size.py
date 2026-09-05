#!/usr/bin/env python3
"""Frozen nested 30/100/300-candidate simulation recovery and calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from hipporeplayimm.replay_coverage_data import array_sha256
from hipporeplayimm.replay_coverage_map_mismatch import mismatch_observations
from hipporeplayimm.replay_coverage_recovery import map_interpolator, simulate_path
from hipporeplayimm.replay_speed_panel_size import decode_event_moments, prefix_panels, profile_cycle
from hipporeplayimm.replay_speed_population_transfer import evaluate_transfers, fit_transfers

from scripts._provenance import build_script_provenance, file_sha256
from scripts.calibrate_replay_speed_identifiability import (
    IDENTITY,
    TEST_CONDITIONS,
    draw_schedule,
    evaluate_panels,
    summarize_decisions,
    validate_panel_contract,
)
from scripts.simulate_replay_coverage_recovery import seed_parts
from scripts.validate_replay_speed_population_transfer import load_source


def profile_plan(profiles, maximum, draw):
    return profile_cycle(profiles, maximum, seed_parts(int(draw["draw_seed"], 16), "duration_cycles", stream=20))


def draw_observations(model, profiles, draw, maximum):
    seed = int(draw["draw_seed"], 16)
    plan = profile_plan(profiles, maximum, draw)
    conditions = TEST_CONDITIONS if draw["phase"] == "test" else [("A", "poisson")]
    evaluators = {"A": map_interpolator(model["training_full_rates_hz"], model["x_edges_cm"], model["y_edges_cm"]),
        "B": map_interpolator(model["generator_rates_hz"], model["generator_x_edges_cm"], model["generator_y_edges_cm"])}
    chunks, path_hashes = {key: [] for key in conditions}, []
    for record in plan.itertuples(index=False):
        ordinal = int(record.synthetic_ordinal)
        path = simulate_path(int(record.n_base_bins), model["domain_cm"], "continuous", float(draw["gradient"]),
            1000., seed_parts(seed, "path", ordinal, 21))
        path_hashes.append(array_sha256(path["midpoints_cm"]))
        for generator in sorted({g for g, _ in conditions}):
            observations, _ = mismatch_observations(evaluators[generator](path["midpoints_cm"]),
                seed_parts(seed, generator, ordinal, 22), 3.)
            for name, counts in observations.items():
                if (generator, name) in chunks:
                    chunks[generator, name].append(counts)
    return plan, chunks, path_hashes


def prefix_digest(values, budget):
    return hashlib.sha256("".join(values[:budget]).encode()).hexdigest()


def simulate_nested(model, profiles, draw, budgets):
    plan, chunks, path_hashes = draw_observations(model, profiles, draw, max(budgets))
    output, saved_moments, saved_diagnostics = [], [], []
    for condition_index, ((generator, observation), fine) in enumerate(chunks.items()):
        moments, diagnostics = decode_event_moments(fine, model)
        seed = int(np.random.SeedSequence(seed_parts(int(draw["draw_seed"], 16), f"{generator}:{observation}", stream=23)).generate_state(1, dtype=np.uint64)[0])
        panel = prefix_panels(moments, diagnostics, budgets, seed, draw["phase"] == "test")
        panel["generator"], panel["observation"] = generator, observation
        panel["condition_index"], panel["bootstrap_seed"] = condition_index, hex(seed)
        count_hashes = [array_sha256(counts) for counts in fine]
        for budget in budgets:
            mask = panel.candidate_budget.eq(budget)
            panel.loc[mask, "path_sha256"] = prefix_digest(path_hashes, budget)
            panel.loc[mask, "counts_sha256"] = prefix_digest(count_hashes, budget)
            panel.loc[mask, "spikes"] = sum(int(counts.sum()) for counts in fine[:budget])
        for key, value in draw.items():
            panel[key] = value
        output.append(panel)
        saved_moments.append(moments)
        saved_diagnostics.append(diagnostics)
    return pd.concat(output, ignore_index=True), np.array(saved_moments), np.array(saved_diagnostics), plan


def source_inputs(root):
    _, _, paths, meta = load_source(root)
    if meta["parameters"].get("max_profiles") is not None:
        raise ValueError("uncapped source profiles required")
    profile_path = Path(meta["input_file_paths"]["profiles"])
    if file_sha256(profile_path) != meta["input_file_sha256"]["profiles"]:
        raise ValueError("source duration profiles changed")
    paths["profiles"] = profile_path
    profiles = pd.read_csv(profile_path, usecols=IDENTITY + ["source_event_index", "n_base_bins"]).drop_duplicates()
    if profiles.duplicated(IDENTITY + ["source_event_index"]).any():
        raise ValueError("ambiguous duration profiles")
    batches = pd.read_csv(root / "speed_identifiability_batches.csv")
    for row in batches.itertuples(index=False):
        path = root / f"model__{row.label}.npz"
        if file_sha256(path) != meta["output_sha256"][path.name]:
            raise ValueError("frozen A/B model changed")
        paths[f"model__{row.label}"] = path
        choose = np.logical_and.reduce([profiles[k].eq(getattr(row, k)) for k in IDENTITY])
        if choose.sum() != row.profiles:
            raise ValueError("source duration membership/count differs")
    return batches, profiles, paths, meta


def write_csv(frame, path):
    frame.to_csv(path, index=False, compression={"method": "gzip", "mtime": 0} if path.suffix == ".gz" else None)


def score_session(payload):
    row, model_path, profiles, args = payload
    start, out = time.monotonic(), args.output_dir.resolve()
    identity = {k: row[k] for k in IDENTITY}
    key, label = ":".join(identity.values()), row["label"]
    with np.load(model_path, allow_pickle=False) as handle:
        model = dict(handle)
    schedule = draw_schedule(args.seed, key, args.fit_draws, args.calibration_draws, args.test_draws, args.fixed_draws)
    schedule.to_csv(out / f"schedule__{label}.csv", index=False)
    profiles.to_csv(out / f"profiles__{label}.csv", index=False)
    frames, all_moments, all_diagnostics, plans = [], [], [], []
    event_row = 0
    for index, draw in enumerate(schedule.to_dict("records")):
        panel, moments, diagnostics, plan = simulate_nested(model, profiles, draw, args.budgets)
        panel["event_statistics_row"] = panel.condition_index + event_row
        event_row += len(moments)
        for name, value in identity.items():
            panel[name] = value
        plan["phase"], plan["draw_id"] = draw["phase"], draw["draw_id"]
        frames.append(panel)
        all_moments.append(moments)
        all_diagnostics.append(diagnostics)
        plans.append(plan)
        if (index + 1) % 20 == 0 or index == 0:
            print(f"{label}: {index + 1}/{len(schedule)} draws at max {max(args.budgets)} candidates", flush=True)
    np.savez_compressed(out / f"events__{label}.npz", moments=np.concatenate(all_moments), diagnostics=np.concatenate(all_diagnostics))
    write_csv(pd.concat(plans, ignore_index=True), out / f"duration_plan__{label}.csv.gz")
    panels = pd.concat(frames, ignore_index=True)
    decisions, fits, summaries = [], [], []
    for budget, part in panels.groupby("candidate_budget", sort=True):
        validate_panel_contract(part, schedule, identity)
        dec, fit = evaluate_panels(part)
        dec["calibration_scope"], fit["calibration_scope"] = "within_session", "within_session"
        fit["candidate_budget"] = budget
        for name, value in identity.items():
            fit[name] = value
        summary = summarize_decisions(dec)
        summary["candidate_budget"], summary["calibration_scope"] = budget, "within_session"
        decisions.append(dec)
        fits.append(fit)
        summaries.append(summary)
    for name, frame in [("panels", panels), ("local_decisions", pd.concat(decisions, ignore_index=True)), ("local_fits", pd.concat(fits, ignore_index=True))]:
        write_csv(frame, out / f"{name}__{label}.csv.gz")
    return {**identity, "label": label, "status": "scored", "duration_profiles": len(profiles),
        "panel_rows": len(panels), "local_decision_rows": sum(map(len, decisions)), "event_statistic_rows": event_row,
        "synthetic_candidates": event_row * max(args.budgets), "runtime_s": time.monotonic() - start}, pd.concat(summaries, ignore_index=True)


def run(args):
    start = time.monotonic()
    source, profiles, paths, parent = source_inputs(args.input_dir.resolve())
    if args.sessions:
        source = source[source.apply(lambda r: ":".join(r[IDENTITY]) in args.sessions, axis=1)]
        if len(source) != len(set(args.sessions)):
            raise ValueError("requested sessions missing or duplicated")
    if source.empty or (source.groupby("dataset").animal.nunique() < 2).any():
        raise ValueError("at least two animals per included dataset required for transfer")
    paths.update({"runner": Path(__file__), "protocol": ROOT / "docs/replay_speed_panel_size_protocol.md",
        "library": ROOT / "src/hipporeplayimm/replay_speed_panel_size.py",
        "transfer_library": ROOT / "src/hipporeplayimm/replay_speed_population_transfer.py",
        "transfer_loader": ROOT / "scripts/validate_replay_speed_population_transfer.py"})
    for name, old_path in parent["input_file_paths"].items():
        if Path(old_path).suffix == ".py":
            path = Path(old_path)
            if file_sha256(path) != parent["input_file_sha256"][name]:
                raise ValueError(f"frozen source dependency changed: {name}")
            paths[f"source_code__{name}"] = path
    meta = build_script_provenance(input_paths=paths, cwd=ROOT)
    if meta["git_dirty"] and not args.allow_dirty:
        raise ValueError("commit before production; dirty runs are technical smokes")
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "inputs").mkdir()
    for name, path in paths.items():
        shutil.copy2(path, out / "inputs" / (name + path.suffix))
    tasks = []
    for row in source.to_dict("records"):
        choose = np.logical_and.reduce([profiles[k].eq(row[k]) for k in IDENTITY])
        tasks.append((row, paths[f"model__{row['label']}"], profiles[choose].sort_values("source_event_index"), args))
    batches, summaries = [], []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for future in as_completed([pool.submit(score_session, task) for task in tasks]):
            batch, summary = future.result()
            batches.append(batch)
            summaries.append(summary)
            write_csv(pd.DataFrame(batches).sort_values(IDENTITY), out / "speed_panel_size_batches.csv")
    panels = pd.concat([pd.read_csv(out / f"panels__{row['label']}.csv.gz", float_precision="round_trip") for row in batches], ignore_index=True)
    for budget, part in panels.groupby("candidate_budget", sort=True):
        fits = fit_transfers(part)
        decisions = evaluate_transfers(part, fits)
        fits["candidate_budget"] = budget
        summary = summarize_decisions(decisions)
        summary["candidate_budget"], summary["calibration_scope"] = budget, "leave_one_animal_out"
        summaries.append(summary)
        write_csv(fits, out / f"transfer_fits__{budget}.csv")
        write_csv(decisions, out / f"transfer_decisions__{budget}.csv.gz")
    write_csv(pd.concat(summaries, ignore_index=True), out / "speed_panel_size_session_summary.csv")
    per_budget = (args.fit_draws + args.calibration_draws + 4 * (args.test_draws + 5 * args.fixed_draws)) * 8
    gates = {"sessions_complete": len(batches) == len(source) and len(batches) > 0,
        "all_budget_panel_rows": all(b["panel_rows"] == per_budget * len(args.budgets) for b in batches),
        "all_local_decisions": all(b["local_decision_rows"] == (args.test_draws + 5 * args.fixed_draws) * 32 * 3 * len(args.budgets) for b in batches),
        "all_event_statistics": all(b["event_statistic_rows"] == per_budget // 8 for b in batches),
        "inputs_unchanged": all(file_sha256(path) == meta["input_file_sha256"][name] for name, path in paths.items())}
    gates["overall_technical"] = all(gates.values())
    counts = source.groupby("dataset").agg(sessions=("session", "size"), animals=("animal", "nunique")).to_dict("index")
    full = counts == {"pfeiffer_foster": {"sessions": 8, "animals": 4}, "tanni2022": {"sessions": 25, "animals": 5}}
    full &= all(getattr(args, k) == v for k, v in {"fit_draws": 40, "calibration_draws": 99, "test_draws": 100, "fixed_draws": 20, "budgets": [30, 100, 300], "seed": 20260917}.items())
    full &= not args.allow_dirty and not meta["git_dirty"] and not args.sessions
    gates["full_33_session_scope"] = full
    write_csv(pd.DataFrame([{"gate": k, "passed": bool(v)} for k, v in gates.items()]), out / "speed_panel_size_gate_summary.csv")
    meta.update(status="complete" if gates["overall_technical"] else "failed", scope="all33_nested_candidate_budgets" if full else "technical_smoke",
        parameters={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, runtime_s=time.monotonic() - start,
        sessions=len(source), animals=len(source[["dataset", "animal"]].drop_duplicates()),
        claim_boundary="known-gradient candidate-budget sensitivity; no real replay truth or biological uniformity",
        snapshot_sha256={str(p.relative_to(out)): file_sha256(p) for p in (out / "inputs").iterdir()},
        output_sha256={p.name: file_sha256(p) for p in out.iterdir() if p.is_file()})
    (out / "speed_panel_size_manifest.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({k: meta[k] for k in ["status", "scope", "sessions", "animals", "runtime_s"]}), flush=True)
    if not gates["overall_technical"]:
        raise RuntimeError("technical gates failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--budgets", type=int, nargs="+", default=[30, 100, 300])
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--fit-draws", type=int, default=40)
    parser.add_argument("--calibration-draws", type=int, default=99)
    parser.add_argument("--test-draws", type=int, default=100)
    parser.add_argument("--fixed-draws", type=int, default=20)
    parser.add_argument("--sessions", nargs="*")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    if args.budgets != sorted(set(args.budgets)) or min(args.budgets) < 5 or min(args.workers, args.fit_draws, args.calibration_draws, args.test_draws, args.fixed_draws) < 1:
        parser.error("increasing unique budgets >=5 and positive counts required")
    run(args)


if __name__ == "__main__":
    main()
