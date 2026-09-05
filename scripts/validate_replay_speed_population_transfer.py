#!/usr/bin/env python3
"""Evaluate animal-excluded speed calibration on existing frozen simulation panels."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
from hipporeplayimm.replay_speed_population_transfer import (
    IDENTITY,
    PANEL_KEY,
    evaluate_transfers,
    fit_transfers,
    paired_comparison,
    summarize_animals,
)

from scripts._provenance import build_script_provenance, file_sha256
from scripts.calibrate_replay_speed_identifiability import summarize_decisions, validate_panel_contract


def load_source(root):
    paths = {"source_manifest": root / "speed_identifiability_manifest.json",
        "source_audit": root / "speed_identifiability_reconstruction_audit.json"}
    meta, audit = [json.loads(paths[k].read_text()) for k in ["source_manifest", "source_audit"]]
    if meta["status"] != "complete" or audit["status"] != "pass" or audit["input_file_sha256"]["scoring_manifest"] != file_sha256(paths["source_manifest"]):
        raise ValueError("complete source with linked successful reconstruction required")
    for key, path in {"replay_speed_identifiability": ROOT / "src/hipporeplayimm/replay_speed_identifiability.py",
        "script": ROOT / "scripts/calibrate_replay_speed_identifiability.py"}.items():
        if file_sha256(path) != meta["input_file_sha256"][key]:
            raise ValueError("frozen calibration/summary code differs from source experiment")
    def read(name):
        path = root / name
        if file_sha256(path) != meta["output_sha256"][name]:
            raise ValueError(f"source output hash changed: {name}")
        paths[name] = path
        return pd.read_csv(path, float_precision="round_trip")
    batches = read("speed_identifiability_batches.csv")
    reference = read("speed_identifiability_session_summary.csv")
    if batches.empty or batches.duplicated(IDENTITY).any() or not batches.status.eq("scored").all() or len(batches) != meta["sessions"]:
        raise ValueError("complete unique source sessions required")
    frames = []
    for row in batches.to_dict("records"):
        panels = read(f"panels__{row['label']}.csv.gz")
        schedule = read(f"schedule__{row['label']}.csv")
        validate_panel_contract(panels, schedule, {k: row[k] for k in IDENTITY})
        if len(panels) != row["panel_rows"]:
            raise ValueError("panel count differs from source batch")
        frames.append(panels)
    panels = pd.concat(frames, ignore_index=True).sort_values(PANEL_KEY).reset_index(drop=True)
    if set(map(tuple, reference[IDENTITY].drop_duplicates().to_numpy())) != set(map(tuple, batches[IDENTITY].to_numpy())):
        raise ValueError("reference session membership mismatch")
    return panels, reference, paths, meta


def run(args):
    started = time.monotonic()
    panels, reference, paths, source = load_source(args.input_dir.resolve())
    paths.update({"runner": Path(__file__), "library": ROOT / "src/hipporeplayimm/replay_speed_population_transfer.py",
        "frozen_calibration": ROOT / "src/hipporeplayimm/replay_speed_identifiability.py",
        "frozen_summary": ROOT / "scripts/calibrate_replay_speed_identifiability.py",
        "provenance": ROOT / "scripts/_provenance.py", "protocol": ROOT / "docs/replay_speed_population_transfer_protocol.md"})
    meta = build_script_provenance(input_paths=paths, cwd=ROOT)
    if meta["git_dirty"] and not args.allow_dirty:
        raise ValueError("commit before production; --allow-dirty is technical smoke only")
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "inputs").mkdir()
    snapshots = {}
    for key, path in paths.items():
        target = out / "inputs" / (key + path.suffix)
        shutil.copy2(path, target)
        snapshots[str(target.relative_to(out))] = file_sha256(target)
    fits = fit_transfers(panels)
    decisions = evaluate_transfers(panels, fits)
    transfer = summarize_decisions(decisions)
    transfer["calibration_scope"] = "leave_one_animal_out"
    reference["calibration_scope"] = "within_session"
    sessions = pd.concat([transfer, reference], ignore_index=True)
    animal, summary = summarize_animals(sessions, args.seed, args.bootstraps)
    paired = paired_comparison(sessions, args.seed, args.bootstraps)
    outputs = {"fits": fits, "decisions": decisions, "session_summary": sessions,
        "animal_summary": animal, "summary": summary, "paired_comparison": paired}
    for name, frame in outputs.items():
        suffix = ".csv.gz" if name == "decisions" else ".csv"
        frame.to_csv(out / f"speed_population_transfer_{name}{suffix}", index=False,
            compression={"method": "gzip", "mtime": 0} if name == "decisions" else None)
    actual = panels[IDENTITY].drop_duplicates()
    technical = {"source_sessions_complete": len(actual) == source["sessions"] and len(actual) > 0,
        "frozen_baseline_code_matches_source": True,
        "excluded_animals_absent_from_fit": all(r.heldout_animal not in json.loads(r.source_animals) for r in fits.itertuples()),
        "all_readout_fits_present": len(fits) == 8 * len(actual[["dataset", "animal"]].drop_duplicates()),
        "all_test_method_rows_present": len(decisions) == 2 * panels.phase.eq("test").sum(),
        "reference_cohorts_match": len(paired) > 0,
        "inputs_unchanged": all(file_sha256(path) == meta["input_file_sha256"][key] for key, path in paths.items())}
    technical["overall_technical"] = all(technical.values())
    counts = actual.groupby("dataset").agg(sessions=("session", "size"), animals=("animal", "nunique")).to_dict("index")
    full = counts == {"pfeiffer_foster": {"sessions": 8, "animals": 4}, "tanni2022": {"sessions": 25, "animals": 5}} and source["parameters"].get("max_profiles") is None
    full &= all(source["parameters"].get(k) == v for k, v in {"fit_draws": 40, "calibration_draws": 99, "test_draws": 100, "fixed_draws": 20}.items())
    full &= not meta["git_dirty"] and not args.allow_dirty
    gates = {**technical, "full_33_session_scope": full}
    pd.DataFrame([{"gate": name, "passed": bool(value)} for name, value in gates.items()]).to_csv(out / "speed_population_transfer_gate_summary.csv", index=False)
    meta.update(status="complete" if technical["overall_technical"] else "failed", sessions=len(actual), animals=len(actual[["dataset", "animal"]].drop_duplicates()),
        fits=len(fits), decisions=len(decisions), runtime_s=time.monotonic() - started,
        seed=args.seed, bootstraps=args.bootstraps, scope="all33_retrospective_transfer" if full else "technical_smoke",
        snapshot_sha256=snapshots, claim_boundary="pooled inverse calibration transfer to excluded animals' simulated panels; no new-animal conformal guarantee or biological uniformity",
        output_sha256={p.name: file_sha256(p) for p in out.glob("speed_population_transfer_*.csv*")})
    (out / "speed_population_transfer_manifest.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({k: meta[k] for k in ["status", "scope", "sessions", "animals", "fits", "decisions", "runtime_s"]}), flush=True)
    if not technical["overall_technical"]:
        raise RuntimeError("technical gates failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--bootstraps", type=int, default=5000)
    parser.add_argument("--allow-dirty", action="store_true")
    opts = parser.parse_args()
    if opts.bootstraps < 1:
        parser.error("positive bootstrap count required")
    run(opts)
