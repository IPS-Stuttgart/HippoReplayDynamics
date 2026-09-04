#!/usr/bin/env python3
"""Freeze real populations and ALL high-MUA candidates for coverage calibration."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from hipporeplayimm.replay_coverage_data import (
    CoverageInputConfig,
    canonical_candidates,
    count_candidate_bins,
    fit_coverage_population,
    index_spike_times,
    load_pf_coverage_session,
    load_tanni_coverage_session,
    prepare_position_support,
)
from scripts._provenance import build_script_provenance, file_sha256


def session_records(pf_root, pf_candidates, tanni_manifest, tanni_candidates):
    pf = canonical_candidates(pf_candidates, "pfeiffer_foster")
    tanni = canonical_candidates(tanni_candidates, "tanni2022")
    if tanni_manifest.duplicated(["animal", "session"]).any():
        raise ValueError("duplicate Tanni manifest sessions")
    manifest_keys = set(zip(tanni_manifest.animal, tanni_manifest.session, strict=True))
    candidate_keys = set(zip(tanni.animal, tanni.session, strict=True))
    if not candidate_keys.issubset(manifest_keys):
        raise ValueError("candidate sessions missing from native input manifest")
    records = []
    for (animal, name), events in pf.groupby(["animal", "session"], sort=True):
        path = Path(pf_root) / name
        if name != f"{animal}/{path.name}":
            raise ValueError("PF animal/session mismatch")
        records.append(("pfeiffer_foster", animal, name, path, events.copy()))
    for row in tanni_manifest.sort_values(["animal", "session"]).itertuples(index=False):
        events = tanni[tanni.animal.eq(row.animal) & tanni.session.eq(row.session)].copy()
        if hasattr(row, "mua_candidates") and int(row.mua_candidates) != len(events):
            raise ValueError(f"candidate denominator mismatch for {row.animal}/{row.session}")
        records.append(("tanni2022", row.animal, row.session, Path(row.nwb_path), events))
    return records


def export_session(dataset, animal, name, source, events, out, config):
    start = time.monotonic()
    session, provenance = load_pf_coverage_session(source) if dataset == "pfeiffer_foster" else load_tanni_coverage_session(source)
    if session.rat != animal or session.name != name.split("/")[-1]:
        raise ValueError("raw recording identity differs from candidate identity")
    bounds = provenance["arena_bounds_cm"]
    session, position_qc = prepare_position_support(session, config.maximum_position_gap_s, bounds)
    arrays, units, metadata = fit_coverage_population(session, config)
    events = events.reset_index(drop=True).copy()
    counts, starts, durations, offsets = [], [], [], [0]
    indexed = index_spike_times(session.spikes, arrays["cell_ids"])
    clock_valid, n_spikes, n_qc_spikes, n_qc_active = [], [], [], []
    for row in events.itertuples(index=False):
        edges, local = count_candidate_bins(indexed, arrays["cell_ids"], row.start_s, row.end_s, config.base_bin_s)
        counts.append(local)
        starts.append(edges[:-1])
        durations.append(np.diff(edges))
        offsets.append(offsets[-1] + len(local))
        clock_valid.append(bool(row.start_s >= position_qc["position_start_s"] and row.end_s <= position_qc["position_end_s"]))
        n_spikes.append(int(local.sum()))
        selected = local[:, arrays["unit_qc_mask"]]
        n_qc_spikes.append(int(selected.sum()))
        n_qc_active.append(int(np.count_nonzero(selected.sum(axis=0))))
    arrays.update({
        "candidate_base_counts": np.concatenate(counts) if counts else np.empty((0, len(arrays["cell_ids"])), dtype=np.int32),
        "candidate_base_starts_s": np.concatenate(starts) if starts else np.empty(0),
        "candidate_base_durations_s": np.concatenate(durations) if durations else np.empty(0),
        "candidate_offsets": np.asarray(offsets, dtype=np.int64),
        "candidate_event_indices": events.event_index.to_numpy(np.int64),
        "candidate_start_s": events.start_s.to_numpy(float),
        "candidate_end_s": events.end_s.to_numpy(float),
        "arena_bounds_cm": np.asarray(bounds) if bounds is not None else np.full((2, 2), np.nan),
        "tracking_extent_cm": np.asarray(position_qc["tracking_extent_cm"]),
    })
    key = f"{dataset}__{animal}__{name.split('/')[-1]}"
    artifact = out / "sessions" / f"{key}.npz"
    np.savez_compressed(artifact, **arrays)
    metadata = {
        "dataset": dataset, "animal": animal, "session": name,
        **provenance, **position_qc, **metadata,
        "candidates": len(events), "candidate_clock_valid": int(sum(clock_valid)),
        "artifact_path": str(artifact), "artifact_sha256": file_sha256(artifact),
        "candidate_selection": "all_source_candidates_no_replay_content_filter",
        "cache_schema_version": 1,
    }
    (artifact.with_suffix(".json")).write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    events["cache_path"] = str(artifact)
    events["cache_event_row"] = np.arange(len(events))
    events["clock_valid"] = clock_valid
    events["n_spikes_all_sorted"] = n_spikes
    events["n_spikes_qc_units"] = n_qc_spikes
    events["n_active_qc_units"] = n_qc_active
    events["status"] = "cached"
    events["failure_reason"] = ""
    for column, value in [("dataset", dataset), ("animal", animal), ("session", name)]:
        units[column] = value
    summary = {key: value for key, value in metadata.items() if not isinstance(value, (dict, list))}
    if bounds is not None:
        summary["arena_area_m2"] = float(np.prod(np.diff(np.asarray(bounds), axis=0)) / 10000)
    summary.update(status="cached", failure_reason="", runtime_s=time.monotonic() - start)
    return events, units, summary


def run(args):
    config = CoverageInputConfig(
        bin_size_cm=args.bin_size_cm, smoothing_sigma_bins=args.smoothing_sigma_bins,
        min_occupancy_s=args.min_occupancy_s,
    )
    inputs = {
        "pf_candidates": args.pf_candidates, "tanni_manifest": args.tanni_manifest,
        "tanni_candidates": args.tanni_candidates, "script": Path(__file__),
        "input_library": ROOT / "src/hipporeplayimm/replay_coverage_data.py",
        "encoding_library": ROOT / "src/hipporeplayimm/encoding.py",
        "data_library": ROOT / "src/hipporeplayimm/data.py",
        "provenance_library": ROOT / "scripts/_provenance.py",
    }
    hashes = {key: file_sha256(path) for key, path in inputs.items()}
    records = session_records(args.pf_root, pd.read_csv(args.pf_candidates), pd.read_csv(args.tanni_manifest), pd.read_csv(args.tanni_candidates))
    if args.sessions:
        requested = set(args.sessions)
        records = [row for row in records if f"{row[0]}:{row[1]}:{row[2]}" in requested]
        if len(records) != len(requested):
            raise ValueError("requested sessions not found exactly once")
    if not records:
        raise ValueError("empty cohort")
    out = args.output_dir.resolve()
    (out / "sessions").mkdir(parents=True, exist_ok=False)
    provenance = build_script_provenance(input_paths=inputs, cwd=ROOT)
    provenance["created_at_utc"] = datetime.now(UTC).isoformat()
    event_frames, unit_frames, summaries = [], [], []
    for dataset, animal, name, path, events in records:
        print(f"cache {dataset} {animal} {name}: {len(events)} candidates", flush=True)
        try:
            local, units, summary = export_session(dataset, animal, name, path, events, out, config)
            unit_frames.append(units)
        except (OSError, ValueError, RuntimeError, KeyError, ImportError) as exc:
            local = events.copy()
            local["status"] = "failed"
            local["failure_reason"] = f"{type(exc).__name__}: {exc}"
            summary = {"dataset": dataset, "animal": animal, "session": name, "candidates": len(events), "status": "failed", "failure_reason": f"{type(exc).__name__}: {exc}"}
            print(summary["failure_reason"], flush=True)
        event_frames.append(local)
        summaries.append(summary)
    events = pd.concat(event_frames, ignore_index=True)
    units = pd.concat(unit_frames, ignore_index=True) if unit_frames else pd.DataFrame()
    sessions = pd.DataFrame(summaries)
    inputs_unchanged = all(file_sha256(path) == hashes[key] for key, path in inputs.items())
    gates = [
        ("all_candidate_rows_retained", len(events) == sum(len(row[-1]) for row in records)),
        ("all_sessions_cached", len(sessions) > 0 and sessions.status.eq("cached").all()),
        ("all_candidate_clocks_valid", len(events) > 0 and "clock_valid" in events and events.clock_valid.fillna(False).all()),
        ("at_least_five_qc_units_each_session", "encoding_cells_qc" in sessions and (sessions.encoding_cells_qc.fillna(0) >= 5).all()),
        ("source_code_and_candidate_tables_unchanged", inputs_unchanged),
    ]
    gates.append(("overall_input_readiness", all(bool(value) for _, value in gates)))
    pd.DataFrame([{"gate": name, "passed": bool(value)} for name, value in gates]).to_csv(out / "coverage_input_gate_summary.csv", index=False)
    events.to_csv(out / "coverage_input_candidates.csv", index=False)
    units.to_csv(out / "coverage_input_units.csv", index=False)
    sessions.to_csv(out / "coverage_input_sessions.csv", index=False)
    manifest = {
        **provenance, "input_sha256_at_start": hashes, "inputs_unchanged": bool(inputs_unchanged),
        "settings": asdict(config), "sessions": len(sessions), "candidates": len(events),
        "cohort_scope": "explicit_session_subset" if args.sessions else "all_input_sessions",
        "requested_sessions": args.sessions,
        "versions": {name: version(name) for name in ["numpy", "scipy", "pandas", "h5py"]},
        "claim_boundary": "input cache, not decoder validation or evidence for biological continuity/speed",
    }
    (out / "coverage_input_manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    lines = [
        "# Replay Coverage Input Audit", "", f"Sessions: {len(sessions)}. Candidate events retained: {len(events)}.",
        f"Cached sessions: {int(sessions.status.eq('cached').sum())}.",
        "", "All supplied high-MUA core windows are retained, including events with zero QC-unit spikes.",
        "No continuity filter, posterior criterion, or model evidence was used for inclusion.",
        "Unit QC uses RUN only; all sorted units and their spikes are cached for sensitivity analysis.",
        "These common-pipeline maps need held-out RUN validation before biological interpretation.",
        "", "PF physical wall coordinates are not established by this cache; observed position extrema are labeled tracking extent only.",
        "Tanni walls use native arena metadata, independently of occupancy or decoding.",
        "", "Final partial 5 ms bins are retained with explicit durations; a downstream 20 ms-window decoder must not treat partial bins as full.",
        "", "## Gates", "",
    ]
    lines.extend(f"- {name}: {'pass' if value else 'FAIL'}" for name, value in gates)
    (out / "coverage_input_report.md").write_text("\n".join(lines) + "\n")
    if not all(value for _, value in gates):
        raise RuntimeError("input readiness gates failed; see written audit tables")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pf-root", type=Path, required=True)
    parser.add_argument("--pf-candidates", type=Path, required=True)
    parser.add_argument("--tanni-manifest", type=Path, required=True)
    parser.add_argument("--tanni-candidates", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sessions", nargs="*")
    parser.add_argument("--bin-size-cm", type=float, default=8.0)
    parser.add_argument("--smoothing-sigma-bins", type=float, default=1.5)
    parser.add_argument("--min-occupancy-s", type=float, default=0.05)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
