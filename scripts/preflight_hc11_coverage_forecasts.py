#!/usr/bin/env python3
"""Freeze native hc-11 POST-NREM inputs; do not classify or score events."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import _hc11_native_encoding as native
from _provenance import build_script_provenance, file_sha256
from audit_hc11_count_conditioned_prediction import ENCODING_PARAMETERS

from hipporeplayimm.independent_rejected_forecast import (
    detect_candidates,
    event_counts,
    full_counts,
    partitions,
    seed,
)

SESSIONS = (
    "Achilles_10252013",
    "Achilles_11012013",
    "Buddy_06272013",
    "Cicero_09012014",
    "Cicero_09102014",
    "Cicero_09172014",
    "Gatsby_08022013",
    "Gatsby_08282013",
)
CATALOG_COLUMNS = (
    "event_id",
    "start_s",
    "end_s",
    "peak_s",
    "duration_s",
    "detector_spikes",
    "detector_active_cells",
    "detector_peak_z",
    "detection_domain",
    "selected",
    "overlaps_previous_cohort",
)


def intersect_nrem(nrem, post, guard_s=0.05):
    arrays = []
    for name, value in (("NREM", nrem), ("POST", post)):
        x = np.asarray(value, float).reshape(-1, 2)
        if not len(x) or not np.isfinite(x).all() or (x[:, 1] <= x[:, 0]).any():
            raise ValueError(f"invalid {name} intervals")
        x = x[np.argsort(x[:, 0], kind="stable")]
        if (x[1:, 0] < x[:-1, 1]).any():
            raise ValueError(f"overlapping {name} intervals")
        arrays.append(x)
    if not np.isfinite(guard_s) or guard_s < 0:
        raise ValueError("nonnegative finite guard required")
    result = []
    for a, b in arrays[0]:
        for c, d in arrays[1]:
            lo, hi = max(a, c) + guard_s, min(b, d) - guard_s
            if hi > lo:
                result.append((lo, hi))
    return np.asarray(result, float).reshape(-1, 2)


def freeze_selection(candidates, session, previous, cap=200):
    if cap < 5:
        raise ValueError("at least five event slots required")
    selected = np.arange(len(candidates))
    if len(selected) > cap:
        selected = np.sort(np.random.default_rng(seed(20260919, session, "selection")).choice(selected, cap, replace=False))
    selected = set(selected.tolist())
    rows = []
    for i, event in enumerate(candidates):
        row = {k: v for k, v in event.items() if k != "mean_speed_cm_s"}
        row.update(detection_domain="native_POST_NREM", selected=i in selected)
        row["overlaps_previous_cohort"] = any(row["start_s"] < b and row["end_s"] > a for a, b in previous)
        rows.append(row)
    table = pd.DataFrame(rows, columns=CATALOG_COLUMNS)
    for column in ("selected", "overlaps_previous_cohort"):
        table[column] = table[column].astype(bool)
    return table


def prepare_session(folder, destination, previous):
    session, animal = folder.name, folder.parent.name
    destination.mkdir(exist_ok=False)
    result = {"dataset": "hc11", "animal": animal, "session": session, "status": "running"}
    started = time.monotonic()
    sources = [
        folder / (session + suffix)
        for suffix in (
            ".position.behavior.mat",
            ".spikes.cellinfo.mat",
            ".SleepState.states.mat",
        )
    ]
    hashes = {}
    try:
        hashes = {str(p): file_sha256(p) for p in sources}
        track = native.load_track_samples(folder)
        spikes = native.load_spikes(folder)
        maps, unit_qc = native.build_session_encodings(track, spikes, **ENCODING_PARAMETERS)
        unit_qc.to_csv(destination / "unit_qc.csv", index=False)
        encoding = maps["pooled"][0]
        units = np.asarray(encoding.unit_ids, int)
        result.update(n_encoding_units=len(units), topology=track.topology, track_length_cm=track.track_length_cm)
        detector, population, splits = partitions(len(units), ("hc11", animal, session))
        state = native.mat_struct(sources[2], "SleepState")
        intervals = intersect_nrem(state.ints.NREMstate, track.post_epoch)
        if not len(intervals):
            raise ValueError("no guarded POST-NREM intervals")
        start, end = float(track.post_epoch[:, 0].min()), float(track.post_epoch[:, 1].max())
        clock = start + (np.arange(int(np.floor((end - start) / 0.001))) + 0.5) * 0.001
        # The existing detector uses finite speed <5 as an eligibility mask.
        # Here that mask denotes native NREM, NOT measured animal speed.
        domain_mask = np.full(len(clock), np.nan)
        for lo, hi in intervals:
            a, b = np.searchsorted(clock, [lo, hi], side="left")
            domain_mask[a:b] = 0
        detector_spikes = np.concatenate([np.column_stack((spikes.times_by_unit[int(uid)], np.full(len(spikes.times_by_unit[int(uid)]), uid))) for uid in units[detector]])
        candidates, diagnostics = detect_candidates(detector_spikes, units[detector], clock, domain_mask)
        diagnostics["nrem_bins"] = diagnostics.pop("immobile_bins")
        catalog = freeze_selection(candidates, session, previous)
        catalog.to_csv(destination / "catalog.csv", index=False)
        selected = catalog[catalog.selected].copy()
        selected.to_csv(destination / "selection.csv", index=False)
        cache = {
            "detector_unit_ids": units[detector],
            "unit_ids": units[population],
            "rates": encoding.rates_hz[population],
            "centers": encoding.bin_centers_cm,
            "topology": np.array(track.topology),
            "track_length_cm": np.array(track.track_length_cm),
            "nrem_intervals": intervals,
            "post_intervals": track.post_epoch,
        }
        for split, (train, half, held) in enumerate(splits):
            for label, ids in (("train", train), ("half", half), ("held", held)):
                cache[f"{label}_{split}"] = ids
        for event in selected.itertuples(index=False):
            base, durations = event_counts(spikes.times_by_unit, units[population], event.start_s, event.end_s)
            counts, discarded = full_counts(base, durations)
            cache[f"base_{event.event_id}"] = base
            cache[f"durations_{event.event_id}"] = durations
            cache[f"counts_{event.event_id}"] = counts
            cache[f"discarded_{event.event_id}"] = np.array(discarded)
        np.savez_compressed(destination / "cache.npz", **cache)
        result.update(
            status="eligible" if len(selected) >= 5 else "insufficient_candidates",
            n_detector_units=len(detector),
            n_prediction_units=len(population),
            n_candidates=len(catalog),
            n_selected=len(selected),
            n_selected_overlapping_previous=int(selected.overlaps_previous_cohort.sum()),
            post_nrem_duration_s=float(np.diff(intervals, axis=1).sum()),
            min_inference_units=min(len(s[0]) for s in splits),
            min_half_inference_units=min(len(s[1]) for s in splits),
            min_evaluation_units=min(len(s[2]) for s in splits),
            detection_diagnostics=diagnostics,
        )
    except (ValueError, OSError, KeyError, TypeError, AttributeError) as exc:
        result.update(status="failed", error=repr(exc))
    finally:
        unchanged = bool(hashes) and all(p.exists() and file_sha256(p) == hashes.get(str(p)) for p in sources)
        if not unchanged:
            result.update(status="failed", provenance_error="source missing or changed")
        result.update(raw_input_sha256=hashes, inputs_unchanged=unchanged, runtime_s=time.monotonic() - started)
        result["output_sha256"] = {p.name: file_sha256(p) for p in destination.iterdir() if p.is_file() and p.name != "session_manifest.json"}
        (destination / "session_manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--previous-selection", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise ValueError("clean source commit required")
    output = args.output_dir.resolve()
    if output.exists():
        raise ValueError("refusing to overwrite an existing bank")
    old = pd.read_csv(args.previous_selection)
    required = {"session", "start_time_s", "end_time_s"}
    if not required.issubset(old) or not set(SESSIONS).issubset(set(old.session)):
        raise ValueError("incomplete previous selection provenance")
    if not np.isfinite(old[["start_time_s", "end_time_s"]]).all().all() or (old.end_time_s <= old.start_time_s).any():
        raise ValueError("invalid previous event times")
    sources = {}
    for session in SESSIONS:
        matches = list(args.dataset_root.glob(f"*/{session}"))
        if len(matches) != 1:
            raise ValueError(f"missing/ambiguous recording: {session}")
        sources[session] = matches[0]
    output.mkdir(parents=True, exist_ok=False)
    protocol = ROOT / "docs/hc11_coverage_forecast_protocol.md"
    manifest = build_script_provenance(input_paths={"previous_selection": args.previous_selection, "protocol": protocol})
    manifest.update(
        status="running",
        created_at_utc=datetime.now(UTC).isoformat(),
        sessions=[],
        selection_seed=20260919,
        max_events_per_session=200,
        encoding_parameters=ENCODING_PARAMETERS,
        experiment="hc11_coverage_forecast_preflight_only",
        dataset_root=str(args.dataset_root.resolve()),
    )
    path = output / "preflight_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    for session, folder in sources.items():
        previous = old[old.session.eq(session)][["start_time_s", "end_time_s"]].to_numpy(float)
        record = prepare_session(folder, output / session, previous)
        manifest["sessions"].append(record)
        path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps({k: v for k, v in record.items() if not isinstance(v, dict)}), flush=True)
    table = pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, dict)} for r in manifest["sessions"]])
    table.to_csv(output / "preflight_sessions.csv", index=False)
    eligible = table[table.status.eq("eligible")]
    gates = pd.DataFrame(
        [
            {"gate": "all_recordings_attempted", "passed": set(table.session) == set(SESSIONS)},
            {"gate": "input_hashes_unchanged", "passed": bool(len(table)) and table.inputs_unchanged.all()},
            {"gate": "all_eight_sessions_eligible", "passed": len(eligible) == 8},
            {"gate": "all_four_animals_eligible", "passed": eligible.animal.nunique() == 4},
        ]
    )
    gates.to_csv(output / "preflight_gate_summary.csv", index=False)
    manifest.update(status="complete", all_sessions_eligible=bool(len(eligible) == 8), completed_at_utc=datetime.now(UTC).isoformat())
    manifest["output_sha256"] = {p.name: file_sha256(p) for p in output.glob("*.csv")}
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "preflight_summary.md").write_text(
        "# hc-11 coverage preflight\n\n"
        f"Eligible sessions: {len(eligible)}/8; animals: {eligible.animal.nunique()}/4.\n\n"
        "Native POST-NREM bursts detected with reserved neurons. No trajectory "
        "classification or predictive scoring has occurred. Feasibility is not "
        "scientific replication; all exclusions remain in preflight_sessions.csv.\n"
    )
    return 0 if gates.passed.all() else 2


if __name__ == "__main__":
    raise SystemExit(main())
