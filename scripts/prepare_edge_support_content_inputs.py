#!/usr/bin/env python3
"""Refit training-only maps and freeze PF/hc11 edge-support benchmark inputs."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np
import pandas as pd

from hipporeplayimm.data import ReplaySession
from hipporeplayimm.replay_coverage_data import (
    CoverageInputConfig, count_candidate_bins, fit_coverage_population,
    index_spike_times, prepare_position_support, load_pf_coverage_session,
)
from scripts._provenance import build_script_provenance, file_sha256
from scripts.measure_population_content_stability import seed_for


def freeze_candidates(events, dataset, session, seed, cap=200):
    if events.event_index.duplicated().any() or events.empty:
        raise ValueError("empty/duplicate candidate catalog")
    if not np.isfinite(events[["event_index", "start_s", "end_s"]].to_numpy(float)).all():
        raise ValueError("nonfinite candidate metadata")
    if not (events.end_s > events.start_s).all():
        raise ValueError("invalid candidate windows")
    ranked = events.copy()
    ranked["sampling_hash"] = [str(seed_for(seed, "edge_support_sample", dataset, session, int(i)))
                               for i in events.event_index]
    ranked["_key"] = ranked.sampling_hash.map(int)
    return ranked.sort_values(["_key", "event_index"]).head(cap).drop(columns="_key").sort_values("event_index").reset_index(drop=True)


def training_session(session):
    start = session.run_times[:, 0].min()
    end = session.run_times[:, 1].max()
    midpoint = (start + end)/2
    intervals = np.array([(a, min(b, np.nextafter(midpoint, -np.inf))) for a, b in session.run_times if a < midpoint])
    if not len(intervals):
        raise ValueError("no first-half RUN")
    return replace(session, run_times=intervals), midpoint, (start, end)


def load_native(row, source, dataset):
    if dataset == "pfeiffer_foster":
        path = Path(row.artifact_path)
        if file_sha256(path) != row.artifact_sha256:
            raise ValueError("PF source checksum differs")
        session, raw_provenance = load_pf_coverage_session(Path(row.source_path))
        with np.load(path, allow_pickle=False) as z:
            events = pd.DataFrame(dict(event_index=z["candidate_event_indices"],
                                       start_s=z["candidate_start_s"], end_s=z["candidate_end_s"]))
        source_info = dict(source_cache=str(path), source_cache_sha256=row.artifact_sha256,
                           native_path=str(session.path))
        # Hash native files consumed by the canonical reader, not just the old cache.
        source_info["native_mat_sha256"] = {p: meta["sha256"] for p, meta in raw_provenance["consumed_files"].items()}
    else:
        folder = Path(row.artifact_dir)
        manifest = json.loads((folder/"manifest.json").read_text())
        if file_sha256(folder/"count_arrays.npz") != manifest["count_arrays_sha256"]:
            raise ValueError("hc11 native cache checksum differs")
        if file_sha256(folder/"candidate_events.csv") != manifest["output_sha256"]["candidate_events.csv"]:
            raise ValueError("hc11 candidate checksum differs")
        if file_sha256(folder/"native_unit_audit.csv") != manifest["output_sha256"]["native_unit_audit.csv"]:
            raise ValueError("hc11 unit metadata checksum differs")
        with np.load(folder/"count_arrays.npz", allow_pickle=False) as z:
            position = z["position_cm"]
            spikes = np.column_stack([z["spike_times_s"], z["spike_unit_ids"]])
            run = z["epochs_s"][1].reshape(1, 2)
        units = pd.read_csv(folder/"native_unit_audit.csv")
        ids = units.loc[units.native_included, "unit_id"].to_numpy(int)
        empty = np.empty((0, 2))
        session = ReplaySession(rat=row.animal, name=row.session, path=folder, position=position, spikes=spikes,
            tetrode_cell_ids=units.loc[units.native_included, ["shank_id", "unit_id"]].to_numpy(int),
            excitatory_neurons=ids, inhibitory_neurons=np.empty(0, int), ripple_events=np.empty((0, 6)),
            run_times=run, sleep_box_immobile_times=empty, sleep_times=empty, rem_times=empty,
            well_sequence=None, metadata={"source_dataset": "hc11", "native_curated_physiology": True})
        events = pd.read_csv(folder/"candidate_events.csv", float_precision="round_trip")
        source_info = dict(native_manifest=manifest, native_manifest_path=str(folder/"manifest.json"),
                           native_manifest_sha256=file_sha256(folder/"manifest.json"))
    return session, events, source_info


def prepare_one(row, source, output, dataset, seed):
    session, candidates, native = load_native(row, source, dataset)
    selected = freeze_candidates(candidates, dataset, row.session, seed)
    target = output / (row.animal+"__"+row.session.replace("/", "_"))
    target.mkdir(exist_ok=False)
    selected.to_csv(target/"frozen_candidates.csv", index=False)
    (target/"sampling_manifest.json").write_text(json.dumps(dict(
        created_at_utc=datetime.now(UTC).isoformat(), seed=seed, source_candidates=len(candidates),
        selected_candidates=len(selected), selection="hash_rank_before_map_fit",
        candidate_sha256=file_sha256(target/"frozen_candidates.csv")), indent=2)+"\n")
    session, qc = prepare_position_support(session, .1)
    training, midpoint, run_bounds = training_session(session)
    arrays, units, fit = fit_coverage_population(training, CoverageInputConfig())
    # Full here means the full TRAINING half. Restore only tracking/RUN intervals
    # for later truth validation, never rates or unit selection from held-out RUN.
    arrays["position"] = session.position
    arrays["supported_run_intervals"] = session.run_times
    arrays["training_end_s"] = np.array(midpoint)
    arrays["run_bounds_s"] = np.asarray(run_bounds)
    ids = arrays["cell_ids"]
    indexed = index_spike_times(session.spikes, ids)
    counts, starts, durations, offsets = [], [], [], [0]
    for event in selected.itertuples(index=False):
        edges, count = count_candidate_bins(indexed, ids, event.start_s, event.end_s, .005)
        counts.append(count)
        starts.append(edges[:-1])
        durations.append(np.diff(edges))
        offsets.append(offsets[-1]+len(count))
    arrays.update(candidate_base_counts=np.concatenate(counts), candidate_base_starts_s=np.concatenate(starts),
                  candidate_base_durations_s=np.concatenate(durations), candidate_offsets=np.asarray(offsets),
                  candidate_event_indices=selected.event_index.to_numpy(int),
                  candidate_start_s=selected.start_s.to_numpy(), candidate_end_s=selected.end_s.to_numpy())
    np.savez_compressed(target/"encoding_inputs.npz", **arrays)
    units.to_csv(target/"training_unit_qc.csv", index=False)
    (target/"encoding_manifest.json").write_text(json.dumps(dict(native=native, position_qc=qc,
        encoding=fit, training_only=True, first_half_end_s=midpoint,
        holdout_spikes_used_for_rate_or_unit_selection=False), indent=2, default=str)+"\n")
    return dict(dataset=dataset, animal=row.animal, session=row.session, status="cached",
        source_candidates=len(candidates), selected_candidates=len(selected),
        native_cells=len(ids), training_qc_cells=int(arrays["unit_qc_mask"].sum()),
        spatial_bins=int(arrays["valid_spatial_bins"].sum()), artifact_path=str(target/"encoding_inputs.npz"),
        artifact_sha256=file_sha256(target/"encoding_inputs.npz"), sampling_sha256=file_sha256(target/"frozen_candidates.csv"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--dataset", choices=["pfeiffer_foster", "hc11"], required=True)
    p.add_argument("--seed", type=int, default=20260914)
    args = p.parse_args()
    catalog = args.input_dir/("coverage_input_sessions.csv" if args.dataset == "pfeiffer_foster" else "source_sessions.csv")
    inputs = dict(catalog=catalog, script=Path(__file__), protocol=ROOT/"docs/edge_support_content_protocol.md",
                  encoding=ROOT/"src/hipporeplayimm/encoding.py", shared=ROOT/"src/hipporeplayimm/replay_coverage_data.py")
    manifest = build_script_provenance(input_paths=inputs, cwd=ROOT)
    manifest.update(status="running", dataset=args.dataset, seed=args.seed, created_at_utc=datetime.now(UTC).isoformat())
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    table = pd.read_csv(catalog)
    if "dataset" in table:
        table = table.loc[table.dataset.eq(args.dataset)]
    if len(table) != 8 or table.duplicated(["animal", "session"]).any():
        raise ValueError("expected eight unique frozen sessions")
    rows = []
    for row in table.itertuples(index=False):
        try:
            if row.status not in ("cached", "complete"):
                raise ValueError("source unavailable")
            result = prepare_one(row, args.input_dir, args.output_dir, args.dataset, args.seed)
        except (ValueError, OSError, KeyError) as exc:
            result = dict(dataset=args.dataset, animal=row.animal, session=row.session, status="failed",
                          source_candidates=getattr(row, "candidates", np.nan), selected_candidates=0, reason=str(exc))
        rows.append(result)
        print(json.dumps(result), flush=True)
        pd.DataFrame(rows).to_csv(args.output_dir/"encoding_sessions.csv", index=False)
    unchanged = all(file_sha256(path) == manifest["input_file_sha256"][key] for key, path in inputs.items())
    manifest.update(status="complete" if unchanged else "failed", inputs_unchanged=unchanged, results=rows,
                    completed_at_utc=datetime.now(UTC).isoformat())
    (args.output_dir/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    if not unchanged:
        raise ValueError("code/protocol changed during preparation")


if __name__ == "__main__":
    main()
