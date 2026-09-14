#!/usr/bin/env python3
"""Cache native Blackstad recordings using frozen MUA windows and common encoding."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"src")]

import numpy as np
import pandas as pd

from hipporeplayimm.data import ReplaySession
from hipporeplayimm.replay_coverage_data import (
    CoverageInputConfig,
    count_candidate_bins,
    fit_coverage_population,
    index_spike_times,
    prepare_position_support,
)
from scripts._provenance import build_script_provenance, file_sha256


def native_reader(legacy):
    path = legacy/"scripts/evaluate_blackstad_moser_open_field_replay_feasibility.py"
    sys.path.append(str(legacy/"scripts"))
    spec = importlib.util.spec_from_file_location("blackstad_native_reader", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def convert(raw):
    cell_ids = np.arange(1, len(raw.spikes_by_unit)+1)
    unit_names = sorted(raw.spikes_by_unit)
    spike_parts = [np.column_stack([raw.spikes_by_unit[name], np.full(len(raw.spikes_by_unit[name]), cell)])
                   for cell, name in zip(cell_ids, unit_names, strict=True)]
    spikes = np.concatenate(spike_parts)
    spikes = spikes[np.argsort(spikes[:, 0], kind="stable")]
    xy = raw.position_cm.copy()
    xy[~raw.valid_position] = np.nan
    empty = np.empty((0, 2))
    return ReplaySession(rat=raw.animal, name=raw.session, path=raw.path,
        position=np.column_stack([raw.position_times_s, xy]), spikes=spikes,
        tetrode_cell_ids=np.column_stack([np.zeros(len(cell_ids), int), cell_ids]),
        excitatory_neurons=np.empty(0, int), inhibitory_neurons=np.empty(0, int),
        ripple_events=np.empty((0, 6)), run_times=np.array([[raw.position_times_s[0], raw.position_times_s[-1]]]),
        sleep_box_immobile_times=empty, sleep_times=empty, rem_times=empty, well_sequence=None,
        metadata={"source_dataset": "blackstad_moser", "cell_type_source": "untyped_sorted_units",
                  "native_unit_names": unit_names, "position_transform": raw.position_transform}), unit_names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--legacy-loader-root", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    inputs = dict(candidates=args.candidates, adapter=Path(__file__),
        loader=args.legacy_loader_root/"scripts/evaluate_blackstad_moser_open_field_replay_feasibility.py",
        loader_helpers=args.legacy_loader_root/"scripts/evaluate_hc3_open_field_replay_feasibility.py",
        axona=ROOT/"src/hipporeplayimm/olafsdottir2016.py",
        encoding=ROOT/"src/hipporeplayimm/encoding.py", cache=ROOT/"src/hipporeplayimm/replay_coverage_data.py",
        protocol=ROOT/"docs/spatially_balanced_content_remedy_protocol.md")
    provenance = build_script_provenance(input_paths=inputs, cwd=ROOT)
    provenance.update(created_at_utc=datetime.now(UTC).isoformat(), status="running")
    reader = native_reader(args.legacy_loader_root)
    candidates = pd.read_csv(args.candidates, dtype={"animal": str, "session": str})
    if candidates.duplicated(["animal", "session", "event_index"]).any() or not candidates.event_definition.eq("immobile_high_mua").all():
        raise ValueError("invalid frozen candidate catalog")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir/"sessions").mkdir()
    (args.output_dir/"manifest.json").write_text(json.dumps(provenance, indent=2)+"\n")
    config = CoverageInputConfig()
    rows, unit_tables = [], []
    for folder in reader.selected_session_paths(args.dataset_root):
        paths = list(folder.glob("*.[Tt]64"))+list(folder.glob("*.pos"))+list(folder.glob("*.[Nn][Vv][Tt]"))
        hashes = {str(path): file_sha256(path) for path in paths}
        raw = reader.load_session(folder, argparse.Namespace(position_quantile=.005, position_smoothing_s=.1))
        session, names = convert(raw)
        local = candidates.loc[candidates.animal.eq(session.rat) & candidates.session.eq(session.name)].sort_values("event_index")
        if local.empty:
            raise ValueError("native recording absent from frozen catalog")
        session, position_qc = prepare_position_support(session, .1, [[0, 0], [150, 150]])
        arrays, units, metadata = fit_coverage_population(session, config)
        indexed = index_spike_times(session.spikes, arrays["cell_ids"])
        counts, starts, durations, offsets = [], [], [], [0]
        for event in local.itertuples(index=False):
            if event.event_start_s < session.position[:, 0].min() or event.event_end_s > session.position[:, 0].max():
                raise ValueError("candidate clock outside native position")
            edges, spike_counts = count_candidate_bins(indexed, arrays["cell_ids"], event.event_start_s, event.event_end_s, .005)
            counts.append(spike_counts)
            starts.append(edges[:-1])
            durations.append(np.diff(edges))
            offsets.append(offsets[-1]+len(spike_counts))
        arrays.update(candidate_base_counts=np.concatenate(counts), candidate_base_starts_s=np.concatenate(starts),
            candidate_base_durations_s=np.concatenate(durations), candidate_offsets=np.array(offsets),
            candidate_event_indices=local.event_index.to_numpy(int), candidate_start_s=local.event_start_s.to_numpy(),
            candidate_end_s=local.event_end_s.to_numpy(), arena_bounds_cm=np.array([[0, 0], [150, 150]]))
        path = args.output_dir/"sessions"/f"blackstad_moser__{session.rat}__{session.name}.npz"
        np.savez_compressed(path, **arrays)
        units["native_unit_name"] = [names[int(cell)-1] for cell in units.cell_id]
        units["animal"], units["session"] = session.rat, session.name
        unit_tables.append(units)
        detail = dict(dataset="blackstad_moser", animal=session.rat, session=session.name,
            source_files=hashes, position_transform=raw.position_transform, position_qc=position_qc,
            encoding=metadata, unit_identity="sorted_native_filename_assigned_stable_integer",
            cell_types="not supplied; eligible manual sorted cells pass common rate/stability QC")
        path.with_suffix(".json").write_text(json.dumps(detail, indent=2, default=str)+"\n")
        if any(file_sha256(path) != value for path, value in hashes.items()):
            raise ValueError("raw recording changed while preparing cache")
        rows.append(dict(dataset="blackstad_moser", animal=session.rat, session=session.name,
            candidates=len(local), encoding_cells_qc=int(arrays["unit_qc_mask"].sum()),
            artifact_path=str(path), artifact_sha256=file_sha256(path), status="cached"))
        pd.DataFrame(rows).to_csv(args.output_dir/"coverage_input_sessions.csv", index=False)
        print(json.dumps(rows[-1]), flush=True)
    pd.concat(unit_tables).to_csv(args.output_dir/"coverage_input_units.csv", index=False)
    unchanged = all(file_sha256(path) == provenance["input_file_sha256"][name] for name, path in inputs.items())
    assert sum(row["candidates"] for row in rows) == len(candidates)
    provenance.update(status="complete" if unchanged else "failed", inputs_unchanged=unchanged,
                      sessions=len(rows), candidates=len(candidates), completed_at_utc=datetime.now(UTC).isoformat())
    (args.output_dir/"manifest.json").write_text(json.dumps(provenance, indent=2)+"\n")
    if not unchanged:
        raise RuntimeError("input changed")


if __name__ == "__main__":
    main()
