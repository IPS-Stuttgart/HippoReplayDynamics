#!/usr/bin/env python3
"""Fit the frozen common encoding on native AutoPI foraging-only data."""
from __future__ import annotations

import argparse
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


def cache_session(row, native_root, output):
    folder = native_root/row.animal/row.session
    paths = dict(arrays=folder/"native_inputs.npz", candidates=folder/"candidates.csv", metadata=folder/"native_manifest.json")
    for key, expected in (("arrays", row.artifact_sha256), ("candidates", row.candidate_sha256),
                          ("metadata", row.native_manifest_sha256)):
        if file_sha256(paths[key]) != expected:
            raise ValueError(f"native source checksum differs: {key}")
    with np.load(paths["arrays"], allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    events = pd.read_csv(paths["candidates"])
    if len(events) != row.candidates or events.event_index.duplicated().any():
        raise ValueError("candidate catalog changed")
    ids = data["cell_ids"]
    empty = np.empty((0, 2))
    session = ReplaySession(rat=row.animal, name=row.session, path=folder, position=data["position"], spikes=data["spikes"],
        tetrode_cell_ids=np.column_stack([data["shank_ids"], ids]), excitatory_neurons=np.empty(0, int),
        inhibitory_neurons=np.empty(0, int), ripple_events=np.empty((0, 6)), run_times=data["run_interval"].reshape(1, 2),
        sleep_box_immobile_times=empty, sleep_times=empty, rem_times=empty, well_sequence=None,
        metadata={"source_dataset": "autopi_ca1", "cell_type_source": "author_good_untyped_CA1_sorted_units"})
    session, position_qc = prepare_position_support(session, .1)
    arrays, units, metadata = fit_coverage_population(session, CoverageInputConfig())
    if not np.array_equal(arrays["cell_ids"], ids):
        # A good unit silent in both retained epochs is still part of the source
        # denominator, but cannot supply RUN evidence; never silently renumber it.
        if not np.isin(arrays["cell_ids"], ids).all():
            raise ValueError("encoding invented native unit IDs")
    indexed = index_spike_times(session.spikes, arrays["cell_ids"])
    counts, starts, durations, offsets = [], [], [], [0]
    for event in events.itertuples(index=False):
        if event.start_s < data["rest_interval"][0] or event.end_s > data["rest_interval"][1]:
            raise ValueError("candidate outside native following rest")
        edges, count = count_candidate_bins(indexed, arrays["cell_ids"], event.start_s, event.end_s, .005)
        counts.append(count)
        starts.append(edges[:-1])
        durations.append(np.diff(edges))
        offsets.append(offsets[-1]+len(count))
    arrays.update(candidate_base_counts=np.concatenate(counts) if counts else np.empty((0, len(arrays["cell_ids"])), int),
        candidate_base_starts_s=np.concatenate(starts) if starts else np.empty(0),
        candidate_base_durations_s=np.concatenate(durations) if durations else np.empty(0),
        candidate_offsets=np.array(offsets), candidate_event_indices=events.event_index.to_numpy(int),
        candidate_start_s=events.start_s.to_numpy(), candidate_end_s=events.end_s.to_numpy())
    target = output/"sessions"/f"{row.animal}__{row.session}.npz"
    np.savez_compressed(target, **arrays)
    units["animal"], units["session"] = row.animal, row.session
    units.to_csv(target.with_suffix(".units.csv"), index=False)
    detail = dict(dataset="autopi_ca1", animal=row.animal, session=row.session, position_qc=position_qc,
                  encoding=metadata, native_metadata=json.loads(paths["metadata"].read_text()),
                  native_inputs={key: dict(path=str(path), sha256=file_sha256(path)) for key, path in paths.items()},
                  native_units_absent_retained_epochs=np.setdiff1d(ids, arrays["cell_ids"]).tolist())
    target.with_suffix(".json").write_text(json.dumps(detail, indent=2, default=str)+"\n")
    return dict(dataset="autopi_ca1", animal=row.animal, session=row.session, candidates=len(events),
                n_author_good_ca1=len(ids), encoding_cells_qc=int(arrays["unit_qc_mask"].sum()),
                artifact_path=str(target), artifact_sha256=file_sha256(target), status="cached")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frozen-model", type=Path, required=True)
    args = parser.parse_args()
    frozen = json.loads(args.frozen_model.read_text())
    if frozen.get("training_dataset") != "pfeiffer_foster" or frozen.get("status") != "frozen":
        raise ValueError("PF diagnostic must be frozen before external encoding/readouts")
    inputs = dict(native_catalog=args.native_dir/"native_sessions.csv", native_manifest=args.native_dir/"manifest.json",
                  frozen_model=args.frozen_model, producer=Path(__file__),
                  encoding=ROOT/"src/hipporeplayimm/replay_coverage_data.py",
                  protocol=ROOT/"docs/three_population_content_protocol.md")
    manifest = build_script_provenance(input_paths=inputs, cwd=ROOT)
    manifest.update(status="running", created_at_utc=datetime.now(UTC).isoformat())
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir/"sessions").mkdir()
    (args.output_dir/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    catalog = pd.read_csv(inputs["native_catalog"])
    rows = []
    for row in catalog.itertuples(index=False):
        try:
            if row.status != "extracted":
                raise ValueError(f"native_extraction_failure:{row.reason}")
            result = cache_session(row, args.native_dir, args.output_dir)
        except (ValueError, KeyError, OSError) as exc:
            result = dict(dataset="autopi_ca1", animal=row.animal, session=row.session, candidates=row.candidates,
                          status="failed", reason=str(exc))
        rows.append(result)
        pd.DataFrame(rows).to_csv(args.output_dir/"coverage_input_sessions.csv", index=False)
        print(json.dumps(result), flush=True)
    unchanged = all(file_sha256(path) == manifest["input_file_sha256"][key] for key, path in inputs.items())
    manifest.update(status="complete" if unchanged else "failed", inputs_unchanged=unchanged, all_recordings_attempted=True,
                    failed_recordings=sum(row["status"] == "failed" for row in rows), completed_at_utc=datetime.now(UTC).isoformat())
    (args.output_dir/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    if not unchanged:
        raise RuntimeError("inputs changed")


if __name__ == "__main__":
    main()
