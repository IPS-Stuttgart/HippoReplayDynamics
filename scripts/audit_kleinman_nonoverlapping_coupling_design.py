#!/usr/bin/env python3
"""Coverage-only audit of non-overlapping ripple/RUN coupling packets."""

from __future__ import annotations

import argparse
import itertools
import json
import re
import socket
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _provenance import build_script_provenance, file_sha256
from audit_kleinman_ripple_run_opportunities import (
    counts_in_intervals,
    duration,
    intersect_intervals,
    interval_run_mask,
)
from preflight_kleinman_biological_endpoints import binary_label
from validate_kleinman_run_decoder import (
    PARAMETERS,
    align_behavior,
    fit_maps,
    interval_counts,
    interval_data,
    make_traversals,
    matrix,
    split_units,
)

PROTOCOL = ROOT / "docs/kleinman_nonoverlapping_coupling_protocol.md"


def candidate_packets(runs):
    if len({r["traversal"] for r in runs}) != len(runs):
        raise ValueError("duplicate_traversal")
    ordered = sorted(runs, key=lambda r: r["start_s"])
    for r in ordered:
        if not np.isfinite([r["start_s"], r["end_s"]]).all() or r["end_s"] <= r["start_s"]:
            raise ValueError("invalid_traversal")
    if any(a["end_s"] > b["start_s"] for a, b in itertools.pairwise(ordered)):
        raise ValueError("overlapping_traversals")
    rows = []
    for epoch, direction in sorted({(r["epoch"], r["direction"]) for r in runs}):
        same = [r for r in ordered if (r["epoch"], r["direction"]) == (epoch, direction)]
        for i in range(5, len(same)):
            chunk = same[i - 5 : i + 1]
            row = {
                "packet_id": f"e{epoch}_d{direction}_t{chunk[-1]['traversal']}",
                "epoch": epoch,
                "direction": direction,
                "reference": chunk[:3],
                "past": chunk[3],
                "baseline": chunk[4],
                "target": chunk[5],
            }
            row.update(span_start_s=chunk[0]["start_s"], span_end_s=chunk[-1]["end_s"])
            rows.append(row)
    end = -np.inf
    for row in sorted(rows, key=lambda r: (r["span_end_s"], r["span_start_s"], r["direction"], r["packet_id"])):
        row["selected"] = row["span_start_s"] >= end
        row["selection_reason"] = "chronological_nonoverlap" if row["selected"] else "overlapping_selected_span"
        if row["selected"]:
            end = row["span_end_s"]
    return rows


def metadata(path):
    info = loadmat(path, simplify_cells=True)["session_info"]
    experiment, animal, session = path.parts[-4:-1]
    match = re.fullmatch(r"(\d{8})_run(\d+)", session)
    track_fields = [k for k in info if "track" in k.lower() or "environment" in k.lower()]
    return {
        "experiment": experiment,
        "animal": animal,
        "session": session,
        "drug": binary_label(info["drug"], "drug"),
        "novel": binary_label(info["novel"], "novel") if "novel" in info else None,
        "date": match[1] if match else "",
        "session_ordinal": int(match[2]) if match else None,
        "fields": json.dumps(sorted(info)),
        "track_candidate_fields": json.dumps(track_fields),
        "authoritative_track_id_available": False,
        "sorted_spikes_present": path.with_name("spike_data.mat").is_file(),
        "ripple_present": path.with_name("ripple_events.mat").is_file(),
    }


def packet_coverage(folder):
    info = loadmat(folder / "session_info.mat", simplify_cells=True)["session_info"]
    t, x, speed, ends, visits, epochs = align_behavior(info)
    runs = make_traversals(visits, epochs)
    packets = candidate_packets(runs)  # Freeze selection before examining neural data.
    _, trains, _ = split_units(loadmat(folder / "spike_data.mat", simplify_cells=True)["spike_data"])
    ripples = matrix(loadmat(folder / "ripple_events.mat", simplify_cells=True)["ripple_events"], 4, "ripples")
    if np.any(ripples[:, 1] <= ripples[:, 0]) or np.any((ripples[:, 2] < ripples[:, 0]) | (ripples[:, 2] > ripples[:, 1])):
        raise ValueError("invalid_native_ripple_interval")
    edges = np.arange(np.floor(x.min() / 2) * 2, np.ceil(x.max() / 2) * 2 + 2, 2)
    counts = interval_counts(trains, t)
    dt, valid, trainable, _, direction, spatial = interval_data(t, x, speed, runs, edges)
    n_bins = len(edges) - 1
    immobile = valid & (np.maximum(speed[:-1], speed[1:]) <= 8)
    allowed = intersect_intervals(np.column_stack((t[:-1][immobile], t[1:][immobile])), [[v["start_s"], min(v["end_s"], v["start_s"] + 10)] for v in visits])
    readout = trainable & ((speed[:-1] + speed[1:]) / 2 > 20) & ((x[:-1] + x[1:]) / 2 > ends[0] + 20) & ((x[:-1] + x[1:]) / 2 < ends[1] - 20)
    rows = []
    for packet in packets:
        row = {k: v for k, v in packet.items() if k not in ("reference", "past", "baseline", "target")}
        row["reference_traversals"] = json.dumps([r["traversal"] for r in packet["reference"]])
        for name in ("past", "baseline", "target"):
            for field in ("traversal", "start_s", "end_s"):
                row[name + "_" + field] = packet[name][field]
        row.update(availability_descriptor=False, minimum_units_descriptor=False, coverage_status="not_selected")
        if not packet["selected"]:
            rows.append(row)
            continue
        ref = trainable & interval_run_mask(t, packet["reference"])
        _, _, units, _ = fit_maps(counts, dt, direction, spatial, ref, n_bins)
        row["n_reference_units"] = int(units.sum())
        row["reference_spikes"] = int(counts[ref][:, units].sum())
        row["minimum_units_descriptor"] = bool(units.sum() >= 5)
        occupancy = {}
        for name in ("past", "baseline", "target"):
            mask = readout & interval_run_mask(t, [packet[name]])
            occupancy[name] = np.bincount(spatial[mask], weights=dt[mask], minlength=n_bins)
            row[name + "_readout_s"] = float(dt[mask].sum())
            row[name + "_spikes"] = int(counts[mask][:, units].sum())
        for label, first, second in (("preceding", "past", "baseline"), ("prospective", "baseline", "target")):
            row[label + "_shared_bins"] = int(((occupancy[first] > 0) & (occupancy[second] > 0)).sum())
        gap = [[packet["baseline"]["end_s"], packet["target"]["start_s"]]]
        eligible = intersect_intervals(allowed, gap)
        ripple = intersect_intervals(eligible, ripples[:, :2])
        rt, bg = duration(ripple), duration(eligible) - duration(ripple)
        rc = counts_in_intervals(trains, ripple)[units]
        bc = counts_in_intervals(trains, eligible)[units] - rc
        if np.any(bc < 0):
            raise ValueError("negative_background_count")
        predictor = np.log((rc + 0.5) / rt) - np.log((bc + 0.5) / bg) if rt > 0 and bg > 0 else np.full(len(rc), np.nan)
        variable = bool(len(predictor) >= 2 and np.isfinite(predictor).all() and np.std(predictor) > 0)
        row.update(eligible_ripple_s=rt, eligible_background_s=bg, ripple_reference_spikes=int(rc.sum()), background_reference_spikes=int(bc.sum()), variable_recruitment=variable)
        reasons = []
        for failed, reason in (
            (units.sum() < 5, "few_reference_units"),
            (rt <= 0, "no_ripple_exposure"),
            (bg <= 0, "no_background_exposure"),
            (not variable, "no_variable_recruitment"),
            (row["preceding_shared_bins"] == 0, "no_preceding_shared_occupancy"),
            (row["prospective_shared_bins"] == 0, "no_prospective_shared_occupancy"),
        ):
            if failed:
                reasons.append(reason)
        row["availability_descriptor"] = not reasons
        row["coverage_status"] = ";".join(reasons) or "available_not_yet_scored"
        rows.append(row)
    return rows


def coverage_table(sessions, packets):
    rows = []
    for animal, drug, novel in itertools.product(sorted(sessions.animal.unique()), (0, 1), (0, 1)):
        s = sessions.loc[sessions.animal.eq(animal) & sessions.drug.eq(drug) & sessions.novel.eq(novel)]
        p = packets.loc[packets.animal.eq(animal) & packets.drug.eq(drug) & packets.novel.eq(novel)]
        selected = p.loc[p.selected]
        available = selected.loc[selected.availability_descriptor]
        rows.append(
            {
                "animal": animal,
                "drug": drug,
                "novel": novel,
                "source_sessions": len(s),
                "run_pass_sessions": int(s.run_pass.sum()),
                "audit_failures": int(s.status.eq("failed").sum()),
                "candidate_packets": len(p),
                "selected_packets": len(selected),
                "reference_min5_packets": int(selected.minimum_units_descriptor.sum()),
                "available_packets": len(available),
                "available_sessions": available.session.nunique(),
                "available_days": available.date.nunique(),
                "direction0_available": int(available.direction.eq(0).sum()),
                "direction1_available": int(available.direction.eq(1).sum()),
            }
        )
    return pd.DataFrame(rows)


def run(args):
    root, output = args.dataset_root.resolve(), args.output_dir.resolve()
    frozen = pd.read_csv(args.run_qc, float_precision="round_trip")
    paths = sorted(root.glob("Experiment_*/*/*/session_info.mat"))
    spikes = sorted(root.glob("Experiment_1/*/*/spike_data.mat"))
    if frozen.duplicated(["animal", "session"]).any():
        raise ValueError("duplicate_RUN_session")
    if set(zip(frozen.animal, frozen.session, strict=True)) != {(p.parent.parent.name, p.parent.name) for p in spikes}:
        raise ValueError("RUN_inventory_mismatch")
    if not frozen.decoder_pass.isin([True, False]).all():
        raise ValueError("invalid_RUN_pass")
    rm = args.run_qc.with_name("manifest.json")
    if json.loads(rm.read_text())["outputs"][args.run_qc.name] != file_sha256(args.run_qc):
        raise ValueError("RUN_hash_mismatch")
    inputs = {str(p.relative_to(root)): p for p in paths}
    for p in spikes:
        inputs[str(p.relative_to(root))] = p
        inputs[str(p.with_name("ripple_events.mat").relative_to(root))] = p.with_name("ripple_events.mat")
    inputs.update(producer=Path(__file__), protocol=PROTOCOL, run_qc=args.run_qc, run_manifest=rm)
    for name in ("validate_kleinman_run_decoder.py", "audit_kleinman_ripple_run_opportunities.py", "preflight_kleinman_biological_endpoints.py", "_provenance.py"):
        inputs[name] = ROOT / "scripts" / name
    provenance = build_script_provenance(cwd=ROOT, input_paths=inputs)
    if provenance["git_dirty"] or provenance["code_commit"] == "unavailable":
        raise ValueError("clean_committed_producer_required")
    if any(v is None for v in provenance["input_file_sha256"].values()):
        raise ValueError("missing_input")
    inventory = pd.DataFrame([metadata(p) for p in paths])
    if inventory.track_candidate_fields.ne("[]").any():
        raise ValueError("unexpected_track_metadata_requires_review")
    output.mkdir(parents=True, exist_ok=False)
    sessions, rows = [], []
    bykey = inventory.set_index(["experiment", "animal", "session"])
    for f in frozen.sort_values(["animal", "session"]).itertuples():
        row = bykey.loc[("Experiment_1", f.animal, f.session)].to_dict()
        if row["drug"] != f.drug or row["novel"] != f.novel:
            raise ValueError("RUN_metadata_mismatch")
        row.update(animal=f.animal, session=f.session, run_pass=bool(f.decoder_pass))
        if not f.decoder_pass:
            row["status"] = "excluded_frozen_RUN_QC"
        else:
            try:
                ps = packet_coverage(root / "Experiment_1" / f.animal / f.session)
                keys = {k: row[k] for k in ("animal", "session", "drug", "novel", "date", "session_ordinal")}
                rows.extend({**keys, **p} for p in ps)
                row.update(status="audited", n_selected=sum(p["selected"] for p in ps))
            except (ValueError, KeyError, IndexError) as exc:
                row.update(status="failed", failure_reason=str(exc))
        sessions.append(row)
        print(json.dumps({k: row[k] for k in ("animal", "session", "status")}), flush=True)
    sf = pd.DataFrame(sessions)
    pf = pd.DataFrame(rows)
    if pf.empty:
        pf = pd.DataFrame(columns=["animal", "session", "drug", "novel", "date", "direction", "selected", "availability_descriptor", "minimum_units_descriptor"])
    cf = coverage_table(sf, pf)
    for name, frame in (("metadata_inventory.csv", inventory), ("sessions.csv", sf), ("packets.csv", pf), ("coverage.csv", cf)):
        frame.to_csv(output / name, index=False)
    gates = [
        ("metadata_readable", len(inventory) > 0, len(inventory)),
        ("nonempty_selected_packets", int(pf.selected.sum()) > 0, int(pf.selected.sum())),
        ("no_RUN_pass_audit_failures", not sf.status.eq("failed").any(), int(sf.status.eq("failed").sum())),
        ("all_animal_drug_novelty_cells_available", len(cf) > 0 and cf.available_packets.gt(0).all(), int(cf.available_packets.gt(0).sum())),
        ("track_ids_available_for_within_track_drug_contrast", False, 0),
        ("drug_contrast_authorized", False, 0),
    ]
    pd.DataFrame([{"gate": g, "passed": bool(p), "value": int(v)} for g, p, v in gates]).to_csv(output / "gates.csv", index=False)
    for key, path in inputs.items():
        if file_sha256(path) != provenance["input_file_sha256"][key]:
            raise ValueError("input_changed_during_run")
    manifest = dict(
        **provenance,
        host=socket.gethostname(),
        run_parameters=PARAMETERS,
        selection_rule="earliest_finish_full_span_across_directions_before_neural_QC",
        outcome_scored=False,
        association_scored=False,
        drug_effect_scored=False,
        track_identity_inferred=False,
        causal_interpretation_authorized=False,
        outputs={p.name: file_sha256(p) for p in output.iterdir()},
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--run-qc", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    run(parser.parse_args())
