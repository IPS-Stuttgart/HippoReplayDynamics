#!/usr/bin/env python3
"""Non-scoring upper-bound audit of already archived Kleinman packet metadata."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _provenance import file_sha256
from kleinman_vta_intervention import ANIMALS, KEY, PACKET_KEY, boolean, import_metadata, prior_readouts, validate_packets
from run_kleinman_vta_intervention import provenance, read_json, write_json

HISTORY = (
    ("kleinman_spatial_expression_20260924/selected_anchors.csv", "kleinman_coupling_pilot_20260924/native/manifest.json", "selected_anchors"),
    ("kleinman_joint_temporal_20260924/calibration/selected_packets.csv", "kleinman_joint_temporal_20260924/native/manifest.json", "selected_packets.csv"),
)


def verified_table(path, manifest_path, consumer=None, input_key=None):
    digest = file_sha256(path)
    producer = read_json(manifest_path)
    if digest is None or producer["outputs"].get(path.name) != digest:
        raise ValueError("archived_table_hash_mismatch:" + path.name)
    if consumer is not None and read_json(consumer)["input_file_sha256"].get(input_key) != digest:
        raise ValueError("native_selection_hash_mismatch:" + path.name)
    return pd.read_csv(path, float_precision="round_trip")


def readout_rows(frame, source):
    if not {"animal", "session", "baseline_traversal", "target_traversal"}.issubset(frame):
        raise ValueError("previous_readout_columns_missing")
    rows = []
    for role in ("past", "baseline", "target"):
        name = role + "_traversal"
        if name in frame:
            part = frame[KEY + [name]].rename(columns={name: "traversal"}).copy()
            part["role"], part["source_artifact"] = role, source
            rows.append(part)
    result = pd.concat(rows, ignore_index=True)
    prior_readouts(result)
    return result


def coverage_tables(request, packets, previous):
    """Only counts and identities; never estimates U/V or a treatment contrast."""
    metadata = import_metadata(request)
    validate_packets(packets)
    prior = prior_readouts(previous)
    identities = set(map(tuple, metadata[KEY].to_numpy()))
    if not set(map(tuple, packets[KEY].to_numpy())).issubset(identities) or not set(map(tuple, previous[KEY].to_numpy())).issubset(identities):
        raise ValueError("session_not_in_request_inventory")
    fields = PACKET_KEY + ["drug", "novel", "selected", "availability_descriptor", "past_traversal", "baseline_traversal", "target_traversal"]
    ledger = packets[fields].merge(metadata[KEY + ["released_drug", "released_novel"]], on=KEY, validate="many_to_one")
    if not ledger.drug.eq(ledger.released_drug.map({"saline": 0, "CNO": 1})).all() or not ledger.novel.eq(ledger.released_novel.map({"familiar": 0, "novel": 1})).all():
        raise ValueError("packet_release_metadata_conflict")
    ledger["known_previous_readout_overlap"] = [any((r.animal, r.session, getattr(r, role + "_traversal")) in prior for role in ("past", "baseline", "target")) for r in ledger.itertuples()]
    ledger["remaining_upper_bound_only"] = ledger.selected.map(boolean) & ledger.availability_descriptor.map(boolean) & ledger.novel.eq(0) & ~ledger.known_previous_readout_overlap
    ledger = ledger.loc[ledger.selected.map(boolean)].copy()
    rows = []
    for animal, group in ANIMALS.items():
        for drug, label in ((0, "saline"), (1, "CNO")):
            release = metadata.loc[metadata.animal.eq(animal) & metadata.drug.eq(drug) & metadata.released_novel.eq("familiar")]
            selected = ledger.loc[ledger.animal.eq(animal) & ledger.drug.eq(drug) & ledger.novel.eq(0)]
            available = selected.loc[selected.availability_descriptor.map(boolean)]
            remaining = available.loc[available.remaining_upper_bound_only]
            rows.append({
                "animal": animal, "group": group, "drug": label,
                "released_familiar_sessions": len(release), "selected_familiar_packets": len(selected),
                "available_familiar_packets": len(available),
                "excluded_known_previous_readout_packets": int(available.known_previous_readout_overlap.sum()),
                "remaining_packets_upper_bound": len(remaining), "remaining_sessions_upper_bound": remaining.session.nunique(),
                "remaining_days_upper_bound": remaining.session.str[:8].nunique(),
                "same_track_support": "unknown", "complete_scoring_history_verified": False,
            })
    return ledger, pd.DataFrame(rows)


def audit(paper_root, output):
    base = paper_root.resolve() / "research/updates"
    request_path = base / "kleinman_metadata_request_20260924/kleinman_track_metadata_request.csv"
    request_manifest = request_path.parent / "request_manifest.json"
    packet_path = base / "kleinman_nonoverlapping_design_20260924/packets.csv"
    packet_manifest = packet_path.parent / "manifest.json"
    request = verified_table(request_path, request_manifest)
    packets = verified_table(packet_path, packet_manifest)
    inputs = {"request": request_path, "request_manifest": request_manifest, "packets": packet_path, "packet_manifest": packet_manifest, "archive_auditor": Path(__file__).resolve()}
    frames, sources, summaries = [], [], []
    for i, (relative, native, key) in enumerate(HISTORY):
        path, consumer = base / relative, base / native
        producer = path.parent / "manifest.json"
        frame = verified_table(path, producer, consumer, key)
        rows = readout_rows(frame, str(path))
        frames.append(rows)
        sources.append({"path": str(path), "sha256": file_sha256(path)})
        summaries.append({"source": relative, "native_manifest": native, "selected_rows": len(frame), "unique_readouts": len(prior_readouts(rows)), "producer_and_consumer_hashes_match": True})
        inputs.update({f"selection_{i}": path, f"selection_producer_{i}": producer, f"native_consumer_{i}": consumer})
    previous = pd.concat(frames, ignore_index=True)
    ledger, coverage = coverage_tables(request, packets, previous)
    p = provenance(inputs, committed=True)
    decision = {
        "status": "archived_upper_bound_only",
        "scope": "local_archived_csv_inventory_not_server_evaluation",
        "author_metadata_required": True, "complete_scoring_history_verified": False,
        "cohort_freezable": False, "native_outcomes_scored": False, "drug_effect_scored": False,
        "all_six_have_both_drugs_in_upper_bound": bool(coverage.remaining_packets_upper_bound.gt(0).all()),
        "remaining_familiar_packets_upper_bound": int(coverage.remaining_packets_upper_bound.sum()),
        "known_previous_unique_readouts": len(prior_readouts(previous)),
        "next_action": "review_unsent_author_request_and_verify_history_on_gpuserver4090",
    }
    output.mkdir(parents=True, exist_ok=False)
    for name, frame in (("coverage_upper_bound.csv", coverage), ("selected_packet_ledger.csv", ledger), ("known_previous_readouts.csv", previous), ("source_history.csv", pd.DataFrame(summaries))):
        frame.to_csv(output / name, index=False)
    write_json(output / "previous_scoring_registry_provisional.json", {"history_complete_confirmed_by": None, "sources": sources, "not_for_freeze": True})
    write_json(output / "decision.json", decision)
    table = ["| Animal | Drug | Remaining packets (upper bound) | Sessions | Days |", "|---|---|---:|---:|---:|"]
    table.extend(f"| {r.animal} | {r.drug} | {r.remaining_packets_upper_bound} | {r.remaining_sessions_upper_bound} | {r.remaining_days_upper_bound} |" for r in coverage.itertuples())
    (output / "report.md").write_text(
        "# Archived readout coverage: provisional upper bound\n\n"
        "This inspects archived CSV metadata only; it is not a new server evaluation. "
        "Producer and native-consumer hashes bind both earlier selection tables. "
        "Every selected R3/R4/R5 readout overlapping either known pilot is excluded. "
        "Reference-only reuse is not excluded. No native spatial-change scores were read or computed.\n\n"
        + "\n".join(table) + "\n\n"
        "These are not author-confirmed same-track pairs. Track identity, experience-order "
        "overlap, dose/timing review and complete scoring history remain unresolved. "
        "Additional prior scoring can only reduce this remaining-packet upper bound. "
        "Availability is an existing descriptor, not proof of positive conditional information "
        "or adequate statistical power. The provisional registry intentionally cannot unlock "
        "the production freeze. No primary cohort, calibration or drug contrast is authorized.\n"
    )
    p.update(**decision, outputs={path.name: file_sha256(path) for path in sorted(output.iterdir())})
    write_json(output / "manifest.json", p)
    return decision


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.paper_root, args.output_dir)))
