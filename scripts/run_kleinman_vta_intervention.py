#!/usr/bin/env python3
"""Fail-closed feasibility, freeze and one-shot native VTA intervention analysis."""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _provenance import build_script_provenance, file_sha256
from kleinman_vta_intervention import (
    ANIMALS,
    AUTHOR_FIELDS,
    PACKET_KEY,
    animal_contrasts,
    boolean,
    calibration_summary,
    cohort_tables,
    contrast_inference,
    import_metadata,
    request_status,
    validate_packets,
)

PROTOCOL = ROOT / "docs/kleinman_vta_intervention_protocol.md"
SOURCE_NAMES = (
    "run_kleinman_vta_intervention.py",
    "kleinman_vta_intervention.py",
    "calibrate_kleinman_vta_intervention.py",
    "score_kleinman_native_joint_temporal_pilot.py",
    "verify_kleinman_native_joint_temporal_pilot.py",
    "verify_kleinman_conditional_coupling.py",
    "verify_kleinman_ripple_run_opportunities.py",
    "verify_kleinman_spatial_expression.py",
    "kleinman_conditional_spatial_score.py",
    "calibrate_kleinman_joint_temporal_specificity.py",
    "calibrate_kleinman_conditional_coupling.py",
    "calibrate_kleinman_spatial_expression.py",
    "calibrate_kleinman_replay_content.py",
    "validate_kleinman_run_decoder.py",
    "audit_kleinman_ripple_run_opportunities.py",
    "audit_kleinman_nonoverlapping_coupling_design.py",
    "_provenance.py",
)


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def provenance(inputs, committed=False):
    sources = {name: ROOT / "scripts" / name for name in SOURCE_NAMES}
    p = build_script_provenance(cwd=ROOT, input_paths={**inputs, **sources, "protocol": PROTOCOL})
    # A CI environment variable must not stand in for the executing checkout.
    p["code_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    p.update(created_at_utc=datetime.now(UTC).isoformat(), host=socket.gethostname())
    if any(v is None for v in p["input_file_sha256"].values()):
        raise ValueError("missing_input_file")
    if committed and p["git_dirty"] is not False:
        raise ValueError("clean_committed_producer_required")
    return p


def verify_inputs(manifest):
    for name, path in manifest["input_file_paths"].items():
        if file_sha256(path) != manifest["input_file_sha256"][name]:
            raise ValueError("changed_or_missing_frozen_input:" + name)


def finish(out, result):
    verify_inputs(result)
    result["outputs"] = {p.name: file_sha256(p) for p in out.iterdir() if p.is_file() and p.name != "manifest.json"}
    write_json(out / "manifest.json", result)


def checked_artifact(folder):
    manifest = read_json(folder / "manifest.json")
    for name, digest in manifest["outputs"].items():
        if file_sha256(folder / name) != digest:
            raise ValueError("changed_artifact:" + name)
    return manifest


def previous_registry(path):
    spec = read_json(path)
    if not spec.get("history_complete_confirmed_by") or not spec.get("sources"):
        raise ValueError("complete_previous_scoring_history_required")
    frames, inputs = [], {}
    for i, source in enumerate(spec["sources"]):
        p = Path(source["path"])
        if not p.is_absolute():
            p = path.parent / p
        if file_sha256(p) != source["sha256"]:
            raise ValueError("changed_previous_selection")
        f = pd.read_csv(p)
        if not {"baseline_traversal", "target_traversal"}.issubset(f):
            raise ValueError("previous_readout_columns_missing")
        inputs[f"previous_selection_{i}"] = p
        for role in ("past", "baseline", "target"):
            column = role + "_traversal"
            if column in f:
                g = f[["animal", "session", column]].rename(columns={column: "traversal"}).copy()
                g["role"], g["source_artifact"] = role, str(p)
                frames.append(g)
    return pd.concat(frames, ignore_index=True), inputs


def prepare(args):
    inputs = {"request_table": args.request_table, "request_state": args.request_state}
    if args.verified_metadata:
        inputs["verified_metadata"] = args.verified_metadata
    state = read_json(args.request_state)
    waiting = request_status(state)
    request = pd.read_csv(args.request_table, dtype=str, keep_default_na=False)
    author = pd.read_csv(args.verified_metadata, dtype=str, keep_default_na=False) if args.verified_metadata else None
    metadata = import_metadata(request, author)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    metadata.to_csv(args.output_dir / "cohort_session_ledger.csv", index=False)
    metadata.loc[~metadata.metadata_verified].to_csv(args.output_dir / "metadata_exclusions.csv", index=False)
    template = request.copy()
    for field in AUTHOR_FIELDS:
        if field not in template:
            template[field] = ""
    template.to_csv(args.output_dir / "author_metadata_template.csv", index=False)
    metadata.groupby(["animal", "group", "released_drug", "released_novel"]).agg(
        release_sessions=("session", "size"), author_verified_sessions=("metadata_verified", "sum")
    ).reset_index().to_csv(args.output_dir / "metadata_coverage.csv", index=False)
    reasons = []
    if waiting == "stopped_feasibility_budget":
        reasons.append(waiting)
    eligible = metadata.loc[metadata.metadata_verified & metadata.released_novel.eq("familiar")]
    metadata_complete = set(eligible.animal) == set(ANIMALS) and all(set(g.drug) == {0, 1} for _, g in eligible.groupby("animal"))
    status, ready = waiting, False
    gates = [{"gate": "author_confirmed_familiar_both_drugs_all_six", "passed": metadata_complete}]
    if not metadata_complete:
        reasons.append("author_confirmed_track_drug_design_incomplete")
    elif waiting != "stopped_feasibility_budget":
        required = (args.packet_inventory, args.previous_selections, args.design_review)
        if not all(required):
            status = "blocked_design_inputs"
            reasons.append("packet_inventory_previous_history_and_design_review_required")
        else:
            inv = checked_artifact(args.packet_inventory.parent)
            if inv["outputs"].get(args.packet_inventory.name) != file_sha256(args.packet_inventory):
                raise ValueError("unverified_packet_inventory")
            previous, prior_inputs = previous_registry(args.previous_selections)
            inputs.update(prior_inputs)
            inputs.update(
                packet_inventory=args.packet_inventory,
                packet_manifest=args.packet_inventory.parent / "manifest.json",
                previous_selections=args.previous_selections,
                design_review=args.design_review,
            )
            packets = pd.read_csv(args.packet_inventory, float_precision="round_trip")
            ledger, confounding, coverage = cohort_tables(metadata, packets, previous)
            ledger.to_csv(args.output_dir / "cohort_packet_ledger.csv", index=False)
            ledger.loc[~ledger.primary_eligible].to_csv(args.output_dir / "exclusions.csv", index=False)
            confounding.to_csv(args.output_dir / "design_confounding.csv", index=False)
            coverage.to_csv(args.output_dir / "cohort_by_animal.csv", index=False)
            previous.to_csv(args.output_dir / "previous_readouts.csv", index=False)
            review = read_json(args.design_review)
            acknowledged = all(
                review.get(k) is True for k in ("design_approved", "dose_timing_reviewed", "allocation_reviewed", "recording_days_reviewed", "experience_overlap_reviewed")
            )
            acknowledged &= bool(review.get("reviewer") and review.get("assessment"))
            acknowledged &= review.get("metadata_sha256") == file_sha256(args.verified_metadata)
            acknowledged &= review.get("packet_inventory_sha256") == file_sha256(args.packet_inventory)
            gates.extend([{"gate": "usable_paired_drugs_all_six", "passed": bool(coverage.paired_drugs.all())}, {"gate": "design_review_approved", "passed": bool(acknowledged)}])
            ready = bool(coverage.paired_drugs.all() and acknowledged)
            status = "ready_for_calibration" if ready else "stopped_design_feasibility"
            if ready:
                cohort = ledger.loc[ledger.primary_eligible].copy()
                cohort.to_csv(args.output_dir / "frozen_cohort.csv", index=False)
                for a in cohort.itertuples():
                    for name in ("session_info.mat", "spike_data.mat", "ripple_events.mat"):
                        inputs[f"raw:{a.animal}/{a.session}/{name}"] = args.dataset_root / "Experiment_1" / a.animal / a.session / name
            else:
                reasons.append("unsupported_or_unapproved_paired_design")
    p = provenance(inputs, committed=True)
    gates.extend([{"gate": "drug_contrast_authorized", "passed": False}, {"gate": "cohort_freezable", "passed": ready}])
    pd.DataFrame(gates).to_csv(args.output_dir / "gate_summary.csv", index=False)
    decision = {
        "status": status,
        "reasons": reasons,
        "native_outcomes_scored": False,
        "drug_effect_scored": False,
        "next_action": "calibrate_frozen_contrast" if ready else "resolve_metadata_or_record_feasibility_stop",
        "clarification_rounds": state.get("clarification_rounds", 0),
        "sent_at_utc": state.get("sent_at_utc"),
        "feasibility_working_days_used": state.get("feasibility_working_days_used", 0),
    }
    write_json(args.output_dir / "decision.json", decision)
    (args.output_dir / "report.md").write_text(
        "# Bounded VTA updating feasibility\n\n"
        f"Status: `{status}`. Release sessions: {len(metadata)}. Author-verified sessions: {int(metadata.metadata_verified.sum())}.\n\n"
        "No native spatial-change scores or drug contrast were computed. Missing information is not a biological zero.\n\n"
        "Primary endpoint: reference-aligned spatial redistribution, not general field stability, synaptic plasticity or confirmed replay.\n\n"
        "Track identity, experience order, allocation and dose cannot be inferred from session names. All three experimental and three control animals must retain paired drug observations.\n\n"
        "The author-response clock starts only when the approved request is sent. Allow one clarification round and stop at two weeks if essential metadata remain unavailable.\n\n"
        "Prior pooled calibration does not validate this contrast. Frozen-cohort simulation and a separate adequacy review must pass before native scoring.\n\n"
        + "Reasons: "
        + ("; ".join(reasons) or "none")
        + "\n"
    )
    if ready:
        write_json(
            args.output_dir / "freeze.json",
            {
                **p,
                "status": status,
                "cohort_sha256": file_sha256(args.output_dir / "frozen_cohort.csv"),
                "analysis": "track_equal_then_animal_equal_DID_Welch_two_sided",
                "benchmark_035": "engineering_only",
            },
        )
    finish(args.output_dir, {**p, **decision})
    print(json.dumps(decision))


def checked_freeze(folder, dataset_root):
    freeze = read_json(folder / "freeze.json")
    if freeze["status"] != "ready_for_calibration" or file_sha256(folder / "frozen_cohort.csv") != freeze["cohort_sha256"]:
        raise ValueError("valid_frozen_cohort_required")
    verify_inputs(freeze)
    current = provenance({}, committed=True)
    if current["code_commit"] != freeze["code_commit"]:
        raise ValueError("producer_changed_after_freeze")
    cohort = pd.read_csv(folder / "frozen_cohort.csv", float_precision="round_trip")
    validate_packets(cohort)
    if not cohort.primary_eligible.map(boolean).all() or cohort.previous_readout_overlap.map(boolean).any() or set(cohort.animal) != set(ANIMALS):
        raise ValueError("invalid_primary_cohort")
    for a in cohort.itertuples():
        for name in ("session_info.mat", "spike_data.mat", "ripple_events.mat"):
            key = f"raw:{a.animal}/{a.session}/{name}"
            p = dataset_root / "Experiment_1" / a.animal / a.session / name
            if file_sha256(p) != freeze["input_file_sha256"].get(key):
                raise ValueError("dataset_root_not_frozen_data")
    return {**freeze, "freeze_sha256": file_sha256(folder / "freeze.json")}, cohort


def lock_stage(folder, stage, out, p):
    if stage == "validation":
        development = read_json(folder / "development.lock.json")
        dm = checked_artifact(Path(development["output_dir"]))
        if (
            dm.get("bank") != "development"
            or dm.get("status") != "development_complete"
            or dm["code_commit"] != p["code_commit"]
            or dm.get("freeze_sha256") != file_sha256(folder / "freeze.json")
        ):
            raise ValueError("complete_independent_development_required")
    # Exclusive locks live beside the cohort, so changing output directories cannot rescue a failed bank.
    with (folder / (stage + ".lock.json")).open("x") as handle:
        json.dump({"stage": stage, "output_dir": str(out.resolve()), "code_commit": p["code_commit"], "created_at_utc": p["created_at_utc"]}, handle, indent=2)


def checked_native_packet(folder, a):
    import score_kleinman_native_joint_temporal_pilot as native
    import verify_kleinman_native_joint_temporal_pilot as verifier

    cells, scores = native.score_packet(folder, a)
    with (
        patch.object(native, "packet_data", verifier.independent_data),
        patch.object(native, "estimate_reference", verifier.independent_reference),
        patch.object(native, "spatial_score", verifier.independent_score),
        patch.object(native, "anchor_score", verifier.independent_anchor),
    ):
        other_cells, other_scores = native.score_packet(folder, a)
    for key in ("preceding_score", "preceding_information", "prospective_score", "prospective_information"):
        np.testing.assert_allclose(scores[key], other_scores[key], atol=2e-4, rtol=1e-5)
    if scores["status"] != other_scores["status"]:
        raise ValueError("independent_packet_status_mismatch")
    if not cells.unit_id.equals(other_cells.unit_id):
        raise ValueError("independent_unit_identity_mismatch")
    for key in ("reference_included", "reference_spikes", "past_spikes", "baseline_spikes", "target_spikes"):
        np.testing.assert_array_equal(cells[key], other_cells[key])
    return cells, scores


def native_score(args):
    frozen, cohort = checked_freeze(args.frozen_cohort, args.dataset_root)
    calibration = checked_artifact(args.calibration_dir)
    if (
        calibration.get("bank") != "validation"
        or not calibration.get("validation_passed")
        or calibration.get("freeze_sha256") != frozen["freeze_sha256"]
        or calibration.get("code_commit") != frozen["code_commit"]
        or "replicate_contrasts.csv" not in calibration["outputs"]
    ):
        raise ValueError("same_cohort_passing_intervention_validation_required")
    validation_lock = read_json(args.frozen_cohort / "validation.lock.json")
    if Path(validation_lock["output_dir"]).resolve() != args.calibration_dir.resolve():
        raise ValueError("not_original_frozen_validation_bank")
    validation = pd.read_csv(args.calibration_dir / "replicate_contrasts.csv")
    summary = calibration_summary(validation)
    if not summary.complete.all() or not summary.loc[summary.effect.eq(0), "null_pass"].all():
        raise ValueError("failed_independent_calibration_recheck")
    review = read_json(args.validation_review)
    if (
        review.get("adequacy_approved") is not True
        or not review.get("reviewer")
        or not review.get("precision_and_power_assessment")
        or review.get("calibration_manifest_sha256") != file_sha256(args.calibration_dir / "manifest.json")
    ):
        raise ValueError("precontrast_calibration_adequacy_review_required")
    p = provenance({"freeze": args.frozen_cohort / "freeze.json", "calibration": args.calibration_dir / "manifest.json", "adequacy_review": args.validation_review}, committed=True)
    lock_stage(args.frozen_cohort, "native", args.output_dir, p)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rows, cells = [], []
    for a in cohort.itertuples():
        cell, scores = checked_native_packet(args.dataset_root / "Experiment_1" / a.animal / a.session, a)
        meta = {k: getattr(a, k) for k in PACKET_KEY + ["drug", "physical_track_id", "group"]}
        rows.append({**meta, **scores})
        cells.append(cell.assign(**meta))
        print(json.dumps({**meta, "status": scores["status"]}), flush=True)
    packet = pd.DataFrame(rows)
    tracks, animals = animal_contrasts(packet)
    primary = contrast_inference(animals)
    robust = []
    for animal in ANIMALS:
        subset = packet.loc[packet.animal.ne(animal)]
        _, aa = animal_contrasts(subset, require_all=False)
        robust.append({"sensitivity": "leave_one_animal_out", "excluded": animal, **contrast_inference(aa, require_all=False)})
    for unit in ("session", "day"):
        labels = packet.animal + "/" + (packet.session if unit == "session" else packet.session.str[:8])
        for label in sorted(labels.unique()):
            _, aa = animal_contrasts(packet.loc[labels.ne(label)])
            robust.append({"sensitivity": "leave_one_" + unit + "_out", "excluded": label, **contrast_inference(aa)})
    for name, frame in (
        ("packet_scores.csv", packet),
        ("cell_scores.csv.gz", pd.concat(cells)),
        ("by_track.csv", tracks),
        ("per_animal_contrasts.csv", animals),
        ("primary_contrast.csv", pd.DataFrame([primary])),
        ("robustness_summary.csv", pd.DataFrame(robust)),
    ):
        frame.to_csv(args.output_dir / name, index=False)
    exposure = cohort.merge(packet[PACKET_KEY + ["status"]], on=PACKET_KEY, validate="one_to_one")
    exposure.groupby(["animal", "drug"]).agg(
        n_frozen_packets=("packet_id", "size"),
        n_informative=("status", lambda x: int(x.eq("paired_information").sum())),
        ripple_exposure_s=("eligible_ripple_s", "sum"),
        background_exposure_s=("eligible_background_s", "sum"),
    ).reset_index().to_csv(args.output_dir / "exposure_missingness.csv", index=False)
    stable = all(r["status"] == "scored" and r["D"] < 0 for r in robust if r["sensitivity"] == "leave_one_animal_out")
    session_stable = all(r["status"] == "scored" and r["D"] < 0 for r in robust if r["sensitivity"] != "leave_one_animal_out")
    positive = primary["status"] == "scored" and primary["ci_high"] < 0 and stable and session_stable
    decision = {
        "status": "discovery_association_requires_replication" if positive else "inconclusive_or_no_supported_directional_effect",
        "native_outcomes_scored": True,
        "drug_effect_scored": True,
        "causal_ripple_mediation_claim": False,
        "absence_or_equivalence_claim": False,
        "all_leave_one_animal_out_same_direction": stable,
        "all_session_day_sensitivities_supported_same_direction": session_stable,
        "independent_packet_scores_verified": True,
    }
    write_json(args.output_dir / "decision.json", decision)
    (args.output_dir / "report.md").write_text(
        "# Bounded VTA updating result\n\n"
        + f"Decision: `{decision['status']}`.\n\n"
        + "The endpoint is reference-aligned spatial redistribution. Recruitment is post-treatment: this is an intervention-modified association, not ripple-mediated causation.\n\n"
        + "D and the two-sided animal-level working interval are in primary_contrast.csv; all six animal effects, exposure/missingness and every sensitivity are retained. Cells were not bootstrapped.\n\n"
        + "A nonsignificant result is not equivalence. No biologically meaningful absence bound has been declared; beta=0.35 is only an engineering benchmark.\n\n"
        + "No subgroup replaces the primary endpoint. Positive discovery evidence requires independent replication.\n"
    )
    verify_inputs(frozen)
    finish(args.output_dir, {**p, **decision, "freeze_sha256": frozen["freeze_sha256"]})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    for name in ("dataset-root", "request-table", "request-state", "output-dir"):
        prep.add_argument("--" + name, type=Path, required=True)
    for name in ("verified-metadata", "packet-inventory", "previous-selections", "design-review"):
        prep.add_argument("--" + name, type=Path)
    score = sub.add_parser("score")
    for name in ("dataset-root", "frozen-cohort", "calibration-dir", "validation-review", "output-dir"):
        score.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    prepare(args) if args.command == "prepare" else native_score(args)


if __name__ == "__main__":
    main()
