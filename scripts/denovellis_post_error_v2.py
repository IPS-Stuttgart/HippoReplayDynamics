"""V2 temporal feasibility and fail-closed gates for the existing staged driver."""

from __future__ import annotations

import itertools
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scripts import analyze_denovellis_post_error_content as legacy
    from scripts._provenance import build_script_provenance, file_sha256
    from scripts.denovellis_post_error_core import build_transitions, contains_event, valid_intervals
except ModuleNotFoundError:
    import analyze_denovellis_post_error_content as legacy
    from _provenance import build_script_provenance, file_sha256
    from denovellis_post_error_core import build_transitions, contains_event, valid_intervals

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "denovellis_post_error_"
FROZEN = ROOT / "docs/denovellis_post_error_protocol_v2.json"


def elapsed_thirds(time):
    time = np.asarray(time, float)
    if time.ndim != 1 or len(time) < 2 or not np.isfinite(time).all() or np.any(np.diff(time) <= 0):
        raise ValueError("Recorded epoch clock must be finite and strictly increasing")
    return np.linspace(time[0], time[-1], 4)


def traversal_intervals(visits, start, end):
    """Whole well-to-well movements, never crossing a history or block boundary."""
    rows = []
    for left, right in itertools.pairwise(visits):
        a, b = float(left["departure_s"]), float(right["arrival_s"])
        if left["history_segment"] == right["history_segment"] and left["well"] != right["well"] and start <= a < b <= end:
            rows.append({"start_s": a, "end_s": b, "from_well": left["well"], "to_well": right["well"], "from_visit": left["visit_index"], "to_visit": right["visit_index"]})
    return rows


def full_window_mask(starts, ends, intervals):
    starts, ends = np.asarray(starts, float), np.asarray(ends, float)
    if starts.shape != ends.shape or not np.isfinite([starts, ends]).all() or np.any(ends <= starts):
        raise ValueError("Finite positive observation windows required")
    mask = np.zeros(len(starts), bool)
    for interval in intervals:
        mask |= (starts >= interval["start_s"]) & (ends <= interval["end_s"])
    return mask


def restrict_transition(row, visits, boundaries):
    """Use complete movements, not just the time of arrival at the error well."""
    result = dict(row)
    result["v1_behavior_eligible"] = bool(row["eligible"])
    result["final_third_eligible"] = False
    result["primary_cohort_eligible"] = False
    result["v2_exclusion_reason"] = row.get("exclusion_reason", "")
    result["error_departure_s"] = None
    if not row["eligible"]:
        return result
    by_index = {v["visit_index"]: v for v in visits}
    error = by_index[row["error_visit_index"]]
    before = by_index.get(row["error_visit_index"] - 1)
    center = by_index[row["center_visit_index"]]
    choice = by_index[row["next_visit_index"]]
    if before is None or len({v["history_segment"] for v in (before, error, center, choice)}) != 1:
        result["v2_exclusion_reason"] = "missing_complete_error_history"
        return result
    result["error_departure_s"] = before["departure_s"]
    clock = [before["departure_s"], error["arrival_s"], error["departure_s"], center["arrival_s"], center["pause_end_s"], center["departure_s"], choice["arrival_s"]]
    if not np.isfinite(clock).all() or np.any(np.diff(clock) < 0):
        raise ValueError("Error/pause/choice clock is not ordered")
    if clock[0] < boundaries[2] or clock[-1] >= boundaries[3]:
        result["v2_exclusion_reason"] = "error_to_choice_not_wholly_in_final_third"
        return result
    result["final_third_eligible"] = True
    result["v2_exclusion_reason"] = ""
    return result


def coverage_bound(trials, animals, p, eligibility="final_third_eligible"):
    frame = trials.copy()
    if frame.empty:
        frame = pd.DataFrame(columns=["animal", "day", "next_outcome", eligibility])
    if frame[eligibility].isna().any() or not frame[eligibility].isin([True, False]).all():
        raise ValueError("Eligibility cannot be missing or a string")
    eligible = frame.loc[frame[eligibility].astype(bool)]
    rows = []
    for animal in sorted(set(animals)):
        group = eligible.loc[eligible.animal == animal]
        corrected = int(group.next_outcome.eq("correction").sum())
        repeated = int(group.next_outcome.eq("repeated_error").sum())
        days = int(group.day.nunique())
        support = corrected > 0 and repeated > 0 and days >= p["min_recording_days_per_animal"]
        rows.append(
            {
                "animal": animal,
                "transitions": len(group),
                "corrections": corrected,
                "repetitions": repeated,
                "recording_days": days,
                "animal_supported": support,
                "reason": "" if support else "requires_both_outcomes_and_two_days",
            }
        )
    summary = pd.DataFrame(rows)
    kept = set(summary.loc[summary.animal_supported, "animal"]) if len(summary) else set()
    frame["primary_cohort_eligible"] = frame[eligibility].astype(bool) & frame.animal.isin(kept)
    actual = frame.loc[frame.primary_cohort_eligible]
    gates = []
    for name, value, minimum in [
        ("qualified_animals_upper_bound", len(kept), p["min_animals"]),
        ("transitions_upper_bound", len(actual), p["min_transitions"]),
        ("corrections_upper_bound", int(actual.next_outcome.eq("correction").sum()), p["min_corrections"]),
        ("repetitions_upper_bound", int(actual.next_outcome.eq("repeated_error").sum()), p["min_repeated_errors"]),
    ]:
        gates.append({"gate": name, "passed": value >= minimum, "observed": value, "criterion": minimum, "gate_type": "necessary_coverage_upper_bound"})
    return frame, summary, gates


def working_deadline(now, days):
    deadline = now
    for _ in range(days):
        deadline += timedelta(days=1)
        while deadline.weekday() >= 5:
            deadline += timedelta(days=1)
    return deadline


def verify_manifest(root, expected_version=None):
    path = root / (PREFIX + "manifest.json")
    manifest = json.loads(path.read_text())
    if expected_version and manifest["protocol"]["protocol_version"] != expected_version:
        raise ValueError("Upstream protocol version mismatch")
    if not manifest.get("output_sha256"):
        raise ValueError("Upstream output hashes are required")
    for name, digest in manifest["output_sha256"].items():
        if Path(name).name != name or file_sha256(root / name) != digest:
            raise ValueError(f"Upstream artifact hash mismatch: {name}")
    return manifest


def check_raw_inputs(rows):
    for row in rows:
        if row.get("sha256"):
            path = Path(row["path"])
            if not path.is_file() or file_sha256(path) != row["sha256"]:
                raise ValueError(f"Source input hash mismatch: {path}")


def audit_session(root, key, events, p):
    packet = legacy.audit_epoch(root, key, events, p)
    animal, day, epoch = key
    time, xy, speed, wells, _, center, outers, _, _, _ = legacy.epoch_data(root / legacy.ANIMALS[animal], day, epoch, p)
    bounds = elapsed_thirds(time)
    visits = packet["visits"]
    center_distance = np.linalg.norm(xy - wells[center - 1], axis=1)
    immobile = valid_intervals(time, np.isfinite(speed) & (speed >= 0) & (speed < p["max_immobile_speed_cm_s"]) & (center_distance <= p["well_radius_cm"]), p["max_tracking_gap_s"])
    transitions = build_transitions(visits, center=center, outers=outers, immobile_intervals=immobile, max_window=p["max_window_s"], min_exposure=p["min_exposure_s"])
    old = {r["error_visit_index"]: r for r in packet["transitions"]}
    trials, assignments = [], []
    for transition in transitions:
        base = old[transition["error_visit_index"]]
        valid = transition.pop("immobile_intervals")
        row = {**base, **transition, "immobile_intervals_json": json.dumps(valid)}
        if not packet["inventory"]["cohort_eligible"]:
            row.update(eligible=False, exclusion_reason="epoch_failed_frozen_cohort_eligibility")
        row = restrict_transition(row, visits, bounds)
        row["native_candidate_count"] = 0
        row["sequence_duration_capable_candidates"] = 0
        if row["final_third_eligible"]:
            for event in events.itertuples(index=False):
                if row["window_start_s"] <= event.start_time_s < event.end_time_s <= row["window_end_s"] and contains_event(valid, event.start_time_s, event.end_time_s):
                    row["native_candidate_count"] += 1
                    capable = event.end_time_s - event.start_time_s >= p["time_bin_s"] * p["min_spike_supported_bins"]
                    row["sequence_duration_capable_candidates"] += int(capable)
                    assignments.append(
                        {
                            "animal": animal,
                            "day": day,
                            "epoch": epoch,
                            "session": row["session"],
                            "trial_id": row["trial_id"],
                            "ripple_number": event.ripple_number,
                            "start_time_s": event.start_time_s,
                            "end_time_s": event.end_time_s,
                            "sequence_duration_capable": capable,
                            "sequence_validated": None,
                            "correct_route_content": None,
                            "mistaken_route_content": None,
                        }
                    )
        trials.append(row)
    traversals = []
    for block, start, end in zip(("train", "validation", "analysis"), bounds[:-1], bounds[1:], strict=True):
        for interval in traversal_intervals(visits, start, end):
            traversals.append({**interval, "session": packet["inventory"]["session"], "block": block})
    packet["inventory"].update(
        epoch_start_s=bounds[0],
        train_end_s=bounds[1],
        validation_end_s=bounds[2],
        epoch_end_s=bounds[3],
        n_final_third_transitions=sum(r["final_third_eligible"] for r in trials),
        n_train_traversals=sum(r["block"] == "train" for r in traversals),
        n_validation_traversals=sum(r["block"] == "validation" for r in traversals),
        decoder_qc_passed=None,
        decoder_status="not_run_necessary_bound_first",
    )
    return {
        "inventory": packet["inventory"],
        "trials": trials,
        "assignments": assignments,
        "traversals": traversals,
        "inputs": packet["inputs"],
        "visits": visits,
        "geometry": packet["geometry"],
    }


def run_audit(args, p):
    source = args.prerequisite_dir.resolve()
    prior = verify_manifest(source, "1.0")
    if not prior.get("feasibility_passed") or prior["biological_status"] != "not_tested":
        raise ValueError("Audit requires the verified pre-content v1 behavioral inventory")
    out = args.output_dir.resolve()
    if out == source or out in source.parents or source in out.parents:
        raise ValueError("V2 outputs must be separate from v1 artifacts")
    if args.dataset_root is None or args.native_events is None:
        raise ValueError("Audit requires dataset root and native event table")
    if file_sha256(args.native_events) != prior["provenance"]["input_file_sha256"]["native_events"]:
        raise ValueError("Frozen native event table changed")
    provenance = build_script_provenance(input_paths={"protocol": args.protocol, "v1_manifest": source / (PREFIX + "manifest.json"), "native_events": args.native_events}, cwd=ROOT)
    if provenance["git_dirty"] is not False or provenance["code_commit"] == "unavailable":
        raise ValueError("A clean committed producer is mandatory")
    identity = {"code_commit": provenance["code_commit"], **provenance["input_file_sha256"], "dataset_root": str(args.dataset_root.resolve())}
    out.mkdir(parents=True, exist_ok=True)
    control = out / "audit_control.json"
    if control.exists():
        frozen = json.loads(control.read_text())
        if frozen["identity"] != identity:
            raise ValueError("Audit producer/input mismatch; use a separate output directory")
    else:
        now = datetime.now(UTC)
        frozen = {"identity": identity, "started_at_utc": now.isoformat(), "deadline_utc": working_deadline(now, p["development_working_days"]).isoformat()}
        legacy.atomic_json(control, frozen)
    check_raw_inputs(pd.read_csv(source / (PREFIX + "input_inventory.csv")).replace({np.nan: None}).to_dict("records"))
    native = legacy.native_events(args.native_events)
    original = pd.read_csv(source / (PREFIX + "cohort_inventory.csv"))
    if original.session.duplicated().any() or not original.cohort_eligible.isin([True, False]).all():
        raise ValueError("Invalid source cohort ledger")
    checkpoints = out / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    packets = []
    for entry in original.sort_values(["animal", "day", "epoch"]).to_dict("records"):
        if datetime.now(UTC) > datetime.fromisoformat(frozen["deadline_utc"]):
            raise RuntimeError("Frozen five-working-day development deadline exceeded")
        path = checkpoints / (entry["session"] + ".json")
        if path.exists():
            saved = json.loads(path.read_text())
            if saved["identity"] != identity:
                raise ValueError("Checkpoint fingerprint mismatch")
            packet = saved["packet"]
            check_raw_inputs(packet["inputs"])
        elif not entry["cohort_eligible"]:
            inventory = {k: None if isinstance(v, float) and np.isnan(v) else v for k, v in entry.items()}
            inventory.update(v2_audit_status="excluded_by_frozen_source_metadata", decoder_qc_passed=None)
            packet = {"inventory": inventory, "trials": [], "assignments": [], "traversals": [], "visits": [], "inputs": []}
            legacy.atomic_json(path, {"identity": identity, "packet": packet})
        else:
            key = (entry["animal"], int(entry["day"]), int(entry["epoch"]))
            events = native.loc[(native.animal == key[0]) & (native.day == key[1]) & (native.epoch == key[2])]
            # A newly failing previously qualified source is an audit error, not zero coverage.
            packet = audit_session(args.dataset_root, key, events, p)
            if not packet["inventory"]["cohort_eligible"]:
                raise ValueError("Reconstructed metadata qualification differs from frozen source")
            packet["inventory"]["v2_audit_status"] = "audited_from_position"
            legacy.atomic_json(path, {"identity": identity, "packet": packet})
        packets.append(packet)
        legacy.atomic_json(out / "progress.json", {"stage": "audit", "completed_epochs": len(packets), "total_epochs": len(original), "last_session": entry["session"]})
        print(json.dumps({"session": entry["session"], "audited": len(packets), "total": len(original)}), flush=True)
    trials = pd.DataFrame([r for packet in packets for r in packet["trials"]])
    trials, animals, gates = coverage_bound(trials, original.animal.unique(), p)
    if not trials.empty:
        no_animal = trials.final_third_eligible & ~trials.primary_cohort_eligible
        trials.loc[no_animal, "v2_exclusion_reason"] = "animal_lacks_both_outcomes_or_two_days"
    passed = all(row["passed"] for row in gates)
    gates.append(
        {
            "gate": "audit_overall",
            "passed": passed,
            "observed": "upper_bound_pass" if passed else "upper_bound_fail",
            "criterion": "all necessary coverage floors",
            "gate_type": "audit",
        }
    )
    for name, rows in [
        ("cohort_inventory", [x["inventory"] for x in packets]),
        ("trial_inventory", trials),
        ("native_event_assignments", [r for x in packets for r in x["assignments"]]),
        ("traversal_inventory", [r for x in packets for r in x["traversals"]]),
        ("all_well_visits", [r for x in packets for r in x["visits"]]),
        ("input_inventory", [r for x in packets for r in x["inputs"]]),
        ("by_animal", animals),
        ("gate_summary", gates),
    ]:
        legacy.csv(out, name, rows)
    exclusions = trials.groupby("v2_exclusion_reason", dropna=False).size().rename("n_transitions").reset_index() if not trials.empty else []
    legacy.csv(out, "exclusions", exclusions)
    decision = {
        "decision": "ready_for_run_only_readout" if passed else "insufficient_v2_behavioral_coverage",
        "basis": "necessary_upper_bound_before_decoder_qc",
        "biological_association_tested": False,
        "run_readout_performed": False,
        "calibration_performed": False,
        "sequence_validation_performed": False,
    }
    legacy.atomic_json(out / (PREFIX + "decision.json"), decision)
    legacy.atomic_json(
        out / (PREFIX + "manifest.json"),
        {
            "protocol": p,
            "stage": "audit",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "provenance": provenance,
            "source_directory": str(source),
            "source_manifest_sha256": file_sha256(source / (PREFIX + "manifest.json")),
            "development_deadline_utc": frozen["deadline_utc"],
            "audit_passed": passed,
            "feasibility_passed": False,
            "calibration_passed": False,
            "biological_status": "not_tested",
            "next_action": decision["decision"],
            "output_sha256": {f.name: file_sha256(f) for f in sorted(out.glob("*.csv"))} | {PREFIX + "decision.json": file_sha256(out / (PREFIX + "decision.json"))},
        },
    )
    report(out)


def report(out):
    manifest = verify_manifest(out, "2.0")
    animals = pd.read_csv(out / (PREFIX + "by_animal.csv"))
    trials = pd.read_csv(out / (PREFIX + "trial_inventory.csv"))
    lines = [
        "# Post-error replay content balance: v2",
        "",
        f"Decision: `{manifest['next_action']}`.",
        "",
        "This is a temporal feasibility upper bound, not a biological result or decoder validation.",
        "V1 remains stopped and unchanged. No replay content/outcome association has been opened.",
        "",
        (
            f"Transitions audited: {len(trials)}; final-third eligible: {int(trials.final_third_eligible.sum())}; "
            f"after per-animal support rules: {int(trials.primary_cohort_eligible.sum())}."
        ),
        "",
        "| Animal | Final-third transitions | Corrections | Repetitions | Days | Animal support |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    lines += [f"| {r.animal} | {r.transitions} | {r.corrections} | {r.repetitions} | {r.recording_days} | {r.animal_supported} |" for r in animals.itertuples()]
    lines += [
        "",
        "The entire incorrect outbound movement, center pause and next outbound choice must be in the final third.",
        "The first/middle thirds are reserved for RUN training/validation. No boundaries were moved.",
        "Coverage assumes all eligible encoders could pass; later QC cannot increase these counts.",
        "These floors are engineering prerequisites, not a power calculation.",
        "",
        "## Work withheld",
        "",
        "RUN fits, sequence validation, the frozen calibration bank, biological regression and",
        "biological panels have not been run. Downstream inference is not implemented past a failed audit.",
        "Missing values are not zero neural evidence. This stop neither supports nor refutes corrective replay.",
        "",
        f"Producer: `{manifest['provenance']['code_commit']}`. Seed: {manifest['protocol']['seed']}.",
        "",
    ]
    (out / (PREFIX + "report.md")).write_text("\n".join(lines))


def run(args, p):
    if p != json.loads(FROZEN.read_text()):
        raise ValueError("V2 protocol differs from the committed frozen registration")
    if args.stage == "report":
        return report(args.output_dir)
    if args.prerequisite_dir is None:
        raise ValueError("V2 requires a hashed upstream manifest")
    if args.stage == "audit":
        return run_audit(args, p)
    prerequisite = verify_manifest(args.prerequisite_dir, "2.0")
    if prerequisite["protocol"] != p or not prerequisite.get("audit_passed"):
        raise ValueError("V2 necessary coverage audit failed; downstream work is blocked")
    raise ValueError("Validated within-epoch readout and frozen calibration are required; no biological scores produced")
