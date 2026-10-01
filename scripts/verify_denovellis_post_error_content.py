#!/usr/bin/env python3
"""Independent accounting verifier; does not import the study's scoring code."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

PREFIX = "denovellis_post_error_"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(root):
    manifest = json.loads((root / (PREFIX + "manifest.json")).read_text())
    p = manifest["protocol"]
    checks = []
    def check(name, passed, detail):
        checks.append({"check": name, "passed": bool(passed), "detail": str(detail)})
    for name, expected in manifest["output_sha256"].items():
        check("output_hash:" + name, digest(root / name) == expected, expected)
    for key, path in manifest["provenance"]["input_file_paths"].items():
        expected = manifest["provenance"]["input_file_sha256"].get(key)
        if expected:
            check("direct_input_hash:" + key, digest(path) == expected, expected)
    inventory = pd.read_csv(root / (PREFIX + "cohort_inventory.csv"))
    trials = pd.read_csv(root / (PREFIX + "trial_inventory.csv"))
    visits = pd.read_csv(root / (PREFIX + "all_well_visits.csv"))
    events = pd.read_csv(root / (PREFIX + "native_event_assignments.csv"))
    gate = pd.read_csv(root / (PREFIX + "gate_summary.csv"))
    check("unique_trials", not trials.trial_id.duplicated().any(), len(trials))
    check("unique_event_assignments", not events.duplicated(["animal", "day", "epoch", "ripple_number"]).any() if len(events) else True, len(events))
    check("no_route_content_imputation", not len(events) or events[["correct_route_content", "mistaken_route_content", "sequence_validated"]].isna().all().all(), "feasibility is not neural decoding")
    check("no_biological_association", manifest["biological_status"] == "not_tested", manifest["biological_status"])
    truth = {}
    bad_rules = []
    for session, frame in visits.groupby("session"):
        packet = json.loads((root / "checkpoints" / (session + ".json")).read_text())
        geom = np.asarray(packet["geometry"]["segments"], float)
        wells = np.asarray(packet["geometry"]["wells"], float)
        # The center well is the endpoint shared by the central-stem segment metadata.
        central_start = geom[0, :2]
        center = int(np.argmin(np.linalg.norm(wells-central_start, axis=1)))+1
        outers = set((1, 2, 3)) - {center}
        last, outer, segment = None, None, None
        scored = []
        for v in frame.sort_values("visit_index").to_dict("records"):
            if v["history_segment"] != segment:
                last, outer = None, None
            required, kind = None, "unknown"
            if last in outers:
                required, kind = center, "inbound"
            elif last == center and outer is not None:
                required, kind = next(x for x in outers if x != outer), "outbound"
            correct = None if required is None else v["well"] == required
            saved_correct = None if pd.isna(v["task_correct"]) else bool(v["task_correct"])
            if v["task_kind"] != kind or correct != saved_correct:
                bad_rules.append((session, v["visit_index"]))
            scored.append((v, kind, correct))
            if v["well"] in outers:
                outer = v["well"]
            last, segment = v["well"], v["history_segment"]
        for i, (v, kind, correct) in enumerate(scored):
            if kind != "outbound" or correct is not False:
                continue
            subsequent = scored[i+1:i+3]
            outcome = "unclassifiable"
            if len(subsequent) == 2:
                a, b = subsequent[0][0], subsequent[1][0]
                if a["history_segment"] == b["history_segment"] == v["history_segment"] and a["well"] == center and b["well"] in outers and subsequent[1][1] == "outbound":
                    outcome = "repeated_error" if b["well"] == v["well"] else "correction"
            truth[(session, int(v["visit_index"]))] = outcome
    check("all_visit_task_rule", not bad_rules, bad_rules[:20])
    observed = {(r.session, int(r.error_visit_index)): r.next_outcome for r in trials.itertuples()}
    # Cohort exclusions do not rewrite the chronological outcome itself.
    check("independent_post_error_outcomes", observed == truth, f"observed {len(observed)}, independently reconstructed {len(truth)}")
    raw = pd.read_csv(manifest["provenance"]["input_file_paths"]["native_events"], low_memory=False).rename(columns={"Animal ID": "animal"})
    raw["animal"] = raw.animal.str.lower()
    raw["a"] = pd.to_timedelta(raw.start_time).dt.total_seconds()
    raw["b"] = pd.to_timedelta(raw.end_time).dt.total_seconds()
    expected_assignments = set()
    expected_counts = {}
    for trial in trials.itertuples():
        intervals = json.loads(trial.immobile_intervals_json)
        exposure = sum(b-a for a, b in intervals)
        check("exposure:" + trial.trial_id, np.isclose(exposure, trial.usable_exposure_s, rtol=0, atol=1e-8), exposure)
        if not trial.behavior_eligible:
            expected_counts[trial.trial_id] = 0
            continue
        selected = raw[(raw.animal == trial.animal) & (raw.day == trial.day) & (raw.epoch == trial.epoch)]
        keys = {(trial.animal, int(trial.day), int(trial.epoch), int(e.ripple_number), trial.trial_id)
                for e in selected.itertuples() if any(a <= e.a < e.b <= b for a, b in intervals)
                and trial.window_start_s <= e.a < e.b <= trial.window_end_s}
        expected_assignments.update(keys)
        expected_counts[trial.trial_id] = len(keys)
    observed_assignments = {(r.animal, int(r.day), int(r.epoch), int(r.ripple_number), r.trial_id) for r in events.itertuples()}
    check("independent_whole_event_assignment", observed_assignments == expected_assignments, f"expected {len(expected_assignments)}, observed {len(observed_assignments)}")
    check("trial_candidate_counts", all(int(r.native_candidate_count) == expected_counts[r.trial_id] for r in trials.itertuples()), len(trials))
    eligible = trials[trials.eligible]
    counts = {"animals_represented": eligible.animal.nunique(), "eligible_transitions": len(eligible),
              "corrections_present": int((eligible.next_outcome == "correction").sum()), "repeated_errors_present": int((eligible.next_outcome == "repeated_error").sum())}
    limits = dict(animals_represented=p["min_animals"], eligible_transitions=p["min_transitions"], corrections_present=p["min_corrections"], repeated_errors_present=p["min_repeated_errors"])
    for name, value in counts.items():
        saved = gate[gate.gate == name].iloc[0]
        check("independent_gate:" + name, float(saved.observed) == value and bool(saved.passed) == (value >= limits[name]), value)
    check("overall_nonvacuous", bool(gate.iloc[-1].passed) == all(gate.iloc[:-1].passed) and (not manifest["feasibility_passed"] or len(eligible) > 0), manifest["feasibility_passed"])
    check("epoch_accounting", len(inventory) == len(pd.read_csv(root / (PREFIX + "task_inventory.csv"))), len(inventory))
    summary = {"verification_passed": all(x["passed"] for x in checks), "n_checks": len(checks), "failed_checks": [x for x in checks if not x["passed"]], "independent_counts": counts}
    pd.DataFrame(checks).to_csv(root / (PREFIX + "verification_checks.csv"), index=False)
    (root / (PREFIX + "verification.json")).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not verify(args.output_dir)["verification_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
