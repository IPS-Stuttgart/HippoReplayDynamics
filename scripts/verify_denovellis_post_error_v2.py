"""Independently rebuild v2 temporal inclusion and its necessary coverage bound."""

from __future__ import annotations

import argparse
import itertools
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scripts._provenance import build_script_provenance, file_sha256
    from scripts.analyze_denovellis_post_error_content import ANIMALS, PREFIX, atomic_json, epoch_data
    from scripts.verify_denovellis_post_error_content import position_visits
except ModuleNotFoundError:
    from _provenance import build_script_provenance, file_sha256
    from analyze_denovellis_post_error_content import ANIMALS, PREFIX, atomic_json, epoch_data
    from verify_denovellis_post_error_content import position_visits


def independent_trials(time, xy, speed, wells, center, outers, p):
    packet = {"figure_data": {"time": np.asarray(time).tolist(), "xy": np.asarray(xy).tolist()}, "geometry": {"wells": np.asarray(wells).tolist()}}
    visits = [{**v, "visit_index": i + 1} for i, v in enumerate(position_visits(packet, p))]
    t = np.asarray(time)
    lower = (t[0] + 2 * t[-1]) / 3
    rows = []
    last_outer = None
    for i, visit in enumerate(visits):
        if i == 0 or visit["history_segment"] != visits[i - 1]["history_segment"]:
            last_outer = None
        erroneous = (
            i > 0 and visits[i - 1]["history_segment"] == visit["history_segment"] and visits[i - 1]["well"] == center and last_outer is not None and visit["well"] == last_outer
        )
        if visit["well"] in outers:
            last_outer = visit["well"]
        if not erroneous:
            continue
        row = {"error_visit_index": i + 1, "eligible": False, "outcome": "unclassifiable", "exposure_s": 0.0}
        if i + 2 < len(visits):
            pause, following = visits[i + 1 : i + 3]
            if pause["well"] == center and following["well"] in outers and pause["history_segment"] == following["history_segment"] == visit["history_segment"]:
                row["outcome"] = "repeated_error" if following["well"] == visit["well"] else "correction"
                start, end = pause["arrival_s"], min(pause["pause_end_s"], pause["arrival_s"] + p["max_window_s"])
                near = np.linalg.norm(xy - wells[center - 1], axis=1) <= p["well_radius_cm"]
                supported = near & np.isfinite(speed) & (speed >= 0) & (speed < p["max_immobile_speed_cm_s"])
                good_edges = supported[:-1] & supported[1:] & (np.diff(t) > 0) & (np.diff(t) <= p["max_tracking_gap_s"])
                exposure = np.maximum(0, np.minimum(t[1:], end) - np.maximum(t[:-1], start))
                row["exposure_s"] = float(exposure[good_edges].sum())
                row["eligible"] = bool(row["exposure_s"] >= p["min_exposure_s"] and visits[i - 1]["departure_s"] >= lower and following["arrival_s"] < t[-1])
        rows.append(row)
    return rows, visits


def verify(root, dataset_root, output):
    manifest_path = root / (PREFIX + "manifest.json")
    manifest = json.loads(manifest_path.read_text())
    p = manifest["protocol"]
    if p["protocol_version"] == "2.0" and manifest["stage"] == "feasibility":
        return verify_readout(root, output)
    if p["protocol_version"] != "2.0" or manifest["stage"] != "audit":
        raise ValueError("This verifier only checks the v2 temporal audit")
    checks = []

    def require(name, passed):
        checks.append({"check": name, "passed": bool(passed)})
        if not passed:
            raise AssertionError(name)

    for name, digest in manifest["output_sha256"].items():
        require("output_hash:" + name, Path(name).name == name and file_sha256(root / name) == digest)
    source = Path(manifest["source_directory"])
    require("source_manifest_hash", file_sha256(source / (PREFIX + "manifest.json")) == manifest["source_manifest_sha256"])
    original = json.loads((source / (PREFIX + "manifest.json")).read_text())
    for name, digest in original["output_sha256"].items():
        require("source_output_hash:" + name, file_sha256(source / name) == digest)
    inputs = pd.read_csv(root / (PREFIX + "input_inventory.csv"))
    for record in inputs.dropna(subset=["sha256"]).drop_duplicates("path").itertuples():
        require("raw_hash:" + record.path, file_sha256(record.path) == record.sha256)
    inventory = pd.read_csv(root / (PREFIX + "cohort_inventory.csv"))
    original_inventory = pd.read_csv(source / (PREFIX + "cohort_inventory.csv"))
    require("all_source_epochs_retained", list(inventory.session) == list(original_inventory.sort_values(["animal", "day", "epoch"]).session))
    trials = pd.read_csv(root / (PREFIX + "trial_inventory.csv"))
    traversals = pd.read_csv(root / (PREFIX + "traversal_inventory.csv"))
    require("unique_trial_ids", not trials.trial_id.duplicated().any())
    rebuilt = []
    for epoch in inventory.loc[inventory.v2_audit_status.eq("audited_from_position")].itertuples():
        t, xy, speed, wells, _, center, outers, _, _, _ = epoch_data(dataset_root / ANIMALS[epoch.animal], epoch.day, epoch.epoch, p)
        expected, visits = independent_trials(t, xy, speed, wells, center, outers, p)
        observed = trials.loc[trials.session == epoch.session].set_index("error_visit_index")
        require(epoch.session + ":error_ids", sorted(observed.index) == sorted(r["error_visit_index"] for r in expected))
        require(
            epoch.session + ":fixed_thirds",
            np.allclose(
                [epoch.epoch_start_s, epoch.train_end_s, epoch.validation_end_s, epoch.epoch_end_s],
                [t[0], (2 * t[0] + t[-1]) / 3, (t[0] + 2 * t[-1]) / 3, t[-1]],
                rtol=0,
                atol=1e-9,
            ),
        )
        for row in expected:
            got = observed.loc[row["error_visit_index"]]
            key = f"{epoch.session}:{row['error_visit_index']}"
            require(key + ":eligibility", bool(got.final_third_eligible) == row["eligible"])
            require(key + ":outcome", got.next_outcome == row["outcome"])
            require(key + ":exposure", np.isclose(got.usable_exposure_s, row["exposure_s"], rtol=0, atol=1e-8))
            if row["eligible"]:
                rebuilt.append({"animal": epoch.animal, "day": epoch.day, "outcome": row["outcome"], "trial_id": got.trial_id})
        blocks = {"train": (t[0], (2 * t[0] + t[-1]) / 3), "validation": ((2 * t[0] + t[-1]) / 3, (t[0] + 2 * t[-1]) / 3), "analysis": ((t[0] + 2 * t[-1]) / 3, t[-1])}
        for block, (start, end) in blocks.items():
            whole = [
                (a["visit_index"], b["visit_index"])
                for a, b in itertools.pairwise(visits)
                if a["history_segment"] == b["history_segment"] and a["well"] != b["well"] and start <= a["departure_s"] < b["arrival_s"] <= end
            ]
            got = traversals.loc[(traversals.session == epoch.session) & (traversals.block == block)]
            require(epoch.session + ":traversals:" + block, whole == list(zip(got.from_visit, got.to_visit, strict=True)))
    eligible = pd.DataFrame(rebuilt, columns=["animal", "day", "outcome", "trial_id"])
    animal_table = pd.read_csv(root / (PREFIX + "by_animal.csv"))
    retained = []
    for row in animal_table.itertuples():
        group = eligible.loc[eligible.animal == row.animal]
        c, r = int(group.outcome.eq("correction").sum()), int(group.outcome.eq("repeated_error").sum())
        qualified = c > 0 and r > 0 and group.day.nunique() >= 2
        require(
            row.animal + ":animal_accounting",
            (len(group), c, r, group.day.nunique(), qualified) == (row.transitions, row.corrections, row.repetitions, row.recording_days, row.animal_supported),
        )
        if qualified:
            retained.append(row.animal)
    primary = eligible.loc[eligible.animal.isin(retained)]
    require("primary_trial_ids", set(primary.trial_id) == set(trials.loc[trials.primary_cohort_eligible, "trial_id"]))
    counts = [len(retained), len(primary), int(primary.outcome.eq("correction").sum()), int(primary.outcome.eq("repeated_error").sum())]
    floors = [p["min_animals"], p["min_transitions"], p["min_corrections"], p["min_repeated_errors"]]
    gates = pd.read_csv(root / (PREFIX + "gate_summary.csv"))
    for index, (count, floor) in enumerate(zip(counts, floors, strict=True)):
        require("gate:" + str(index), int(gates.iloc[index].observed) == count and bool(gates.iloc[index].passed) == (count >= floor))
    require("decision", manifest["audit_passed"] == all(c >= f for c, f in zip(counts, floors, strict=True)))
    require("no_biological_claim", manifest["biological_status"] == "not_tested" and manifest["calibration_passed"] is False)
    provenance = build_script_provenance(input_paths={"manifest": manifest_path}, cwd=Path(__file__).resolve().parents[1])
    result = {
        "status": "verified",
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "provenance": provenance,
        "n_checks": len(checks),
        "independently_reconstructed_epochs": int(inventory.v2_audit_status.eq("audited_from_position").sum()),
        "n_reconstructed_final_third_transitions": len(eligible),
        "n_primary_upper_bound": len(primary),
        "biological_association_tested": False,
        "checks": checks,
    }
    atomic_json(output, result)
    return result


def verify_readout(root, output):
    manifest_path = root / (PREFIX + "manifest.json")
    manifest = json.loads(manifest_path.read_text())
    p = manifest["protocol"]
    checks = []

    def require(name, passed):
        checks.append({"check": name, "passed": bool(passed)})
        if not passed:
            raise AssertionError(name)

    for name, digest in manifest["output_sha256"].items():
        require("output:" + name, Path(name).name == name and file_sha256(root / name) == digest)
    for name, digest in manifest["checkpoint_sha256"].items():
        require("checkpoint:" + name, Path(name).name == name and file_sha256(root / "checkpoints" / name) == digest)
    source = Path(manifest["source_directory"])
    require("audit_manifest", file_sha256(source / (PREFIX + "manifest.json")) == manifest["source_manifest_sha256"])
    audit = json.loads((source / (PREFIX + "manifest.json")).read_text())
    require("same_protocol_and_passed_audit", audit["protocol"] == p and audit["audit_passed"])
    for name, digest in audit["output_sha256"].items():
        require("audit_output:" + name, file_sha256(source / name) == digest)
    inputs = pd.read_csv(root / (PREFIX + "input_inventory.csv"))
    for row in inputs.dropna(subset=["sha256"]).drop_duplicates("path").itertuples():
        require("input:" + row.path, file_sha256(row.path) == row.sha256)
    trials = pd.read_csv(root / (PREFIX + "trial_inventory.csv"))
    qc = pd.read_csv(root / (PREFIX + "decoder_audit.csv"))
    require("every_target_epoch_evaluated", set(qc.session) == set(trials.loc[trials.final_third_eligible, "session"]) and not qc.session.duplicated().any())
    recomputed_pass = {}
    total_windows = 0
    for row in qc.to_dict("records"):
        session = row["session"]
        windows = pd.read_csv(root / "checkpoints" / (session + "_windows.csv"))
        require(session + ":status", row["status"] in {"passed", "failed_accuracy", "unavailable"})
        passed = False
        if row["status"] == "unavailable":
            require(session + ":missing_not_zero", len(windows) == 0 and bool(row["reason"]))
        else:
            total_windows += len(windows)
            recalls = []
            for arm in (0, 1):
                selected = windows.loc[windows.true_arm == arm]
                require(session + f":arm{arm}_support", len(selected) > 0)
                value = float((selected.predicted_arm == arm).sum() / len(selected))
                recalls.append(value)
                require(session + f":arm{arm}_recall", np.isclose(value, row[f"arm{arm}_recall"], rtol=0, atol=1e-12))
            balanced = sum(recalls) / 2
            passed = balanced >= p["min_run_balanced_accuracy"] and min(recalls) >= p["min_run_arm_recall"]
            require(session + ":balanced_accuracy", np.isclose(balanced, row["balanced_accuracy"], rtol=0, atol=1e-12))
            require(session + ":bin_count", len(windows) == row["n_arm_windows"])
            require(session + ":silent_bins", int((windows.n_spikes == 0).sum()) == row["zero_spike_windows"])
            require(session + ":middle_only", bool((windows.start_time_s >= row["train_end_s"]).all() and (windows.end_time_s <= row["validation_end_s"]).all()))
            require(
                session + ":nonoverlapping",
                not windows.start_time_s.duplicated().any() and np.all(windows.start_time_s.to_numpy()[1:] >= windows.end_time_s.to_numpy()[:-1] - 1e-9),
            )
            require(session + ":twenty_ms", np.allclose(windows.end_time_s - windows.start_time_s, 0.020, rtol=0, atol=1e-9))
            require(session + ":posterior_arm_choice", np.array_equal((windows.arm1_mass > windows.arm0_mass).astype(int), windows.predicted_arm))
        recomputed_pass[session] = passed
        require(session + ":threshold_decision", passed == row["decoder_qc_passed"])
    eligible = trials.final_third_eligible & trials.session.map(recomputed_pass).fillna(False).astype(bool)
    require("trial_decoder_qualification", eligible.equals(trials.decoder_qualified_transition))
    by_animal = pd.read_csv(root / (PREFIX + "by_animal.csv"))
    retained = []
    for row in by_animal.itertuples():
        group = trials.loc[eligible & trials.animal.eq(row.animal)]
        corrected, repeated = int(group.next_outcome.eq("correction").sum()), int(group.next_outcome.eq("repeated_error").sum())
        supported = corrected > 0 and repeated > 0 and group.day.nunique() >= p["min_recording_days_per_animal"]
        require(
            row.animal + ":cohort",
            (len(group), corrected, repeated, group.day.nunique(), supported) == (row.transitions, row.corrections, row.repetitions, row.recording_days, row.animal_supported),
        )
        if supported:
            retained.append(row.animal)
    primary = eligible & trials.animal.isin(retained)
    require("primary_trial_qualification", primary.equals(trials.primary_cohort_eligible))
    selected = trials.loc[primary]
    counts = [len(retained), len(selected), int(selected.next_outcome.eq("correction").sum()), int(selected.next_outcome.eq("repeated_error").sum())]
    floors = [p["min_animals"], p["min_transitions"], p["min_corrections"], p["min_repeated_errors"]]
    gates = pd.read_csv(root / (PREFIX + "gate_summary.csv"))
    for i, (count, floor) in enumerate(zip(counts, floors, strict=True)):
        require(f"gate:{i}", int(gates.iloc[i].observed) == count and bool(gates.iloc[i].passed) == (count >= floor))
    require("overall", manifest["feasibility_passed"] == all(c >= f for c, f in zip(counts, floors, strict=True)))
    require("biology_not_tested", manifest["biological_status"] == "not_tested" and not manifest["calibration_passed"])
    result = {
        "status": "verified",
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "n_checks": len(checks),
        "n_evaluated_epochs": len(qc),
        "n_validation_windows": total_windows,
        "n_passing_epochs": sum(recomputed_pass.values()),
        "n_decoder_qualified_transitions": int(eligible.sum()),
        "n_primary_transitions": int(primary.sum()),
        "biological_association_tested": False,
        "scope": "Independent per-bin confusion counts, frozen thresholds, chronology, cohort/gate accounting and hashes; not an independent raw decoder refit",
        "provenance": build_script_provenance(input_paths={"manifest": manifest_path}, cwd=Path(__file__).resolve().parents[1]),
        "checks": checks,
    }
    atomic_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.audit_dir, args.dataset_root, args.output)
    print(json.dumps({k: v for k, v in result.items() if k not in {"checks", "provenance"}}))


if __name__ == "__main__":
    main()
