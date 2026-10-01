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
    visits = position_visits(packet, p)
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
