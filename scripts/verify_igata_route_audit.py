#!/usr/bin/env python3
"""Independently verify accounting/hashes in a completed, non-comparative Igata audit."""

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def reference_distance(a, b):
    """Full-matrix recurrence, independent of the producer's rolling-row implementation."""
    table = [[0.0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        table[i][0] = .2 * i
    for j in range(len(b) + 1):
        table[0][j] = .2 * j
    for i, c in enumerate(a, 1):
        for j, d in enumerate(b, 1):
            ci, di = ord(c) - ord("A"), ord(d) - ord("A")
            cost = math.sqrt((ci % 5 - di % 5) ** 2 + (ci // 5 - di // 5) ** 2) * .2 / 1.13
            table[i][j] = min(table[i - 1][j] + .2, table[i][j - 1] + .2, table[i - 1][j - 1] + cost)
    return table[-1][-1]


def verify(output, dataset):
    manifest = json.loads((output / "manifest.json").read_text())
    decision = json.loads((output / "decision.json").read_text())
    rows = read(output / "trial_inventory.csv")
    labels = read(output / "trial_route_labels_audit.csv")
    transitions = read(output / "observable_transition_audit.csv")
    cohort = read(output / "cohort_inventory.csv")
    checks = []

    def check(name, ok, detail):
        checks.append({"check": name, "status": "pass" if ok else "fail", "detail": detail})

    check("manifest_output_hashes", all(sha(output / name) == digest for name, digest in manifest["output_file_sha256"].items()), "Producer outputs have not changed")
    check("input_npz_count", len(rows) == manifest["n_input_npz"], f"{len(rows)} records")
    actual = {p.relative_to(dataset).as_posix() for p in dataset.glob("*/*/*_data/*.npz")}
    check("actual_file_inventory", {r["relative_path"] for r in rows} == actual, "Compare the filesystem with producer accounting")
    safe = all(not Path(r["relative_path"]).is_absolute() and ".." not in Path(r["relative_path"]).parts for r in rows)
    check("input_paths_safe", safe, "Release-relative paths only")
    check("input_raw_hashes", safe and all(sha(dataset / r["relative_path"]) == r["file_sha256"] for r in rows), "Independent hash pass over every NPZ")
    key = lambda r: (r["animal"], r["date"], r["recording_block"], int(r["trial_number"]), r["record_kind"])
    check("unique_trial_identity", len({key(r) for r in rows}) == len(rows), "Animal/block/trial/kind unique")
    check("route_label_accounting", {r["relative_path"] for r in labels} == {r["relative_path"] for r in rows if r["record_kind"] == "trial_data"}, "Every task trial retained including unclassifiable")
    check("allowed_route_categories", all(r["route_label"] in {"new", "obsolete", "other", "unclassifiable"} for r in labels), "No implicit missing-to-other conversion")
    by_path = {r["relative_path"]: r for r in rows}
    valid_transitions = True
    for t in transitions:
        a, b = by_path[t["previous_relative_path"]], by_path[t["next_relative_path"]]
        valid_transitions &= all(a[k] == b[k] == t[k] for k in ("animal", "date", "recording_block"))
        valid_transitions &= int(b["trial_number"]) == int(a["trial_number"]) + 1 and a["optimized_new_success"] == "True"
        valid_transitions &= float(b["start_time_ms"]) > float(a["end_time_ms"]) and t["next_route_label"] == b["route_label"]
    check("transition_chronology_and_previous_success", valid_transitions, "No missing-number or recording-clock bridge")
    lookup = {key(r)[:4]: r for r in labels}
    distances_ok = True
    for r in read(output / "published_distance_algorithm_check.csv"):
        base = (r["animal"], r["date"], r["recording_block"])
        a = lookup[(*base, int(r["previous_trial_number"]))]
        b = lookup[(*base, int(r["next_trial_number"]))]
        distance = reference_distance(a["field_string"], b["field_string"])
        distances_ok &= math.isclose(distance, float(r["modified_levenshtein_distance"]), abs_tol=1e-10)
        distances_ok &= (distance >= 2) == (r["behavior_change_distance_ge_2"] == "True")
    check("published_distance_independent_recalculation", distances_ok, "Independent full-matrix recurrence; no group-level claim")
    check("cohort_inventory_counts", all(int(c["n_trials"]) == sum(r["animal"] == c["animal"] and r["released_group"] == c["released_group"] for r in labels) for c in cohort), "Animal counts agree with task inventory")
    check("feasibility_stop_enforced", decision["decision"] == "stop_unverified_public_trial_design" and not decision["biological_contrast_run"] and decision["validation_replicates_run"] == 0 and all(t["primary_eligible"] == "False" for t in transitions), "No fitted contrast or invented-design validation after stop")
    check("actual_commit_recorded", len(manifest["code_commit"]) == 40 and not manifest["git_dirty"], "Committed clean producer checkout")
    result = {"verified_at_utc": datetime.now(UTC).isoformat(), "overall": "pass" if all(r["status"] == "pass" for r in checks) else "fail", "checks": checks, "route_label_counts_all_task_records": dict(Counter(r["route_label"] for r in labels)), "primary_calculation": "not_executed_due_to_feasibility_stop"}
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, required=True)
    a = p.parse_args()
    result = verify(a.output_dir.resolve(), a.dataset_root.resolve())
    (a.output_dir / "independent_verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    if result["overall"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
