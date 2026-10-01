#!/usr/bin/env python3
"""Independent accounting verifier; does not refit or decode any data."""

import argparse
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

PREFIX = "denovellis_run_audit_"


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(root):
    manifest = json.loads((root/(PREFIX+"manifest.json")).read_text())
    p = manifest["protocol"]
    frames = {name: pd.read_csv(root/(PREFIX+name+".csv")) for name in ("metrics", "windows", "confusion", "encoder_candidates", "encoder_selection", "parity")}
    checks = []

    def check(name, passed):
        checks.append({"check": name, "passed": bool(passed)})

    check("no_replay_or_association", not manifest["replay_decoded"] and not manifest["behavioral_association_tested"] and manifest["original_no_go_preserved"])
    m, w, c = [frames[name] for name in ("metrics", "windows", "confusion")]
    cases = {f"{x['animal']}-{x['day']:02d}-{x['epoch']:02d}" for x in p["cases"]}
    check("frozen_cases", set(m.case) == cases and len(m) == len(cases)*len(p["variants"]))
    check("no_duplicate_windows", not w.duplicated(["case", "variant", "start_time_s"]).any())
    check("fixed_window_duration", np.allclose(w.end_time_s-w.start_time_s, p["time_bin_s"], rtol=0, atol=1e-9))
    check("support_consistent", ((w.n_spikes >= w.n_active_tetrodes) & (w.n_active_tetrodes >= 0)).all())
    for row in m.itertuples(index=False):
        key = f"{row.case}/{row.variant}"
        bins = w[(w.case == row.case) & (w.variant == row.variant)]
        reported = c[(c.case == row.case) & (c.variant == row.variant)]
        counts = np.array([[int(((bins.true_arm == a) & (bins.predicted_arm == b)).sum()) for b in (0, 1)] for a in (0, 1)])
        recall = np.diag(counts)/counts.sum(axis=1)
        check(key+"/counts", len(bins) == row.n_windows and (counts.sum(axis=1) <= p["max_windows_per_arm"]).all())
        check(key+"/chronology", (bins.start_time_s >= row.train_end_s).all())
        check(key+"/confusion", all(int(reported[(reported.true_arm == a) & (reported.predicted_arm == b)].n_windows.iloc[0]) == counts[a, b] for a in (0, 1) for b in (0, 1)))
        check(key+"/accuracy", np.allclose([row.arm0_recall, row.arm1_recall, row.balanced_accuracy], [*recall, recall.mean()]))
        threshold = recall.mean() >= p["min_run_balanced_accuracy"] and recall.min() >= p["min_run_arm_recall"]
        check(key+"/unchanged_threshold", row.diagnostic_threshold_met == threshold and not row.biological_cohort_promoted)
        check(key+"/silent_windows", row.zero_spike_windows == (bins.n_spikes == 0).sum())
        check(key+"/support_medians", np.allclose([row.median_spikes, row.median_active_tetrodes], [bins.n_spikes.median(), bins.n_active_tetrodes.median()]))
        baseline = w[(w.case == row.case) & (w.variant == "frozen_geodesic")]
        check(key+"/same_test_windows", np.array_equal(bins.start_time_s.to_numpy(), baseline.start_time_s.to_numpy()) and np.array_equal(bins.true_arm.to_numpy(), baseline.true_arm.to_numpy()))
    for row in frames["encoder_selection"].itertuples(index=False):
        candidates = frames["encoder_candidates"]
        group = candidates[(candidates.animal == row.animal) & (candidates.day == row.day) & (candidates.target_epoch == row.target_epoch)]
        matches = group[group.context_matches]
        expected = int(matches.source_epoch.max()) if len(matches) else None
        actual = None if pd.isna(row.latest_matching_epoch) else int(row.latest_matching_epoch)
        check(row.session+"/latest_matching", actual == expected and (group.source_epoch < row.target_epoch).all())
        check(row.session+"/not_promoted", not row.decoder_qc_evaluated and not row.biological_cohort_promoted)
    parity = frames["parity"]
    check("parity_tolerance", len(parity) == len(cases) and (parity.max_posterior_l1_error <= p["parity_posterior_l1_tolerance"]).all())
    provenance = manifest["provenance"]
    for key, path in provenance["input_file_paths"].items():
        check("input_hash/"+key, Path(path).is_file() and file_hash(path) == provenance["input_file_sha256"][key])
    check("producer_traceable", provenance["code_commit"] not in {"unavailable", "unknown_without_git"} and provenance["git_dirty"] is False)
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, required=True)
    args = parser.parse_args()
    checks = verify(args.audit_dir)
    pd.DataFrame(checks).to_csv(args.audit_dir/(PREFIX+"verification_checks.csv"), index=False)
    result = {"passed": all(x["passed"] for x in checks), "n_checks": len(checks), "n_passed": sum(x["passed"] for x in checks),
              "verified_at_utc": datetime.now(UTC).isoformat(), "scope": "independent table accounting and raw input rehash; not independent decoder refit"}
    (args.audit_dir/(PREFIX+"verification.json")).write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
