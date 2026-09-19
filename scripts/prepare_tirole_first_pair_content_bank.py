"""Freeze a separate first-pair diagnostic bank without promoting the primary cohort."""

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hipporeplayimm.tirole_ripple import load_supplied_ripple, window_ripple_stats
from hipporeplayimm.tirole_two_track import file_sha256, fit_maps, load_session
from hipporeplayimm.two_track_content import event_bin_counts, split_populations
from scripts.audit_tirole_first_pair_preflight import SESSION, first_pair_view, verify_inputs
from scripts.prepare_tirole_content_bank import detector_candidates

COHORT = "first_pair_source_scope_diagnostic_only"


def validate_preflight(folder, verification):
    m = json.loads((folder / "manifest.json").read_text())
    v = json.loads(verification.read_text())
    if (
        m["session"] != SESSION
        or m["status"] != "complete"
        or m["git_dirty"]
        or not m["RUN_gate_passed"]
        or m["primary_cohort_promoted"]
        or m["scientific_replay_scores_computed"]
        or m["scope"] != "first_two_epochs_before_third_diagnostic"
        or v["status"] != "pass"
        or not v["RUN_gate_passed"]
        or v["primary_cohort_promoted"]
        or v["source_manifest_sha256"] != file_sha256(folder / "manifest.json")
        or v["checks"] != {"map_arrays": 12, "crossvalidation_rows": 10, "source_epochs": 4, "rest_durations": 2}
        or not 0 <= v["max_map_array_absolute_error"] < 1e-8
        or v["lfp_clock"]["clock_offset_applied_s"] != 0
    ):
        raise ValueError("verified first-pair RUN diagnostic required")
    for name, digest in m["output_sha256"].items():
        if Path(name).name != name or file_sha256(folder / name) != digest:
            raise ValueError("changed or unsafe preflight product")
    return m, v


def run(root, author, preflight, verification, output):
    if output.exists() and any(output.iterdir()):
        raise ValueError("new or empty diagnostic bank required")
    repo = Path(__file__).resolve().parents[1]
    git = lambda *args: subprocess.check_output(["git", *args], cwd=repo, text=True).strip()
    if git("status", "--porcelain"):
        raise ValueError("freeze clean code before preparing bank")
    commit = git("rev-parse", "HEAD")
    config_path = repo / "docs/two_track_content_configuration.json"
    config = json.loads(config_path.read_text())
    if SESSION in config["primary_cohort"] or SESSION not in config["quarantined_sessions"]:
        raise ValueError("original primary cohort/quarantine must remain unchanged")
    verify_inputs(root, author)
    pm, vm = validate_preflight(preflight, verification)
    data, intervals, cutoff = first_pair_view(load_session(root, SESSION))
    maps = fit_maps(data)
    with np.load(preflight / "first_pair_RUN_maps.npz") as expected:
        for k, v in maps.items():
            np.testing.assert_array_equal(v, expected[k], err_msg=k)
    detector, splits = split_populations(maps["common_units"], SESSION, config["seed"])
    events, detector_meta = detector_candidates(data, detector)
    lfp = root / f"{SESSION}_extracted_CSC.mat"
    lfp_sha = file_sha256(lfp)
    if lfp_sha != vm["lfp_input"]["sha256"]:
        raise ValueError("LFP differs from verified clock input")
    ripple = load_supplied_ripple(lfp, float(data.times[0]) - 1, float(data.times[-1]) + 1)
    overlap = max(0.0, min(ripple["full_end_s"], data.times[-1]) - max(ripple["full_start_s"], data.times[0])) / (data.times[-1] - data.times[0])
    if overlap < 0.99:
        raise ValueError("LFP overlap below unchanged 99% gate; no offset permitted")
    counts, offsets = [], [0]
    for e in events:
        if not data.times[0] <= e["start_s"] < e["end_s"] < cutoff:
            raise ValueError("candidate outside first-pair scope")
        if not (e["end_s"] <= intervals[0][0] or e["start_s"] >= intervals[1][1]):
            raise ValueError("candidate overlaps RUN/inter-track interval")
        e.update(window_ripple_stats(ripple, e["start_s"], e["end_s"]))
        c = event_bin_counts(data, e["start_s"], e["end_s"])
        e.update(n_time_bins=len(c), decoded_duration_s=len(c) * 0.02, primary_ripple_candidate=bool(e["ripple_supported"] and e["ripple_peak_z"] >= 3))
        counts.append(c)
        offsets.append(offsets[-1] + len(c))
    if not events:
        raise ValueError("no detector-only rest candidates")
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(events).to_csv(output / "candidate_events.csv", index=False)
    np.savez_compressed(output / "event_counts.npz", counts=np.concatenate(counts), offsets=offsets, unit_ids=data.unit_ids)
    np.savez_compressed(output / "RUN_maps.npz", **maps, unit_ids=data.unit_ids)
    parts = {
        "detector": detector.tolist(),
        "splits": [{"inference": a.tolist(), "evaluation": b.tolist()} for a, b in splits],
        "seed": config["seed"],
        "indexing": "zero_based_session_unit_index",
    }
    (output / "partitions.json").write_text(json.dumps(parts, indent=2) + "\n")
    if git("rev-parse", "HEAD") != commit or git("status", "--porcelain"):
        raise ValueError("code changed during bank preparation")
    info = {
        "session": SESSION,
        "animal": "RAT2",
        "strict_RUN_preflight_passed": True,
        "cohort_stratum": COHORT,
        "primary_cohort_promoted": False,
        "session_specific_reexposure_flag_available": False,
        "analysis_scope": pm["scope"],
        "source_epoch_indices_one_based": [1, 2],
        "source_time_end_exclusive_s": cutoff,
        "source_track_intervals_s": intervals,
        "preflight_sha256": file_sha256(preflight / "manifest.json"),
        "scope_verification_sha256": file_sha256(verification),
        "configuration_sha256": file_sha256(config_path),
        "lfp_sha256": lfp_sha,
        "raw_input_sha256": {p.name: file_sha256(p) for p in [root / f"{SESSION}_extracted_clusters.mat", root / f"{SESSION}_extracted_position.mat"]},
        "n_candidates": len(events),
        "n_ripple_candidates": sum(e["primary_ripple_candidate"] for e in events),
        "n_detector_units": len(detector),
        "n_common_units": len(maps["common_units"]),
        "detector": detector_meta,
        "ripple_metadata": {k: v for k, v in ripple.items() if k not in {"times", "zscore"}},
        "LFP_epoch_overlap_fraction": float(overlap),
        "clock_offset_applied_s": 0,
        "code_commit": commit,
        "git_dirty": False,
        "command_line": sys.argv,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "replay_sequence_or_content_scored": False,
        "outputs_sha256": {p.name: file_sha256(p) for p in output.iterdir() if p.is_file()},
    }
    (output / "manifest.json").write_text(json.dumps(info, indent=2) + "\n")
    print(
        json.dumps({k: info[k] for k in ["session", "cohort_stratum", "n_candidates", "n_ripple_candidates", "n_detector_units", "n_common_units", "LFP_epoch_overlap_fraction"]}),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", required=True, type=Path)
    p.add_argument("--author-code", required=True, type=Path)
    p.add_argument("--preflight-dir", required=True, type=Path)
    p.add_argument("--verification", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    a = p.parse_args()
    run(a.dataset_root, a.author_code, a.preflight_dir, a.verification, a.output_dir)
