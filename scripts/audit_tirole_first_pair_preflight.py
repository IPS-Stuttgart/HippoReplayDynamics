"""RUN-only, source-pinned first-two-epoch reconstruction of RAT2_SESS1.

This diagnostic does not promote a session or score any rest/replay event.
The source release has four epochs but no session-specific re-exposure flag.
"""

import argparse
import json
import subprocess
import sys
from dataclasses import replace
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hipporeplayimm.tirole_two_track import file_sha256, fit_maps, load_session, nearest_samples, run_crossvalidation
from scripts.audit_tirole_two_track_inputs import session_gate

SESSION = "RAT2_SESS1"
AUTHOR_COMMIT = "44ecf4275c2a7ca33dda6b67f11b4e854c3123e9"
RAW_HASHES = {
    "RAT2_SESS1_extracted_clusters.mat": "23b48eb0f4b2ce2e6658ba11523f64686d8099f45c2f95ec38005ecceb1dcde9",
    "RAT2_SESS1_extracted_position.mat": "e6f428868a82cd19be70f7477d316a6f59b187a51584a72de7b42b3ddac61f58",
}
AUTHOR_PATHS = ["README.md", "batch_remapping_pipeline.m", "Analysis/sort_replay_events.m", "Pipeline/batch_analysis.m"]


def git(repo, *args):
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def first_pair_view(session):
    """Keep epochs 1/2 and the clock before epoch 3; do not merge/relabel 3/4."""
    if session.n_tracks != 4:
        raise ValueError("exactly four source epochs required")
    intervals = []
    for x in session.positions:
        times = session.times[np.isfinite(x)]
        if not len(times):
            raise ValueError("empty source epoch")
        intervals.append((float(times.min()), float(times.max())))
    if any(a[1] >= b[0] for a, b in pairwise(intervals)):
        raise ValueError("source epochs must be strictly chronological and disjoint")
    cutoff = intervals[2][0]
    use = session.times < cutoff
    times = session.times[use].copy()
    keep_spikes = (session.spike_times >= times[0]) & (session.spike_times <= times[-1])
    spike_times = session.spike_times[keep_spikes].copy()
    view = replace(
        session,
        times=times,
        speed=session.speed[use].copy(),
        sleepbox=session.sleepbox[use].copy(),
        positions=session.positions[:2, use].copy(),
        lengths_cm=session.lengths_cm[:2].copy(),
        unit_ids=session.unit_ids.copy(),
        original_ids=session.original_ids.copy(),
        spike_times=spike_times,
        spike_units=session.spike_units[keep_spikes].copy(),
        spike_samples=nearest_samples(times, spike_times),
    )
    if view.n_tracks != 2 or (view.spike_samples < 0).any():
        raise ValueError("invalid reconstructed view")
    return view, intervals, cutoff


def summarize_view(view, maps, intervals, cutoff, cv):
    dt = float(np.median(np.diff(view.times)))
    pre = (view.times < intervals[0][0]) & view.sleepbox & (view.speed <= 5)
    post = (view.times > intervals[1][1]) & view.sleepbox & (view.speed <= 5)
    row = {
        "session": view.name,
        "animal": view.name.split("_")[0],
        "analysis_scope": "first_two_epochs_before_third_diagnostic",
        "duplicate_identity": False,
        "n_tracks": view.n_tracks,
        "n_units": len(view.unit_ids),
        "n_common_units": len(maps["common_units"]),
        "n_union_units": len(maps["union_units"]),
        "minimum_track_occupancy_fraction": float(maps["valid_bins"].mean(axis=1).min()),
        "pre_rest_s": float(pre.sum() * dt),
        "post_rest_s": float(post.sum() * dt),
        "nominal_post_start_s": intervals[1][1],
        "nominal_post_end_exclusive_s": cutoff,
        "nominal_post_duration_min": (cutoff - intervals[1][1]) / 60,
        "position_end_s": float(view.times[-1]),
        "n_cv_rows": len(cv),
        "source_reexposure_flag_available": False,
        "primary_cohort_promoted": False,
        "replay_events_scored": 0,
    }
    reasons = session_gate(row, cv)
    row.update(RUN_gate_passed=not reasons, exclusion_reason=";".join(reasons))
    return row


def verify_inputs(root, author):
    if git(author, "rev-parse", "HEAD") != AUTHOR_COMMIT or git(author, "status", "--porcelain"):
        raise ValueError("clean pinned original-study author code required")
    inputs = []
    for name, expected in RAW_HASHES.items():
        path = root / name
        if file_sha256(path) != expected:
            raise ValueError("pinned input mismatch: " + name)
    for path in [*(root / name for name in RAW_HASHES), root / "README.txt", root / "release_file_metadata.json", *(author / p for p in AUTHOR_PATHS)]:
        inputs.append({"path": str(path.resolve()), "size": path.stat().st_size, "sha256": file_sha256(path)})
    # Check the pinned release, not merely a locally supplied filename.
    pages = json.loads((root / "release_file_metadata.json").read_text())
    expected = {r["path"]: r for p in pages for r in p["_embedded"]["stash:files"]}
    for name, digest in RAW_HASHES.items():
        if expected[name]["digest"] != digest or expected[name]["size"] != (root / name).stat().st_size:
            raise ValueError("release identity mismatch")
        if any(r["path"] != name and r["digest"] == digest for r in expected.values()):
            raise ValueError("duplicate release identity cannot be rehabilitated")
    return inputs


def run(root, author, output):
    if output.exists() and any(output.iterdir()):
        raise ValueError("new or empty output directory required")
    repo = Path(__file__).resolve().parents[1]
    commit = git(repo, "rev-parse", "HEAD")
    if git(repo, "status", "--porcelain"):
        raise ValueError("freeze a clean code commit before real-data validation")
    inputs = verify_inputs(root, author)
    session = load_session(root, SESSION)
    view, intervals, cutoff = first_pair_view(session)
    maps = fit_maps(view)
    print("FIRST_PAIR_COMMON_UNITS", len(maps["common_units"]), flush=True)
    cv = run_crossvalidation(view)
    row = summarize_view(view, maps, intervals, cutoff, cv)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(output / "first_pair_session_qc.csv", index=False)
    pd.DataFrame(cv).to_csv(output / "first_pair_RUN_crossvalidation.csv", index=False)
    pd.DataFrame(
        [
            {
                "session": SESSION,
                "source_epoch": k + 1,
                "start_s": a,
                "end_s": b,
                "duration_s": b - a,
                "included_in_RUN_maps": k < 2,
                "context_identity": f"first_pair_track_{k + 1}" if k < 2 else "unresolved_not_used",
            }
            for k, (a, b) in enumerate(intervals)
        ]
    ).to_csv(output / "source_epoch_audit.csv", index=False)
    np.savez_compressed(output / "first_pair_RUN_maps.npz", **maps, unit_ids=view.unit_ids)
    gates = [
        {"gate": "original_study_code_pinned", "passed": True},
        {"gate": "later_exposures_excluded", "passed": bool(view.times[-1] < cutoff and view.spike_times.max() < cutoff)},
        {"gate": "unchanged_RUN_calibration", "passed": row["RUN_gate_passed"]},
        {"gate": "session_specific_reexposure_metadata_resolved", "passed": False},
        {"gate": "primary_cohort_promotion", "passed": False},
    ]
    pd.DataFrame(gates).to_csv(output / "first_pair_gate_summary.csv", index=False)
    lines = [
        "# First-pair RUN-only reconstruction",
        "",
        "This is a diagnostic, not an additional independent primary session or a replay result.",
        "",
        f"Source session: {SESSION}; original-study code: {AUTHOR_COMMIT}.",
        "The source batch pipeline reads a per-session re-exposure flag from folders(:,2). That lookup table is absent.",
        "When that flag is set, Analysis/sort_replay_events.m ends POST at epoch 3, after epoch 2.",
        "Only the chronological first two exposures are used here; epochs 3/4 are neither merged nor assigned context labels.",
        f"The nominal interval between epochs 2/3 is {row['nominal_post_duration_min']:.3f} min; paper Table 1 reports 125 min for this session.",
        "Timing agreement is corroboration, not a recovered session-specific flag. Published decoding accuracy is not substituted for held-out validation.",
        "",
        f"Common RUN units: {row['n_common_units']}; unchanged RUN gate: {row['RUN_gate_passed']}; failures: {row['exclusion_reason'] or 'none'}.",
        "All ten track/fold rows are reported. Thresholds and estimator are exactly the frozen preflight session_gate/run_crossvalidation functions.",
        "Later spikes, later positions, and later exposure maps cannot enter this reconstructed view.",
        "The original position clock, supplied cm/s speed and first-two track coordinates are retained unchanged; no clock correction is inferred.",
        "No rest-window counts, replay scores, ripple classifications or content outcomes are inspected.",
        "PRE/POST rest durations use the source sleepbox and immobility flags, not sleep-stage truth.",
        "",
        "## Decision",
        "",
        "Keep the existing primary cohort and all frozen biological results unchanged. A passing RUN gate alone does not resolve task metadata or validate replay content.",
        "If RUN fails, do not tune thresholds or try alternative epoch pairings against replay outcomes.",
        "Author clarification is still needed for the session-specific epoch mapping, duplicated release files and missing recording identities.",
        "",
        "Sources: https://elifesciences.org/articles/79031 ; https://doi.org/10.5061/dryad.ksn02v76h ;",
        f"https://github.com/bendor-lab/Elife_Tirole_Huelin_Gorriz_2022/tree/{AUTHOR_COMMIT}",
    ]
    (output / "first_pair_audit.md").write_text("\n".join(lines) + "\n")
    if git(repo, "rev-parse", "HEAD") != commit or git(repo, "status", "--porcelain"):
        raise ValueError("code changed during validation")
    for item in inputs:
        if file_sha256(item["path"]) != item["sha256"]:
            raise ValueError("input changed during validation")
    manifest = {
        "status": "complete",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "code_commit": commit,
        "git_dirty": False,
        "author_code_commit": AUTHOR_COMMIT,
        "source_doi": "10.5061/dryad.ksn02v76h",
        "source_version": 238435,
        "session": SESSION,
        "command_line": sys.argv,
        "working_directory": str(Path.cwd()),
        "input_files": inputs,
        "scientific_replay_scores_computed": False,
        "primary_cohort_promoted": False,
        "scope": row["analysis_scope"],
        "RUN_gate_passed": row["RUN_gate_passed"],
        "code_sha256": {
            p: file_sha256(repo / p) for p in ["scripts/audit_tirole_first_pair_preflight.py", "scripts/audit_tirole_two_track_inputs.py", "src/hipporeplayimm/tirole_two_track.py"]
        },
        "output_sha256": {p.name: file_sha256(p) for p in sorted(output.iterdir()) if p.is_file()},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(row), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--author-code", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    run(args.dataset_root, args.author_code, args.output_dir)
