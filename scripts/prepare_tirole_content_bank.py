"""Freeze detector-only PRE/POST candidates and independent cell partitions."""

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hipporeplayimm.independent_rejected_forecast import detect_candidates
from hipporeplayimm.tirole_ripple import load_supplied_ripple, window_ripple_stats
from hipporeplayimm.tirole_two_track import file_sha256, fit_maps, load_session, nearest_samples
from hipporeplayimm.two_track_content import event_bin_counts, split_populations


def detector_candidates(session, detector):
    dt = 0.001
    centers = session.times[0] + (np.arange(int(np.floor((session.times[-1] - session.times[0]) / dt))) + 0.5) * dt
    sample = nearest_samples(session.times, centers)
    safe = np.maximum(sample, 0)
    speed = np.interp(centers, session.times, session.speed)
    first = min(session.times[np.isfinite(x)].min() for x in session.positions)
    last = max(session.times[np.isfinite(x)].max() for x in session.positions)
    rest = (sample >= 0) & session.sleepbox[safe] & ((centers < first) | (centers > last))
    speed[~rest] = np.nan
    choose = np.isin(session.spike_units, detector)
    spikes = np.c_[session.spike_times[choose], session.spike_units[choose]]
    rows, meta = detect_candidates(spikes, detector, centers, speed)
    selected = []
    for row in rows:
        if not 0.1 - 1e-9 <= row["duration_s"] <= 0.75 + 1e-9 or row["detector_active_cells"] < 3:
            continue
        row["source_detector_event_id"] = row["event_id"]
        row["event_id"] = len(selected)
        row["epoch"] = "PRE" if row["end_s"] <= first else "POST"
        selected.append(row)
    meta["source_candidate_count"] = len(rows)
    meta["after_duration_active_cell_gate"] = len(selected)
    return selected, meta


def run(root, preflight, session_name, output, seed=20260918):
    if output.exists() and any(output.iterdir()):
        raise ValueError("new output directory required")
    qc = pd.read_csv(preflight / "two_track_session_qc.csv")
    found = qc.loc[qc.session == session_name]
    if len(found) != 1:
        raise ValueError("unique preflight session required")
    row = found.iloc[0]
    if row.duplicate_identity or row.n_tracks != 2 or row.n_common_units < 20 or row.minimum_track_occupancy_fraction < 0.8:
        raise ValueError("ambiguous identity or insufficient training maps")
    manifest = json.loads((preflight / "two_track_preflight_manifest.json").read_text())
    for src in manifest["input_files"]:
        if src["session"] == session_name and file_sha256(root / Path(src["path"]).name) != src["sha256"]:
            raise ValueError("spike/position data changed after preflight")
    lfp = root / f"{session_name}_extracted_CSC.mat"
    pages = json.loads((root / "release_file_metadata.json").read_text())
    expected = {r["path"]: r for p in pages for r in p["_embedded"]["stash:files"]}[lfp.name]
    lfp_sha = file_sha256(lfp)
    if lfp_sha != expected["digest"] or lfp.stat().st_size != expected["size"]:
        raise ValueError("LFP download incomplete or checksum mismatch")
    data = load_session(root, session_name)
    maps = fit_maps(data)
    detector, splits = split_populations(maps["common_units"], session_name, seed)
    events, detector_meta = detector_candidates(data, detector)
    ripple = load_supplied_ripple(lfp, float(data.times[0]) - 1, float(data.times[-1]) + 1)
    overlap = max(0.0, min(ripple["full_end_s"], data.times[-1]) - max(ripple["full_start_s"], data.times[0])) / (data.times[-1] - data.times[0])
    if overlap < 0.99:
        raise ValueError("LFP clock overlaps less than 99% of the spike/position epoch; no offset correction is permitted")
    counts = []
    offset = [0]
    for event in events:
        event.update(window_ripple_stats(ripple, event["start_s"], event["end_s"]))
        c = event_bin_counts(data, event["start_s"], event["end_s"])
        event["n_time_bins"] = len(c)
        event["decoded_duration_s"] = len(c) * 0.02
        event["primary_ripple_candidate"] = bool(event["ripple_supported"] and event["ripple_peak_z"] >= 3)
        counts.append(c)
        offset.append(offset[-1] + len(c))
    if not events:
        raise ValueError("no independent-detector rest events")
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(events).to_csv(output / "candidate_events.csv", index=False)
    np.savez_compressed(output / "event_counts.npz", counts=np.concatenate(counts), offsets=offset, unit_ids=data.unit_ids)
    np.savez_compressed(output / "RUN_maps.npz", **maps, unit_ids=data.unit_ids)
    parts = {
        "detector": detector.tolist(),
        "splits": [{"inference": a.tolist(), "evaluation": b.tolist()} for a, b in splits],
        "seed": seed,
        "indexing": "zero_based_session_unit_index",
    }
    (output / "partitions.json").write_text(json.dumps(parts, indent=2) + "\n")
    repo = Path(__file__).resolve().parents[1]
    git = lambda *args: subprocess.check_output(["git", *args], cwd=repo, text=True).strip()
    info = {
        "session": session_name,
        "animal": row.animal,
        "strict_RUN_preflight_passed": bool(row.preflight_passed),
        "cohort_stratum": "strict_RUN_pass" if row.preflight_passed else "RUN_quality_diagnostic_only",
        "preflight_sha256": file_sha256(preflight / "two_track_preflight_manifest.json"),
        "lfp_sha256": lfp_sha,
        "n_candidates": len(events),
        "n_ripple_candidates": sum(e["primary_ripple_candidate"] for e in events),
        "n_detector_units": len(detector),
        "n_common_units": len(maps["common_units"]),
        "detector": detector_meta,
        "ripple_metadata": {k: v for k, v in ripple.items() if k not in {"times", "zscore"}},
        "LFP_epoch_overlap_fraction": float(overlap),
        "code_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "command_line": sys.argv,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "replay_sequence_or_content_scored": False,
        "outputs_sha256": {p.name: file_sha256(p) for p in output.iterdir() if p.is_file()},
    }
    (output / "manifest.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps({k: info[k] for k in ("session", "cohort_stratum", "n_candidates", "n_ripple_candidates", "n_common_units", "n_detector_units")}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", required=True, type=Path)
    p.add_argument("--preflight-dir", required=True, type=Path)
    p.add_argument("--session", required=True)
    p.add_argument("--output-dir", required=True, type=Path)
    a = p.parse_args()
    run(a.dataset_root, a.preflight_dir, a.session, a.output_dir)
