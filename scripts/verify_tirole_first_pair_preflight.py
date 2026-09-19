"""Verify first-pair scope with a full-clock reconstruction; audit LFP clock only."""

import argparse
import json
import subprocess
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.io import loadmat

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hipporeplayimm.tirole_ripple import _text, _vector
from hipporeplayimm.tirole_two_track import file_sha256, fit_maps, load_session, run_crossvalidation
from scripts.audit_tirole_two_track_inputs import session_gate


def clock_check(times, sr, required_start, required_end):
    times = np.asarray(times, float)
    if times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all() or not np.isfinite(sr) or sr <= 0:
        raise ValueError("invalid clock")
    diff = np.diff(times)
    if not (diff > 0).all() or not np.isclose(np.median(diff), 1 / sr, rtol=0.02):
        raise ValueError("nonmonotonic clock or sample-rate mismatch")
    if not np.isfinite([required_start, required_end]).all() or required_end <= required_start:
        raise ValueError("invalid required interval")
    a = max(0, int(np.searchsorted(times, required_start)) - 1)
    b = min(len(times), int(np.searchsorted(times, required_end)) + 1)
    interval = times[a:b]
    gap = float(np.diff(interval).max()) if len(interval) > 1 else None
    complete = bool(times[0] <= required_start and times[-1] + 1 / sr >= required_end and gap is not None and gap <= 2.01 / sr)
    return {
        "sample_rate_hz": sr,
        "clock_start_s": float(times[0]),
        "clock_end_s": float(times[-1]),
        "n_samples": len(times),
        "median_sample_spacing_s": float(np.median(diff)),
        "required_start_s": required_start,
        "required_end_s": required_end,
        "max_gap_in_required_interval_s": gap,
        "required_interval_covered": complete,
        "clock_offset_applied_s": 0,
        "independent_spike_lfp_synchronization_established": False,
    }


def read_clock(path, required_start, required_end):
    """Read time/channel metadata from either MATLAB storage format."""
    if h5py.is_hdf5(path):
        with h5py.File(path, "r") as f:
            c = f["CSC"]
            labels = [_text(f[r]) for r in c["channel_label"][:].ravel()]
            if labels.count("best_ripple") != 1:
                raise ValueError("exactly one best_ripple channel required")
            k = labels.index("best_ripple")
            get = lambda name: f[c[name][:].ravel()[k]]
            if _text(get("time_scale")) != "seconds":
                raise ValueError("clock must be labelled seconds")
            times = _vector(get("CSCtime"))
            if get("ripple_zscore").size != len(times):
                raise ValueError("clock/signal length mismatch")
            result = clock_check(times, float(_vector(get("SR"))[0]), required_start, required_end)
            result.update(channel=int(_vector(get("channel"))[0]), channel_label=labels[k], filename=_text(get("filename")), storage_format="MATLAB_HDF5")
            return result
    c = loadmat(path, variable_names=["CSC"], simplify_cells=True)["CSC"]
    channels = c if isinstance(c, list) else [c]
    chosen = [x for x in channels if x["channel_label"] == "best_ripple"]
    if len(chosen) != 1:
        raise ValueError("exactly one best_ripple channel required")
    channel = chosen[0]
    if channel["time_scale"] != "seconds":
        raise ValueError("clock must be labelled seconds")
    times = np.asarray(channel["CSCtime"]).reshape(-1)
    if np.asarray(channel["ripple_zscore"]).size != len(times):
        raise ValueError("clock/signal length mismatch")
    result = clock_check(times, float(channel["SR"]), required_start, required_end)
    result.update(channel=int(channel["channel"]), channel_label=channel["channel_label"], filename=str(channel["filename"]), storage_format="MATLAB_classic")
    return result


def verify(source, dataset, output):
    if output.exists():
        raise ValueError("verification output already exists")
    repo = Path(__file__).resolve().parents[1]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).strip():
        raise ValueError("clean verification commit required")
    meta = json.loads((source / "manifest.json").read_text())
    if meta["status"] != "complete" or meta["git_dirty"] or meta["primary_cohort_promoted"] or meta["scientific_replay_scores_computed"]:
        raise ValueError("invalid diagnostic manifest")
    for name, digest in meta["output_sha256"].items():
        if Path(name).name != name or file_sha256(source / name) != digest:
            raise ValueError("output checksum mismatch")
    for entry in meta["input_files"]:
        if file_sha256(entry["path"]) != entry["sha256"]:
            raise ValueError("input checksum mismatch")
    data = load_session(dataset, meta["session"])
    assert data.n_tracks == 4
    # Independently retain the entire source clock/spike train, but expose only
    # the first two position arrays. RUN masking must make later data irrelevant.
    reference = replace(data, positions=data.positions[:2], lengths_cm=data.lengths_cm[:2])
    maps = fit_maps(reference)
    max_error = 0.0
    with np.load(source / "first_pair_RUN_maps.npz") as stored:
        assert set(stored.files) == {*maps, "unit_ids"}
        for key, expected in {**maps, "unit_ids": data.unit_ids}.items():
            actual = stored[key]
            np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
            if np.asarray(actual).size and actual.dtype.kind != "b":
                max_error = max(max_error, float(np.max(np.abs(actual - expected))))
    cv = run_crossvalidation(reference)
    recorded = pd.read_csv(source / "first_pair_RUN_crossvalidation.csv")
    pd.testing.assert_frame_equal(recorded, pd.DataFrame(cv), check_dtype=False, rtol=1e-10, atol=1e-10)
    assert len(cv) == 10
    epochs = [[float(data.times[np.isfinite(x)].min()), float(data.times[np.isfinite(x)].max())] for x in data.positions]
    audit = pd.read_csv(source / "source_epoch_audit.csv")
    np.testing.assert_allclose(audit[["start_s", "end_s"]], epochs, rtol=0, atol=1e-8)
    row = pd.read_csv(source / "first_pair_session_qc.csv").iloc[0].to_dict()
    assert not row["source_reexposure_flag_available"] and not row["primary_cohort_promoted"]
    assert row["replay_events_scored"] == 0 and not row["duplicate_identity"]
    assert row["n_tracks"] == 2 and row["n_common_units"] == len(maps["common_units"])
    assert row["RUN_gate_passed"] == (not session_gate(row, cv)) == meta["RUN_gate_passed"]
    cutoff = epochs[2][0]
    before = data.times < cutoff
    dt = float(np.median(np.diff(data.times)))
    for name, mask in [
        ("pre_rest_s", data.times < epochs[0][0]),
        ("post_rest_s", data.times > epochs[1][1]),
    ]:
        expected = float((mask & before & data.sleepbox & (data.speed <= 5)).sum() * dt)
        np.testing.assert_allclose(row[name], expected)
    np.testing.assert_allclose(row["position_end_s"], data.times[before][-1])
    np.testing.assert_allclose(row["nominal_post_duration_min"], (cutoff - epochs[1][1]) / 60)
    lfp_path = dataset / f"{meta['session']}_extracted_CSC.mat"
    pages = json.loads((dataset / "release_file_metadata.json").read_text())
    release_row = next(r for page in pages for r in page["_embedded"]["stash:files"] if r["path"] == lfp_path.name)
    lfp_sha = file_sha256(lfp_path)
    assert lfp_sha == release_row["digest"] and lfp_path.stat().st_size == release_row["size"]
    clock = read_clock(lfp_path, float(data.times[0]), cutoff)
    result = {
        "status": "pass",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "code_commit": commit,
        "source_manifest_sha256": file_sha256(source / "manifest.json"),
        "verifier_sha256": file_sha256(__file__),
        "RUN_gate_passed": bool(row["RUN_gate_passed"]),
        "primary_cohort_promoted": False,
        "replay_scores_computed": False,
        "verification_scope": "independent full-clock view construction; shared frozen RUN fit/decoder kernels",
        "checks": {"map_arrays": len(maps) + 1, "crossvalidation_rows": len(cv), "source_epochs": len(epochs), "rest_durations": 2},
        "max_map_array_absolute_error": max_error,
        "lfp_input": {"path": str(lfp_path), "sha256": lfp_sha},
        "lfp_clock": clock,
        "clock_claim_boundary": "Matching absolute clocks and interval coverage do not independently prove spike/LFP synchronization. No offset is inferred, no LFP amplitude or ripple-event outcome was inspected.",
    }
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip() != commit:
        raise ValueError("verification code changed")
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    verify(args.source_dir, args.dataset_root, args.output)
