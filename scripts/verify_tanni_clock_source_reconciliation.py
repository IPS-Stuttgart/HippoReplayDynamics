"""Recompute clock lookup independently; geometry remains the pinned source routine."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

try:
    from scripts._provenance import build_script_provenance, file_sha256
    from scripts.audit_tanni_clock_source_reconciliation import inspect_file
    from scripts.audit_tanni_replay_run_inputs import source_combiner
except ModuleNotFoundError:
    from _provenance import build_script_provenance, file_sha256
    from audit_tanni_clock_source_reconciliation import inspect_file
    from audit_tanni_replay_run_inputs import source_combiner


def reference_nearest(frames, clock):
    """Search monotonic raw-order runs, without the producer's sorted lookup."""
    frames, clock = np.asarray(frames), np.asarray(clock)
    if clock.ndim != 1 or len(clock) < 2 or not np.isfinite(clock).all():
        raise ValueError("Invalid clock vector")
    if len(np.unique(clock)) != len(clock) or not np.isfinite(frames).all():
        raise ValueError("Duplicate clock value or nonfinite frame")
    boundaries = np.r_[0, np.flatnonzero(np.diff(clock) < 0) + 1, len(clock)]
    distance = np.full(len(frames), np.inf)
    best = np.zeros(len(frames), dtype=int)
    for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
        run = clock[start:end]
        right = np.minimum(np.searchsorted(run, frames), len(run) - 1)
        for offset in (np.maximum(right - 1, 0), right):
            index = start + offset
            d = np.abs(frames - clock[index])
            update = (d < distance) | ((d == distance) & (clock[index] > clock[best]))
            best[update] = index[update]
            distance[update] = d[update]
    return best


def reference_convert(oe, clock, frames, other_times_divider):
    index = reference_nearest(frames, clock)
    return np.asarray(oe)[index] + (np.asarray(frames) - np.asarray(clock)[index]) / other_times_divider


def verify(root):
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["diagnostic_only"] and not manifest["cohort_changed"] and not manifest["source_repair_applied"]
    assert not manifest["association_fit"] and not manifest["goal_complete"]
    for name, expected in manifest["outputs_sha256"].items():
        assert file_sha256(root / name) == expected, name
    for name, path in manifest["input_file_paths"].items():
        assert file_sha256(path) == manifest["input_file_sha256"][name], name
    p = json.loads(Path(manifest["input_file_paths"]["protocol"]).read_text())
    source = Path(manifest["input_file_paths"]["acquisition_source"])
    for name, digest in manifest["source_function_sha256"].items():
        path = source / ("openEPhys_DACQ/TrackingDataProcessing.py" if name == "TrackingDataProcessing.py" else name)
        assert file_sha256(path) == digest, name
    combine, crop, _ = source_combiner(source, p["acquisition_source_commit"])
    sessions = pd.read_csv(root / "sessions.csv", float_precision="round_trip")
    cameras = pd.read_csv(root / "cameras.csv", float_precision="round_trip")
    arrays = pd.read_csv(root / "consumed_arrays.csv")
    parent = pd.read_csv(manifest["input_file_paths"]["parent_session_inventory"])
    assert len(sessions) == len(parent) == manifest["recordings"] > 0
    assert not sessions.session.duplicated().any()
    assert set(sessions.session) == set(parent.session) and not sessions.eligible_for_neural_analysis.any()
    checked_arrays, checked_sessions, checked_cameras = 0, 0, 0
    for row in sessions.itertuples():
        path = Path(row.path)
        assert path.stat().st_size == row.source_size_bytes and path.stat().st_mtime_ns == row.source_mtime_ns
        with h5py.File(path, "r") as h:
            for field in arrays[arrays.session == row.session].itertuples():
                a = np.asarray(h[field.hdf5_path][()])
                digest = hashlib.sha256(json.dumps({"shape": a.shape, "dtype": a.dtype.str}, sort_keys=True).encode())
                digest.update(a.tobytes(order="C"))
                assert digest.hexdigest() == field.array_sha256
                checked_arrays += 1
        if row.status == "unresolved_source_reproduction":
            assert not row.source_clock_reconciliation_passed
            continue
        result, cr, _ = inspect_file(path, p, combine, crop, reference_nearest, reference_convert)
        assert result["status"] == row.status and result["source_clock_reconciliation_passed"] == row.source_clock_reconciliation_passed
        for key in ("processed_rows", "raw_backward_steps", "mismatched_processed_rows",
                    "maximum_time_error_s", "maximum_coordinate_error_cm"):
            if key in result:
                assert result[key] == getattr(row, key), (row.session, key)
        observed = cameras[cameras.session == row.session].set_index("camera_id")
        assert len(observed) == len(cr)
        for camera in cr:
            match = observed.loc[int(camera["camera_id"])]
            assert camera["mode"] == match["mode"]
            for key in camera.keys() - {"camera_id", "mode"}:
                assert camera[key] == match[key], (row.session, camera["camera_id"], key)
            checked_cameras += 1
        checked_sessions += 1
        print(f"VERIFIED CLOCK REPRODUCTION {row.session}: {row.status}", flush=True)
    return {"verified": True, "recordings_in_parent_inventory": len(parent), "native_arrays_rehashed": checked_arrays,
            "clock_and_position_reproductions_checked": checked_sessions, "camera_rows_checked": checked_cameras,
            "position_reproduced_recordings": int(sessions.source_clock_reconciliation_passed.sum()),
            "independent_clock_algorithm": "nearest lookup across original-order monotonic runs",
            "geometry_algorithm": "same pinned author functions; not an independent geometry implementation",
            "safe_intervals_verified": False, "hardware_alignment_verified": False,
            "source_repair_applied": False, "cohort_changed": False, "association_fit": False, "goal_complete": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("Use a new verification path")
    provenance = build_script_provenance(input_paths={"audit_manifest": args.audit_dir / "manifest.json"})
    if provenance["git_dirty"] or provenance["code_commit"] == "unavailable":
        raise ValueError("Clean committed verifier required")
    result = {**provenance, **verify(args.audit_dir)}
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
