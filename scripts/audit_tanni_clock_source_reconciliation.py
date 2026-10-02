"""Reproduce pinned camera conversion diagnostically; never repair or admit data.

Raw pulse identities and chronology defects remain explicit. The upstream nearest
pulse lookup can be reproduced even when camera pulses are backward, but agreement
with deposited positions alone does not establish a safe biological clock.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
import pandas as pd

try:
    from scripts._provenance import build_script_provenance, file_sha256
    from scripts.audit_tanni_replay_run_inputs import (
        increasing, read_array, reconcile_processed, reconstruct_tracking, source_combiner,
    )
except ModuleNotFoundError:
    from _provenance import build_script_provenance, file_sha256
    from audit_tanni_replay_run_inputs import (
        increasing, read_array, reconcile_processed, reconstruct_tracking, source_combiner,
    )


def pinned_converter(source, commit):
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    if actual != commit:
        raise ValueError("Pinned acquisition commit differs")
    specs = {"openEPhys_DACQ/HelperFunctions.py": "closest_argmin",
             "openEPhys_DACQ/NWBio.py": "estimate_open_ephys_timestamps_from_other_timestamps"}
    functions, hashes = [], {}
    for name, function in specs.items():
        path = source / name
        if subprocess.check_output(["git", "diff", "HEAD", "--", name], cwd=source, text=True):
            raise ValueError("Pinned clock conversion source was modified")
        selected = [node for node in ast.parse(path.read_text()).body
                    if isinstance(node, ast.FunctionDef) and node.name == function]
        if len(selected) != 1:
            raise ValueError("Missing or ambiguous pinned clock conversion function")
        functions.extend(selected)
        hashes[name] = file_sha256(path)
    namespace = {"np": np}
    exec(compile(ast.Module(body=functions, type_ignores=[]), "pinned_clock_functions", "exec"), namespace)
    return (namespace["closest_argmin"],
            namespace["estimate_open_ephys_timestamps_from_other_timestamps"], hashes)


def source_conversion(oe, camera, frames, divider, nearest_function, convert_function):
    oe = increasing(oe, "acquisition pulses")
    frames = increasing(frames, "camera frames")
    camera = np.asarray(camera, dtype=float)
    if camera.ndim != 1 or not np.isfinite(camera).all() or len(camera) < 2:
        raise ValueError("Invalid camera pulse vector")
    if len(oe) != len(camera):
        raise ValueError("Pulse counts differ; upstream truncation is prohibited")
    if len(np.unique(camera)) != len(camera):
        raise ValueError("Duplicate camera pulse identities are ambiguous")
    if not np.isfinite(divider) or divider <= 0:
        raise ValueError("Invalid camera clock scale")
    # Execute the exact published lookup, retaining the raw pulse order separately.
    nearest = nearest_function(frames, camera)
    converted = convert_function(oe, camera, frames, other_times_divider=divider)
    if not np.isfinite(converted).all():
        raise ValueError("Nonfinite converted frame times")
    defects = np.flatnonzero(np.diff(camera) < 0)
    implicated = np.unique(np.r_[defects, defects + 1])
    flags = np.isin(nearest, implicated)
    return converted, {"pulse_count": len(camera), "raw_backward_steps": len(defects),
                       "converted_backward_steps": int(np.sum(np.diff(converted) < 0)),
                       "converted_duplicate_steps": int(np.sum(np.diff(converted) == 0)),
                       "frame_count": len(frames), "frames_nearest_defective_pulse": int(flags.sum())}, nearest


def inspect_file(path, p, combine, crop, nearest, converter):
    inventory, clocks = [], []
    with h5py.File(path, "r") as h:
        recordings = h["acquisition/timeseries"]
        if len(recordings) != 1:
            raise ValueError("Ambiguous recording epoch")
        prefix = f"acquisition/timeseries/{next(iter(recordings))}"
        tracking = h[f"{prefix}/tracking"]
        processed = read_array(h, f"{prefix}/tracking/ProcessedPos", inventory)
        if processed.ndim != 2 or processed.shape[1] < 5:
            raise ValueError("Unknown ProcessedPos columns")
        arena = read_array(h, "general/data_collection/Settings/General/arena_size", inventory)
        cameras, settings = {}, {"General": {}, "CameraSpecific": {}}
        pulse_times = None
        for name in sorted(k for k in tracking if k.isdigit()):
            group = tracking[name]
            if isinstance(group, h5py.Dataset):
                data = read_array(h, group.name, inventory)
                cameras[name] = data
                clocks.append({"camera_id": name, "mode": "source_declared_acquisition_tracking",
                               "raw_backward_steps": 0, "frame_count": len(data)})
                continue
            if pulse_times is None:
                times = read_array(h, f"{prefix}/events/ttl1/timestamps", inventory)
                channels = read_array(h, f"{prefix}/events/ttl1/data", inventory)
                if channels.shape != times.shape:
                    raise ValueError("TTL channel/time dimensions differ")
                pulse_times = times[channels == p["global_clock_rising_channel"]]
            pulses = read_array(h, f"{group.name}/GlobalClock_timestamps", inventory)
            frames = read_array(h, f"{group.name}/OnlineTrackerData_timestamps", inventory)
            xy = read_array(h, f"{group.name}/OnlineTrackerData", inventory)
            if xy.ndim != 2 or xy.shape[1] < 4 or len(xy) != len(frames):
                raise ValueError("Camera frame/position dimensions differ")
            increasing(frames, "all raw frame times")
            keep = ~np.all(np.isnan(xy), axis=1)
            converted, stats, _ = source_conversion(pulse_times, pulses, frames[keep],
                                                    p["camera_clock_divider"], nearest, converter)
            clocks.append({"camera_id": name, "mode": "recorded_global_clock_source_reproduction", **stats})
            cameras[name] = np.column_stack((converted, xy[keep]))
        if len(cameras) > 1:
            base = "general/data_collection/Settings/CameraSettings"
            settings["General"]["camera_transfer_radius"] = float(read_array(
                h, f"{base}/General/camera_transfer_radius", inventory))
            for name in cameras:
                settings["CameraSpecific"][name] = {"location_xy": read_array(
                    h, f"{base}/CameraSpecific/{name}/location_xy", inventory)}
        try:
            rebuilt = reconstruct_tracking(cameras, settings, arena, combine, crop,
                                           p["multicamera_processed_sampling_rate_hz"])
            result = reconcile_processed(processed, rebuilt, p["clock_tolerance_s"], p["coordinate_tolerance_cm"])
            result["status"] = "source_reproduced_diagnostic_only" if result["source_clock_reconciliation_passed"] else "position_reproduction_mismatch"
        except ValueError as exc:
            result = {"status": "converted_chronology_unresolved", "error": str(exc),
                      "source_clock_reconciliation_passed": False, "processed_rows": len(processed)}
        result["raw_backward_steps"] = sum(row["raw_backward_steps"] for row in clocks)
        result["eligible_for_neural_analysis"] = False
    return result, clocks, inventory


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-audit", required=True, type=Path)
    parser.add_argument("--acquisition-source", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    p = json.loads(args.protocol.read_text())
    parent = json.loads((args.parent_audit / "manifest.json").read_text())
    inventory_path = args.parent_audit / "session_inventory.csv"
    if file_sha256(inventory_path) != parent["outputs_sha256"]["session_inventory.csv"]:
        raise ValueError("Parent source inventory changed")
    inputs = {"protocol": args.protocol, "parent_manifest": args.parent_audit / "manifest.json",
              "parent_session_inventory": inventory_path, "acquisition_source": args.acquisition_source}
    provenance = build_script_provenance(input_paths=inputs)
    if provenance["git_dirty"] or provenance["code_commit"] == "unavailable":
        raise ValueError("Clean committed diagnostic required")
    combine, crop, geometry_hash = source_combiner(args.acquisition_source, p["acquisition_source_commit"])
    nearest, converter, source_hashes = pinned_converter(args.acquisition_source, p["acquisition_source_commit"])
    args.output_dir.mkdir(parents=True, exist_ok=False)
    sessions, clocks, arrays = [], [], []
    original = pd.read_csv(inventory_path)
    if original.empty or original.session.duplicated().any():
        raise ValueError("Empty or duplicate parent session identity")
    for row in original.itertuples():
        path = Path(row.path)
        identity = {"animal": row.animal, "session": row.session, "path": str(path),
                    "parent_status": row.status, "source_size_bytes": path.stat().st_size,
                    "source_mtime_ns": path.stat().st_mtime_ns}
        try:
            if path.stat().st_size != row.size_bytes or path.stat().st_mtime_ns != row.mtime_ns:
                raise ValueError("Source identity changed since parent audit")
            result, cr, ar = inspect_file(path, p, combine, crop, nearest, converter)
            clocks.extend([{**identity, **r} for r in cr])
            arrays.extend([{**identity, **r} for r in ar])
        except (ValueError, KeyError, OSError) as exc:
            result = {"status": "unresolved_source_reproduction", "error": f"{type(exc).__name__}: {exc}",
                      "source_clock_reconciliation_passed": False, "eligible_for_neural_analysis": False}
        sessions.append({**identity, **result})
        print(f"SOURCE {row.session}: {result['status']}", flush=True)
        (args.output_dir / "progress.json").write_text(json.dumps({"completed": len(sessions), "total": len(original)}, indent=2) + "\n")
    outputs = {"sessions.csv": sessions, "cameras.csv": clocks, "consumed_arrays.csv": arrays}
    for name, rows in outputs.items():
        pd.DataFrame(rows).to_csv(args.output_dir / name, index=False)
    manifest = {**provenance, "diagnostic_only": True, "source_repair_applied": False,
                "cohort_changed": False, "association_fit": False, "goal_complete": False,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "environment": {"python": sys.version, "numpy": np.__version__, "pandas": pd.__version__, "h5py": h5py.__version__},
                "source_function_sha256": {**source_hashes, "TrackingDataProcessing.py": geometry_hash},
                "recordings": len(sessions), "position_reproduced_recordings": sum(r["source_clock_reconciliation_passed"] for r in sessions),
                "claim_boundary": "Source-code reproduction does not prove hardware alignment or safe intervals. No defective recording is admitted.",
                "outputs_sha256": {name: file_sha256(args.output_dir / name) for name in outputs}}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
