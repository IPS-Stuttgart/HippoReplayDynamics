"""Test fixed invalid-clock exclusions without admitting a neural cohort."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import pandas as pd

try:
    from scripts._provenance import build_script_provenance, file_sha256
    from scripts.audit_tanni_clock_source_reconciliation import pinned_converter, source_conversion
    from scripts.audit_tanni_replay_run_inputs import increasing, read_array, source_combiner
except ModuleNotFoundError:
    from _provenance import build_script_provenance, file_sha256
    from audit_tanni_clock_source_reconciliation import pinned_converter, source_conversion
    from audit_tanni_replay_run_inputs import increasing, read_array, source_combiner


def native_nearest(times, queries):
    """Match upstream argmin(abs(...)), returning unchanged original indices."""
    times, queries = np.asarray(times, float), np.asarray(queries, float)
    if times.ndim != 1 or not len(times) or not np.isfinite(times).all() or not np.isfinite(queries).all():
        raise ValueError("Invalid native nearest lookup")
    order = np.argsort(times, kind="stable")
    values, first = np.unique(times[order], return_index=True)
    original = order[first]
    right = np.minimum(np.searchsorted(values, queries), len(values) - 1)
    left = np.maximum(right - 1, 0)
    dl, dr = np.abs(queries - values[left]), np.abs(queries - values[right])
    use_left = (dl < dr) | ((dl == dr) & (original[left] < original[right]))
    return original[np.where(use_left, left, right)]


def merge_intervals(intervals):
    if not intervals:
        return np.empty((0, 2), float)
    ordered = sorted((float(a), float(b)) for a, b in intervals)
    if any(not np.isfinite([a, b]).all() or a > b for a, b in ordered):
        raise ValueError("Invalid exclusion interval")
    result = [list(ordered[0])]
    for a, b in ordered[1:]:
        if a <= result[-1][1]:
            result[-1][1] = max(result[-1][1], b)
        else:
            result.append([a, b])
    return np.asarray(result)


def excluded(times, intervals):
    times = np.asarray(times, float)
    mask = np.zeros(times.shape, bool)
    for a, b in intervals:
        mask |= (times >= a) & (times <= b)
    return mask


def pulse_exclusions(oe, camera, converted, nearest, p):
    oe, camera = np.asarray(oe, float), np.asarray(camera, float)
    guard = p["pulse_guard_count"]
    padding = p["acquisition_guard_s"]
    if isinstance(guard, bool) or not isinstance(guard, int) or guard < 0 or not np.isfinite(padding) or padding < 0:
        raise ValueError("Invalid fixed guard")
    intervals, defects = [], []
    pulse_bad = np.zeros(len(camera), bool)
    for index in np.flatnonzero(np.diff(camera) < 0):
        lo, hi = max(0, index - guard), min(len(camera) - 1, index + 1 + guard)
        pulse_bad[lo:hi + 1] = True
        frame_times = converted[(nearest >= lo) & (nearest <= hi)]
        a = min(oe[lo], frame_times.min()) if len(frame_times) else oe[lo]
        b = max(oe[hi], frame_times.max()) if len(frame_times) else oe[hi]
        intervals.append((a - padding, b + padding))
        defects.append({"backward_pulse_index": int(index), "left_pulse_value": camera[index],
                        "right_pulse_value": camera[index + 1], "guard_first_pulse_index": lo,
                        "guard_last_pulse_index": hi, "exclusion_start_s": a - padding,
                        "exclusion_end_s": b + padding})
    return merge_intervals(intervals), pulse_bad[nearest], defects


def rebuild(cameras, settings, arena, combine, crop, rate, intervals, frame_good,
            maximum_distance, lookup=native_nearest):
    names = sorted(cameras)
    if not names or not np.isfinite(rate) or rate <= 0 or not np.isfinite(maximum_distance) or maximum_distance <= 0:
        raise ValueError("Invalid reconstruction settings")
    if len(names) == 1:
        full = crop(cameras[names[0]].copy(), arena, max_error=10)
        indices = lookup(cameras[names[0]][:, 0], full[:, 0])
        good = frame_good[names[0]][indices] & ~excluded(full[:, 0], intervals)
        safe = full.copy()
        safe[~good, 1:] = np.nan
        return full, safe, good
    start = min(cameras[name][0, 0] for name in names)
    end = max(cameras[name][-1, 0] for name in names)
    grid = np.arange(start, end, 1 / rate)
    nearest = {name: lookup(cameras[name][:, 0], grid) for name in names}
    good = ~excluded(grid, intervals)
    for name in names:
        good &= frame_good[name][nearest[name]]
        good &= np.abs(cameras[name][nearest[name], 0] - grid) <= maximum_distance
    full_rows, safe = [], np.column_stack((grid, np.full((len(grid), 4), np.nan)))
    previous_full, previous_safe = None, None
    for i, t in enumerate(grid):
        positions = [cameras[name][nearest[name][i], 1:5] for name in names]
        point = combine(positions, previous_full, names, settings, arena)
        previous_full = point
        if point is not None:
            full_rows.append(np.r_[t, point])
        if not good[i]:
            previous_safe = None
            continue
        point = combine(positions, previous_safe, names, settings, arena)
        previous_safe = point
        if point is not None:
            safe[i, 1:] = point
    if not full_rows:
        raise ValueError("Published source reconstruction has no positions")
    return crop(np.asarray(full_rows), arena, max_error=10), safe, good


def row_match(processed, rebuilt, lookup, time_tolerance, coordinate_tolerance):
    if not len(rebuilt):
        raise ValueError("Empty source reconstruction")
    index = lookup(rebuilt[:, 0], processed[:, 0])
    time_error = np.abs(processed[:, 0] - rebuilt[index, 0])
    a, b = processed[:, 1:3], rebuilt[index, 1:3]
    finite = np.isfinite(a) & np.isfinite(b)
    error = np.where(finite, np.abs(a - b), 0.)
    matches = (time_error <= time_tolerance) & (np.isnan(a) == np.isnan(b)).all(axis=1)
    matches &= ~np.any(np.isinf(a) | np.isinf(b), axis=1)
    matches &= (error <= coordinate_tolerance).all(axis=1)
    return matches, time_error, error, index


def mask_runs(mask, timestamps):
    edges = np.diff(np.r_[False, np.asarray(mask, bool), False].astype(int))
    return [{"first_processed_row": int(a), "last_processed_row": int(b - 1),
             "excluded_rows": int(b - a), "first_timestamp_s": timestamps[a],
             "last_timestamp_s": timestamps[b - 1]}
            for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True)]


def inspect_file(path, parent, p, combine, crop, nearest, converter, lookup=native_nearest,
                 exclusion_function=pulse_exclusions):
    inventory, camera_rows, defect_rows = [], [], []
    with h5py.File(path, "r") as h:
        recordings = h["acquisition/timeseries"]
        if len(recordings) != 1:
            raise ValueError("Ambiguous recording epoch")
        prefix = f"acquisition/timeseries/{next(iter(recordings))}"
        tracking = h[f"{prefix}/tracking"]
        processed = read_array(h, f"{prefix}/tracking/ProcessedPos", inventory)
        if processed.ndim != 2 or processed.shape[1] < 5:
            raise ValueError("Unknown ProcessedPos columns")
        increasing(processed[:, 0], "deposited position chronology")
        arena = read_array(h, "general/data_collection/Settings/General/arena_size", inventory)
        cameras, frame_good, interval_list = {}, {}, []
        settings = {"General": {}, "CameraSpecific": {}}
        oe = None
        for name in sorted(k for k in tracking if k.isdigit()):
            group = tracking[name]
            if isinstance(group, h5py.Dataset):
                cameras[name] = read_array(h, group.name, inventory)
                increasing(cameras[name][:, 0], "source-declared native tracking")
                frame_good[name] = np.ones(len(cameras[name]), bool)
                camera_rows.append({"camera_id": name, "raw_backward_steps": 0,
                                    "excluded_frames": 0, "frame_count": len(cameras[name])})
                continue
            if oe is None:
                times = read_array(h, f"{prefix}/events/ttl1/timestamps", inventory)
                channels = read_array(h, f"{prefix}/events/ttl1/data", inventory)
                if channels.shape != times.shape:
                    raise ValueError("TTL channel/time dimensions differ")
                oe = increasing(times[channels == parent["global_clock_rising_channel"]], "acquisition pulses")
            pulses = read_array(h, f"{group.name}/GlobalClock_timestamps", inventory)
            frames = read_array(h, f"{group.name}/OnlineTrackerData_timestamps", inventory)
            xy = read_array(h, f"{group.name}/OnlineTrackerData", inventory)
            if xy.ndim != 2 or xy.shape[1] < 4 or len(xy) != len(frames):
                raise ValueError("Camera frame/position dimensions differ")
            increasing(frames, "all raw frame times")
            keep = ~np.all(np.isnan(xy), axis=1)
            converted, stats, indices = source_conversion(oe, pulses, frames[keep],
                parent["camera_clock_divider"], nearest, converter)
            intervals, nearest_bad, defects = exclusion_function(oe, pulses, converted, indices, p)
            good = ~nearest_bad & ~excluded(converted, intervals)
            increasing(converted[good], "unexcluded converted chronology")
            cameras[name] = np.column_stack((converted, xy[keep]))
            frame_good[name] = good
            interval_list.extend(intervals.tolist())
            camera_rows.append({"camera_id": name, **stats, "excluded_frames": int((~good).sum())})
            defect_rows.extend([{"camera_id": name, **row} for row in defects])
        if len(cameras) > 1:
            base = "general/data_collection/Settings/CameraSettings"
            settings["General"]["camera_transfer_radius"] = float(read_array(
                h, f"{base}/General/camera_transfer_radius", inventory))
            for name in cameras:
                settings["CameraSpecific"][name] = {"location_xy": read_array(
                    h, f"{base}/CameraSpecific/{name}/location_xy", inventory)}
        intervals = merge_intervals(interval_list)
        full, safe, good_grid = rebuild(cameras, settings, arena, combine, crop,
            parent["multicamera_processed_sampling_rate_hz"], intervals, frame_good,
            p["maximum_camera_sample_distance_s"], lookup)
        full_match, full_time, full_error, _ = row_match(processed, full, lookup,
            parent["clock_tolerance_s"], parent["coordinate_tolerance_cm"])
        matches, time_error, coord_error, safe_index = row_match(processed, safe, lookup,
            parent["clock_tolerance_s"], parent["coordinate_tolerance_cm"])
        # Exclusions depend on clock/support only, never on coordinate agreement.
        bad_clock = excluded(processed[:, 0], intervals)
        unsupported = ~good_grid[safe_index]
        excluded_rows = bad_clock | unsupported
        retained = ~excluded_rows
        passed = bool(retained.any() and matches[retained].all() and full_match[retained].all())
        result = {"status": "retained_tracking_reconciled_diagnostic_only" if passed else "retained_tracking_unresolved",
                  "processed_rows": len(processed), "retained_rows": int(retained.sum()),
                  "excluded_rows": int(excluded_rows.sum()), "clock_excluded_rows": int(bad_clock.sum()),
                  "unsupported_camera_rows": int(unsupported.sum()), "merged_invalid_intervals": len(intervals),
                  "retained_mismatched_rows": int((~matches[retained]).sum()),
                  "retained_published_mismatched_rows": int((~full_match[retained]).sum()),
                  "full_published_mismatched_rows": int((~full_match).sum()),
                  "maximum_retained_time_error_s": float(time_error[retained].max()) if retained.any() else None,
                  "maximum_retained_coordinate_error_cm": float(coord_error[retained].max()) if retained.any() else None,
                  "maximum_full_time_error_s": float(full_time.max()),
                  "maximum_full_coordinate_error_cm": float(full_error.max()),
                  "tracking_exclusion_reconciled": passed, "eligible_for_neural_analysis": False}
    return result, camera_rows, defect_rows, mask_runs(excluded_rows, processed[:, 0]), inventory


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-audit", required=True, type=Path)
    parser.add_argument("--acquisition-source", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    p = json.loads(args.protocol.read_text())
    parent_protocol_path = Path(p["parent_protocol"])
    parent = json.loads(parent_protocol_path.read_text())
    parent_manifest = json.loads((args.parent_audit / "manifest.json").read_text())
    inventory_path = args.parent_audit / "session_inventory.csv"
    if file_sha256(inventory_path) != parent_manifest["outputs_sha256"]["session_inventory.csv"]:
        raise ValueError("Parent source inventory changed")
    inputs = {"protocol": args.protocol, "parent_protocol": parent_protocol_path,
              "parent_manifest": args.parent_audit / "manifest.json", "parent_session_inventory": inventory_path,
              "acquisition_source": args.acquisition_source}
    provenance = build_script_provenance(input_paths=inputs)
    if provenance["git_dirty"] or provenance["code_commit"] == "unavailable":
        raise ValueError("Clean committed diagnostic required")
    combine, crop, geometry_hash = source_combiner(args.acquisition_source, parent["acquisition_source_commit"])
    nearest, converter, clock_hashes = pinned_converter(args.acquisition_source, parent["acquisition_source_commit"])
    original = pd.read_csv(inventory_path)
    if original.empty or original.session.duplicated().any():
        raise ValueError("Empty or duplicate parent session identity")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    sessions, cameras, defects, masks, arrays = [], [], [], [], []
    for row in original.itertuples():
        path = Path(row.path)
        identity = {"animal": row.animal, "session": row.session, "path": str(path), "parent_status": row.status,
                    "source_size_bytes": path.stat().st_size, "source_mtime_ns": path.stat().st_mtime_ns}
        try:
            if path.stat().st_size != row.size_bytes or path.stat().st_mtime_ns != row.mtime_ns:
                raise ValueError("Native source changed since parent audit")
            result, cr, dr, mr, ar = inspect_file(path, parent, p, combine, crop, nearest, converter)
            for collection, rows in ((cameras, cr), (defects, dr), (masks, mr), (arrays, ar)):
                collection.extend([{**identity, **record} for record in rows])
        except (ValueError, KeyError, OSError) as exc:
            result = {"status": "unresolved_source_inputs", "error": f"{type(exc).__name__}: {exc}",
                      "tracking_exclusion_reconciled": False, "eligible_for_neural_analysis": False}
        sessions.append({**identity, **result})
        print(f"INTERVAL AUDIT {row.session}: {result['status']}", flush=True)
        (args.output_dir / "progress.json").write_text(json.dumps({"completed": len(sessions), "total": len(original)}) + "\n")
    outputs = {"sessions.csv": sessions, "cameras.csv": cameras, "defects.csv": defects,
               "excluded_processed_row_runs.csv": masks, "consumed_arrays.csv": arrays}
    for name, rows in outputs.items():
        pd.DataFrame(rows).to_csv(args.output_dir / name, index=False)
    manifest = {**provenance, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "source_function_sha256": {**clock_hashes, "TrackingDataProcessing.py": geometry_hash},
                "recordings": len(sessions), "reconciled_recordings": sum(r["tracking_exclusion_reconciled"] for r in sessions),
                "diagnostic_only": True, "timestamps_modified": False, "cohort_changed": False,
                "association_fit": False, "goal_complete": False, "hardware_alignment_verified": False,
                "environment": {"python": sys.version, "numpy": np.__version__, "pandas": pd.__version__, "h5py": h5py.__version__},
                "claim_boundary": p["claim_boundary"],
                "outputs_sha256": {name: file_sha256(args.output_dir / name) for name in outputs}}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
