"""Independently check raw-index lookups and bounds; geometry is shared source."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

try:
    from scripts._provenance import build_script_provenance, file_sha256
    from scripts.audit_tanni_clock_interval_exclusion import inspect_file
    from scripts.audit_tanni_replay_run_inputs import array_digest, source_combiner
    from scripts.verify_tanni_clock_source_reconciliation import reference_nearest, reference_convert
except ModuleNotFoundError:
    from _provenance import build_script_provenance, file_sha256
    from audit_tanni_clock_interval_exclusion import inspect_file
    from audit_tanni_replay_run_inputs import array_digest, source_combiner
    from verify_tanni_clock_source_reconciliation import reference_nearest, reference_convert


def reference_frame_nearest(times, queries):
    times, queries = np.asarray(times, float), np.asarray(queries, float)
    if times.ndim != 1 or not len(times) or not np.isfinite(times).all() or not np.isfinite(queries).all():
        raise ValueError("Invalid native frame lookup")
    boundaries = np.r_[0, np.flatnonzero(np.diff(times) < 0) + 1, len(times)]
    distance, best = np.full(len(queries), np.inf), np.full(len(queries), len(times), int)
    for a, b in zip(boundaries[:-1], boundaries[1:], strict=True):
        values = times[a:b]
        right = np.minimum(np.searchsorted(values, queries, side="left"), len(values) - 1)
        left = np.maximum(right - 1, 0)
        left = np.searchsorted(values, values[left], side="left")
        for candidate in (a + left, a + right):
            d = np.abs(times[candidate] - queries)
            update = (d < distance) | ((d == distance) & (candidate < best))
            distance[update], best[update] = d[update], candidate[update]
    return best


def reference_exclusions(oe, camera, converted, nearest, p):
    intervals, rows = [], []
    bad = np.zeros(len(converted), bool)
    for index in range(len(camera) - 1):
        if camera[index + 1] >= camera[index]:
            continue
        first = max(0, index - p["pulse_guard_count"])
        last = min(len(camera) - 1, index + 1 + p["pulse_guard_count"])
        frame_index = np.flatnonzero(np.asarray([first <= n <= last for n in nearest]))
        bad[frame_index] = True
        bounds = [oe[first], oe[last], *converted[frame_index]]
        start = min(bounds) - p["acquisition_guard_s"]
        end = max(bounds) + p["acquisition_guard_s"]
        intervals.append([start, end])
        rows.append({"backward_pulse_index": index, "left_pulse_value": camera[index],
                     "right_pulse_value": camera[index + 1], "guard_first_pulse_index": first,
                     "guard_last_pulse_index": last, "exclusion_start_s": start, "exclusion_end_s": end})
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end])
    return np.asarray(merged, float).reshape(-1, 2), bad, rows


def compare_rows(actual, expected, identity_columns):
    if not actual:
        assert expected.empty
        return
    observed, saved = pd.DataFrame(actual), expected.copy()
    if "camera_id" in identity_columns:
        observed["camera_id"] = observed.camera_id.astype(str)
        saved["camera_id"] = saved.camera_id.astype(str)
    observed = observed.sort_values(identity_columns).reset_index(drop=True)
    saved = saved.sort_values(identity_columns).reset_index(drop=True)
    assert len(observed) == len(saved)
    for column in observed.columns:
        a, b = observed[column].to_numpy(), saved[column].to_numpy()
        if a.dtype.kind in "iuf":
            np.testing.assert_allclose(a.astype(float), b.astype(float), rtol=0, atol=0, equal_nan=True)
        else:
            np.testing.assert_array_equal(a, b)


def verify(root):
    m = json.loads((root / "manifest.json").read_text())
    assert m["diagnostic_only"] and not m["timestamps_modified"] and not m["cohort_changed"]
    assert not m["association_fit"] and not m["goal_complete"] and not m["hardware_alignment_verified"]
    for name, digest in m["outputs_sha256"].items():
        assert file_sha256(root / name) == digest, name
    for name, path in m["input_file_paths"].items():
        assert file_sha256(path) == m["input_file_sha256"][name], name
    p = json.loads(Path(m["input_file_paths"]["protocol"]).read_text())
    parent = json.loads(Path(m["input_file_paths"]["parent_protocol"]).read_text())
    assert p["pulse_guard_count"] == 1 and p["acquisition_guard_s"] == .1
    assert p["maximum_camera_sample_distance_s"] == .1 and not p["neural_analysis_enabled"]
    source = Path(m["input_file_paths"]["acquisition_source"])
    for name, digest in m["source_function_sha256"].items():
        path = source / ("openEPhys_DACQ/TrackingDataProcessing.py" if name == "TrackingDataProcessing.py" else name)
        assert file_sha256(path) == digest, name
    combine, crop, _ = source_combiner(source, parent["acquisition_source_commit"])
    tables = {name: pd.read_csv(root / name, float_precision="round_trip") for name in
              ("sessions.csv", "cameras.csv", "defects.csv", "excluded_processed_row_runs.csv", "consumed_arrays.csv")}
    sessions = tables["sessions.csv"]
    original = pd.read_csv(m["input_file_paths"]["parent_session_inventory"])
    assert len(sessions) == len(original) == m["recordings"] > 0
    assert not sessions.session.duplicated().any() and set(sessions.session) == set(original.session)
    assert not sessions.eligible_for_neural_analysis.any()
    checked_arrays, checked_files, retained_rows, excluded_rows = 0, 0, 0, 0
    for row in sessions.itertuples():
        path = Path(row.path)
        assert path.stat().st_size == row.source_size_bytes and path.stat().st_mtime_ns == row.source_mtime_ns
        with h5py.File(path, "r") as h:
            table = tables["consumed_arrays.csv"]
            for record in table[table.session == row.session].itertuples():
                assert array_digest(h[record.hdf5_path][()]) == record.array_sha256
                checked_arrays += 1
        try:
            result, cameras, defects, masks, _ = inspect_file(path, parent, p, combine, crop,
                reference_nearest, reference_convert, reference_frame_nearest, reference_exclusions)
        except (ValueError, KeyError, OSError) as exc:
            assert row.status == "unresolved_source_inputs"
            assert row.error == f"{type(exc).__name__}: {exc}"
            continue
        for name, value in result.items():
            saved = getattr(row, name)
            if value is None:
                assert pd.isna(saved)
            else:
                assert value == saved, (row.session, name, value, saved)
        assert result["processed_rows"] == result["retained_rows"] + result["excluded_rows"]
        for rows, name, keys in ((cameras, "cameras.csv", ["camera_id"]),
                                 (defects, "defects.csv", ["camera_id", "backward_pulse_index"]),
                                 (masks, "excluded_processed_row_runs.csv", ["first_processed_row"])):
            saved = tables[name][tables[name].session == row.session]
            compare_rows(rows, saved, keys)
        retained_rows += result["retained_rows"]
        excluded_rows += result["excluded_rows"]
        checked_files += 1
        print(f"VERIFIED EXCLUSIONS {row.session}: {row.status}", flush=True)
    assert int(sessions.tracking_exclusion_reconciled.sum()) == m["reconciled_recordings"]
    return {"verified": True, "recordings_in_inventory": len(sessions), "reconstructions_checked": checked_files,
            "native_arrays_rehashed": checked_arrays, "reconciled_recordings": m["reconciled_recordings"],
            "retained_rows_in_reconstructed_files": retained_rows, "excluded_rows_in_reconstructed_files": excluded_rows,
            "independent_components": "Raw-order monotonic-run frame/pulse lookup and independently derived interval bounds",
            "shared_components": "File I/O, geometry driver and pinned author geometry; not independent geometry validation",
            "timestamps_modified": False, "cohort_changed": False, "association_fit": False,
            "hardware_alignment_verified": False, "neural_cohort_admitted": False, "goal_complete": False}


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
