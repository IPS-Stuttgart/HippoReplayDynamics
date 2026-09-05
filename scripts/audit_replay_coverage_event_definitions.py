#!/usr/bin/env python3
"""Verify event sources, baseline clocks, window support and overlap identities."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import h5py
import numpy as np
import pandas as pd
from scipy.io import loadmat

from hipporeplayimm.replay_coverage_data import array_sha256
from hipporeplayimm.replay_coverage_event_definitions import EventDefinitionConfig, tracking_speed
from scripts._provenance import build_script_provenance, file_sha256
from scripts.prepare_replay_coverage_event_definitions import load_ripple_envelope


def reference_episodes(times, z, fs, config):
    positive = np.flatnonzero(np.isfinite(z) & (z > config.ripple_onset_z))
    if not len(positive):
        return []
    breaks = np.flatnonzero(np.diff(positive) > 1)
    starts, stops = np.r_[positive[0], positive[breaks + 1]], np.r_[positive[breaks] + 1, positive[-1] + 1]
    rows = []
    a, b, episode_id = int(starts[0]), int(stops[0]), 0
    def finish(a, b, episode_id):
        peak = a + int(np.nanargmax(z[a:b]))
        if z[peak] >= config.ripple_peak_z:
            rows.append([episode_id, times[a], times[b - 1] + 1 / fs, times[peak], z[peak], a, b])
    for start, stop in zip(starts[1:], stops[1:], strict=True):
        if (start - b) / fs <= config.ripple_merge_gap_s + 1e-12 and np.isfinite(z[b:start]).all():
            b = int(stop)
        else:
            finish(a, b, episode_id)
            a, b, episode_id = int(start), int(stop), episode_id + 1
    finish(a, b, episode_id)
    return rows


def verify_session(record, config, source_mua, refilter=False):
    for key in ["windows", "overlaps", "position_support", "source_cache"]:
        assert file_sha256(record[f"{key}_path"]) == record[f"{key}_sha256"], key
    windows = pd.read_csv(record["windows_path"], float_precision="round_trip")
    pairs = pd.read_csv(record["overlaps_path"], float_precision="round_trip")
    with np.load(record["source_cache_path"], allow_pickle=False) as a:
        position, spikes, ids, mask = a["position"], a["spikes"], a["cell_ids"], a["unit_qc_mask"]
    with np.load(record["position_support_path"], allow_pickle=False) as a:
        times, speed, supported, immobile = a["times_s"], a["speed_cm_s"], a["supported_run_intervals"], a["immobile_intervals"]
    np.testing.assert_array_equal(times, position[:, 0])
    np.testing.assert_allclose(speed, tracking_speed(position, supported, config), equal_nan=True)
    assert len(immobile) > 0 and np.all(immobile[:, 1] > immobile[:, 0])
    assert np.all(immobile[1:, 0] >= immobile[:-1, 1])
    for lo, hi in immobile:
        internal = (times >= lo) & (times <= hi)
        assert np.isfinite(speed[internal]).all()
        assert np.all(speed[internal] <= config.maximum_speed_cm_s + 1e-8)
        assert np.any((supported[:, 0] <= lo) & (supported[:, 1] >= hi))
    assert windows.window_uid.is_unique and windows.groupby("source_uid").size().eq(2).all()
    core = windows[windows.window_variant.eq("detected_core")]
    mua = core[core.detector.eq("source_high_mua")].sort_values("source_event_id")
    source_mua = source_mua.sort_values("event_index")
    np.testing.assert_array_equal(mua.source_event_id, source_mua.event_index)
    np.testing.assert_allclose(mua[["start_s", "end_s", "peak_s"]], source_mua[["start_s", "end_s", "peak_s"]], rtol=0, atol=1e-9)
    ripple = core[~core.detector.eq("source_high_mua")].sort_values("source_event_id")
    raw = record["raw_ripple_source"]
    arrays_hashed, baseline_samples = 0, 0
    if record["dataset"] == "pfeiffer_foster":
        assert file_sha256(raw["path"]) == raw["sha256"]
        values = np.asarray(loadmat(raw["path"])["Ripple_Events"], float)
        np.testing.assert_allclose(ripple[["start_s", "end_s", "peak_s"]], values[:, :3], rtol=0, atol=1e-9)
        np.testing.assert_array_equal(ripple.source_event_id, np.arange(len(values)))
    else:
        with h5py.File(raw["path"], "r") as f:
            for path, description in raw["consumed_datasets"].items():
                assert array_sha256(np.asarray(f[path][()])) == description["sha256"], path
                arrays_hashed += 1
        assert file_sha256(raw["envelope_cache_path"]) == raw["envelope_cache_sha256"]
        with np.load(raw["envelope_cache_path"], allow_pickle=False) as a:
            lt, envelope, z, baseline, fs = a["times_s"], a["envelope"], a["z"], a["baseline"], float(a["sampling_rate_hz"])
        if refilter:
            reconstructed_t, reconstructed_envelope, reconstructed_fs, _, _ = load_ripple_envelope(Path(raw["path"]), config)
            np.testing.assert_array_equal(lt, reconstructed_t)
            np.testing.assert_allclose(envelope, reconstructed_envelope, rtol=0, atol=0)
            assert fs == reconstructed_fs
        expected_mask = np.zeros(len(lt), bool)
        for lo, hi in immobile:
            a, b = np.searchsorted(lt, [lo, hi], side="left")
            b += int(b < len(lt) and lt[b] == hi)
            expected_mask[a:b] = True
        edge = (lt >= lt[0] + config.filter_edge_guard_s) & (lt < lt[-1] - config.filter_edge_guard_s)
        expected_mask &= edge
        np.testing.assert_array_equal(baseline, expected_mask)
        baseline_samples = int(baseline.sum())
        np.testing.assert_allclose(baseline_samples / fs, raw["baseline_duration_s"])
        if record["ripple_status"] == "available":
            assert baseline_samples / fs >= config.minimum_baseline_s
            expected_z = (envelope - np.mean(envelope[baseline])) / np.std(envelope[baseline])
            expected_z[~edge] = np.nan
            np.testing.assert_allclose(z, expected_z, rtol=0, atol=1e-12, equal_nan=True)
            expected = np.asarray(reference_episodes(lt, z, fs, config)).reshape(-1, 7)
            np.testing.assert_allclose(ripple[["source_event_id", "start_s", "end_s", "peak_s", "ripple_peak_z", "lfp_start_sample", "lfp_end_sample_exclusive"]], expected, rtol=0, atol=1e-9)
            durations = (expected[:, 6] - expected[:, 5]) / fs
            np.testing.assert_array_equal(ripple.detector_duration_pass, (durations >= config.ripple_minimum_duration_s) & (durations <= config.ripple_maximum_duration_s))
        else:
            assert baseline_samples / fs < config.minimum_baseline_s
            assert np.isnan(z).all() and len(ripple) == 0 and record["source_ripple_events"] is None
            assert windows.other_detector_overlap_count.isna().all()
            assert windows.loc[windows.eligible, "overlap_class"].eq("ripple_unavailable").all()
    qc_ids = ids[mask]
    for w in windows.itertuples(index=False):
        if w.window_variant == "peak_centered_200ms":
            np.testing.assert_allclose([w.start_s, w.end_s], [w.peak_s - .1, w.peak_s + .1], rtol=0, atol=1e-9)
        else:
            assert w.start_s == w.core_start_s and w.end_s == w.core_end_s
        tracking = np.any((supported[:, 0] <= w.start_s) & (supported[:, 1] + 1e-9 >= w.end_s))
        motion = np.any((immobile[:, 0] <= w.start_s) & (immobile[:, 1] + 1e-9 >= w.end_s))
        clock = w.start_s >= position[0, 0] and w.end_s <= position[-1, 0]
        peak_immobile = np.any((immobile[:, 0] <= w.peak_s) & (immobile[:, 1] + 1e-9 >= w.peak_s))
        lfp_supported = record["dataset"] == "pfeiffer_foster" or (w.start_s >= lt[0] + config.filter_edge_guard_s and w.end_s <= lt[-1] - config.filter_edge_guard_s)
        assert bool(w.position_clock_valid) == clock and bool(w.peak_immobile) == peak_immobile
        assert bool(w.lfp_window_supported) == lfp_supported
        assert bool(w.tracking_supported) == tracking and bool(w.whole_window_immobile) == motion
        assert bool(w.eligible) == all([w.detector_duration_pass, w.position_clock_valid, tracking, motion, w.lfp_window_supported])
        a, b = np.searchsorted(spikes[:, 0], [w.start_s, w.end_s], side="left")
        unit = spikes[a:b, 1].astype(int)
        selected = unit[np.isin(unit, qc_ids)]
        # The source all-sorted count includes only the cached cell universe.
        assert w.n_spikes_all_sorted == np.isin(unit, ids).sum()
        assert w.n_spikes_qc_units == len(selected) and w.n_active_qc_units == len(np.unique(selected))
    expected_pairs = set()
    for variant in ["detected_core", "peak_centered_200ms"]:
        active = windows[windows.eligible & windows.window_variant.eq(variant)]
        mua = active[active.detector.eq("source_high_mua")]
        ripple = active[~active.detector.eq("source_high_mua")]
        overlap = np.maximum(0, np.minimum(mua.end_s.to_numpy()[:, None], ripple.end_s.to_numpy()[None, :]) - np.maximum(mua.start_s.to_numpy()[:, None], ripple.start_s.to_numpy()[None, :]))
        for i, j in zip(*np.nonzero(overlap), strict=True):
            expected_pairs.add((mua.window_uid.iloc[i], ripple.window_uid.iloc[j]))
    actual_pairs = set(zip(pairs.mua_window_uid, pairs.ripple_window_uid, strict=True))
    assert len(actual_pairs) == len(pairs) and actual_pairs == expected_pairs
    if record["ripple_status"] == "available":
        for w in windows.itertuples(index=False):
            assert w.other_detector_overlap_count == sum(w.window_uid in pair for pair in expected_pairs)
    return {**{k: record[k] for k in ["dataset", "animal", "session"]}, "status": "pass",
        "window_rows_verified": len(windows), "ripple_source_rows_verified": record["source_ripple_events"],
        "ripple_status": record["ripple_status"], "native_arrays_hashed": arrays_hashed,
        "baseline_samples_verified": baseline_samples, "overlap_edges_verified": len(pairs), "all_channels_refiltered": bool(refilter)}


def audit(root, workers):
    manifest_path = root / "coverage_event_definition_manifest.json"
    meta = json.loads(manifest_path.read_text())
    if meta["status"] not in {"complete", "complete_with_ripple_unavailable"}:
        raise ValueError("terminal successful preparation required")
    for name, digest in meta["output_sha256"].items():
        assert file_sha256(root / name) == digest, name
    for name, path in meta["input_file_paths"].items():
        assert file_sha256(path) == meta["input_file_sha256"][name], name
        assert file_sha256(root / "inputs" / f"{name}{Path(path).suffix}") == meta["input_file_sha256"][name], name
    candidates = pd.read_csv(meta["input_file_paths"]["candidates"])
    config = EventDefinitionConfig(**meta["parameters"])
    rows = []
    tanni = sorted([r for r in meta["results"] if r["dataset"] == "tanni2022"], key=lambda r: (r["animal"], r["session"]))
    refilter_key = (tanni[0]["animal"], tanni[0]["session"]) if tanni else None
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = []
        for record in meta["results"]:
            assert record["status"] == "complete"
            local = candidates[np.logical_and.reduce([candidates[k].eq(record[k]) for k in ["dataset", "animal", "session"]])]
            futures.append(pool.submit(verify_session, record, config, local, (record["animal"], record["session"]) == refilter_key))
        for f in as_completed(futures):
            row = f.result()
            rows.append(row)
            print(json.dumps(row), flush=True)
    table = pd.DataFrame(rows).sort_values(["dataset", "animal", "session"])
    table.to_csv(root / "coverage_event_definition_reconstruction_audit.csv", index=False)
    out = build_script_provenance(input_paths={"preparation_manifest": manifest_path, "auditor": Path(__file__)}, cwd=ROOT)
    out.update(status="pass", sessions=len(rows), window_rows_verified=int(table.window_rows_verified.sum()),
        native_arrays_hashed=int(table.native_arrays_hashed.sum()), overlap_edges_verified=int(table.overlap_edges_verified.sum()),
        all_channel_refilter_sessions=int(table.all_channels_refiltered.sum()),
        audit_scope="all source array/file hashes; all MUA/native ripple identities; independent episode segmentation and baseline mask; direct window spike support; all overlaps and eligibility; all-channel refilter on alphabetically first Tanni session only; no replay decoding")
    (root / "coverage_event_definition_reconstruction_audit.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    audit(args.input_dir.resolve(), args.workers)
