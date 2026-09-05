#!/usr/bin/env python3
"""Freeze native/LFP-ripple and high-MUA windows without decoding any replay."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import h5py
import numpy as np
import pandas as pd
from scipy.io import loadmat

from hipporeplayimm.replay_coverage_data import array_sha256, index_spike_times
from hipporeplayimm.replay_coverage_event_definitions import (
    EventDefinitionConfig,
    InsufficientBaselineError,
    annotate_overlap,
    detect_ripple_episodes,
    immobile_intervals,
    make_windows,
    overlap_pairs,
    ripple_envelope,
    ripple_window_metrics,
    source_events,
    standardize_envelope,
    tracking_speed,
    validate_lfp_clock,
    window_spike_support,
)
from scripts._provenance import build_script_provenance, file_sha256

IDENTITY = ["dataset", "animal", "session"]
LFP_GROUP = "/acquisition/timeseries/recording1/continuous/processor102_100"
CHANNEL_MAP = "/general/data_collection/Settings/General/channel_map"


def load_ripple_envelope(path, config):
    before = path.stat()
    consumed, channel_rows = {}, []
    with h5py.File(path, "r") as f:
        def read(key):
            value = np.asarray(f[key][()])
            consumed[key] = {"shape": list(value.shape), "dtype": value.dtype.str, "sha256": array_sha256(value)}
            return value
        times = read(f"{LFP_GROUP}/downsampled_timestamps").reshape(-1)
        fs = float(read(f"{LFP_GROUP}/downsampling_info/downsampled_sampling_rate").reshape(-1)[0])
        channels = read(f"{LFP_GROUP}/downsampling_info/downsampled_channels").reshape(-1)
        regions = [read(f"{CHANNEL_MAP}/{name}/list").reshape(-1) for name in sorted(f[CHANNEL_MAP]) if name.startswith("CA1")]
        if not regions:
            raise ValueError("native CA1 channel map missing")
        ca1 = np.concatenate(regions)
        validate_lfp_clock(times, fs)
        data = read(f"{LFP_GROUP}/downsampled_tetrode_data")
        if data.shape != (len(times), len(channels)) or len(np.unique(channels)) != len(channels):
            raise ValueError("LFP channel/time alignment failed")
        if not np.issubdtype(data.dtype, np.integer):
            raise ValueError("integer raw LFP expected for declared saturation screen")
        rails = np.iinfo(data.dtype)
        aggregate = np.zeros(len(times), float)
        n_channels = 0
        for j, channel_id in enumerate(channels):
            raw = data[:, j]
            fraction = float(np.mean((raw == rails.min) | (raw == rails.max)))
            constant = bool(np.min(raw) == np.max(raw))
            selected = bool(channel_id in ca1 and not constant and fraction <= config.maximum_channel_rail_fraction)
            channel_rows.append({"stored_column": j, "native_channel_id": int(channel_id), "ca1_metadata_member": bool(channel_id in ca1),
                "constant": constant, "rail_fraction": fraction, "selected": selected})
            if selected:
                aggregate += ripple_envelope(raw, fs, config)
                n_channels += 1
        if n_channels == 0:
            raise ValueError("no technically usable native CA1 LFP channels")
        aggregate /= n_channels
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError("raw LFP source changed during read")
    provenance = {"path": str(path), "size_bytes": before.st_size, "mtime_ns": before.st_mtime_ns,
        "hash_scope": "consumed_HDF5_arrays_not_whole_NWB", "consumed_datasets": consumed,
        "channels_selected": n_channels, "sampling_rate_hz": fs,
        "start_s": float(times[0]), "end_s": float(times[-1]),
        "clock_min_dt_s": float(np.min(np.diff(times))), "clock_max_dt_s": float(np.max(np.diff(times)))}
    return times, aggregate, fs, channel_rows, provenance


def native_ripples(source):
    path = source / "Ripple_Events.mat"
    before = file_sha256(path)
    values = np.asarray(loadmat(path)["Ripple_Events"], float)
    if values.ndim != 2 or values.shape[1] < 3:
        raise ValueError("invalid native ripple array")
    if file_sha256(path) != before:
        raise RuntimeError("native ripple file changed during read")
    frame = pd.DataFrame(values[:, :3], columns=["start_s", "end_s", "peak_s"])
    frame["source_event_id"] = np.arange(len(frame))
    return frame, {"path": str(path), "sha256": before, "native_rows": len(frame),
        "raw_lfp_available": False, "uninterpreted_other_columns": int(values.shape[1] - 3)}


def summarize_windows(windows):
    rows = []
    for identity, frame in windows.groupby(IDENTITY, sort=True):
        detector_names = ["source_high_mua", "native_ripple_table" if identity[0] == "pfeiffer_foster" else "lfp_ripple_detected"]
        for detector in detector_names:
            thresholds = [3.0, 4.0, 5.0] if detector == "lfp_ripple_detected" else [0.0]
            for variant in ["detected_core", "peak_centered_200ms"]:
                all_rows = frame[frame.detector.eq(detector) & frame.window_variant.eq(variant)]
                available = detector != "lfp_ripple_detected" or "ripple_status" not in frame or frame.ripple_status.eq("available").all()
                overlap_available = "ripple_status" not in frame or frame.ripple_status.eq("available").all()
                for threshold in thresholds:
                    subset = all_rows[all_rows.ripple_peak_z >= threshold] if threshold else all_rows
                    eligible = subset[subset.eligible]
                    rows.append({**dict(zip(IDENTITY, identity, strict=True)), "detector": detector, "window_variant": variant,
                        "peak_threshold_z": threshold, "detector_available": bool(available),
                        "source_windows": len(subset) if available else np.nan, "duration_pass": int(subset.detector_duration_pass.sum()) if available else np.nan,
                        "tracking_supported": int(subset.tracking_supported.sum()), "peak_immobile": int(subset.peak_immobile.sum()),
                        "whole_window_immobile": int(subset.whole_window_immobile.sum()), "eligible_windows": len(eligible) if available else np.nan,
                        "overlap_available": bool(overlap_available),
                        "other_detector_overlap_at_primary_z3": int((eligible.other_detector_overlap_count > 0).sum()) if overlap_available else np.nan,
                        "n_spikes_qc_units_median": float(eligible.n_spikes_qc_units.median()) if len(eligible) else np.nan,
                        "n_active_qc_units_median": float(eligible.n_active_qc_units.median()) if len(eligible) else np.nan,
                        "duration_ms_median": float(1000 * eligible.window_duration_s.median()) if len(eligible) else np.nan})
    return pd.DataFrame(rows)


def prepare_session(record, mua, out, config, reuse=None):
    started = time.monotonic()
    identity = {k: record[k] for k in IDENTITY}
    label = "__".join(str(identity[k]).replace("/", "_") for k in IDENTITY)
    cache = Path(record["artifact_path"])
    if file_sha256(cache) != record["artifact_sha256"]:
        raise ValueError("source encoding cache hash mismatch")
    with np.load(cache, allow_pickle=False) as a:
        position = a["position"]
        supported = a["supported_run_intervals"]
        spikes, ids, mask = a["spikes"], a["cell_ids"], a["unit_qc_mask"]
    speed = tracking_speed(position, supported, config)
    immobile = immobile_intervals(position[:, 0], speed, config)
    if len(immobile) == 0:
        raise ValueError("no common tracking-supported immobility")
    source = Path(record["source_path"])
    times, z, baseline, channels = None, None, None, []
    ripple_status = "available"
    if identity["dataset"] == "tanni2022":
        if reuse is None:
            print(f"filter LFP {label}", flush=True)
            times, envelope, fs, channels, raw_meta = load_ripple_envelope(source, config)
        else:
            raw_meta = dict(reuse["raw_ripple_source"])
            cache_path = Path(raw_meta["envelope_cache_path"])
            if file_sha256(cache_path) != raw_meta["envelope_cache_sha256"]:
                raise ValueError("reused envelope hash mismatch")
            stat = source.stat()
            if str(source) != raw_meta["path"] or (stat.st_size, stat.st_mtime_ns) != (raw_meta["size_bytes"], raw_meta["mtime_ns"]):
                raise ValueError("reused envelope raw input differs")
            with np.load(cache_path, allow_pickle=False) as f:
                times, envelope, fs = f["times_s"], f["envelope"], float(f["sampling_rate_hz"])
            channels = reuse["channels"]
            raw_meta["reused_raw_envelope_manifest"] = reuse["manifest_path"]
            raw_meta["reused_raw_envelope_manifest_sha256"] = file_sha256(reuse["manifest_path"])
        try:
            z, baseline, baseline_meta = standardize_envelope(envelope, times, fs, immobile, config)
        except InsufficientBaselineError as exc:
            ripple_status = "unavailable_insufficient_baseline"
            z, baseline = np.full(len(times), np.nan), exc.baseline
            baseline_meta = {"baseline_duration_s": exc.duration_s, "ripple_unavailable_reason": str(exc)}
        raw_meta.update(baseline_meta)
        ripple = detect_ripple_episodes(times, z, fs, config)
        lfp_cache = out / "sessions" / f"{label}__ripple_envelope.npz"
        np.savez_compressed(lfp_cache, times_s=times, envelope=envelope, z=z, baseline=baseline,
            sampling_rate_hz=fs)
        raw_meta.update(envelope_cache_path=str(lfp_cache), envelope_cache_sha256=file_sha256(lfp_cache))
    else:
        ripple, raw_meta = native_ripples(source)
    mua = mua[["event_index", "start_s", "end_s", "peak_s"]]
    events = source_events(mua, ripple, **identity)
    windows = make_windows(events, position, supported, immobile, config, times)
    windows["ripple_status"] = ripple_status
    pairs = overlap_pairs(windows)
    windows = annotate_overlap(windows, pairs)
    metrics = window_spike_support(windows, index_spike_times(spikes, ids), ids, mask)
    windows = pd.concat([windows, metrics], axis=1)
    if times is not None:
        windows = pd.concat([windows, ripple_window_metrics(windows, times, z)], axis=1)
    windows["source_cache_path"] = str(cache)
    windows["source_cache_sha256"] = record["artifact_sha256"]
    windows["decode_performed"] = False
    events_path = out / "sessions" / f"{label}__windows.csv"
    windows.to_csv(events_path, index=False)
    pairs_path = out / "sessions" / f"{label}__overlaps.csv"
    pairs.to_csv(pairs_path, index=False)
    position_cache = out / "sessions" / f"{label}__position_support.npz"
    np.savez_compressed(position_cache, times_s=position[:, 0], speed_cm_s=speed, supported_run_intervals=supported, immobile_intervals=immobile)
    if file_sha256(cache) != record["artifact_sha256"]:
        raise RuntimeError("source encoding cache changed")
    for row in channels:
        row.update(identity)
    metadata = {**identity, "status": "complete", "source_cache_path": str(cache), "source_cache_sha256": record["artifact_sha256"],
        "source_high_mua_events": len(mua), "source_ripple_events": len(ripple) if ripple_status == "available" else None,
        "ripple_status": ripple_status, "windows": len(windows), "overlap_edges": len(pairs),
        "immobile_duration_s": float(np.sum(immobile[:, 1] - immobile[:, 0])), "raw_ripple_source": raw_meta,
        "windows_path": str(events_path), "windows_sha256": file_sha256(events_path),
        "overlaps_path": str(pairs_path), "overlaps_sha256": file_sha256(pairs_path),
        "position_support_path": str(position_cache), "position_support_sha256": file_sha256(position_cache),
        "runtime_s": time.monotonic() - started}
    (out / "sessions" / f"{label}__manifest.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    return metadata, summarize_windows(windows).to_dict("records"), channels


def run(args):
    config = EventDefinitionConfig()
    sessions_path = args.input_dir / "coverage_input_sessions.csv"
    candidates_path = args.input_dir / "coverage_input_candidates.csv"
    inputs = {"sessions": sessions_path, "candidates": candidates_path,
        "input_manifest": args.input_dir / "coverage_input_manifest.json", "script": Path(__file__),
        "library": ROOT / "src/hipporeplayimm/replay_coverage_event_definitions.py",
        "data_library": ROOT / "src/hipporeplayimm/replay_coverage_data.py",
        "provenance_library": ROOT / "scripts/_provenance.py",
        "protocol": ROOT / "docs/replay_coverage_event_definition_protocol.md"}
    reusable = {}
    if args.reuse_envelope_dir is not None:
        parent_path = args.reuse_envelope_dir / "coverage_event_definition_manifest.json"
        parent = json.loads(parent_path.read_text())
        if parent["status"] not in {"complete", "failed", "complete_with_ripple_unavailable"}:
            raise ValueError("only terminal envelope preparation may be reused")
        if parent["parameters"] != asdict(config):
            raise ValueError("reused envelope parameters differ")
        channel_path = args.reuse_envelope_dir / "coverage_event_definition_channel_qc.csv"
        channel_frame = pd.read_csv(channel_path)
        if file_sha256(channel_path) != parent["output_sha256"][channel_path.name]:
            raise ValueError("reused channel metadata changed")
        inputs["reused_envelope_manifest"] = parent_path
        inputs["reused_channel_qc"] = channel_path
        for record in parent["results"]:
            if record["status"] == "complete" and record["dataset"] == "tanni2022":
                local = channel_frame[np.logical_and.reduce([channel_frame[k].eq(record[k]) for k in IDENTITY])]
                reusable[tuple(record[k] for k in IDENTITY)] = {**record, "channels": local.to_dict("records"), "manifest_path": str(parent_path)}
    sessions = pd.read_csv(sessions_path)
    events = pd.read_csv(candidates_path)
    if args.sessions:
        requested = set(args.sessions)
        sessions = sessions[sessions[IDENTITY].agg(":".join, axis=1).isin(requested)]
        if len(sessions) != len(requested):
            raise ValueError("unknown or duplicated requested sessions")
    if len(sessions) == 0 or sessions.duplicated(IDENTITY).any() or not sessions.status.eq("cached").all():
        raise ValueError("empty, duplicate or uncached session cohort")
    if events.duplicated(IDENTITY + ["event_index"]).any():
        raise ValueError("duplicate source MUA events")
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "sessions").mkdir()
    (out / "inputs").mkdir()
    provenance = build_script_provenance(input_paths=inputs, cwd=ROOT)
    for key, path in inputs.items():
        if provenance["input_file_sha256"][key] is None:
            raise ValueError(f"missing input {key}")
        shutil.copy2(path, out / "inputs" / f"{key}{Path(path).suffix}")
    manifest = {**provenance, "created_at_utc": datetime.now(UTC).isoformat(), "parameters": asdict(config),
        "workers": args.workers, "cohort": "explicit_subset" if args.sessions else "all_input_sessions",
        "requested_sessions": args.sessions, "status": "running", "decode_performed": False,
        "versions": {name: version(name) for name in ["numpy", "scipy", "h5py", "pandas"]},
        "claim_boundary": "candidate-definition preparation only; no replay truth, scoring, speed effect or uniformity claim"}
    manifest_path = out / "coverage_event_definition_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    results, summaries, channel_rows = [], [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = {}
        for record in sessions.to_dict("records"):
            local = events[np.logical_and.reduce([events[k].eq(record[k]) for k in IDENTITY])].copy()
            if len(local) != int(record["candidates"]):
                raise ValueError("source MUA count disagrees with session manifest")
            jobs[pool.submit(prepare_session, record, local, out, config, reusable.get(tuple(record[k] for k in IDENTITY)))] = record
        for future in as_completed(jobs):
            record = jobs[future]
            try:
                metadata, summary, channels = future.result()
                results.append(metadata)
                summaries.extend(summary)
                channel_rows.extend(channels)
                print(json.dumps({k: metadata[k] for k in IDENTITY + ["status", "windows", "runtime_s"]}), flush=True)
            except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
                failure = {**{k: record[k] for k in IDENTITY}, "status": "failed", "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
                results.append(failure)
                print(json.dumps(failure), flush=True)
    complete = [r for r in results if r["status"] == "complete"]
    windows = pd.concat([pd.read_csv(r["windows_path"]) for r in complete], ignore_index=True) if complete else pd.DataFrame()
    overlap = pd.concat([pd.read_csv(r["overlaps_path"]) for r in complete], ignore_index=True) if complete else pd.DataFrame()
    ordered_results = sorted(results, key=lambda r: tuple(r[k] for k in IDENTITY))
    pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, (dict, list))} for r in ordered_results]).to_csv(out / "coverage_event_definition_sessions.csv", index=False)
    summary_table = pd.DataFrame(summaries)
    if len(summary_table):
        summary_table = summary_table.sort_values(IDENTITY + ["detector", "window_variant", "peak_threshold_z"])
    summary_table.to_csv(out / "coverage_event_definition_counts.csv", index=False)
    pd.DataFrame(channel_rows).to_csv(out / "coverage_event_definition_channel_qc.csv", index=False)
    windows.to_csv(out / "coverage_event_definition_windows.csv", index=False)
    overlap.to_csv(out / "coverage_event_definition_overlaps.csv", index=False)
    unchanged = all(file_sha256(path) == provenance["input_file_sha256"][key] for key, path in inputs.items())
    nonempty = len(windows) > 0
    retained = sum(r["source_high_mua_events"] for r in complete)
    gates = {"all_sessions_complete": len(complete) == len(sessions),
        "all_source_mua_events_retained": retained == int(sessions.candidates.sum()),
        "unique_window_identities": nonempty and not windows.window_uid.duplicated().any(),
        "two_variants_per_source_event": nonempty and windows.groupby("source_uid").size().eq(2).all(),
        "explicit_eligibility_and_exclusions": nonempty and windows.loc[~windows.eligible, "exclusion_reason"].notna().all(),
        "no_decoding_or_content_selection": nonempty and not windows.decode_performed.any(),
        "source_code_and_tables_unchanged": unchanged}
    gates["overall_preparation"] = all(bool(v) for v in gates.values())
    availability = len(complete) == len(sessions) and all(r["ripple_status"] == "available" for r in complete)
    pd.DataFrame([{"gate": k, "passed": bool(v), "required_for_preparation": True} for k, v in gates.items()] +
        [{"gate": "all_ripple_definitions_available", "passed": bool(availability), "required_for_preparation": False}]).to_csv(out / "coverage_event_definition_gate_summary.csv", index=False)
    lines = ["# Event-Definition Preparation", "", f"Completed sessions: {len(complete)}/{len(sessions)}.",
        f"Original MUA events retained: {retained}/{int(sessions.candidates.sum())}. Window rows: {len(windows)}.",
        "", "PF ripple events are native tables; Tanni ripple-like events are independently detected from CA1 LFP.",
        "All source events, including out-of-RUN and duration/movement exclusions, remain visible.",
        "Detected cores and fixed 200 ms windows are separate. Many-to-many overlaps are not replay precision/recall.",
        "Envelope amplitude z is not ripple power z. LFP artifacts and sharp waves are not independently validated.",
        "No replay was decoded. Candidate counts and overlap alone do not finish coverage sensitivity.", "", "## Gates", ""]
    lines.extend(f"- {k}: {'pass' if v else 'FAIL'}" for k, v in gates.items())
    lines.extend(["", f"All ripple definitions available: {availability}. Unavailable detection is not zero events."])
    lines.extend(f"- {r['animal']}/{r['session']}: {r['ripple_status']}" for r in complete if r["ripple_status"] != "available")
    (out / "coverage_event_definition_report.md").write_text("\n".join(lines) + "\n")
    manifest.update(status=("complete" if availability else "complete_with_ripple_unavailable") if gates["overall_preparation"] else "failed", results=ordered_results,
        sessions=len(sessions), source_mua_events=retained, windows=len(windows), inputs_unchanged=unchanged,
        output_sha256={p.name: file_sha256(p) for p in sorted(out.iterdir()) if p.is_file() and p != manifest_path})
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    if not gates["overall_preparation"]:
        raise RuntimeError("preparation failed; inspect explicit failures")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sessions", nargs="*")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--reuse-envelope-dir", type=Path)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    run(args)
