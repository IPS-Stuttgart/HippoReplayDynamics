"""Verify source identities and matched-RUN support for optional Tanni replication.

No association is fit, no theta phase is inferred from spikes, and no detected
pause is called replay. Only whitelisted scientific fields are read/exported.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
from itertools import combinations
import json
from pathlib import Path
import subprocess

import h5py
import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean

try:
    from scripts._provenance import build_script_provenance, file_sha256
    from scripts.audit_replay_order_run_coordination import (
        epoch_links, immobile_pauses, link_spikes, matched_run,
    )
except ModuleNotFoundError:
    from _provenance import build_script_provenance, file_sha256
    from audit_replay_order_run_coordination import (
        epoch_links, immobile_pauses, link_spikes, matched_run,
    )


def array_digest(a: np.ndarray) -> str:
    a = np.asarray(a)
    if a.dtype.kind not in "biufcSU":
        raise ValueError("Only numerical or fixed-string scientific arrays may be hashed")
    digest = hashlib.sha256(json.dumps({"shape": a.shape, "dtype": a.dtype.str},
                                     sort_keys=True).encode())
    digest.update(np.ascontiguousarray(a).tobytes())
    return digest.hexdigest()


def read_array(handle: h5py.File, path: str, inventory: list[dict]) -> np.ndarray:
    value = handle[path][()]
    if isinstance(value, bytes):
        value = np.asarray(value)
    a = np.asarray(value)
    inventory.append({"hdf5_path": path, "shape": list(a.shape),
                      "dtype": a.dtype.str, "array_sha256": array_digest(a)})
    return a


def increasing(a: np.ndarray, name: str, *, allow_equal: bool = False) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    if a.ndim != 1 or not len(a) or not np.isfinite(a).all():
        raise ValueError(f"Invalid {name}")
    delta = np.diff(a)
    if np.any(delta < 0 if allow_equal else delta <= 0):
        raise ValueError(f"Backward or duplicate {name}")
    return a


def closest_indices(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Nearest timestamp, retaining the first reference on an exact tie."""
    reference = increasing(reference, "reference timestamps")
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite query timestamp")
    right = np.clip(np.searchsorted(reference, values, side="left"), 0, len(reference) - 1)
    left = np.maximum(0, right - 1)
    return np.where(np.abs(values - reference[left]) <= np.abs(values - reference[right]),
                    left, right)


def camera_to_acquisition(oe: np.ndarray, camera: np.ndarray,
                          frames: np.ndarray, divider: float) -> np.ndarray:
    oe = increasing(oe, "acquisition clock pulses")
    camera = increasing(camera, "camera clock pulses")
    frames = increasing(frames, "camera frame timestamps")
    if len(oe) != len(camera) or len(oe) < 2:
        raise ValueError("Unmatched global-clock pulse counts; no fitted/truncated clock correction")
    if not np.isfinite(divider) or divider <= 0:
        raise ValueError("Invalid camera clock scale")
    nearest = closest_indices(camera, frames)
    converted = oe[nearest] + (frames - camera[nearest]) / divider
    return increasing(converted, "converted camera frames")


def bad_channels(raw: bytes | str) -> set[int]:
    text = raw.decode() if isinstance(raw, bytes) else str(raw)
    result = set()
    if not text.strip():
        return result
    for token in text.split(","):
        bounds = token.strip().split("-")
        if len(bounds) not in (1, 2):
            raise ValueError("Ambiguous bad-channel range")
        start, end = int(bounds[0]), int(bounds[-1])
        if start < 1 or end < start:
            raise ValueError("Invalid one-based bad-channel range")
        for channel in range(start - 1, end):
            if channel in result:
                raise ValueError("Duplicate bad-channel annotation")
            result.add(channel)
    return result


def ca1_mapping(areas: dict[str, np.ndarray]) -> dict[int, str]:
    channels: dict[int, str] = {}
    for area, values in areas.items():
        a = np.asarray(values)
        if a.ndim != 1 or a.dtype.kind not in "iu" or np.any(a < 0):
            raise ValueError("Invalid zero-based channel-map list")
        for c in a:
            if int(c) in channels:
                raise ValueError("Conflicting or duplicate anatomical channel annotation")
            channels[int(c)] = area
    result = {}
    for tetrode in sorted({c // 4 for c in channels}):
        annotations = [channels.get(tetrode * 4 + i) for i in range(4)]
        if len(set(annotations)) != 1 or annotations[0] is None:
            raise ValueError("Incomplete or mixed-area tetrode annotation")
        if annotations[0].startswith("CA1_"):
            result[tetrode] = annotations[0]
    if not result:
        raise ValueError("No explicitly annotated CA1 tetrodes")
    return result


def curated_units(times: np.ndarray, keep: np.ndarray, labels: np.ndarray,
                  tetrode: int) -> tuple[np.ndarray, list[dict]]:
    times = np.asarray(times, dtype=float)
    if times.ndim != 1:
        raise ValueError("Raw spike timestamps must be a vector")
    if len(times):
        times = increasing(times, "raw spike timestamps", allow_equal=True)
    keep, labels = np.asarray(keep), np.asarray(labels)
    if keep.dtype.kind != "b" or keep.shape != times.shape:
        raise ValueError("Artifact mask must be Boolean with one entry per raw spike")
    if labels.ndim != 1 or labels.dtype.kind not in "iu" or len(labels) != int(keep.sum()):
        raise ValueError("Curated labels must index artifact-retained spikes, not raw spikes")
    if np.any(labels < 0) or np.any(labels >= 65536):
        raise ValueError("Invalid curated cluster label")
    filtered = times[keep]
    hits = labels > 0
    ids = tetrode * 65536 + labels[hits].astype(np.int64)
    spikes = np.column_stack((filtered[hits], ids))
    rows = [{"unit_id": int(tetrode * 65536 + cluster),
             "tetrode_zero_based": tetrode, "cluster_id": int(cluster),
             "retained_spikes": int(np.sum(labels == cluster)),
             "identity_verified": True, "cell_type": "not_assumed_from_spatial_firing"}
            for cluster in np.unique(labels[hits])]
    return spikes, rows


def source_combiner(source: Path, expected_commit: str):
    """Execute only the two pinned upstream geometry functions, not package I/O."""
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    if commit != expected_commit:
        raise ValueError("Acquisition source commit differs from frozen protocol")
    path = source / "openEPhys_DACQ/TrackingDataProcessing.py"
    if subprocess.check_output(["git", "diff", "HEAD", "--", str(path)], cwd=source, text=True):
        raise ValueError("Pinned acquisition reconstruction source was modified")
    tree = ast.parse(path.read_text())
    names = {"combineCamerasData", "remove_tracking_data_outside_boundaries"}
    functions = [x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name in names]
    if {x.name for x in functions} != names:
        raise ValueError("Missing pinned acquisition geometry functions")
    namespace = {"np": np, "euclidean": euclidean, "combinations": combinations}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["combineCamerasData"], namespace["remove_tracking_data_outside_boundaries"], file_sha256(path)


def reconstruct_tracking(cameras: dict[str, np.ndarray], settings: dict,
                         arena: np.ndarray, combine, crop, sampling_rate: float) -> np.ndarray:
    names = sorted(cameras)
    if not names:
        raise ValueError("Missing original camera observations")
    for data in cameras.values():
        increasing(data[:, 0], "camera tracking chronology")
    if len(names) == 1:
        return crop(cameras[names[0]].copy(), arena, max_error=10)
    start = min(cameras[k][0, 0] for k in names)
    end = max(cameras[k][-1, 0] for k in names)
    grid = np.arange(start, end, 1.0 / sampling_rate)
    nearest = {k: closest_indices(cameras[k][:, 0], grid) for k in names}
    rows = []
    previous = None
    for i, t in enumerate(grid):
        point = combine([cameras[k][nearest[k][i], 1:5] for k in names],
                        previous, names, settings, arena)
        previous = point
        if point is not None:
            rows.append(np.r_[t, point])
    if not rows:
        raise ValueError("Original camera reconstruction has no valid observations")
    return crop(np.asarray(rows), arena, max_error=10)


def reconcile_processed(processed: np.ndarray, reconstructed: np.ndarray,
                        time_tolerance: float, coordinate_tolerance: float) -> dict:
    increasing(processed[:, 0], "ProcessedPos timestamps")
    nearest = closest_indices(reconstructed[:, 0], processed[:, 0])
    time_error = np.abs(reconstructed[nearest, 0] - processed[:, 0])
    a, b = processed[:, 1:3], reconstructed[nearest, 1:3]
    nan_match = np.isnan(a) == np.isnan(b)
    finite = np.isfinite(a) & np.isfinite(b)
    coord_error = np.where(finite, np.abs(a - b), 0.0)
    mismatch = ~nan_match.all(axis=1) | np.any(np.isinf(a) | np.isinf(b), axis=1)
    mismatch |= np.any(coord_error > coordinate_tolerance, axis=1)
    mismatch |= time_error > time_tolerance
    return {"processed_rows": len(processed), "reconstructed_rows": len(reconstructed),
            "mismatched_processed_rows": int(mismatch.sum()),
            "maximum_time_error_s": float(time_error.max()),
            "maximum_coordinate_error_cm": float(coord_error.max()),
            "source_clock_reconciliation_passed": bool(not mismatch.any()),
            "hardware_latency_measured": False}


def load_source(path: Path, protocol: dict, combine, crop) -> tuple[dict, list[dict]]:
    inventory: list[dict] = []
    with h5py.File(path, "r") as h:
        recordings = h["acquisition/timeseries"]
        if len(recordings) != 1:
            raise ValueError("Expected one recording per source file; never bridge epochs")
        prefix = f"acquisition/timeseries/{next(iter(recordings))}"
        settings_root = "general/data_collection/Settings"
        areas = {area: read_array(h, f"{settings_root}/General/channel_map/{area}/list", inventory)
                 for area in h[f"{settings_root}/General/channel_map"]}
        mapping = ca1_mapping(areas)
        bad_path = f"{settings_root}/General/badChan"
        bad = bad_channels(read_array(h, bad_path, inventory).item()) if bad_path in h else set()
        arena = read_array(h, f"{settings_root}/General/arena_size", inventory)
        processed = read_array(h, f"{prefix}/tracking/ProcessedPos", inventory)
        if processed.ndim != 2 or processed.shape[1] < 5:
            raise ValueError("Unknown ProcessedPos coordinate columns")
        raw_cameras = h[f"{prefix}/tracking"]
        camera_names = sorted(k for k in raw_cameras if k.isdigit())
        cameras, clock_rows = {}, []
        camera_settings = {"CameraSpecific": {}, "General": {}}
        pulses = None
        for name in camera_names:
            obj = raw_cameras[name]
            if isinstance(obj, h5py.Dataset):
                cameras[name] = read_array(h, obj.name, inventory)
                clock_rows.append({"camera_id": name, "mode": "source_declared_acquisition_timestamps",
                                   "pulse_count": 0})
            else:
                if pulses is None:
                    event_times = read_array(h, f"{prefix}/events/ttl1/timestamps", inventory)
                    channels = read_array(h, f"{prefix}/events/ttl1/data", inventory)
                    if channels.shape != event_times.shape:
                        raise ValueError("TTL channel/timestamp shape mismatch")
                    pulses = event_times[channels == protocol["global_clock_rising_channel"]]
                cp = read_array(h, f"{obj.name}/GlobalClock_timestamps", inventory)
                times = read_array(h, f"{obj.name}/OnlineTrackerData_timestamps", inventory)
                positions = read_array(h, f"{obj.name}/OnlineTrackerData", inventory)
                if positions.ndim != 2 or len(positions) != len(times) or positions.shape[1] < 4:
                    raise ValueError("Unknown original camera coordinate columns")
                retained = ~np.all(np.isnan(positions), axis=1)
                converted = camera_to_acquisition(pulses, cp, times[retained], protocol["camera_clock_divider"])
                cameras[name] = np.column_stack((converted, positions[retained]))
                clock_rows.append({"camera_id": name, "mode": "recorded_global_clock_pulse_conversion",
                                   "pulse_count": len(cp)})
        if len(cameras) > 1:
            base = f"{settings_root}/CameraSettings"
            camera_settings["General"]["camera_transfer_radius"] = float(read_array(
                h, f"{base}/General/camera_transfer_radius", inventory))
            for name in cameras:
                camera_settings["CameraSpecific"][name] = {"location_xy": read_array(
                    h, f"{base}/CameraSpecific/{name}/location_xy", inventory)}
        rebuilt = reconstruct_tracking(cameras, camera_settings, arena, combine, crop,
                                       protocol["multicamera_processed_sampling_rate_hz"])
        reconciliation = reconcile_processed(processed, rebuilt, protocol["clock_tolerance_s"],
                                             protocol["coordinate_tolerance_cm"])
        spikes, units = [], []
        for tetrode, area in mapping.items():
            root = f"{prefix}/spikes/electrode{tetrode + 1}"
            if root not in h:
                raise ValueError("Annotated CA1 tetrode has no source spike group")
            times = read_array(h, f"{root}/timestamps", inventory)
            keep = read_array(h, f"{root}/idx_keep", inventory)
            labels = read_array(h, f"{root}/clustering/{protocol['clustering_name']}", inventory)
            if h[f"{root}/data"].shape[0] != len(times):
                raise ValueError("Raw waveform/timestamp count disagreement")
            rows, crosswalk = curated_units(times, keep, labels, tetrode)
            spikes.append(rows)
            units.extend([{**u, "area": area} for u in crosswalk])
        spikes = np.concatenate(spikes) if spikes else np.empty((0, 2))
        # Each native tetrode chronology was checked first; interleaving tetrodes is not repairing a clock error.
        spikes = spikes[np.argsort(spikes[:, 0], kind="stable")]
        references = []
        continuous = h[f"{prefix}/continuous"]
        if len(continuous) != 1:
            raise ValueError("Ambiguous continuous acquisition processor")
        cp = next(iter(continuous.values())).name
        channels = read_array(h, f"{cp}/downsampling_info/downsampled_channels", inventory)
        rate = float(read_array(h, f"{cp}/downsampling_info/downsampled_sampling_rate", inventory))
        data = h[f"{cp}/downsampled_tetrode_data"]
        if channels.ndim != 1 or len(channels) != data.shape[1] or len(np.unique(channels)) != len(channels):
            raise ValueError("Ambiguous downsampled-channel to matrix-column mapping")
        if channels.dtype.kind not in "iu" or np.any(channels < 0):
            raise ValueError("Invalid downsampled channel identity")
        # Source uses explicit zero-based channels, not anatomical list strings or inferred column numbers.
        for area in sorted(set(mapping.values())):
            hits = [(int(c), column) for column, c in enumerate(channels)
                    if int(c) not in bad and mapping.get(int(c) // 4) == area]
            if hits:
                channel, column = min(hits)
                references.append({"area": area, "channel_zero_based": channel, "lfp_column": column})
        clock = read_array(h, f"{cp}/downsampled_timestamps", inventory)
        increasing(clock, "LFP timestamps")
        if len(clock) != data.shape[0] or not np.isfinite(rate) or rate <= 22:
            raise ValueError("LFP data/clock/rate inconsistency")
        maximum_step_error = float(np.abs(np.diff(clock) - 1 / rate).max())
        lfp_regular = maximum_step_error <= 1e-7
        within_bounds = processed[0, 0] >= clock[0] and processed[-1, 0] <= clock[-1]
        result = {"position": processed[:, :3], "spikes": spikes, "units": units,
                  "reconciliation": reconciliation, "camera_clock_rows": clock_rows,
                  "theta_references": references, "lfp_sampling_rate_hz": rate,
                  "lfp_maximum_clock_step_error_s": maximum_step_error,
                  "lfp_regular_clock": bool(lfp_regular), "tracking_within_lfp_bounds": bool(within_bounds),
                  "source_inputs_verified": bool(reconciliation["source_clock_reconciliation_passed"]
                                                 and lfp_regular and within_bounds and len(references)),
                  "theta_phase_extracted": False, "theta_phase_validated": False}
    return result, inventory


def pause_support(position: np.ndarray, spikes: np.ndarray, matching: dict) -> tuple[list[dict], dict]:
    links = epoch_links(position, np.asarray([[position[0, 0], position[-1, 0]]]), matching)
    pauses = immobile_pauses(links, matching)
    rows = []
    last_selected_end = -np.inf
    for pause in pauses:
        metrics, before, _ = matched_run(links, pause, matching)
        pre = spikes[link_spikes(spikes[:, 0], links, before)]
        _, counts = np.unique(pre[:, 1].astype(np.int64), return_counts=True)
        n_units = int(np.sum(counts >= matching["minimum_preceding_run_spikes_per_unit"]))
        supported = (metrics["matched_exposure_s"] >= matching["minimum_matched_run_exposure_s"]
                     and n_units >= matching["minimum_eligible_units"])
        start = pause["start_s"] - matching["run_search_window_s"]
        selected = supported and start >= last_selected_end
        if selected:
            last_selected_end = pause["end_s"] + matching["run_search_window_s"]
        rows.append({"pause_id": f"tracking:{pause['tracking_start_index']}", **pause, **metrics,
                     "eligible_preceding_units": n_units, "run_supported": bool(supported),
                     "earliest_nonoverlap_supported": bool(selected),
                     "candidate_events_tested": False, "theta_control_tested": False})
    return rows, {"immobile_pauses": len(rows),
                  "matched_run_unit_supported_pauses": sum(x["run_supported"] for x in rows),
                  "earliest_nonoverlap_supported_pauses": sum(x["earliest_nonoverlap_supported"] for x in rows)}


def decision_gates(sessions: list[dict]) -> list[dict]:
    verified = [x for x in sessions if x.get("source_inputs_verified", False)]
    supported = [x for x in verified if x.get("earliest_nonoverlap_supported_pauses", 0) > 0]
    # These are measurement gates, not a statistical power certificate or replication claim.
    return [
        {"gate": "sessions_present", "passed": bool(sessions), "observed": len(sessions)},
        {"gate": "all_inventory_sessions_resolved", "passed": bool(sessions) and len(verified) == len(sessions),
         "observed": len(verified)},
        {"gate": "source_verified_matched_pauses_present", "passed": bool(supported),
         "observed": sum(x["earliest_nonoverlap_supported_pauses"] for x in supported)},
        {"gate": "multiple_animals_with_verified_matched_pauses", "passed": len({x["animal"] for x in supported}) > 1,
         "observed": len({x["animal"] for x in supported})},
        {"gate": "theta_control_implemented_and_validated", "passed": False, "observed": 0},
        {"gate": "replay_order_and_run_change_tested", "passed": False, "observed": 0},
        {"gate": "overall_goal_complete", "passed": False, "observed": 0},
    ]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--acquisition-source", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--matching-protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    p, matching = json.loads(args.protocol.read_text()), json.loads(args.matching_protocol.read_text())
    provenance = build_script_provenance(input_paths={"protocol": args.protocol,
        "matching_protocol": args.matching_protocol, "acquisition_source": args.acquisition_source})
    combine, crop, source_hash = source_combiner(args.acquisition_source, p["acquisition_source_commit"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.exists() or (args.output_dir / "session_inventory.csv").exists():
        raise ValueError("Use a new audit directory; do not overwrite frozen results")
    files = sorted(args.dataset_root.rglob("*.nwb"))
    sessions, pause_rows, unit_rows, clock_rows, inputs = [], [], [], [], []
    for path in files:
        relative = str(path.relative_to(args.dataset_root))
        identity = {"animal": path.relative_to(args.dataset_root).parts[0], "session": relative,
                    "path": str(path), "size_bytes": path.stat().st_size,
                    "mtime_ns": path.stat().st_mtime_ns}
        print(f"START {relative}", flush=True)
        try:
            loaded, input_rows = load_source(path, p, combine, crop)
            pauses, counts = pause_support(loaded["position"], loaded["spikes"], matching)
            session = {**identity, **counts, **loaded["reconciliation"],
                       **{key: loaded[key] for key in ("lfp_sampling_rate_hz", "lfp_regular_clock",
                           "lfp_maximum_clock_step_error_s", "tracking_within_lfp_bounds",
                           "source_inputs_verified", "theta_phase_extracted", "theta_phase_validated")},
                       "curated_ca1_units": len(loaded["units"]), "theta_reference_count": len(loaded["theta_references"]),
                       "theta_references_json": json.dumps(loaded["theta_references"], sort_keys=True),
                       "status": "source_verified_measurement_only" if loaded["source_inputs_verified"]
                                 else "unresolved_source_reconciliation"}
            pause_rows.extend([{**identity, **x} for x in pauses])
            unit_rows.extend([{**identity, **x} for x in loaded["units"]])
            clock_rows.extend([{**identity, **x} for x in loaded["camera_clock_rows"]])
            inputs.extend([{**identity, **x} for x in input_rows])
        except (ValueError, KeyError, OSError) as exc:
            session = {**identity, "source_inputs_verified": False,
                       "status": "unresolved_input", "error": f"{type(exc).__name__}: {exc}"}
        sessions.append(session)
        print(f"END {relative}: {session['status']} supported={session.get('earliest_nonoverlap_supported_pauses', 0)}",
              flush=True)
        (args.output_dir / "progress.json").write_text(json.dumps(
            {"finished_sessions": len(sessions), "total_sessions": len(files), "latest": session}, indent=2) + "\n")
    gates = decision_gates(sessions)
    frames = {"session_inventory.csv": sessions, "matched_pause_inventory.csv": pause_rows,
              "unit_crosswalk.csv": unit_rows, "camera_clock_inventory.csv": clock_rows,
              "consumed_array_inventory.csv": inputs, "gate_summary.csv": gates}
    for name, rows in frames.items():
        pd.DataFrame(rows).to_csv(args.output_dir / name, index=False)
    manifest = {**provenance, "protocol_id": p["protocol_id"],
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "acquisition_reconstruction_file_sha256": source_hash,
                "scientific_array_integrity_scope": "SHA256 of every consumed numerical/fixed-string dataset; not full NWB waveform/file integrity",
                "dataset_files": [{key: x[key] for key in ("animal", "session", "path", "size_bytes", "mtime_ns")}
                                  for x in sessions],
                "outputs_sha256": {name: file_sha256(args.output_dir / name) for name in frames},
                "association_fit": False, "theta_phase_validated": False, "pf_stopgate_unchanged": True}
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    supported = [x for x in sessions if x.get("source_inputs_verified", False)
                 and x.get("earliest_nonoverlap_supported_pauses", 0) > 0]
    (args.output_dir / "README.md").write_text(
        "# Tanni Replay-to-RUN Input Audit\n\n"
        f"Files: {len(sessions)}; source-resolved files: {sum(x.get('source_inputs_verified', False) for x in sessions)}.\n\n"
        f"Verified earliest nonoverlapping supported pauses: {sum(x['earliest_nonoverlap_supported_pauses'] for x in supported)} "
        f"across {len({x['animal'] for x in supported})} animals.\n\n"
        "Thresholds are inherited unchanged from the PF measurement protocol. Pauses are not replay events. "
        "Neither theta extraction/QC nor the requested RUN-coordination change association was completed. "
        "Source reconstruction reconciles processed timestamps and coordinates with documented acquisition conventions; "
        "it does not independently measure camera hardware latency. Missing/unresolved recordings remain in the denominator. "
        "The original PF missing-theta and historical negative stopgates remain unchanged.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
