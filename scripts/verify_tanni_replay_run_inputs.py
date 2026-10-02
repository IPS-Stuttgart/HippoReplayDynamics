"""Independently check consumed-array hashes, unit counts and RUN matching.

This does not independently validate theta, replay sequences or an association.
Source camera reconstruction is checked by the producer, not repeated here.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

try:
    from scripts._provenance import build_script_provenance, file_sha256
except ModuleNotFoundError:
    from _provenance import build_script_provenance, file_sha256


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def reference_pauses(position: np.ndarray, p: dict) -> list[tuple[int, float, float]]:
    t, xy = position[:, 0], position[:, 1:3]
    require(np.isfinite(t).all() and np.all(np.diff(t) > 0), "Invalid native position chronology")
    rows = []
    start = None
    for i in range(len(position) - 1):
        dt = t[i + 1] - t[i]
        good = dt <= p["tracking_max_gap_s"] and np.isfinite(xy[i:i + 2]).all()
        immobile = good and np.linalg.norm(xy[i + 1] - xy[i]) / dt <= p["pause_max_speed_cm_s"]
        if immobile and start is None:
            start = i
        if start is not None and (not immobile or i == len(position) - 2):
            end = i if not immobile else i + 1
            if t[end] - t[start] >= p["pause_min_duration_s"]:
                rows.append((start, float(t[start]), float(t[end])))
            start = None
    return rows


def reference_match(position: np.ndarray, spikes: np.ndarray, pause: tuple[float, float], p: dict) -> dict:
    t = position[:, 0]
    pre, post = defaultdict(float), defaultdict(float)
    pre_links: dict[int, tuple] = {}
    start, end = pause
    first = max(0, int(np.searchsorted(t, start - p["run_search_window_s"]) - 1))
    last = min(len(t) - 1, int(np.searchsorted(t, end + p["run_search_window_s"], side="right")))
    for i in range(first, last):
        dt = t[i + 1] - t[i]
        if dt > p["tracking_max_gap_s"] or not np.isfinite(position[i:i + 2, 1:3]).all():
            continue
        displacement = position[i + 1, 1:3] - position[i, 1:3]
        speed = np.linalg.norm(displacement) / dt
        if not p["run_min_speed_cm_s"] < speed <= p["run_max_speed_cm_s"]:
            continue
        is_pre = t[i] >= start - p["run_search_window_s"] and t[i + 1] <= start
        is_post = t[i] >= end and t[i + 1] <= end + p["run_search_window_s"]
        if not (is_pre or is_post):
            continue
        xy = np.floor(position[i, 1:3] / p["match_position_bin_cm"]).astype(int)
        angle = np.arctan2(displacement[1], displacement[0]) % (2 * np.pi)
        sector = int(np.floor(angle * p["match_direction_bins"] / (2 * np.pi)))
        speed_bin = min(sum(speed >= edge for edge in p["match_speed_edges_cm_s"]) - 1,
                        len(p["match_speed_edges_cm_s"]) - 2)
        key = (int(xy[0]), int(xy[1]), sector, speed_bin)
        if is_pre:
            pre[key] += dt
            pre_links[i] = key
        if is_post:
            post[key] += dt
    common = set(pre) & set(post)
    counts = Counter()
    if len(spikes):
        near = spikes[(spikes[:, 0] >= start - p["run_search_window_s"]) & (spikes[:, 0] < start)]
        indices = np.searchsorted(t, near[:, 0], side="right") - 1
        for index, cell in zip(indices, near[:, 1], strict=True):
            if pre_links.get(int(index)) in common:
                counts[int(cell)] += 1
    return {"common_strata": len(common),
            "matched_exposure_s": sum(min(pre[key], post[key]) for key in common),
            "before_run_exposure_s": sum(pre.values()),
            "after_run_exposure_s": sum(post.values()),
            "eligible_preceding_units": sum(n >= p["minimum_preceding_run_spikes_per_unit"] for n in counts.values())}


def source_spikes_and_units(h: h5py.File, prefix: str) -> tuple[np.ndarray, dict[int, int]]:
    source = h["general/data_collection/Settings/General/channel_map"]
    tetrodes = set()
    for area in source:
        if area.startswith("CA1_"):
            tetrodes.update(int(c) // 4 for c in source[area]["list"][()])
    spikes, unit_counts = [], {}
    for tetrode in sorted(tetrodes):
        group = h[f"{prefix}/spikes/electrode{tetrode + 1}"]
        keep = group["idx_keep"][()]
        timestamps = group["timestamps"][()][keep]
        labels = group["clustering/manual_1"][()]
        require(len(timestamps) == len(labels), "Curated timestamp/label mismatch")
        require(keep.dtype.kind == "b", "Non-Boolean source artifact mask")
        for cluster in np.unique(labels):
            if cluster <= 0:
                continue
            cell = tetrode * 65536 + int(cluster)
            selected = timestamps[labels == cluster]
            require(cell not in unit_counts, "Duplicate canonical unit identity")
            unit_counts[cell] = len(selected)
            spikes.append(np.column_stack((selected, np.full(len(selected), cell))))
    return np.concatenate(spikes) if spikes else np.empty((0, 2)), unit_counts


def verify(root: Path) -> dict:
    manifest = json.loads((root / "manifest.json").read_text())
    for name, expected in manifest["outputs_sha256"].items():
        require(file_sha256(root / name) == expected, f"Output checksum mismatch: {name}")
    matching_path = Path(manifest["input_file_paths"]["matching_protocol"])
    require(file_sha256(matching_path) == manifest["input_file_sha256"]["matching_protocol"],
            "Changed frozen matching protocol")
    p = json.loads(matching_path.read_text())
    sessions = pd.read_csv(root / "session_inventory.csv")
    pauses = pd.read_csv(root / "matched_pause_inventory.csv")
    units = pd.read_csv(root / "unit_crosswalk.csv")
    arrays = pd.read_csv(root / "consumed_array_inventory.csv")
    require(not sessions.empty and not sessions["session"].duplicated().any(), "Empty/duplicate session inventory")
    require(not pauses[["session", "pause_id"]].duplicated().any(), "Duplicate pause identity")
    require(not units[["session", "unit_id"]].duplicated().any(), "Duplicate unit identity")
    verified_arrays, checked_pauses = 0, 0
    supported_animals, supported_pauses = set(), 0
    failures = []
    for _, session in sessions.iterrows():
        identity = session["session"]
        print(f"VERIFY {identity}", flush=True)
        if session["status"] == "unresolved_input":
            require(identity not in set(pauses["session"]), "Unresolved source acquired usable pause rows")
            failures.append({"session": identity, "error": session["error"]})
            continue
        path = Path(session["path"])
        require(path.stat().st_size == session["size_bytes"] and path.stat().st_mtime_ns == session["mtime_ns"],
                "Source identity changed since producer run")
        with h5py.File(path, "r") as h:
            for _, field in arrays[arrays["session"] == identity].iterrows():
                a = np.asarray(h[field["hdf5_path"]][()])
                if a.dtype.kind == "O" and a.ndim == 0 and isinstance(a.item(), bytes):
                    a = np.asarray(a.item())
                header = json.dumps({"shape": a.shape, "dtype": a.dtype.str}, sort_keys=True).encode()
                digest = hashlib.sha256(header)
                digest.update(a.tobytes(order="C"))
                require(digest.hexdigest() == field["array_sha256"], f"Consumed source array changed: {field['hdf5_path']}")
                verified_arrays += 1
            recordings = h["acquisition/timeseries"]
            require(len(recordings) == 1, "Ambiguous source epoch")
            prefix = f"acquisition/timeseries/{next(iter(recordings))}"
            position = h[f"{prefix}/tracking/ProcessedPos"][()]
            spikes, raw_units = source_spikes_and_units(h, prefix)
            observed_units = units[units["session"] == identity]
            require({int(r.unit_id): int(r.retained_spikes) for r in observed_units.itertuples()} == raw_units,
                    "Source unit count/canonical ID disagreement")
            actual_pauses = reference_pauses(position, p)
            observed = pauses[pauses["session"] == identity]
            require(len(actual_pauses) == len(observed) == int(session["immobile_pauses"]), "Pause accounting mismatch")
            last_end = -np.inf
            supported, selected = 0, 0
            for (index, start, end), row in zip(actual_pauses, observed.itertuples(), strict=True):
                require(row.pause_id == f"tracking:{index}"
                        and np.isclose(row.start_s, start, atol=1e-9, rtol=0)
                        and np.isclose(row.end_s, end, atol=1e-9, rtol=0),
                        "Reconstructed pause identity/boundary mismatch")
                metrics = reference_match(position, spikes, (start, end), p)
                for key, value in metrics.items():
                    require(np.isclose(getattr(row, key), value, atol=1e-8, rtol=1e-10), f"RUN match disagreement: {key}")
                good = metrics["matched_exposure_s"] >= p["minimum_matched_run_exposure_s"]
                good &= metrics["eligible_preceding_units"] >= p["minimum_eligible_units"]
                chosen = good and start - p["run_search_window_s"] >= last_end
                require(bool(row.run_supported) == bool(good) and bool(row.earliest_nonoverlap_supported) == bool(chosen),
                        "Support/nonoverlap flag disagreement")
                supported += int(good)
                selected += int(chosen)
                if chosen:
                    last_end = end + p["run_search_window_s"]
                checked_pauses += 1
            require(supported == session["matched_run_unit_supported_pauses"]
                    and selected == session["earliest_nonoverlap_supported_pauses"], "Session supported-pause totals differ")
            if session["source_inputs_verified"] and selected:
                supported_animals.add(session["animal"])
                supported_pauses += selected
    gates = pd.read_csv(root / "gate_summary.csv").set_index("gate")
    require(int(gates.loc["source_verified_matched_pauses_present", "observed"]) == supported_pauses,
            "Verified matched-pause gate denominator mismatch")
    require(int(gates.loc["multiple_animals_with_verified_matched_pauses", "observed"]) == len(supported_animals),
            "Animal coverage gate denominator mismatch")
    require(not bool(gates.loc["overall_goal_complete", "passed"]), "Input audit must not claim complete goal")
    return {"verified": True, "consumed_arrays_verified": verified_arrays,
            "pauses_independently_reconstructed_and_matched": checked_pauses,
            "verified_supported_nonoverlap_pauses": supported_pauses,
            "verified_supported_animals": sorted(supported_animals), "unresolved_sessions": failures,
            "source_camera_reconstruction_independently_verified": False,
            "theta_phase_validated": False, "replay_or_run_change_association_tested": False,
            "verification_scope": "All exported consumed-array hashes, curated unit counts, native pause boundaries and RUN match metrics; not full NWB integrity or biological calibration"}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    require(not args.output.exists(), "Do not overwrite a frozen verification")
    result = verify(args.audit_dir)
    result.update(build_script_provenance(input_paths={"manifest": args.audit_dir / "manifest.json"}))
    result["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
