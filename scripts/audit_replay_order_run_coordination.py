"""Audit matched RUN support and count-preserving candidate pair-order controls.

This entry point does not fit the biological association. In particular, ripple
power is not theta phase, and these candidates are not validated replay events.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import pandas as pd
from scipy.io import whosmat

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hipporeplayimm.data import load_replay_session  # noqa: E402

try:
    from scripts._provenance import build_script_provenance, file_sha256
except ModuleNotFoundError:
    from _provenance import build_script_provenance, file_sha256


def epoch_links(position: np.ndarray, epochs: np.ndarray, p: dict) -> dict:
    """Keep native chronology and invalidate links rather than sorting errors."""
    a = np.asarray(position, dtype=float)
    if a.ndim != 2 or a.shape[1] < 3 or len(a) < 2:
        raise ValueError("Position requires at least two timestamp/x/y rows")
    t, xy = a[:, 0], a[:, 1:3]
    if not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError("Nonfinite, duplicate or backward tracking timestamps")
    dt = np.diff(t)
    good = (dt <= p["tracking_max_gap_s"]) & np.isfinite(xy[:-1]).all(axis=1)
    good &= np.isfinite(xy[1:]).all(axis=1)
    delta = np.diff(xy, axis=0)
    speed = np.linalg.norm(delta, axis=1) / dt
    ep = np.full(len(dt), -1, dtype=int)
    intervals = np.asarray(epochs, dtype=float).reshape(-1, 2)
    if not np.isfinite(intervals).all() or np.any(intervals[:, 1] <= intervals[:, 0]):
        raise ValueError("Invalid RUN epochs")
    for j, (start, end) in enumerate(intervals):
        hits = (t[:-1] >= start) & (t[1:] <= end)
        if np.any(hits & (ep >= 0)):
            raise ValueError("Overlapping RUN epochs")
        ep[hits] = j
    good &= ep >= 0
    direction = np.mod(np.arctan2(delta[:, 1], delta[:, 0]), 2 * np.pi)
    return {"t": t, "xy": xy, "dt": dt, "speed": speed,
            "direction": direction, "good": good, "epoch": ep}


def immobile_pauses(links: dict, p: dict) -> list[dict]:
    valid = links["good"] & (links["speed"] <= p["pause_max_speed_cm_s"])
    groups: list[dict] = []
    i = 0
    while i < len(valid):
        if not valid[i]:
            i += 1
            continue
        end = i + 1
        while end < len(valid) and valid[end] and links["epoch"][end] == links["epoch"][i]:
            end += 1
        start_s, end_s = links["t"][i], links["t"][end]
        if end_s - start_s >= p["pause_min_duration_s"]:
            groups.append({"tracking_start_index": i, "epoch_index": int(links["epoch"][i]),
                           "start_s": float(start_s), "end_s": float(end_s)})
        i = end
    return groups


def matched_run(links: dict, pause: dict, p: dict) -> tuple[dict, np.ndarray, np.ndarray]:
    t0, t1 = links["t"][:-1], links["t"][1:]
    run = links["good"] & (links["epoch"] == pause["epoch_index"])
    run &= (links["speed"] > p["run_min_speed_cm_s"]) & (links["speed"] <= p["run_max_speed_cm_s"])
    window = p["run_search_window_s"]
    before = run & (t0 >= pause["start_s"] - window) & (t1 <= pause["start_s"])
    after = run & (t0 >= pause["end_s"]) & (t1 <= pause["end_s"] + window)
    indices = np.flatnonzero(before | after)
    if len(indices) == 0:
        return {"common_strata": 0, "matched_exposure_s": 0.0,
                "before_run_exposure_s": 0.0, "after_run_exposure_s": 0.0}, before, after
    position = np.floor(links["xy"][indices] / p["match_position_bin_cm"]).astype(int)
    direction = np.floor(links["direction"][indices] * p["match_direction_bins"] / (2 * np.pi)).astype(int)
    speed = np.searchsorted(p["match_speed_edges_cm_s"], links["speed"][indices], side="right") - 1
    speed = np.minimum(speed, len(p["match_speed_edges_cm_s"]) - 2)
    keys = np.column_stack((position, direction, speed))
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    n = int(inverse.max()) + 1
    pre = np.bincount(inverse, weights=links["dt"][indices] * before[indices], minlength=n)
    post = np.bincount(inverse, weights=links["dt"][indices] * after[indices], minlength=n)
    common = (pre > 0) & (post > 0)
    selected = indices[common[inverse]]
    matched_before = np.zeros_like(before)
    matched_after = np.zeros_like(after)
    matched_before[selected] = before[selected]
    matched_after[selected] = after[selected]
    metrics = {"common_strata": int(common.sum()),
               "matched_exposure_s": float(np.minimum(pre, post).sum()),
               "before_run_exposure_s": float(links["dt"][before].sum()),
               "after_run_exposure_s": float(links["dt"][after].sum())}
    return metrics, matched_before, matched_after


def link_spikes(times: np.ndarray, links: dict, mask: np.ndarray) -> np.ndarray:
    indices = np.searchsorted(links["t"], times, side="right") - 1
    inside = (indices >= 0) & (indices < len(mask))
    result = np.zeros(len(times), dtype=bool)
    result[inside] = mask[indices[inside]]
    return result


def event_counts(spikes: np.ndarray, ids: np.ndarray, start: float, end: float, target_bin: float) -> tuple[np.ndarray, float]:
    if not np.isfinite([start, end, target_bin]).all() or end <= start or target_bin <= 0:
        raise ValueError("Invalid event boundaries/bin size")
    n = max(1, int(np.ceil((end - start) / target_bin)))
    width = (end - start) / n
    spikes = np.asarray(spikes, dtype=float).reshape(-1, 2)
    hits = spikes[(spikes[:, 0] >= start) & (spikes[:, 0] < end)]
    counts = np.zeros((n, len(ids)), dtype=np.int64)
    mapping = {int(cid): j for j, cid in enumerate(ids)}
    for t, cid in hits:
        if int(cid) in mapping:
            b = min(n - 1, int((t - start) / width))
            counts[b, mapping[int(cid)]] += 1
    return counts, width


def order_asymmetry(counts: np.ndarray, width_s: float, min_lag_s: float, max_lag_s: float) -> np.ndarray:
    counts = np.asarray(counts)
    if counts.ndim != 2 or not np.isfinite(counts).all() or np.any(counts < 0):
        raise ValueError("Invalid event count matrix")
    if not np.isfinite([width_s, min_lag_s, max_lag_s]).all() or width_s <= 0 or min_lag_s < 0 or max_lag_s < min_lag_s:
        raise ValueError("Invalid lag specification")
    forward = np.zeros((counts.shape[1], counts.shape[1]), dtype=float)
    for lag in range(1, len(counts)):
        delta = lag * width_s
        if delta + 1e-12 >= min_lag_s and delta <= max_lag_s + 1e-12:
            forward += counts[:-lag].T @ counts[lag:]
    totals = counts.sum(axis=0).astype(float)
    denominator = np.outer(totals, totals)
    return np.divide(forward - forward.T, denominator,
                     out=np.zeros_like(forward), where=denominator > 0)


def whole_bin_shuffles(counts: np.ndarray, n: int, seed: int) -> list[np.ndarray]:
    if isinstance(n, bool) or int(n) != n or n < 1:
        raise ValueError("Positive integer shuffle count required")
    rng = np.random.default_rng(seed)
    return [np.asarray(counts)[rng.permutation(len(counts))].copy() for _ in range(n)]


def stable_seed(seed: int, identity: str) -> int:
    digest = hashlib.sha256(f"{seed}:{identity}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def variable_inventory(folder: Path) -> list[dict]:
    rows = []
    for f in sorted(folder.iterdir()):
        if not f.is_file():
            continue
        variables = []
        if f.suffix.lower() == ".mat":
            if h5py.is_hdf5(f):
                with h5py.File(f, "r") as handle:
                    for key, value in handle.items():
                        if key != "#refs#":
                            variables.append({"name": key, "shape": list(value.shape) if isinstance(value, h5py.Dataset) else None})
            else:
                variables = [{"name": key, "shape": list(shape)} for key, shape, _ in whosmat(f)]
        rows.append({"path": str(f), "sha256": file_sha256(f), "size_bytes": f.stat().st_size,
                     "variables": variables})
    return rows


def tanni_lfp_headers(root: Path) -> list[dict]:
    """Bounded header/clock reads only: availability is not synchronization."""
    rows = []
    for f in sorted(root.rglob("*.nwb")):
        with h5py.File(f, "r") as handle:
            recordings = handle.get("acquisition/timeseries")
            if recordings is None:
                rows.append({"path": str(f), "status": "unrecognized_source_schema"})
                continue
            for name, recording in recordings.items():
                pos = recording.get("tracking/ProcessedPos")
                continuous = recording.get("continuous")
                if pos is None or continuous is None:
                    rows.append({"path": str(f), "recording": name, "status": "missing_tracking_or_lfp"})
                    continue
                for processor, group in continuous.items():
                    data = group.get("downsampled_tetrode_data")
                    clock = group.get("downsampled_timestamps")
                    rate = group.get("downsampling_info/downsampled_sampling_rate")
                    if data is None or clock is None or not len(clock):
                        continue
                    tracking_start, tracking_end = float(pos[0, 0]), float(pos[-1, 0])
                    lfp_start, lfp_end = float(clock[0]), float(clock[-1])
                    row = {"path": str(f), "size_bytes": f.stat().st_size,
                           "recording": name, "processor": processor,
                           "lfp_samples": len(clock), "lfp_channels": data.shape[1],
                           "sampling_rate_hz": float(rate[()]) if rate is not None else None,
                           "tracking_rows": len(pos), "tracking_start_s": tracking_start,
                           "tracking_end_s": tracking_end, "lfp_start_s": lfp_start, "lfp_end_s": lfp_end,
                           "tracking_within_lfp_clock_bounds": tracking_start >= lfp_start and tracking_end <= lfp_end,
                           "clock_alignment_verified": False, "ca1_channel_identity_verified": False,
                           "matched_run_replication_tested": False,
                           "status": "lfp_available_not_theta_or_replication_validation",
                           "integrity_scope": "header_and_endpoint_reads_not_full_file_checksum"}
                    row["header_sha256"] = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()
                    rows.append(row)
    return rows


def audit_session(folder: Path, p: dict) -> tuple[list, list, list, list, dict]:
    session = load_replay_session(folder)
    links = epoch_links(session.position, session.run_times, p)
    pauses = immobile_pauses(links, p)
    spikes = session.excitatory_spikes()
    if not np.isfinite(spikes).all() or np.any(np.diff(spikes[:, 0]) < 0):
        raise ValueError("Nonfinite or backward spike timestamps")
    inventory = variable_inventory(folder)
    phase_inputs = [f["path"] for f in inventory if
                    any(word in Path(f["path"]).name.lower() for word in ("theta", "lfp", "eeg")) or
                    any("theta" in v["name"].lower() or "lfp" in v["name"].lower()
                        for v in f["variables"])]
    theta_status = ("potential_phase_input_requires_adapter_and_clock_verification" if phase_inputs else
                    "no_lfp_or_theta_phase_in_verified_pf_source_schema")
    pause_rows, event_rows, pair_rows = [], [], []
    last_selected_end = -np.inf
    for pause in pauses:
        identity = f"{session.session_id}:pause:{pause['tracking_start_index']}"
        metrics, before, _ = matched_run(links, pause, p)
        pre_hits = link_spikes(spikes[:, 0], links, before)
        ids, n = np.unique(spikes[pre_hits, 1].astype(int), return_counts=True)
        ids = ids[n >= p["minimum_preceding_run_spikes_per_unit"]]
        event_indices = np.flatnonzero((session.ripple_events[:, 0] >= pause["start_s"]) &
                                      (session.ripple_events[:, 1] <= pause["end_s"]))
        matched = metrics["matched_exposure_s"] >= p["minimum_matched_run_exposure_s"]
        supported = matched and len(ids) >= p["minimum_eligible_units"]
        selected = supported and pause["start_s"] - p["run_search_window_s"] >= last_selected_end
        if selected:
            last_selected_end = pause["end_s"] + p["run_search_window_s"]
        row = {"animal": session.rat, "session": session.session_id, "pause_id": identity,
               **pause, **metrics, "eligible_preceding_units": len(ids),
               "native_contained_events": len(event_indices), "matched_run_supported": supported,
               "nonoverlapping_measurement_subset": selected, "theta_phase_available": False,
               "eligible_candidate_events": 0, "candidate_pair_rows": 0}
        for index in event_indices:
            event = session.ripple(int(index))
            counts, width = event_counts(spikes, ids, event.start, event.end, p["order_target_bin_s"])
            active = counts.sum(axis=0) > 0
            reasons = []
            duration = event.end - event.start
            if not p["candidate_min_duration_s"] <= duration <= p["candidate_max_duration_s"]:
                reasons.append("duration")
            if counts.sum() < p["candidate_min_spikes"]:
                reasons.append("spikes")
            if active.sum() < p["candidate_min_active_units"]:
                reasons.append("active_units")
            if not supported:
                reasons.append("matched_run_or_past_unit_support")
            event_id = f"{session.session_id}:native:{int(index)}"
            erow = {"animal": session.rat, "session": session.session_id,
                    "pause_id": identity, "event_id": event_id, "event_index": int(index),
                    "start_s": event.start, "end_s": event.end, "duration_s": duration,
                    "n_eligible_spikes": int(counts.sum()), "n_active_eligible_units": int(active.sum()),
                    "eligible_candidate": not reasons, "exclusion_reason": ";".join(reasons),
                    "label": p["candidate_label"], "order_measured": False,
                    "shuffle_invariants_verified": False}
            if not reasons:
                row["eligible_candidate_events"] += 1
            if not reasons and selected:
                counts, active_ids = counts[:, active], ids[active]
                original = order_asymmetry(counts, width, p["order_min_lag_s"], p["order_max_lag_s"])
                shuffled = whole_bin_shuffles(counts, p["order_shuffles"], stable_seed(p["seed"], event_id))
                invariant = all(np.array_equal(c.sum(axis=0), counts.sum(axis=0)) and
                                sorted(map(tuple, c)) == sorted(map(tuple, counts)) for c in shuffled)
                if not invariant:
                    raise AssertionError("Whole-bin shuffle changed participation/count vectors")
                nulls = np.stack([order_asymmetry(c, width, p["order_min_lag_s"], p["order_max_lag_s"])
                                 for c in shuffled])
                for i, a in enumerate(active_ids):
                    for j in range(i + 1, len(active_ids)):
                        pair_rows.append({"animal": session.rat, "session": session.session_id,
                                          "pause_id": identity, "event_id": event_id,
                                          "cell_a": int(a), "cell_b": int(active_ids[j]),
                                          "a_before_b_asymmetry": original[i, j],
                                          "shuffle_mean_asymmetry": float(nulls[:, i, j].mean()),
                                          "shuffle_sd_asymmetry": float(nulls[:, i, j].std(ddof=1)),
                                          "n_shuffles": len(shuffled), "bin_width_s": width,
                                          "a_spikes": int(counts[:, i].sum()), "b_spikes": int(counts[:, j].sum()),
                                          "candidate_not_validated_replay": True,
                                          "independent_biological_replicate": False})
                row["candidate_pair_rows"] += len(active_ids) * (len(active_ids) - 1) // 2
                erow["order_measured"] = True
                erow["shuffle_invariants_verified"] = True
            event_rows.append(erow)
        pause_rows.append(row)
    summary = {"animal": session.rat, "session": session.session_id,
               "native_tracking_rows": len(session.position), "native_spikes": len(session.spikes),
               "excitatory_cells": len(session.excitatory_neurons), "immobile_pauses": len(pauses),
               "matched_run_supported_pauses": sum(x["matched_run_supported"] for x in pause_rows),
               "nonoverlapping_supported_pauses": sum(x["nonoverlapping_measurement_subset"] for x in pause_rows),
               "order_measured_events": sum(x["order_measured"] for x in event_rows),
               "candidate_pair_rows": len(pair_rows), "theta_phase_available": False,
               "full_requested_control_stack_available": False,
               "theta_status": theta_status, "potential_phase_inputs": json.dumps(phase_inputs)}
    return pause_rows, event_rows, pair_rows, inventory, summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tanni-root", type=Path)
    args = parser.parse_args(argv)
    p = json.loads(args.protocol.read_text())
    if p["primary_analysis_enabled"] or not p["require_theta_phase"]:
        raise ValueError("Measurement audit cannot enable analysis or remove mandatory theta control")
    root = args.output_dir
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise ValueError("Use an empty output directory; preserve completed audits")
    pause_rows, event_rows, pair_rows, inputs, summaries = [], [], [], [], []
    for session in p["sessions"]:
        rows = audit_session(args.dataset_root / session, p)
        pause_rows.extend(rows[0])
        event_rows.extend(rows[1])
        pair_rows.extend(rows[2])
        inputs.extend(rows[3])
        summaries.append(rows[4])
        print(json.dumps(rows[4]), flush=True)
    tables = {"pause_inventory": pause_rows, "candidate_event_inventory": event_rows,
              "candidate_pair_order": pair_rows, "session_summary": summaries}
    for name, rows in tables.items():
        pd.DataFrame(rows).to_csv(root / f"{name}.csv", index=False)
    if args.tanni_root is not None:
        pd.DataFrame(tanni_lfp_headers(args.tanni_root)).to_csv(root / "tanni_replication_input_headers.csv", index=False)
    by_animal = pd.DataFrame(summaries).groupby("animal", as_index=False).agg(
        sessions=("session", "count"), immobile_pauses=("immobile_pauses", "sum"),
        supported_pauses=("matched_run_supported_pauses", "sum"),
        nonoverlapping_pauses=("nonoverlapping_supported_pauses", "sum"),
        order_measured_events=("order_measured_events", "sum"),
        candidate_pair_rows=("candidate_pair_rows", "sum"))
    by_animal.to_csv(root / "animal_support_summary.csv", index=False)
    pd.DataFrame([
        {"gate": "matched_run_measurements_present", "passed": any(s["nonoverlapping_supported_pauses"] > 0 for s in summaries)},
        {"gate": "candidate_pair_order_measured", "passed": bool(pair_rows)},
        {"gate": "whole_bin_shuffle_invariants", "passed": bool(pair_rows) and all(e["shuffle_invariants_verified"] for e in event_rows if e["order_measured"])},
        {"gate": "theta_control_available", "passed": False},
        {"gate": "pair_coordination_change_endpoint_implemented", "passed": False},
        {"gate": "association_calibrated_and_evaluated", "passed": False},
        {"gate": "full_goal_complete", "passed": False}
    ]).to_csv(root / "gate_summary.csv", index=False)
    manifest = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
                "status": "partial_measurement_audit_missing_pf_theta_not_biological_test",
                "protocol": p, "input_files": inputs,
                "provenance": build_script_provenance(input_paths={"protocol": args.protocol}),
                "environment_versions": {"python": sys.version, **{package: version(package)
                    for package in ("numpy", "scipy", "pandas", "h5py")}},
                "outputs": {str(f.name): file_sha256(f) for f in root.glob("*.csv")},
                "remaining_requirements": ["LFP-derived clock-verified PF theta phases", "validated replay identity",
                    "covariate-adjusted pairwise RUN change endpoint", "complete calibration and animal-level inference",
                    "independent verification of saved measurements", "Tanni matching/replication feasibility"]}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (root / "README.md").write_text(
        "# Replay order and subsequent RUN coordination: measurement audit\n\n"
        "This is partial progress, not completion or a biological result. Native immobile ripple candidates "
        "are not validated replay. Pair rows are dependent measurements, not independent subjects.\n\n"
        "RUN exposure matches 8 cm position bins, movement-direction sectors and speed categories. "
        "Matched exposure is the sum of the smaller pre/post exposure in each common stratum. "
        "This inventory does not yet estimate covariate-adjusted RUN coordination.\n\n"
        "The local PF release contains no LFP or theta phase in its verified source schema. "
        "Ripple-power scalars cannot replace theta phase. Full-goal and biological gates remain false.\n\n"
        "Order shuffles permute equal-width population count bins spanning each original event. "
        "Every bin vector, unit spike total, event duration and participation set is preserved. "
        "These order summaries alone cannot show updating in later RUN activity.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
