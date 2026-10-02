"""Measure LFP theta and count-preserving event order for frozen matched pauses.

RUN banks retain zero-spike bins and all frozen pauses. This is not the final
theta-adjusted coordination-change association or independent replay validation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import pandas as pd
from scipy.fftpack import next_fast_len
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, convolve, filtfilt, hilbert, welch
from scipy.signal.windows import gaussian

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._provenance import build_script_provenance, file_sha256  # noqa: E402
from scripts.audit_replay_order_run_coordination import (  # noqa: E402
    epoch_links, event_counts, immobile_pauses, link_spikes, matched_run, order_asymmetry,
    stable_seed, whole_bin_shuffles,
)
from scripts.audit_tanni_replay_run_inputs import (  # noqa: E402
    array_digest, load_source, source_combiner,
)


def runs(mask):
    delta = np.diff(np.r_[False, np.asarray(mask, bool), False].astype(np.int8))
    return list(zip(np.flatnonzero(delta == 1), np.flatnonzero(delta == -1), strict=True))


def checked(path, digest):
    if file_sha256(path) != digest:
        raise ValueError(f"Changed input: {path}")


def source_phase(raw, fs, p):
    raw = np.asarray(raw)
    if raw.ndim != 1 or not np.isfinite(fs) or fs <= 2 * p["theta_band_hz"][1]:
        raise ValueError("One native LFP channel and a valid sampling rate required")
    finite = np.isfinite(raw)
    if raw.dtype.kind in "iu":
        bounds = np.iinfo(raw.dtype)
        finite &= (raw != bounds.min) & (raw != bounds.max)
    phase = np.full(len(raw), np.nan)
    valid = np.zeros(len(raw), bool)
    guard = int(np.ceil(p["lfp_edge_guard_s"] * fs))
    b, a = butter(p["theta_filter_order"], np.asarray(p["theta_band_hz"]) / (fs / 2), btype="band")
    sigma = p["theta_phase_smoothing_sigma_s"] * fs
    width = int(round(sigma * 10))
    width = width - 1 if width % 2 == 0 else width
    kernel = gaussian(max(1, width), sigma)
    kernel /= kernel.sum()
    for start, end in runs(finite):
        if end - start <= max(2 * guard, 3 * max(len(a), len(b))):
            continue
        if np.std(raw[start:end].astype(float)) <= 0:
            continue
        filtered = filtfilt(b, a, raw[start:end].astype(float))
        if np.std(filtered[guard:-guard]) <= 0:
            continue
        analytic = hilbert(filtered, next_fast_len(len(filtered)))[:len(filtered)]
        angle = convolve(np.unwrap(np.angle(analytic)), kernel, mode="same")
        phase[start:end] = (angle + np.pi) % (2 * np.pi) - np.pi
        valid[start + guard:end - guard] = True
    phase[~valid] = np.nan
    return phase, valid


def theta_bouts(raw, clock, fs, run_mask, phase_valid, p):
    """Raw-LFP oscillation check, independent of filtered phase and spike outcomes."""
    support = np.zeros(len(clock), bool)
    rows = []
    for start, end in runs(run_mask & phase_valid):
        n = end - start
        if n / fs < p["theta_spectral_min_run_bout_s"]:
            continue
        f, density = welch(np.asarray(raw[start:end], float), fs=fs,
                           nperseg=min(n, int(round(fs * p["theta_spectral_window_s"]))),
                           detrend="constant")
        lo, hi = p["theta_band_hz"]
        theta = (f >= lo) & (f <= hi)
        adjacent = np.zeros(len(f), bool)
        for a, b in p["theta_spectral_comparison_bands_hz"]:
            adjacent |= (f >= a) & (f <= b)
        search = (f >= p["theta_spectral_peak_search_hz"][0]) & (f <= p["theta_spectral_peak_search_hz"][1])
        denom = density[adjacent].mean() if adjacent.any() else np.nan
        ratio = float(density[theta].mean() / denom) if theta.any() and denom > 0 else np.nan
        peak = float(f[search][np.argmax(density[search])]) if search.any() else np.nan
        ok = bool(np.isfinite(ratio) and ratio >= p["theta_spectral_min_power_density_ratio"]
                  and lo <= peak <= hi)
        support[start:end] = ok
        rows.append({"start_s": float(clock[start]), "end_s": float(clock[end - 1] + 1 / fs),
                     "duration_s": n / fs, "theta_adjacent_density_ratio": ratio,
                     "raw_spectral_peak_hz": peak, "theta_spectral_supported": ok})
    return support, rows


def theta_windows(raw, clock, fs, phase_valid, p):
    """Assess continuous native LFP, without treating short movement as missing LFP."""
    support = np.zeros(len(clock), bool)
    rows = []
    width = int(round(fs * p["theta_spectral_window_s"]))
    minimum = int(np.ceil(fs * p["theta_spectral_min_run_bout_s"]))
    if width < minimum or width <= 0:
        raise ValueError("Spectral windows must accommodate the frozen minimum duration")
    for start in range(0, len(clock), width):
        end = min(start + width, len(clock))
        if end - start < minimum or not phase_valid[start:end].all():
            continue
        f, density = welch(np.asarray(raw[start:end], float), fs=fs,
                           nperseg=end - start, detrend="constant")
        lo, hi = p["theta_band_hz"]
        theta = (f >= lo) & (f <= hi)
        adjacent = np.zeros(len(f), bool)
        for a, b in p["theta_spectral_comparison_bands_hz"]:
            adjacent |= (f >= a) & (f <= b)
        search = (f >= p["theta_spectral_peak_search_hz"][0]) & (f <= p["theta_spectral_peak_search_hz"][1])
        denominator = density[adjacent].mean() if adjacent.any() else np.nan
        ratio = float(density[theta].mean() / denominator) if theta.any() and denominator > 0 else np.nan
        peak = float(f[search][np.argmax(density[search])]) if search.any() else np.nan
        ok = bool(np.isfinite(ratio) and ratio >= p["theta_spectral_min_power_density_ratio"]
                  and lo <= peak <= hi)
        support[start:end] = ok
        rows.append({"start_s": float(clock[start]), "end_s": float(clock[end - 1] + 1 / fs),
                     "duration_s": (end - start) / fs, "theta_adjacent_density_ratio": ratio,
                     "raw_spectral_peak_hz": peak, "theta_spectral_supported": ok,
                     "spectral_scope": "native_grid_lfp_windows", "native_window_index": start // width})
    return support, rows


def sample_phase(clock, phase, valid, times):
    right = np.searchsorted(clock, times, side="left")
    inside = (right > 0) & (right < len(clock))
    result = np.full(len(times), np.nan)
    indices = np.flatnonzero(inside)
    left, right = right[indices] - 1, right[indices]
    ok = valid[left] & valid[right]
    indices, left, right = indices[ok], left[ok], right[ok]
    weight = (times[indices] - clock[left]) / (clock[right] - clock[left])
    z = (1 - weight) * np.exp(1j * phase[left]) + weight * np.exp(1j * phase[right])
    nonzero = np.abs(z) > 1e-12
    result[indices[nonzero]] = np.angle(z[nonzero])
    return result


def grid(links, bin_s):
    n = int(np.floor((links["t"][-1] - links["t"][0]) / bin_s))
    start = links["t"][0] + np.arange(n) * bin_s
    end = start + bin_s
    index = np.searchsorted(links["t"], start, side="right") - 1
    inside = (index >= 0) & (index < len(links["good"]))
    safe = np.clip(index, 0, len(links["good"]) - 1)
    complete = inside & links["good"][safe] & (end <= links["t"][safe + 1] + 1e-10)
    return start, end, safe, complete


def detect_mua(spikes, n_units, links, p):
    starts, ends, indices, complete = grid(links, p["mua_bin_s"])
    # Tracking knots are not gaps: a detection bin may cross two valid links.
    endpoint = np.searchsorted(links["t"], ends - 1e-10, side="right") - 1
    endpoint = np.clip(endpoint, 0, len(links["good"]) - 1)
    complete = links["good"][indices] & links["good"][endpoint] & (endpoint - indices <= 1)
    edges = np.r_[starts, ends[-1]]
    counts = np.histogram(spikes[:, 0], edges)[0]
    sigma = p["mua_smoothing_sigma_s"] / p["mua_bin_s"]
    smoothed = np.zeros(len(counts))
    guarded = np.zeros(len(counts), bool)
    guard = int(np.ceil(sigma * p["mua_gaussian_truncate"]))
    for a, b in runs(complete):
        smoothed[a:b] = gaussian_filter1d(counts[a:b].astype(float), sigma,
                                        mode="constant", truncate=p["mua_gaussian_truncate"])
        if b - a > 2 * guard:
            guarded[a + guard:b - guard] = True
    immobile = guarded & (links["speed"][indices] <= 4.0)
    if not immobile.any() or smoothed[immobile].std() <= 0:
        return [], {"immobile_baseline_bins": int(immobile.sum()), "baseline_mean": None, "baseline_sd": None}
    mean, sd = float(smoothed[immobile].mean()), float(smoothed[immobile].std())
    rows = []
    for a, b in runs(immobile & (smoothed > mean)):
        duration = ends[b - 1] - starts[a]
        peak = a + int(np.argmax(smoothed[a:b]))
        if not p["mua_detector_min_duration_s"] - 1e-10 <= duration <= p["mua_detector_max_duration_s"] + 1e-10:
            continue
        if smoothed[peak] <= mean + p["mua_peak_z"] * sd:
            continue
        local = spikes[(spikes[:, 0] >= starts[a]) & (spikes[:, 0] < ends[b - 1])]
        active = len(np.unique(local[:, 1]))
        if active < max(2, int(np.ceil(p["mua_active_fraction"] * n_units))):
            continue
        rows.append({"event_index": int(a), "start_s": float(starts[a]), "end_s": float(ends[b - 1]),
                     "peak_s": float((starts[peak] + ends[peak]) / 2), "duration_s": float(duration),
                     "detector_spikes": len(local), "detector_active_units": active,
                     "detector_peak_z": float((smoothed[peak] - mean) / sd)})
    return rows, {"immobile_baseline_bins": int(immobile.sum()), "baseline_mean": mean, "baseline_sd": sd}


def run_bank(spikes, ids, links, mask, clock, phases, supports, p):
    starts, ends, link, complete = grid(links, p["run_count_bin_s"])
    hits = np.flatnonzero(complete & mask[link])
    starts, ends, link = starts[hits], ends[hits], link[hits]
    counts = np.zeros((len(hits), len(ids)), dtype=np.int64)
    for col, cid in enumerate(ids):
        times = np.sort(spikes[spikes[:, 1] == cid, 0])
        counts[:, col] = np.searchsorted(times, ends, side="left") - np.searchsorted(times, starts, side="left")
    center = (starts + ends) / 2
    alpha = (center - links["t"][link]) / links["dt"][link]
    xy = links["xy"][link] + alpha[:, None] * (links["xy"][link + 1] - links["xy"][link])
    theta = np.column_stack([sample_phase(clock, phase, support, center)
                             for phase, support in zip(phases, supports, strict=True)])
    return {"time_s": center, "native_link": link, "counts": counts, "position_cm": xy,
            "speed_cm_s": links["speed"][link], "direction_rad": links["direction"][link],
            "theta_phase_rad": theta, "bin_duration_s": np.full(len(hits), p["run_count_bin_s"])}


def frozen_native_pause(row, native):
    pause = native[int(row.tracking_start_index)]
    if row.pause_id != f"tracking:{pause['tracking_start_index']}" or int(row.epoch_index) != pause["epoch_index"]:
        raise ValueError("Frozen native pause identity differs")
    if not np.allclose([row.start_s, row.end_s], [pause["start_s"], pause["end_s"]], rtol=0, atol=1e-9):
        raise ValueError("Frozen pause boundaries differ from native timestamps")
    # A CSV parser's final-bit rounding must not exclude a link at the native boundary.
    return {k: pause[k] for k in ("start_s", "end_s", "epoch_index")}


def support_run_masks(links, pause, matching):
    """Broaden rate training only, within the original period and native epoch."""
    start, stop = links["t"][:-1], links["t"][1:]
    run = links["good"] & (links["epoch"] == pause["epoch_index"])
    run &= (links["speed"] > matching["run_min_speed_cm_s"]) & (links["speed"] <= matching["run_max_speed_cm_s"])
    window = matching["run_search_window_s"]
    return (run & (start >= pause["start_s"] - window) & (stop <= pause["start_s"]),
            run & (start >= pause["end_s"]) & (stop <= pause["end_s"] + window))


def assert_previous_bank(saved, previous):
    if not set(previous.files) <= set(saved):
        raise ValueError("Frozen measurement fields disappeared")
    for key in previous.files:
        if not np.array_equal(saved[key], previous[key], equal_nan=True):
            raise ValueError(f"Frozen measurement bank changed: {key}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-audit", type=Path, required=True)
    parser.add_argument("--input-verification", type=Path, required=True)
    parser.add_argument("--acquisition-source", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--previous-measurement-dir", type=Path)
    parser.add_argument("--previous-measurement-verification", type=Path)
    args = parser.parse_args(argv)
    p = json.loads(args.protocol.read_text())
    parent = json.loads((args.input_audit / "manifest.json").read_text())
    for name, digest in parent["outputs_sha256"].items():
        checked(args.input_audit / name, digest)
    for name in ("protocol", "matching_protocol"):
        checked(Path(parent["input_file_paths"][name]), parent["input_file_sha256"][name])
    source_p = json.loads(Path(parent["input_file_paths"]["protocol"]).read_text())
    matching = json.loads(Path(parent["input_file_paths"]["matching_protocol"]).read_text())
    previous_banks = None
    if p.get("include_rate_training_support", False):
        if args.previous_measurement_dir is None or args.previous_measurement_verification is None:
            raise ValueError("Support sensitivity requires immutable, independently verified previous measurements")
        previous_manifest = args.previous_measurement_dir / "manifest.json"
        previous = json.loads(previous_manifest.read_text())
        previous_verification = json.loads(args.previous_measurement_verification.read_text())
        if not previous_verification.get("verified") or previous_verification["input_file_sha256"]["manifest"] != file_sha256(previous_manifest):
            raise ValueError("Previous measurement verification identity differs")
        for name, digest in previous["outputs_sha256"].items():
            checked(args.previous_measurement_dir / name, digest)
        previous_banks = pd.read_csv(args.previous_measurement_dir / "banks.csv").set_index(["session", "pause_id"])
        if not previous_banks.index.is_unique:
            raise ValueError("Duplicate previous measurement identity")
    if source_p["protocol_id"] != p["parent_input_protocol"]:
        raise ValueError("Unexpected parent protocol")
    verification = json.loads(args.input_verification.read_text())
    if not verification.get("verified") or verification.get("input_file_sha256", {}).get("manifest") != file_sha256(args.input_audit / "manifest.json"):
        raise ValueError("Parent census lacks matching successful independent verification")
    inputs = {"protocol": args.protocol, "parent_manifest": args.input_audit / "manifest.json", "input_verification": args.input_verification}
    if previous_banks is not None:
        inputs.update(previous_measurement_manifest=previous_manifest,
                      previous_measurement_verification=args.previous_measurement_verification)
    provenance = build_script_provenance(input_paths=inputs)
    if provenance["git_dirty"] or provenance["code_commit"] == "unavailable":
        raise ValueError("Run from a clean committed checkout")
    combine, crop, _ = source_combiner(args.acquisition_source, source_p["acquisition_source_commit"])
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "banks").mkdir()
    sessions = pd.read_csv(args.input_audit / "session_inventory.csv")
    pauses = pd.read_csv(args.input_audit / "matched_pause_inventory.csv", float_precision="round_trip")
    source_arrays = pd.read_csv(args.input_audit / "consumed_array_inventory.csv")
    selected_pauses = pauses[pauses.earliest_nonoverlap_supported.eq(True)]
    event_rows, pair_rows, pause_rows, theta_rows, consumed, session_rows, bank_rows = [], [], [], [], [], [], []
    for session in sessions[sessions.source_inputs_verified.eq(True)].itertuples(index=False):
        identity = {"animal": session.animal, "session": session.session}
        path = Path(session.path)
        print(f"START {session.session}", flush=True)
        if path.stat().st_size != session.size_bytes or path.stat().st_mtime_ns != session.mtime_ns:
            raise ValueError("Changed parent source-file identity")
        loaded, inventory = load_source(path, source_p, combine, crop)
        previous = source_arrays[source_arrays.session.eq(session.session)]
        expected = {x.hdf5_path.lstrip("/"): x.array_sha256 for x in previous.itertuples(index=False)}
        actual = {x["hdf5_path"].lstrip("/"): x["array_sha256"] for x in inventory}
        if expected != actual or not loaded["source_inputs_verified"]:
            raise ValueError("Parent source arrays or reconciliation changed")
        consumed.extend([{**identity, "path": str(path), **x} for x in inventory])
        links = epoch_links(loaded["position"], np.array([[loaded["position"][0, 0], loaded["position"][-1, 0]]]), matching)
        native_pauses = {x["tracking_start_index"]: x for x in immobile_pauses(links, matching)}
        spikes = loaded["spikes"]
        references = loaded["theta_references"]
        area_index = {x["area"]: i for i, x in enumerate(references)}
        if any(u["area"] not in area_index for u in loaded["units"]):
            raise ValueError("A CA1 unit hemisphere has no frozen LFP reference")
        unit_areas = {u["unit_id"]: area_index[u["area"]] for u in loaded["units"]}
        phases, supports = [], []
        with h5py.File(path, "r") as h:
            rec = next(iter(h["acquisition/timeseries"].values()))
            processor = next(iter(rec["continuous"].values()))
            clock = processor["downsampled_timestamps"][()]
            raw_run = link_spikes(clock, links, links["good"] & (links["speed"] > matching["run_min_speed_cm_s"])
                                 & (links["speed"] <= matching["run_max_speed_cm_s"]))
            fs = loaded["lfp_sampling_rate_hz"]
            for ref in references:
                raw = processor["downsampled_tetrode_data"][:, ref["lfp_column"]]
                consumed.append({**identity, "path": str(path), "hdf5_path": processor["downsampled_tetrode_data"].name,
                                 "column": ref["lfp_column"], "shape": list(raw.shape), "dtype": raw.dtype.str,
                                 "array_sha256": array_digest(raw)})
                phase, phase_valid = source_phase(raw, fs, p)
                if p.get("theta_spectral_scope") == "native_grid_lfp_windows":
                    support, spectral = theta_windows(raw, clock, fs, phase_valid, p)
                else:
                    support, spectral = theta_bouts(raw, clock, fs, raw_run, phase_valid, p)
                phases.append(phase)
                supports.append(support)
                theta_rows.extend([{**identity, **ref, **x} for x in spectral])
        candidates, detection = detect_mua(spikes, len(loaded["units"]), links, p)
        local_pauses = selected_pauses[selected_pauses.session.eq(session.session)]
        for pause in local_pauses.itertuples(index=False):
            key = {**identity, "pause_id": pause.pause_id}
            frozen = frozen_native_pause(pause, native_pauses)
            metrics, before, after = matched_run(links, frozen, matching)
            pre = spikes[link_spikes(spikes[:, 0], links, before)]
            ids, totals = np.unique(pre[:, 1].astype(np.int64), return_counts=True)
            ids = ids[totals >= matching["minimum_preceding_run_spikes_per_unit"]]
            if len(ids) != pause.eligible_preceding_units or not np.isclose(metrics["matched_exposure_s"], pause.matched_exposure_s):
                raise ValueError("Frozen matched-pause population changed")
            banks = {label: run_bank(spikes, ids, links, mask, clock, phases, supports, p)
                     for label, mask in (("pre", before), ("post", after))}
            saved = {"unit_ids": ids, "unit_theta_reference_index": np.array([unit_areas[int(u)] for u in ids]),
                     "pause_start_s": np.array(frozen["start_s"]), "pause_end_s": np.array(frozen["end_s"])}
            fractions = {}
            for label, bank in banks.items():
                saved.update({f"{label}_{k}": v for k, v in bank.items()})
                fractions[label] = np.mean(np.isfinite(bank["theta_phase_rad"]), axis=0).tolist() if len(bank["counts"]) else [0.0] * len(references)
            if previous_banks is not None:
                masks = support_run_masks(links, frozen, matching)
                for label, mask in zip(("pre", "post"), masks, strict=True):
                    pool = run_bank(spikes, ids, links, mask, clock, phases, supports, p)
                    saved.update({f"{label}_rate_support_{k}": v for k, v in pool.items()})
            event_count, pair_count = 0, 0
            for candidate in candidates:
                if candidate["start_s"] < frozen["start_s"] or candidate["end_s"] > frozen["end_s"]:
                    continue
                event_id = f"{session.session}:mua:{candidate['event_index']}"
                counts, width = event_counts(spikes, ids, candidate["start_s"], candidate["end_s"], matching["order_target_bin_s"])
                active = counts.sum(axis=0) > 0
                reasons = []
                if not matching["candidate_min_duration_s"] - 1e-10 <= candidate["duration_s"] <= matching["candidate_max_duration_s"] + 1e-10:
                    reasons.append("duration")
                if counts.sum() < matching["candidate_min_spikes"]:
                    reasons.append("eligible_spikes")
                if active.sum() < matching["candidate_min_active_units"]:
                    reasons.append("eligible_active_units")
                row = {**key, **candidate, "event_id": event_id, "n_eligible_spikes": int(counts.sum()),
                       "n_active_eligible_units": int(active.sum()), "order_measured": not reasons,
                       "exclusion_reason": ";".join(reasons), "label": p["candidate_label"], "validated_replay": False}
                event_rows.append(row)
                if reasons:
                    continue
                counts, active_ids = counts[:, active], ids[active]
                original = order_asymmetry(counts, width, matching["order_min_lag_s"], matching["order_max_lag_s"])
                shuffled_counts = whole_bin_shuffles(counts, matching["order_shuffles"], stable_seed(p["seed"], event_id))
                for shuffled in shuffled_counts:
                    if not np.array_equal(shuffled.sum(axis=0), counts.sum(axis=0)) or sorted(map(tuple, shuffled)) != sorted(map(tuple, counts)):
                        raise ValueError("Whole-bin population shuffle changed counts or participation")
                shuffled_order = np.stack([order_asymmetry(x, width, matching["order_min_lag_s"], matching["order_max_lag_s"])
                                           for x in shuffled_counts])
                prefix = f"event_{candidate['event_index']}"
                saved.update({f"{prefix}_counts": counts, f"{prefix}_unit_ids": active_ids,
                              f"{prefix}_width_s": np.array(width), f"{prefix}_shuffle_order": shuffled_order})
                event_count += 1
                for a in range(len(active_ids)):
                    for b in range(a + 1, len(active_ids)):
                        pair_rows.append({**key, "event_id": event_id, "unit_a": int(active_ids[a]), "unit_b": int(active_ids[b]),
                                          "original_order": float(original[a, b]),
                                          "shuffle_mean_order": float(shuffled_order[:, a, b].mean()),
                                          "shuffle_median_order": float(np.median(shuffled_order[:, a, b])),
                                          "n_shuffles": matching["order_shuffles"], "independent_biological_subject": False})
                        pair_count += 1
            tag = f"{session.animal}_{stable_seed(p['seed'], session.session + pause.pause_id):016x}"
            bank_path = args.output_dir / "banks" / f"{tag}.npz"
            if previous_banks is not None:
                old = previous_banks.loc[(session.session, pause.pause_id)]
                checked(Path(old.bank_path), old.bank_sha256)
                with np.load(old.bank_path, allow_pickle=False) as previous_bank:
                    assert_previous_bank(saved, previous_bank)
            np.savez_compressed(bank_path, **saved)
            bank_rows.append({**key, "bank_path": str(bank_path), "bank_sha256": file_sha256(bank_path)})
            theta_ok = all(x >= p["minimum_theta_supported_run_fraction"] for label in fractions for x in fractions[label])
            pause_rows.append({**key, **frozen, "eligible_units": len(ids), "matched_exposure_s": metrics["matched_exposure_s"],
                               "pre_run_bins": len(banks["pre"]["counts"]), "post_run_bins": len(banks["post"]["counts"]),
                               "pre_theta_supported_fractions": json.dumps(fractions["pre"]),
                               "post_theta_supported_fractions": json.dumps(fractions["post"]),
                               "theta_run_support_screen_passed": bool(theta_ok), "order_measured_candidates": event_count,
                               "dependent_pair_measurements": pair_count, "association_fit": False})
        session_rows.append({**identity, "source_verified": True, "frozen_selected_pauses": len(local_pauses),
                             "whole_file_mua_candidates": len(candidates), **detection})
        print(f"END {session.session}: {len(local_pauses)} frozen pauses, {len(candidates)} whole-file candidates", flush=True)
        (args.output_dir / "progress.json").write_text(json.dumps({"finished_sessions": len(session_rows), "latest": identity}) + "\n")
    names = {"sessions.csv": session_rows, "pauses.csv": pause_rows, "events.csv": event_rows,
             "pairs.csv": pair_rows, "theta_bouts.csv": theta_rows, "consumed_arrays.csv": consumed, "banks.csv": bank_rows}
    empty_columns = {"events.csv": ["animal", "session", "pause_id", "event_id", "order_measured", "validated_replay"],
                     "pairs.csv": ["animal", "session", "pause_id", "event_id", "unit_a", "unit_b", "original_order"]}
    for name, rows in names.items():
        pd.DataFrame(rows, columns=empty_columns.get(name) if not rows else None).to_csv(args.output_dir / name, index=False)
    if len(pause_rows) != len(selected_pauses):
        raise ValueError("Frozen pause accounting incomplete")
    gates = [
        {"gate": "frozen_pauses_complete", "passed": bool(pause_rows) and len(pause_rows) == len(selected_pauses), "observed": len(pause_rows)},
        {"gate": "all_pauses_theta_run_support_screen", "passed": bool(pause_rows) and all(x["theta_run_support_screen_passed"] for x in pause_rows), "observed": sum(x["theta_run_support_screen_passed"] for x in pause_rows)},
        {"gate": "candidate_order_present", "passed": bool(pair_rows), "observed": len(pair_rows)},
        {"gate": "replay_sequences_independently_validated", "passed": False, "observed": 0},
        {"gate": "coordination_change_association_calibrated_and_tested", "passed": False, "observed": 0},
        {"gate": "overall_goal_complete", "passed": False, "observed": 0},
    ]
    pd.DataFrame(gates).to_csv(args.output_dir / "gates.csv", index=False)
    outputs = list(names) + ["gates.csv"]
    manifest = {**provenance, "protocol_id": p["protocol_id"], "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "environment_versions": {k: version(k) for k in ("numpy", "scipy", "h5py", "pandas")},
                "outputs_sha256": {name: file_sha256(args.output_dir / name) for name in outputs},
                "association_fit": False, "validated_replay": False, "pf_theta_blocker_unchanged": True,
                "theta_qc_scope": p.get("theta_spectral_scope", "contiguous_run_bouts"),
                "raw_data_in_compact_archive": False, "banks": bank_rows}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"COMPLETE pauses={len(pause_rows)} events={len(event_rows)} pairs={len(pair_rows)} association_fit=False", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
