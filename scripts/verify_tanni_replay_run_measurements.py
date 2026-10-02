"""Reconstruct saved RUN/event counts and LFP phases directly from native data.

Spectral checks cover the exported bouts, not an independent bout-detector census.
No replay sequence or biological association is validated by this verifier.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.fftpack import next_fast_len
from scipy.signal import butter, convolve, filtfilt, hilbert, welch
from scipy.signal.windows import gaussian

try:
    from scripts._provenance import build_script_provenance, file_sha256
except ModuleNotFoundError:
    from _provenance import build_script_provenance, file_sha256


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest_array(a):
    a = np.asarray(a)
    digest = hashlib.sha256(json.dumps({"shape": a.shape, "dtype": a.dtype.str}, sort_keys=True).encode())
    digest.update(np.ascontiguousarray(a).tobytes())
    return digest.hexdigest()


def reference_phase(raw, fs, protocol):
    good = np.isfinite(raw)
    if raw.dtype.kind in "iu":
        limits = np.iinfo(raw.dtype)
        good &= (raw > limits.min) & (raw < limits.max)
    edges = np.diff(np.r_[0, good.astype(int), 0])
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    result = np.full(len(raw), np.nan)
    margin = int(np.ceil(protocol["lfp_edge_guard_s"] * fs))
    b, a = butter(protocol["theta_filter_order"], protocol["theta_band_hz"], fs=fs, btype="bandpass")
    sigma = protocol["theta_phase_smoothing_sigma_s"] * fs
    length = int(round(10 * sigma))
    length -= int(length % 2 == 0)
    kernel = gaussian(max(1, length), sigma)
    for left, right in zip(starts, ends, strict=True):
        values = np.asarray(raw[left:right], float)
        if len(values) <= max(2 * margin, 3 * max(len(a), len(b))) or np.std(values) == 0:
            continue
        band = filtfilt(b, a, values)
        if np.std(band[margin:-margin]) == 0:
            continue
        angles = np.unwrap(np.angle(hilbert(band, next_fast_len(len(band)))[:len(band)]))
        smoothed = convolve(angles, kernel / kernel.sum(), mode="same")
        result[left + margin:right - margin] = (smoothed[margin:-margin] + np.pi) % (2 * np.pi) - np.pi
    return result


def reference_order(counts, width, low, high):
    counts = np.asarray(counts, float)
    answer = np.zeros((counts.shape[1], counts.shape[1]))
    for i in range(len(counts)):
        for j in range(i + 1, len(counts)):
            if low - 1e-12 <= (j - i) * width <= high + 1e-12:
                answer += np.outer(counts[i], counts[j]) - np.outer(counts[j], counts[i])
    denominator = np.outer(counts.sum(axis=0), counts.sum(axis=0))
    return np.divide(answer, denominator, out=np.zeros_like(answer), where=denominator > 0)


def native_spikes(handle, prefix):
    annotations = handle["general/data_collection/Settings/General/channel_map"]
    tetrodes = set()
    for area in annotations:
        if area.startswith("CA1_"):
            tetrodes.update(int(c) // 4 for c in annotations[area]["list"][()])
    output = {}
    for tetrode in sorted(tetrodes):
        group = handle[f"{prefix}/spikes/electrode{tetrode + 1}"]
        keep = group["idx_keep"][()]
        times = group["timestamps"][()][keep]
        labels = group["clustering/manual_1"][()]
        require(keep.dtype.kind == "b" and len(times) == len(labels), "Curated spike identity mismatch")
        for cluster in np.unique(labels):
            if cluster > 0:
                output[tetrode * 65536 + int(cluster)] = times[labels == cluster]
    return output


def verify(root):
    manifest = json.loads((root / "manifest.json").read_text())
    for name, expected in manifest["outputs_sha256"].items():
        require(file_sha256(root / name) == expected, f"Changed output {name}")
    protocol_path = Path(manifest["input_file_paths"]["protocol"])
    require(file_sha256(protocol_path) == manifest["input_file_sha256"]["protocol"], "Changed measurement protocol")
    p = json.loads(protocol_path.read_text())
    parent_path = Path(manifest["input_file_paths"]["parent_manifest"])
    require(file_sha256(parent_path) == manifest["input_file_sha256"]["parent_manifest"], "Changed parent census")
    parent = json.loads(parent_path.read_text())
    matching_path = Path(parent["input_file_paths"]["matching_protocol"])
    require(file_sha256(matching_path) == parent["input_file_sha256"]["matching_protocol"], "Changed matching protocol")
    matching = json.loads(matching_path.read_text())
    parents = pd.read_csv(parent_path.parent / "session_inventory.csv", float_precision="round_trip").set_index("session")
    bank_rows = pd.read_csv(root / "banks.csv", float_precision="round_trip")
    pauses = pd.read_csv(root / "pauses.csv", float_precision="round_trip").set_index(["session", "pause_id"])
    events = pd.read_csv(root / "events.csv", float_precision="round_trip")
    pairs = pd.read_csv(root / "pairs.csv", float_precision="round_trip")
    bouts = pd.read_csv(root / "theta_bouts.csv", float_precision="round_trip")
    arrays = pd.read_csv(root / "consumed_arrays.csv")
    require(not bank_rows.empty and not bank_rows[["session", "pause_id"]].duplicated().any(), "Empty/duplicate banks")
    require(len(bank_rows) == len(pauses), "Incomplete frozen pause banks")
    count_bins, event_count, pair_count, checked_arrays = 0, 0, 0, 0
    for identity, local_banks in bank_rows.groupby("session"):
        print(f"VERIFY {identity}", flush=True)
        with h5py.File(parents.loc[identity, "path"], "r") as h:
            for row in arrays[arrays.session.eq(identity)].itertuples(index=False):
                array = h[row.hdf5_path][()] if not np.isfinite(row.column) else h[row.hdf5_path][:, int(row.column)]
                require(digest_array(array) == row.array_sha256, "Consumed native array changed")
                checked_arrays += 1
            rec = next(iter(h["acquisition/timeseries"].values()))
            position = rec["tracking/ProcessedPos"][()]
            source = native_spikes(h, rec.name)
            continuous = next(iter(rec["continuous"].values()))
            clock = continuous["downsampled_timestamps"][()]
            fs = float(continuous["downsampling_info/downsampled_sampling_rate"][()])
            references = json.loads(parents.loc[identity, "theta_references_json"])
            phases, spectral_support = [], []
            for reference in references:
                raw = continuous["downsampled_tetrode_data"][:, reference["lfp_column"]]
                phase = reference_phase(raw, fs, p)
                support = np.zeros(len(clock), bool)
                for row in bouts[bouts.session.eq(identity) & bouts.area.eq(reference["area"])].itertuples(index=False):
                    start = int(np.searchsorted(clock, row.start_s - 1e-9))
                    end = int(np.searchsorted(clock, row.end_s - 1e-9, side="right"))
                    require(np.isfinite(phase[start:end]).all(), "Spectral bout crosses unusable LFP phase")
                    frequency, density = welch(np.asarray(raw[start:end], float), fs=fs,
                                               nperseg=min(end - start, int(round(fs * p["theta_spectral_window_s"]))),
                                               detrend="constant")
                    theta = (frequency >= p["theta_band_hz"][0]) & (frequency <= p["theta_band_hz"][1])
                    adjacent = np.zeros(len(frequency), bool)
                    for left, right in p["theta_spectral_comparison_bands_hz"]:
                        adjacent |= (frequency >= left) & (frequency <= right)
                    ratio = density[theta].mean() / density[adjacent].mean()
                    search = (frequency >= p["theta_spectral_peak_search_hz"][0]) & (frequency <= p["theta_spectral_peak_search_hz"][1])
                    peak = frequency[search][np.argmax(density[search])]
                    require(np.isclose(ratio, row.theta_adjacent_density_ratio, rtol=1e-8), "Raw theta power mismatch")
                    require(np.isclose(peak, row.raw_spectral_peak_hz), "Raw theta peak mismatch")
                    ok = (ratio >= p["theta_spectral_min_power_density_ratio"]
                          and p["theta_band_hz"][0] <= peak <= p["theta_band_hz"][1])
                    require(bool(ok) == bool(row.theta_spectral_supported), "Theta spectral flag mismatch")
                    support[start:end] = ok
                phases.append(phase)
                spectral_support.append(support)
            for row in local_banks.itertuples(index=False):
                require(file_sha256(row.bank_path) == row.bank_sha256, "Changed spike/covariate bank")
                meta = pauses.loc[(identity, row.pause_id)]
                with np.load(row.bank_path, allow_pickle=False) as bank:
                    ids = bank["unit_ids"]
                    require(set(map(int, ids)) <= set(source), "Bank includes unknown CA1 identity")
                    for period in ("pre", "post"):
                        times = bank[f"{period}_time_s"]
                        width = bank[f"{period}_bin_duration_s"]
                        link = bank[f"{period}_native_link"]
                        require(np.all(times - width / 2 >= position[link, 0] - 1e-10)
                                and np.all(times + width / 2 <= position[link + 1, 0] + 1e-10), "RUN bin bridges native link")
                        if period == "pre":
                            require(np.all(times + width / 2 <= meta.start_s + 1e-10), "Future spikes enter PRE bank")
                        else:
                            require(np.all(times - width / 2 >= meta.end_s - 1e-10), "Pause spikes enter POST bank")
                        indices = np.rint((times - position[0, 0]) / p["run_count_bin_s"] - .5).astype(np.int64)
                        starts = position[0, 0] + indices * p["run_count_bin_s"]
                        ends = starts + p["run_count_bin_s"]
                        require(np.array_equal((starts + ends) / 2, times), "RUN bin not on frozen native grid")
                        counts = np.column_stack([np.searchsorted(source[int(u)], ends, side="left")
                                                  - np.searchsorted(source[int(u)], starts, side="left") for u in ids])
                        require(np.array_equal(counts, bank[f"{period}_counts"]), "Native RUN spike counts disagree")
                        count_bins += len(times)
                        for column, (phase, support) in enumerate(zip(phases, spectral_support, strict=True)):
                            right = np.searchsorted(clock, times)
                            left = right - 1
                            require(np.all(left >= 0) and np.all(right < len(clock)), "RUN phase outside LFP clock")
                            ok = support[left] & support[right]
                            weights = (times - clock[left]) / (clock[right] - clock[left])
                            z = (1 - weights) * np.exp(1j * phase[left]) + weights * np.exp(1j * phase[right])
                            expected = np.angle(z)
                            expected[~ok | (np.abs(z) <= 1e-12)] = np.nan
                            require(np.allclose(expected, bank[f"{period}_theta_phase_rad"][:, column],
                                                atol=1e-9, equal_nan=True), "Native LFP theta phase disagrees")
                        fractions = np.mean(np.isfinite(bank[f"{period}_theta_phase_rad"]), axis=0)
                        require(np.allclose(fractions, json.loads(meta[f"{period}_theta_supported_fractions"])), "Theta coverage denominator mismatch")
                    local_events = events[events.session.eq(identity) & events.pause_id.eq(row.pause_id) & events.order_measured.eq(True)]
                    for event in local_events.itertuples(index=False):
                        prefix = f"event_{event.event_index}"
                        counts, event_ids = bank[f"{prefix}_counts"], bank[f"{prefix}_unit_ids"]
                        width = float(bank[f"{prefix}_width_s"])
                        edges = np.linspace(event.start_s, event.end_s, len(counts) + 1)
                        expected = np.column_stack([np.histogram(source[int(u)][(source[int(u)] >= event.start_s)
                                                                                & (source[int(u)] < event.end_s)], edges)[0] for u in event_ids])
                        require(np.array_equal(counts, expected), "Raw whole-event counts disagree")
                        original = reference_order(counts, width, matching["order_min_lag_s"], matching["order_max_lag_s"])
                        seed = int.from_bytes(hashlib.sha256(f"{p['seed']}:{event.event_id}".encode()).digest()[:8], "little")
                        rng = np.random.default_rng(seed)
                        shuffled = np.stack([reference_order(counts[rng.permutation(len(counts))], width,
                                                              matching["order_min_lag_s"], matching["order_max_lag_s"])
                                             for _ in range(matching["order_shuffles"])])
                        require(np.allclose(shuffled, bank[f"{prefix}_shuffle_order"], atol=1e-12), "Whole-bin shuffle order disagrees")
                        local_pairs = pairs[pairs.event_id.eq(event.event_id)]
                        require(len(local_pairs) == len(event_ids) * (len(event_ids) - 1) // 2, "Incomplete pair census")
                        for pair in local_pairs.itertuples(index=False):
                            a, b = int(np.flatnonzero(event_ids == pair.unit_a)[0]), int(np.flatnonzero(event_ids == pair.unit_b)[0])
                            require(np.isclose(original[a, b], pair.original_order, atol=1e-12), "Original pair order disagrees")
                            require(np.isclose(shuffled[:, a, b].mean(), pair.shuffle_mean_order, atol=1e-12)
                                    and np.isclose(np.median(shuffled[:, a, b]), pair.shuffle_median_order, atol=1e-12), "Pair shuffle summary disagrees")
                            pair_count += 1
                        event_count += 1
    require(event_count == int(events.order_measured.sum()) and pair_count == len(pairs), "Event/pair total disagreement")
    return {"verified": True, "native_arrays_verified": checked_arrays, "run_bins_reconstructed": count_bins,
            "candidate_events_reconstructed": event_count, "dependent_pair_orders_reconstructed": pair_count,
            "frozen_pause_banks_verified": len(bank_rows), "association_tested": False, "replay_validated": False,
            "verification_scope": "Raw-array hashes, all saved RUN/event counts, source-compatible LFP phase, exported-bout spectra, whole-bin order and summaries; not a detector census, sequence calibration or RUN-change association"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    require(not args.output.exists(), "Do not overwrite a verification")
    result = verify(args.audit_dir)
    result.update(build_script_provenance(input_paths={"manifest": args.audit_dir / "manifest.json"}))
    result["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
