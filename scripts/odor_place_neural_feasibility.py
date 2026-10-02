"""Source-gated RUN validation and ripple opportunities, never replay inference."""

from __future__ import annotations

import io
import json
from pathlib import Path, PurePosixPath
import re
import zipfile

import h5py
import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, hilbert, sosfiltfilt
from scipy.special import logsumexp

from scripts._provenance import git_metadata
from scripts.odor_place_feasibility_core import PREFIX, clean, read_table, write_table
from scripts.odor_place_original_source_audit import epochs, struct, text, tracking
from scripts.odor_place_source_io import atomic_json, check_space, digest, download_verified


def prerequisites(args, protocol):
    from scripts.odor_place_original_source_audit import verify
    verification = verify(args, protocol)
    decision = json.loads((args.output_dir / (PREFIX + "decision.json")).read_text())
    if not decision["source_screen_passed"]:
        return "source_screen_failed"
    if verification.get("raw_source_reconciliation", {}).get("status") != "verified_against_raw_digital_edges":
        return "raw_source_verification_missing"
    if protocol["novelty_review_status"] != "passed_bounded_exact_contrast_review":
        return "novelty_prerequisite_not_passed"
    provenance = git_metadata()
    if provenance["git_dirty"] is not False or provenance["code_commit"] == "unavailable":
        raise ValueError("Conditional neural processing requires a committed clean checkout")
    return ""


def blocked(args, reason):
    source = json.loads((args.output_dir / (PREFIX + "decision.json")).read_text())
    result = {"status": "inconclusive_source_feasibility" if reason == "source_screen_failed" else "inconclusive_feasibility",
              "failure_reason": reason, "ready_for_calibration": False, "neural_processing_performed": False,
              "full_nwb_files_downloaded": 0, "n_source_eligible_transitions": source["n_eligible_source_transitions"],
              "n_decoder_qualified_transitions": None, "n_supported_ripple_opportunities": None,
              "association_fitted": False, "manuscript_claims_changed": False}
    atomic_json(args.output_dir / (PREFIX + "neural_decision.json"), result)
    for name, columns in [
        ("unit_crosswalk", ["asset_id", "unit_id", "status", "failure_reason"]),
        ("run_validation", ["source_trial_key", "decoder_qualified", "status", "failure_reason"]),
        ("ripple_opportunities", ["source_trial_key", "candidate_count", "sequence_testing_opportunities", "status"]),
    ]:
        write_table(args.output_dir, name, [], columns)
    write_table(args.output_dir, "neural_gate_summary", [{"gate": "neural_prerequisites", "passed": False,
                                                         "status": "fail", "reason": reason},
                                                        {"gate": "ready_for_calibration", "passed": False, "status": "fail", "reason": reason}])
    return result


def selected_assets(args):
    packets = json.loads((args.output_dir / (PREFIX + "source_sensor_checkpoints.json")).read_text())
    errors = read_table(args.output_dir, "post_error_transition_inventory")
    ids = {r["asset_id"] for r in errors if r["eligible_source_transition"] == "True"}
    listing = json.loads((args.dataset_root / "metadata/dandi_assets.json").read_text())["results"]
    assets = [r for r in listing if r["asset_id"] in ids]
    if not ids or {r["asset_id"] for r in assets} != ids:
        raise ValueError("Selected asset inventory is empty or incomplete")
    return assets, packets


def acquire_neural(args, protocol):
    reason = prerequisites(args, protocol)
    if reason:
        return blocked(args, reason)
    assets, _ = selected_assets(args)
    root = args.dataset_root / "nwb"
    remaining = 0
    for asset in assets:
        path = root / (asset["asset_id"] + ".nwb")
        partial = path.with_suffix(".nwb.part")
        completed = path.stat().st_size if path.exists() else partial.stat().st_size if partial.exists() else 0
        if completed > asset["size"]:
            raise ValueError("Existing neural asset exceeds its pinned size")
        remaining += asset["size"] - completed
    rows = []
    for asset in assets:
        check_space(root, remaining, protocol["download_reserve_bytes"])
        meta_path = args.dataset_root / "metadata" / (asset["asset_id"] + ".json")
        meta = json.loads(meta_path.read_text())
        pinned = json.loads((args.reference_inventory / "checkpoints" / (asset["asset_id"] + ".json")).read_text())
        if digest(meta_path) != pinned["identity"]["asset_metadata_sha256"]:
            raise ValueError("Asset metadata differs from the immutable pinned header inventory")
        expected = meta["digest"].get("dandi:sha2-256")
        if not expected or meta["path"] != asset["path"] or int(meta["contentSize"]) != asset["size"]:
            raise ValueError("Neural metadata lacks a matching pinned SHA256/size/path")
        urls = [u for u in meta["contentUrl"] if u.startswith("https://dandiarchive.s3.") or u.startswith("https://dandiarchive.s3.amazonaws.com/")]
        if len(urls) != 1:
            raise ValueError("Neural asset has no unique public S3 URL")
        path = root / (asset["asset_id"] + ".nwb")
        partial = path.with_suffix(".nwb.part")
        completed = path.stat().st_size if path.exists() else partial.stat().st_size if partial.exists() else 0
        verified = download_verified(urls[0], path, asset["size"], expected, "sha256", protocol["download_reserve_bytes"])
        remaining -= asset["size"] - completed
        rows.append({"asset_id": asset["asset_id"], "asset_path": asset["path"], "metadata_sha256": digest(meta_path), **verified})
        atomic_json(args.output_dir / (PREFIX + "neural_acquisition.json"), {"assets": rows, "complete": len(rows) == len(assets)})
        print(f"neural_asset_verified {len(rows)}/{len(assets)} {asset['asset_id']}", flush=True)
    return {"status": "neural_assets_verified", "full_nwb_files_downloaded": len(rows), "association_fitted": False}


def graph_grid(routes, bin_cm):
    left, right = [np.asarray(r, float) for r in routes]
    if left.shape != (3, 2) or right.shape != (3, 2) or not np.isfinite([left, right]).all() or not np.allclose(left[:2], right[:2], rtol=0, atol=1e-6):
        raise ValueError("Track graph requires explicitly identical shared start/junction")
    segments = [(left[0], left[1]), (left[1], left[2]), (right[1], right[2])]
    xy, edges, offsets, lengths = [], [], [], []
    for edge, (a, b) in enumerate(segments):
        length = float(np.linalg.norm(b - a))
        if length <= 0 or bin_cm <= 0:
            raise ValueError("Degenerate graph segment or bin width")
        lengths.append(length)
        bounds = np.r_[np.arange(0., length, bin_cm), length]
        for offset in (bounds[:-1] + bounds[1:]) / 2:
            xy.append(a + offset / length * (b - a))
            edges.append(edge)
            offsets.append(offset)
    edges, offsets = np.asarray(edges), np.asarray(offsets)
    junction_distance = np.where(edges == 0, lengths[0] - offsets, offsets)
    distances = junction_distance[:, None] + junction_distance[None, :]
    distances = np.where(edges[:, None] == edges[None, :], np.abs(offsets[:, None] - offsets[None, :]), distances)
    return {"xy": np.asarray(xy), "edges": edges, "offsets": offsets, "distance": distances, "segments": segments}


def project_bins(xy, graph):
    points = np.asarray(xy, float)
    squared, positions = [], []
    for a, b in graph["segments"]:
        fraction = np.clip((points - a) @ (b - a) / np.sum((b - a) ** 2), 0, 1)
        projected = a + fraction[:, None] * (b - a)
        squared.append(np.sum((points - projected) ** 2, axis=1))
        positions.append(fraction * np.linalg.norm(b - a))
    edge = np.argmin(np.asarray(squared), axis=0)
    result = np.empty(len(points), dtype=int)
    for number in range(3):
        subset = np.flatnonzero(edge == number)
        indices = np.flatnonzero(graph["edges"] == number)
        result[subset] = indices[np.argmin(np.abs(np.asarray(positions[number])[subset, None] - graph["offsets"][indices]), axis=1)]
    return result


def chronological_split(traversals, cutoff, config):
    ordered = sorted((r for r in traversals if r["end_s"] <= cutoff), key=lambda r: (r["start_s"], r["end_s"]))
    if any(a["end_s"] > b["start_s"] for a, b in zip(ordered, ordered[1:])):
        raise ValueError("Complete traversals overlap or are duplicated")
    split = int(len(ordered) * config["chronological_train_fraction"])
    train, validation = ordered[:split], ordered[split:]
    for side in ["left", "right"]:
        if sum(r["arm"] == side for r in train) < config["minimum_train_traversals_per_arm"] or sum(r["arm"] == side for r in validation) < config["minimum_validation_traversals_per_arm"]:
            raise ValueError("Insufficient chronological training/validation traversals per arm")
    return train, validation


def run_windows(data, traversals, spikes, graph, config, gap_s):
    results = []
    times = data[:, 0]
    for traversal in traversals:
        starts = np.arange(traversal["start_s"], traversal["end_s"] - config["bin_s"] + 1e-10, config["bin_s"])
        mids = starts + config["bin_s"] / 2
        ends = starts + config["bin_s"]
        right = np.searchsorted(times, mids)
        inside = (right > 0) & (right < len(times))
        safe = np.clip(right, 1, len(times) - 1)
        a, b = times[safe - 1], times[safe]
        valid = inside & (b - a <= gap_s)
        # Check every interval touched by a bin, not just its midpoint bracket.
        for j, (start, end) in enumerate(zip(starts, ends, strict=True)):
            low, high = max(0, np.searchsorted(times, start, side="right") - 1), np.searchsorted(times, end, side="left")
            if low == high or high >= len(times) or np.any(np.diff(times[low:high + 1]) > gap_s) or not np.isfinite(data[low:high + 1, [1, 2, 4]]).all():
                valid[j] = False
        fraction = (mids - a) / (b - a)
        xy = data[safe - 1, 1:3] * (1 - fraction[:, None]) + data[safe, 1:3] * fraction[:, None]
        speed = data[safe - 1, 4] * (1 - fraction) + data[safe, 4] * fraction
        valid &= np.isfinite(xy).all(axis=1) & (speed > config["minimum_speed_cm_s"])
        starts, ends, xy = starts[valid], ends[valid], xy[valid]
        counts = np.column_stack([np.searchsorted(s, ends, side="left") - np.searchsorted(s, starts, side="left") for s in spikes]) if spikes else np.zeros((len(starts), 0), int)
        results.append({"start": starts, "end": ends, "truth": project_bins(xy, graph), "counts": counts, "arm": traversal["arm"]})
    return results


def fit_maps(windows, graph, config):
    if not windows or not sum(len(w["truth"]) for w in windows):
        raise ValueError("No moving training observations")
    truth, counts = np.concatenate([w["truth"] for w in windows]), np.concatenate([w["counts"] for w in windows])
    selected = np.flatnonzero(counts.sum(axis=0) >= config["minimum_training_spikes_per_unit"])
    if len(selected) < config["minimum_verified_ca1_units"]:
        raise ValueError("Too few verified CA1 units with training spike support")
    occupancy = np.bincount(truth, minlength=len(graph["xy"])) * config["bin_s"]
    spike_map = np.column_stack([np.bincount(truth, weights=counts[:, u], minlength=len(occupancy)) for u in selected])
    weights = np.exp(-.5 * (graph["distance"] / config["graph_smoothing_sigma_cm"]) ** 2)
    weights /= weights.sum(axis=1, keepdims=True)
    smooth_occupancy, smooth_spikes = weights @ occupancy, weights @ spike_map
    global_rate = spike_map.sum(axis=0) / occupancy.sum()
    pseudo = config["global_rate_pseudocount_s"]
    rates = (smooth_spikes + pseudo * global_rate) / (smooth_occupancy[:, None] + pseudo)
    support = occupancy >= config["minimum_occupancy_s"]
    if not support.any() or not np.isfinite(rates).all() or np.any(rates <= 0):
        raise ValueError("No valid training-supported spatial likelihood")
    return {"rates": rates, "support": support, "selected_units": selected, "occupancy": occupancy,
            "spike_map": spike_map, "global_rate": global_rate}


def decode(counts, fitted, bin_s):
    rates = fitted["rates"]
    likelihood = np.asarray(counts)[:, fitted["selected_units"]] @ np.log(rates).T - bin_s * rates.sum(axis=1)
    likelihood[:, ~fitted["support"]] = -np.inf
    return np.exp(likelihood - logsumexp(likelihood, axis=1, keepdims=True))


def validation_metrics(windows, fitted, graph, config):
    recalls = {"left": [], "right": []}
    coverages, entropies, counts = [], [], []
    for window in windows:
        truth = window["truth"]
        unique = graph["edges"][truth] != 0
        if not unique.any():
            raise ValueError("Validation traversal has no moving unique-arm observations")
        posterior = decode(window["counts"], fitted, config["bin_s"])
        left = posterior[:, graph["edges"] == 1].sum(axis=1)
        right = posterior[:, graph["edges"] == 2].sum(axis=1)
        true_left = graph["edges"][truth] == 1
        ties = np.isclose(left, right, rtol=0, atol=1e-12)
        correct = np.where(ties, .5, ((left > right) == true_left).astype(float))
        # Unsupported true locations remain in both accuracy and coverage denominators.
        recalls[window["arm"]].append(float(correct[unique].mean()))
        coverages.append(float(fitted["support"][truth[unique]].mean()))
        entropies.extend((-np.sum(posterior * np.log(np.maximum(posterior, 1e-300)), axis=1)).tolist())
        counts.extend(window["counts"].sum(axis=1).tolist())
    if not all(recalls.values()):
        raise ValueError("Both arms must have validation traversals")
    recall = {side: float(np.mean(values)) for side, values in recalls.items()}
    return {"balanced_accuracy": float(np.mean(list(recall.values()))), "left_recall": recall["left"], "right_recall": recall["right"],
            "support_coverage": float(np.mean(coverages)), "mean_posterior_entropy": float(np.mean(entropies)),
            "n_validation_bins": len(counts), "n_zero_spike_bins": sum(c == 0 for c in counts)}


def metrics_pass(metrics, config):
    return (metrics["balanced_accuracy"] >= config["minimum_balanced_accuracy"]
            and min(metrics["left_recall"], metrics["right_recall"]) >= config["minimum_arm_recall"]
            and metrics["support_coverage"] >= config["minimum_support_coverage"])


def verify_unit_identity(unit_ids, nwb_spikes, electrode_candidates, source_units):
    rows, verified_spikes, verified_tetrodes = [], [], set()
    for unit_id, spikes, candidates in zip(unit_ids, nwb_spikes, electrode_candidates, strict=True):
        matches = [u for u in source_units if len(spikes) == len(u["spikes"]) and len(spikes) > 0
                   and np.allclose(spikes, u["spikes"], rtol=0, atol=1e-8)]
        reason, source = "", matches[0] if len(matches) == 1 else None
        if source is None:
            reason = "no_unique_source_spike_train_match"
        elif {(str(c["area"]), int(c["tetrode"])) for c in candidates} != {(source["area"], source["tetrode"])}:
            reason = "electrode_identity_conflicts_or_is_ambiguous"
        elif source["area"] != "CA1":
            reason = "not_source_verified_ca1"
        elif source["tag"].lower() == "mua":
            reason = "source_multiunit_not_sorted_unit"
        rows.append({"unit_id": int(unit_id), "source_tetrode": source["tetrode"] if source else None,
                     "source_cluster": source["cluster"] if source else None, "source_area": source["area"] if source else None,
                     "status": "verified_ca1_spike_identity" if not reason else "excluded", "failure_reason": reason,
                     "n_spikes": len(spikes), "spike_comparison_tolerance_s": 1e-8, "clock_offset_fitted": False})
        if not reason:
            verified_spikes.append(np.asarray(spikes, float))
            verified_tetrodes.add(source["tetrode"])
    return rows, verified_spikes, verified_tetrodes


def merge_overlap(windows, *, touching=False):
    output = []
    for start, stop in sorted(windows):
        if output and (start < output[-1][1] or (touching and start == output[-1][1])):
            output[-1][1] = max(output[-1][1], stop)
        else:
            output.append([float(start), float(stop)])
    return output


def detect_ripples(signal, times, config):
    signal, times = np.asarray(signal, float), np.asarray(times, float)
    if len(times) < 30 or len(signal) != len(times) or not np.isfinite(signal).all() or not np.isfinite(times).all():
        raise ValueError("Insufficient or nonfinite ripple LFP")
    dt = np.diff(times)
    if np.any(dt <= 0) or not np.allclose(dt, np.median(dt), rtol=1e-4, atol=1e-8):
        raise ValueError("Nonuniform or gapped LFP must not be filtered across gaps")
    fs = 1 / np.median(dt)
    if fs <= 2 * config["band_hz"][1]:
        raise ValueError("LFP sampling cannot support the frozen ripple band")
    envelope = np.abs(hilbert(sosfiltfilt(butter(config["filter_order"], config["band_hz"], btype="bandpass", fs=fs, output="sos"), signal)))
    if envelope.std() <= 0:
        return []
    active = (envelope - envelope.mean()) / envelope.std() > config["epochwise_envelope_z_threshold"]
    starts = np.flatnonzero(np.diff(np.r_[False, active].astype(int)) == 1)
    stops = np.flatnonzero(np.diff(np.r_[active, False].astype(int)) == -1) + 1
    return [[float(times[a]), float(times[b - 1] + 1 / fs)] for a, b in zip(starts, stops, strict=True)
            if (b - a) / fs >= config["minimum_duration_s"] and a > 0 and b < len(times)]


def opportunity_counts(events, segments, spikes, config, bin_s):
    contained = [[a, b] for a, b in events if any(c <= a and b <= d for c, d in segments)]
    rows = []
    for start, stop in contained:
        starts = np.arange(start, stop - bin_s + 1e-10, bin_s)
        counts = np.column_stack([np.searchsorted(s, starts + bin_s, side="left") - np.searchsorted(s, starts, side="left") for s in spikes]) if spikes else np.zeros((len(starts), 0), int)
        supported = int(np.sum(counts.sum(axis=1) > 0))
        # Include the trailing partial bin in active-unit and total-spike counts.
        full = np.array([np.searchsorted(s, stop, side="left") - np.searchsorted(s, start, side="left") for s in spikes])
        active = int(np.sum(full > 0))
        rows.append({"start_s": start, "stop_s": stop, "n_spikes": int(full.sum()), "active_ca1_units": active,
                     "spike_supported_bins": supported,
                     "sequence_testing_opportunity": supported >= config["minimum_spike_supported_bins"] and active >= config["minimum_active_ca1_units"],
                     "classification": config["classification"]})
    return rows


def append_report(lines, args, protocol):
    path = args.output_dir / (PREFIX + "neural_decision.json")
    result = json.loads(path.read_text()) if path.exists() else None
    lines.extend(["## Conditional neural feasibility", "", f"Novelty prerequisite: `{protocol['novelty_review_status']}`.",
                  "Acquisition and RUN/ripple processing require a verified source screen and a passed bounded exact-contrast review.",
                  f"Neural status: `{result['status'] if result else 'not_started'}`.",
                  f"Limiting reason: `{result.get('failure_reason', '') if result else 'source/novelty prerequisites not yet passed'}`.",
                  "Zero-event qualified pauses remain observations. Supported ripple candidates are opportunities, not validated replay; no association model is fitted.", ""])
    if result and result["neural_processing_performed"]:
        verify_neural_artifacts(args, protocol)
        lines.extend([f"Decoder-qualified transitions: {result['n_decoder_qualified_transitions']}; animals with both conditions: {', '.join(result['animals_with_both_conditions']) or 'none'}.",
                      f"Sequence-testing opportunities: {result['n_supported_ripple_opportunities']}; qualified zero-event transitions: {result['n_qualified_zero_event_transitions']}.",
                      f"Ready for separately frozen calibration: {result['ready_for_calibration']}. This is not a power certificate or a biological result.", ""])


def dispatch(args, protocol):
    return {"acquire-neural": acquire_neural, "run-qc": run_qc}[args.stage](args, protocol)


def source_member(archive, animal, role, day=None):
    name = f"{animal}{role}{day:02d}.mat" if day is not None else f"{animal}{role}.mat"
    entries = [e for e in archive.infolist() if PurePosixPath(e.filename).name == name]
    if len(entries) != 1:
        raise ValueError("No unique original source member: " + name)
    return loadmat(io.BytesIO(archive.read(entries[0])), struct_as_record=False, squeeze_me=False)[role]


def source_neural_epoch(archive, animal, day, epoch, bin_cm=3.):
    spike_epoch = epochs(source_member(archive, animal, "spikes", day), day)[epoch]
    info_epoch = epochs(source_member(archive, animal, "tetinfo"), day)[epoch]
    units = []
    for tetrode, cell in enumerate(np.asarray(spike_epoch, dtype=object).ravel(), 1):
        if not getattr(cell, "size", 0):
            continue
        info = struct(np.asarray(info_epoch, dtype=object).ravel()[tetrode - 1])
        for cluster, value in enumerate(np.asarray(cell, dtype=object).ravel(), 1):
            if not getattr(value, "size", 0):
                continue
            record = struct(value)
            data = np.asarray(record["data"], float)
            if data.ndim != 2 or data.shape[1] < 1 or not np.isfinite(data[:, 0]).all() or np.any(np.diff(data[:, 0]) < 0):
                raise ValueError("Unresolved original spike time schema")
            units.append({"tetrode": tetrode, "cluster": cluster, "area": text(info["area"]),
                          "tag": text(record.get("tag", np.array([""]))), "spikes": data[:, 0]})
    task = struct(epochs(source_member(archive, animal, "task", day), day)[epoch])
    routes = [np.asarray(v, float)[:, :, 0] for v in np.asarray(task["linearcoord"], dtype=object).ravel()]
    pos = tracking(struct(epochs(source_member(archive, animal, "pos", day), day)[epoch]))
    return units, graph_grid(routes, bin_cm), pos


def nwb_units(handle, source_units, epoch_bounds):
    units, electrodes = handle["units"], handle["general/extracellular_ephys/electrodes"]
    ids, references = units["id"][()], units["electrodes"][()]
    if "electrodes_index" in units:
        stops = units["electrodes_index"][()]
        starts = np.r_[0, stops[:-1]]
        unit_refs = [references[a:b].ravel().tolist() for a, b in zip(starts, stops, strict=True)]
    elif len(references) == len(ids):
        unit_refs = [[int(r)] for r in references]
    else:
        raise ValueError("Unresolved NWB unit-to-electrode reference schema")
    electrode_ids = electrodes["id"][()]
    table_ref = units["electrodes"].attrs.get("table")
    verified_row_reference = table_ref is not None and handle[table_ref].name == electrodes.name
    if table_ref is not None and not verified_row_reference:
        raise ValueError("Unit references point to an unexpected electrode table")
    candidates = []
    for refs in unit_refs:
        rows = set()
        for ref in refs:
            if not verified_row_reference:
                rows.update(np.flatnonzero(electrode_ids == ref).tolist())
            if 0 <= ref < len(electrode_ids):
                rows.add(ref)
        meanings = []
        for row in sorted(rows):
            match = re.fullmatch(r"tetrode(\d+)", clean(electrodes["group_name"][row]))
            if not match:
                raise ValueError("Unverified electrode group naming")
            meanings.append({"area": clean(electrodes["location"][row]), "tetrode": int(match[1])})
        candidates.append(meanings)
    stops = units["spike_times_index"][()]
    trains = []
    for start, stop in zip(np.r_[0, stops[:-1]], stops, strict=True):
        spikes = np.asarray(units["spike_times"][int(start):int(stop)], float)
        trains.append(spikes[(spikes >= epoch_bounds[0]) & (spikes <= epoch_bounds[1])])
    source_units = [{**u, "spikes": u["spikes"][(u["spikes"] >= epoch_bounds[0]) & (u["spikes"] <= epoch_bounds[1])]} for u in source_units]
    return verify_unit_identity(ids, trains, candidates, source_units)


def nwb_ripples(handle, tetrodes, bounds, config):
    electrodes = handle["general/extracellular_ephys/electrodes"]
    series = []
    handle.visititems(lambda _, node: series.append(node) if isinstance(node, h5py.Group) and clean(node.attrs.get("neurodata_type", "")) == "ElectricalSeries" else None)
    all_events, seen = [], set()
    for group in series:
        if "electrodes" not in group or "data" not in group or "referenced" not in electrodes:
            continue
        refs = group["electrodes"]
        table = refs.attrs.get("table")
        if table is None or handle[table].name != electrodes.name:
            raise ValueError("LFP electrode references lack a verified DynamicTableRegion")
        data = group["data"]
        if data.ndim != 2 or data.shape[1] != len(refs):
            raise ValueError("LFP data and electrode dimensions disagree")
        if "timestamps" in group:
            times = group["timestamps"][()]
        elif "starting_time" in group and "rate" in group["starting_time"].attrs:
            rate = float(group["starting_time"].attrs["rate"])
            times = float(group["starting_time"][()]) + np.arange(len(data)) / rate
        else:
            raise ValueError("LFP timestamps/rate unresolved")
        selected = np.flatnonzero((times >= bounds[0]) & (times <= bounds[1]))
        if not len(selected):
            continue
        for column, row in enumerate(refs[()]):
            row = int(row)
            if row < 0 or row >= len(electrodes["id"]):
                raise ValueError("LFP electrode row outside its referenced table")
            group_name = clean(electrodes["group_name"][row])
            match = re.fullmatch(r"tetrode(\d+)", group_name)
            if not match or int(match[1]) not in tetrodes or clean(electrodes["location"][row]) != "CA1" or clean(electrodes["referenced"][row]) != "Ref ON":
                continue
            if row in seen:
                raise ValueError("Multiple LFP series ambiguously reference the same CA1 electrode")
            seen.add(row)
            signal = data[int(selected[0]):int(selected[-1]) + 1, column]
            all_events.extend(detect_ripples(signal, times[selected], config))
    if not seen:
        raise ValueError("No locally referenced source-verified CA1 LFP channel")
    return merge_overlap(all_events)


def run_qc(args, protocol):
    reason = prerequisites(args, protocol)
    if reason:
        return blocked(args, reason)
    acquisition = json.loads((args.output_dir / (PREFIX + "neural_acquisition.json")).read_text())
    if not acquisition["complete"]:
        raise ValueError("Incomplete neural acquisition")
    assets, packets = selected_assets(args)
    inputs = {r["asset_id"]: r for r in acquisition["assets"]}
    if set(inputs) != {r["asset_id"] for r in assets}:
        raise ValueError("Acquisition does not match the eligible source cohort")
    errors = [r for r in read_table(args.output_dir, "post_error_transition_inventory") if r["eligible_source_transition"] == "True"]
    source_trials = [r for p in packets for r in p["source_sensor_trials"]]
    checkpoint = args.output_dir / "neural_checkpoints"
    checkpoint.mkdir(exist_ok=True)
    unit_rows, validation, opportunities, event_rows = [], [], [], []
    with zipfile.ZipFile(args.source_archive) as archive:
        for asset in assets:
            info = inputs[asset["asset_id"]]
            path = Path(info["path"])
            if digest(path) != info["sha256"]:
                raise ValueError("Downloaded NWB identity changed")
            selected = [r for r in errors if r["asset_id"] == asset["asset_id"]]
            with h5py.File(path, "r") as handle:
                for animal, day, epoch in sorted({(r["animal"], int(r["source_day"]), int(r["source_epoch"])) for r in selected}):
                    try:
                        subject = clean(handle["general/subject/subject_id"][()]).removeprefix("Symanski-")
                        if subject != animal:
                            raise ValueError("NWB subject differs from independently reconciled source identity")
                        source_units, graph, pos = source_neural_epoch(archive, animal, day, epoch, protocol["run"]["graph_bin_cm"])
                        crosswalk, spikes, tetrodes = nwb_units(handle, source_units, (pos[0, 0], pos[-1, 0]))
                    except (ValueError, KeyError, OSError) as exc:
                        unit_rows.append({"asset_id": asset["asset_id"], "animal": animal, "source_day": day, "source_epoch": epoch,
                                          "status": "unresolved_epoch_adapter", "failure_reason": str(exc)})
                        for error in [r for r in selected if (int(r["source_day"]), int(r["source_epoch"])) == (day, epoch)]:
                            validation.append({"source_trial_key": error["source_trial_key"], "animal": animal, "asset_id": asset["asset_id"],
                                               "decoder_qualified": False, "status": "unresolved", "failure_reason": str(exc)})
                            opportunities.append({"source_trial_key": error["source_trial_key"], "animal": animal, "asset_id": asset["asset_id"],
                                                  "candidate_count": None, "sequence_testing_opportunities": None, "status": "unresolved", "failure_reason": str(exc)})
                        continue
                    unit_rows.extend({"asset_id": asset["asset_id"], "animal": animal, "source_day": day, "source_epoch": epoch, **r} for r in crosswalk)
                    ripple_failure = ""
                    try:
                        ripples = nwb_ripples(handle, tetrodes, (pos[0, 0], pos[-1, 0]), protocol["ripples"])
                    except ValueError as exc:
                        ripples, ripple_failure = [], str(exc)
                    trials = [r for r in source_trials if (r["animal"], r["source_day"], r["source_epoch"]) == (animal, day, epoch) and r["nwb_asset_id"] == asset["asset_id"]]
                    traversals = [{"start_s": r["nosepoke_stop_s"], "end_s": r["well_onset_s"], "arm": r["chosen_arm"]}
                                  for r in trials if r["trial_verified"] and r["nosepoke_stop_s"] < r["well_onset_s"]]
                    for error in [r for r in selected if (int(r["source_day"]), int(r["source_epoch"])) == (day, epoch)]:
                        key = error["source_trial_key"]
                        identity = {"source_trial_key": key, "protocol_sha256": digest(args.protocol), "nwb_sha256": info["sha256"],
                                    "source_identity_sha256": digest(args.output_dir / (PREFIX + "inventory_identity.json")), "code_commit": git_metadata()["code_commit"]}
                        saved_path = checkpoint / (key.replace(":", "_") + ".json")
                        if saved_path.exists():
                            saved = json.loads(saved_path.read_text())
                            if saved["identity"] != identity or digest(args.output_dir / saved["evidence_file"]) != saved["evidence_sha256"]:
                                raise ValueError("Neural checkpoint identity or evidence differs")
                            validation.append(saved["validation"])
                            opportunities.append(saved["opportunities"])
                            event_rows.extend(saved["events"])
                            continue
                        result = {"source_trial_key": key, "animal": animal, "asset_id": asset["asset_id"], "source_day": day, "source_epoch": epoch,
                                  "decoder_qualified": False, "status": "unresolved", "failure_reason": ""}
                        evidence_name = "neural_checkpoints/" + key.replace(":", "_") + ".npz"
                        evidence = {}
                        try:
                            train, test = chronological_split(traversals, float(error["pause_start_s"]), protocol["run"])
                            train_windows = run_windows(pos, train, spikes, graph, protocol["run"], protocol["maximum_tracking_gap_s"])
                            test_windows = run_windows(pos, test, spikes, graph, protocol["run"], protocol["maximum_tracking_gap_s"])
                            fitted = fit_maps(train_windows, graph, protocol["run"])
                            metrics = validation_metrics(test_windows, fitted, graph, protocol["run"])
                            result.update(metrics, n_encoding_units=len(fitted["selected_units"]), n_training_traversals=len(train),
                                          n_validation_traversals=len(test), training_end_s=max(r["end_s"] for r in train),
                                          validation_start_s=min(r["start_s"] for r in test), status="evaluated", decoder_qualified=metrics_pass(metrics, protocol["run"]))
                            evidence = {**fitted, "edges": graph["edges"], "validation_counts": np.concatenate([w["counts"] for w in test_windows]),
                                        "validation_truth": np.concatenate([w["truth"] for w in test_windows]),
                                        "traversal_lengths": np.array([len(w["truth"]) for w in test_windows]),
                                        "traversal_arms": np.array([w["arm"] for w in test_windows])}
                            if result["decoder_qualified"]:
                                refit = fit_maps(train_windows + test_windows, graph, protocol["run"])
                                evidence.update({"refit_" + k: v for k, v in refit.items()})
                            else:
                                result["failure_reason"] = "frozen_decoder_thresholds_failed"
                        except ValueError as exc:
                            result.update(failure_reason=str(exc), decoder_qualified=False, status="unresolved")
                        events = []
                        if result["decoder_qualified"] and not ripple_failure:
                            original = next(r for r in trials if r["source_trial_key"] == key)
                            events = [{"source_trial_key": key, "animal": animal, "asset_id": asset["asset_id"], **e}
                                      for e in opportunity_counts(ripples, merge_overlap(original["immobile_segments"], touching=True), spikes, protocol["ripples"], protocol["run"]["bin_s"])]
                        opportunity = {"source_trial_key": key, "animal": animal, "asset_id": asset["asset_id"],
                                       "candidate_count": len(events) if result["decoder_qualified"] and not ripple_failure else None,
                                       "sequence_testing_opportunities": sum(e["sequence_testing_opportunity"] for e in events) if result["decoder_qualified"] and not ripple_failure else None,
                                       "status": "counted_not_validated_replay" if result["decoder_qualified"] and not ripple_failure else "unresolved",
                                       "failure_reason": ripple_failure or result["failure_reason"]}
                        np.savez_compressed(args.output_dir / evidence_name, **evidence)
                        atomic_json(saved_path, clean({"identity": identity, "validation": result, "opportunities": opportunity, "events": events,
                                                       "evidence_file": evidence_name, "evidence_sha256": digest(args.output_dir / evidence_name)}))
                        validation.append(result)
                        opportunities.append(opportunity)
                        event_rows.extend(events)
                        print(f"run_qc {key}: decoder={result['decoder_qualified']} opportunities={opportunity['sequence_testing_opportunities']}", flush=True)
    write_table(args.output_dir, "unit_crosswalk", unit_rows)
    write_table(args.output_dir, "run_validation", validation)
    write_table(args.output_dir, "ripple_opportunities", opportunities)
    write_table(args.output_dir, "ripple_event_inventory", event_rows, None if event_rows else ["source_trial_key", "start_s", "stop_s", "classification"])
    return finalize_neural(args, protocol, errors, validation, opportunities)


def finalize_neural(args, protocol, errors, validation, opportunities):
    from scripts.odor_place_original_source_audit import coverage_passed
    keys = [r["source_trial_key"] for r in errors]
    for rows in [validation, opportunities]:
        observed = [r["source_trial_key"] for r in rows]
        if len(observed) != len(set(observed)) or set(observed) != set(keys):
            raise ValueError("Neural accounting must contain exactly one row per eligible source transition")
    counted = {r["source_trial_key"] for r in opportunities if r["status"] == "counted_not_validated_replay"}
    qualified = {r["source_trial_key"] for r in validation if r["decoder_qualified"] and r["status"] == "evaluated"} & counted
    eligible = [r for r in errors if r["source_trial_key"] in qualified]
    animals = [{"animal": a, "source_full_maze": a in protocol["full_maze_animals_source_code"],
                "both_conditions": {r["cue_condition"] for r in eligible if r["animal"] == a} == {"repeat", "changed"},
                "n_qualified_transitions": sum(r["animal"] == a for r in eligible)} for a in protocol["animals"]]
    cells = [{"cue_condition": c, "outcome_relative_to_prior_error_cue": o,
              "n_source_transitions": sum(r["cue_condition"] == c and r["outcome_relative_to_prior_error_cue"] == o for r in eligible)}
             for c in ["repeat", "changed"] for o in ["correction_to_prior_cue_arm", "repeated_mistaken_arm"]]
    passed, covered = coverage_passed(animals, eligible, cells, protocol["screening"])
    result = {"status": "ready_for_calibration" if passed else "inconclusive_feasibility", "ready_for_calibration": passed,
              "failure_reason": "" if passed else "decoder_qualified_cohort_below_frozen_screening_floors",
              "n_source_eligible_transitions": len(errors), "n_decoder_qualified_transitions": len(eligible), "animals_with_both_conditions": covered,
              "n_supported_ripple_opportunities": sum(r["sequence_testing_opportunities"] or 0 for r in opportunities),
              "n_qualified_zero_event_transitions": sum(r["source_trial_key"] in qualified and r["candidate_count"] == 0 for r in opportunities),
              "neural_processing_performed": True, "association_fitted": False, "manuscript_claims_changed": False}
    write_table(args.output_dir, "neural_animal_summary", animals)
    write_table(args.output_dir, "decoder_qualified_coverage", cells)
    write_table(args.output_dir, "decoder_qualified_transitions", [{**r, "decoder_qualified": r["source_trial_key"] in qualified} for r in errors],
                None if errors else ["source_trial_key", "decoder_qualified"])
    independently_verify_neural(args, protocol, validation)
    atomic_json(args.output_dir / (PREFIX + "neural_decision.json"), result)
    write_table(args.output_dir, "neural_gate_summary", [{"gate": "neural_accounting", "passed": len(validation) == len(errors), "status": "pass" if len(validation) == len(errors) else "fail"},
                                                        {"gate": "decoder_qualified_screen", "passed": passed, "status": "pass" if passed else "fail"},
                                                        {"gate": "ready_for_calibration", "passed": passed, "status": "pass" if passed else "fail"}])
    files = [p for p in args.output_dir.glob(PREFIX + "*.csv") if p.name not in json.loads((args.output_dir / (PREFIX + "inventory_identity.json")).read_text())["output_sha256"]]
    files += list((args.output_dir / "neural_checkpoints").glob("*"))
    files += [args.output_dir / (PREFIX + name + ".json") for name in ["neural_decision", "neural_verification", "neural_acquisition"]]
    atomic_json(args.output_dir / (PREFIX + "neural_identity.json"), {
        "protocol_sha256": digest(args.protocol), "source_identity_sha256": digest(args.output_dir / (PREFIX + "inventory_identity.json")),
        "code_commit": git_metadata()["code_commit"], "files_sha256": {str(p.relative_to(args.output_dir)): digest(p) for p in files},
    })
    return result


def verify_neural_artifacts(args, protocol):
    identity = json.loads((args.output_dir / (PREFIX + "neural_identity.json")).read_text())
    if digest(args.protocol) != identity["protocol_sha256"] or digest(args.output_dir / (PREFIX + "inventory_identity.json")) != identity["source_identity_sha256"]:
        raise ValueError("Neural artifacts belong to different inputs")
    for name, expected in identity["files_sha256"].items():
        if digest(args.output_dir / name) != expected:
            raise ValueError("Neural artifact differs: " + name)
    saved = read_table(args.output_dir, "run_validation")
    for row in saved:
        row["decoder_qualified"] = row["decoder_qualified"] == "True"
        if row["status"] == "evaluated":
            for key in ["balanced_accuracy", "left_recall", "right_recall", "support_coverage"]:
                row[key] = float(row[key])
    independently_verify_neural(args, protocol, saved)


def independently_verify_neural(args, protocol, validation):
    checked = 0
    for row in validation:
        if row["status"] != "evaluated":
            continue
        name = row["source_trial_key"].replace(":", "_")
        with np.load(args.output_dir / "neural_checkpoints" / (name + ".npz"), allow_pickle=False) as arrays:
            counts, rates = arrays["validation_counts"][:, arrays["selected_units"]], arrays["rates"]
            logp = np.zeros((len(counts), len(rates)))
            for unit in range(counts.shape[1]):
                logp += counts[:, unit, None] * np.log(rates[None, :, unit]) - protocol["run"]["bin_s"] * rates[None, :, unit]
            logp[:, ~arrays["support"]] = -np.inf
            p = np.exp(logp - np.max(logp, axis=1, keepdims=True))
            p /= p.sum(axis=1, keepdims=True)
            left, right = p[:, arrays["edges"] == 1].sum(axis=1), p[:, arrays["edges"] == 2].sum(axis=1)
            recalls, coverages, cursor = {"left": [], "right": []}, [], 0
            for length, arm in zip(arrays["traversal_lengths"], arrays["traversal_arms"], strict=True):
                section = slice(cursor, cursor + int(length))
                truth = arrays["validation_truth"][section]
                unique = arrays["edges"][truth] != 0
                correct = np.where(np.isclose(left[section], right[section], rtol=0, atol=1e-12), .5,
                                   ((left[section] > right[section]) == (arrays["edges"][truth] == 1)).astype(float))
                recalls[str(arm)].append(float(correct[unique].mean()))
                coverages.append(float(arrays["support"][truth[unique]].mean()))
                cursor += int(length)
            values = {"left_recall": np.mean(recalls["left"]), "right_recall": np.mean(recalls["right"]), "support_coverage": np.mean(coverages)}
            values["balanced_accuracy"] = (values["left_recall"] + values["right_recall"]) / 2
            if any(not np.isclose(row[key], value, rtol=0, atol=1e-10) for key, value in values.items()):
                raise ValueError("Independent saved-likelihood validation metrics differ")
            checked += 1
    atomic_json(args.output_dir / (PREFIX + "neural_verification.json"), {"status": "verified_saved_likelihood_metrics", "n_checked": checked,
                                                                        "association_fitted": False})
