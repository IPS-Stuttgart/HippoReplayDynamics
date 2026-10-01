"""Causal first-third encoding and middle-third RUN QC for the v2 driver."""

from __future__ import annotations

import json
import time as clock
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

try:
    from scripts import analyze_denovellis_post_error_content as legacy
    from scripts._provenance import build_script_provenance, file_sha256
    from scripts.denovellis_post_error_neural import fit_encoding, load_marks, make_graph
    from scripts.denovellis_post_error_v2 import PREFIX, ROOT, check_raw_inputs, coverage_bound, elapsed_thirds, full_window_mask, verify_manifest
except ModuleNotFoundError:
    import analyze_denovellis_post_error_content as legacy
    from _provenance import build_script_provenance, file_sha256
    from denovellis_post_error_neural import fit_encoding, load_marks, make_graph
    from denovellis_post_error_v2 import PREFIX, ROOT, check_raw_inputs, coverage_bound, elapsed_thirds, full_window_mask, verify_manifest


class UnavailableReadout(ValueError):
    """Missing declared information, distinct from a measured poor decoder."""


def training_samples(time, xy, speed, bounds, traversals):
    # Crop first: neither interpolation nor occupancy may inspect a later sample.
    keep = (time >= bounds[0]) & (time < bounds[1])
    t, x, v = time[keep], xy[keep], speed[keep].copy()
    if len(t) < 2:
        raise UnavailableReadout("fewer_than_two_first_third_position_samples")
    edges = full_window_mask(t[:-1], t[1:], traversals)
    edges &= np.isfinite(x[:-1]).all(axis=1) & np.isfinite(x[1:]).all(axis=1)
    edges &= (np.diff(t) <= 0.25) & np.isfinite(v[:-1]) & np.isfinite(v[1:]) & (v[:-1] > 4) & (v[1:] > 4)
    v[~np.r_[edges, False]] = np.nan
    if not np.any(edges):
        raise UnavailableReadout("no_complete_first_third_running_traversals")
    return t, x, v


def validation_windows(graph, time, xy, speed, bounds, traversals, dt):
    starts = np.arange(bounds[1], bounds[2] - dt + 1e-9, dt)
    ends = starts + dt
    mids = starts + dt / 2
    edges_good = np.isfinite(xy[:-1]).all(axis=1) & np.isfinite(xy[1:]).all(axis=1)
    edges_good &= np.isfinite(speed[:-1]) & np.isfinite(speed[1:]) & (speed[:-1] > 4) & (speed[1:] > 4)
    edges_good &= (np.diff(time) > 0) & (np.diff(time) <= 0.25)
    # Every tracking edge touched by a bin must be valid, not only its midpoint.
    first = np.clip(np.searchsorted(time, starts, side="right") - 1, 0, len(time) - 2)
    last = np.clip(np.searchsorted(time, ends, side="left") - 1, 0, len(time) - 2)
    invalid_prefix = np.r_[0, np.cumsum(~edges_good)]
    valid = full_window_mask(starts, ends, traversals) & (ends <= bounds[2])
    valid &= (invalid_prefix[last + 1] - invalid_prefix[first]) == 0
    valid &= (time[first] >= bounds[1]) & (time[last + 1] <= bounds[2])
    position = np.column_stack([np.interp(mids, time, xy[:, j]) for j in (0, 1)])
    finite = np.isfinite(position).all(axis=1)
    nearest = np.zeros(len(starts), int)
    nearest[finite] = cKDTree(graph.xy).query(position[finite])[1]
    truth = np.full(len(starts), -1)
    for arm, mask in enumerate(graph.unique_masks):
        truth[mask[nearest]] = arm
    valid &= finite & (truth >= 0)
    return starts[valid], ends[valid], truth[valid], nearest[valid]


def fit_first_third(graph, time, xy, speed, marks, bounds, traversals, p):
    t, x, v = training_samples(time, xy, speed, bounds, traversals)
    index = cKDTree(graph.xy).query(x[np.isfinite(v)])[1]
    counts = [int(mask[index].sum()) for mask in graph.unique_masks]
    if not all(counts):
        raise UnavailableReadout("unique_arm_missing_first_third_training_support")
    first_marks = {tet: (mt[(mt >= bounds[0]) & (mt < bounds[1])], mf[(mt >= bounds[0]) & (mt < bounds[1])]) for tet, (mt, mf) in marks.items()}
    try:
        model = fit_encoding(graph, t, x, v, first_marks, start=bounds[0], end=bounds[1], spatial_sigma=p["graph_spatial_sigma_cm"], mark_sigma=p["mark_sigma_native_units"])
    except ValueError as exc:
        raise UnavailableReadout(str(exc)) from exc
    return model, counts, float(np.diff(t)[np.isfinite(v[:-1])].sum())


def summarize_windows(windows, p):
    recall = [float((windows.loc[windows.true_arm == a, "predicted_arm"] == a).mean()) if (windows.true_arm == a).any() else None for a in (0, 1)]
    if any(r is None for r in recall):
        raise UnavailableReadout("unique_arm_missing_middle_third_validation_support")
    balanced = float(np.mean(recall))
    passed = balanced >= p["min_run_balanced_accuracy"] and min(recall) >= p["min_run_arm_recall"]
    return {
        "status": "passed" if passed else "failed_accuracy",
        "decoder_qc_passed": passed,
        "arm0_recall": recall[0],
        "arm1_recall": recall[1],
        "balanced_accuracy": balanced,
        "n_arm_windows": len(windows),
        "n_arm0_windows": int((windows.true_arm == 0).sum()),
        "n_arm1_windows": int((windows.true_arm == 1).sum()),
        "zero_spike_windows": int((windows.n_spikes == 0).sum()),
        "median_spikes_per_window": float(windows.n_spikes.median()),
        "median_active_tetrodes_per_window": float(windows.n_active_tetrodes.median()),
    }


def decode_middle_third(model, time, xy, speed, marks, bounds, traversals, p):
    starts, ends, truth, nearest = validation_windows(model.graph, time, xy, speed, bounds, traversals, p["time_bin_s"])
    if not all(np.any(truth == a) for a in (0, 1)):
        raise UnavailableReadout("unique_arm_missing_middle_third_validation_support")
    rows = []
    for i in range(0, len(starts), 256):
        a, b = starts[i : i + 256], ends[i : i + 256]
        posterior, counts, active = model.likelihood(a, b, marks)
        masses = np.column_stack([posterior[:, mask].sum(axis=1) for mask in model.graph.unique_masks])
        if not np.isfinite(posterior).all() or not np.allclose(posterior.sum(axis=1), 1):
            raise ValueError("Invalid normalized flat-prior posterior")
        for j in range(len(a)):
            rows.append(
                {
                    "start_time_s": a[j],
                    "end_time_s": b[j],
                    "true_arm": int(truth[i + j]),
                    "predicted_arm": int(np.argmax(masses[j])),
                    "true_graph_bin": int(nearest[i + j]),
                    "true_bin_training_supported": bool(model.occupied[nearest[i + j]]),
                    "arm0_mass": masses[j, 0],
                    "arm1_mass": masses[j, 1],
                    "n_spikes": int(counts[j]),
                    "n_active_tetrodes": int(active[j]),
                }
            )
    windows = pd.DataFrame(rows)
    return summarize_windows(windows, p), windows


def evaluate_epoch(args, entry, traversals, p, hash_cache):
    tick = clock.monotonic()
    animal, day, epoch = entry["animal"], int(entry["day"]), int(entry["epoch"])
    session = entry["session"]
    folder = args.dataset_root / legacy.ANIMALS[animal]
    time, xy, speed, wells, coords, center, outers, _, _, paths = legacy.epoch_data(folder, day, epoch, p)
    bounds = elapsed_thirds(time)
    if not np.allclose(bounds, [entry[c] for c in ("epoch_start_s", "train_end_s", "validation_end_s", "epoch_end_s")], rtol=0, atol=1e-9):
        raise ValueError("Frozen epoch boundaries changed")
    graph = make_graph(coords, wells, center, outers, p["graph_bin_cm"])
    sources = [*paths.values(), folder / f"{animal}tetinfo.mat"]
    row = dict(
        animal=animal,
        day=day,
        epoch=epoch,
        session=session,
        decoder_qc_passed=False,
        status="unavailable",
        reason="",
        **{c: float(b) for c, b in zip(("epoch_start_s", "train_end_s", "validation_end_s", "epoch_end_s"), bounds, strict=True)},
    )
    windows = pd.DataFrame()
    try:
        try:
            marks, mark_paths = load_marks(folder, animal, day, epoch, p["hippocampal_areas"])
        except (ValueError, FileNotFoundError) as exc:
            # Preserve every inspected mark source even when its schema is unavailable.
            sources.extend(sorted((folder / "EEG").glob(f"{animal}marks{day:02d}-*.mat")))
            raise UnavailableReadout(str(exc)) from exc
        sources.extend(mark_paths)
        row["n_available_tetrodes"] = len(marks)
        model, counts, exposure = fit_first_third(graph, time, xy, speed, marks, bounds, traversals.loc[traversals.block == "train"].to_dict("records"), p)
        row.update(
            n_encoding_tetrodes=len(model.features),
            training_run_exposure_s=exposure,
            arm0_training_samples=counts[0],
            arm1_training_samples=counts[1],
            n_training_marks=sum(len(f) for f in model.features.values()),
        )
        metrics, windows = decode_middle_third(model, time, xy, speed, marks, bounds, traversals.loc[traversals.block == "validation"].to_dict("records"), p)
        row.update(metrics)
    except UnavailableReadout as exc:
        row["reason"] = str(exc)
    inputs = []
    for path in dict.fromkeys(sources):
        if path.exists():
            signature = (str(path), path.stat().st_size, path.stat().st_mtime_ns)
            if signature not in hash_cache:
                hash_cache[signature] = file_sha256(path)
            inputs.append({"path": str(path), "sha256": hash_cache[signature], "size_bytes": path.stat().st_size, "role": "within_epoch_run_qc"})
    row["runtime_s"] = clock.monotonic() - tick
    return row, windows, inputs


def cohort_after_qc(trials, qc, animals, p):
    if qc.session.duplicated().any() or not qc.decoder_qc_passed.isin([True, False]).all():
        raise ValueError("Unique sessions and explicit decoder decisions required")
    if not set(qc.session).issubset(set(trials.loc[trials.final_third_eligible, "session"])):
        raise ValueError("Readout session outside frozen target cohort")
    for row in qc.to_dict("records"):
        metrics = [row.get(k, np.nan) for k in ("balanced_accuracy", "arm0_recall", "arm1_recall")]
        measured = all(x is not None and np.isfinite(x) for x in metrics)
        passed = measured and metrics[0] >= p["min_run_balanced_accuracy"] and min(metrics[1:]) >= p["min_run_arm_recall"]
        if bool(row["decoder_qc_passed"]) != bool(passed):
            raise ValueError("Readout pass inconsistent with fixed numerical thresholds")
    result = trials.copy()
    result["decoder_status"] = result.session.map(qc.set_index("session").status).fillna("not_needed_no_final_third_transition")
    result["decoder_qc_passed"] = result.session.map(qc.set_index("session").decoder_qc_passed).fillna(False).astype(bool)
    result["decoder_qualified_transition"] = result.final_third_eligible & result.decoder_qc_passed
    result, summaries, gates = coverage_bound(result, animals, p, eligibility="decoder_qualified_transition")
    for gate in gates:
        gate["gate"] = gate["gate"].replace("_upper_bound", "")
        gate["gate_type"] = "actual_validated_coverage"
    return result, summaries, gates


def run_feasibility(args, p, prior):
    if prior["stage"] != "audit" or args.dataset_root is None:
        raise ValueError("Within-epoch RUN QC requires a passed v2 audit and dataset root")
    source, out = args.prerequisite_dir.resolve(), args.output_dir.resolve()
    if out == source or out in source.parents or source in out.parents:
        raise ValueError("Do not overwrite upstream audit")
    provenance = build_script_provenance(input_paths={"protocol": args.protocol, "audit_manifest": source / (PREFIX + "manifest.json")}, cwd=ROOT)
    if provenance["git_dirty"] is not False or provenance["code_commit"] == "unavailable":
        raise ValueError("A clean committed producer is mandatory")
    identity = {"code_commit": provenance["code_commit"], **provenance["input_file_sha256"], "dataset_root": str(args.dataset_root.resolve())}
    out.mkdir(parents=True, exist_ok=True)
    control = out / "readout_control.json"
    if control.exists() and json.loads(control.read_text()) != identity:
        raise ValueError("Readout producer/input mismatch")
    legacy.atomic_json(control, identity)
    source_inputs = pd.read_csv(source / (PREFIX + "input_inventory.csv")).replace({np.nan: None}).to_dict("records")
    check_raw_inputs(source_inputs)
    inventory = pd.read_csv(source / (PREFIX + "cohort_inventory.csv"))
    trials = pd.read_csv(source / (PREFIX + "trial_inventory.csv"))
    traversals = pd.read_csv(source / (PREFIX + "traversal_inventory.csv"))
    needed = set(trials.loc[trials.final_third_eligible, "session"])
    rows, inputs, hash_cache = [], list(source_inputs), {}
    checkpoints = out / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    for entry in inventory.loc[inventory.session.isin(needed)].sort_values(["animal", "day", "epoch"]).to_dict("records"):
        if datetime.now(UTC) > datetime.fromisoformat(prior["development_deadline_utc"]):
            raise RuntimeError("Frozen five-working-day development deadline exceeded")
        path = checkpoints / (entry["session"] + ".json")
        window_path = checkpoints / (entry["session"] + "_windows.csv")
        if path.exists():
            packet = json.loads(path.read_text())
            if packet["identity"] != identity or file_sha256(window_path) != packet["windows_sha256"]:
                raise ValueError("Checkpoint context/windows mismatch")
            check_raw_inputs(packet["inputs"])
        else:
            print(json.dumps({"starting": entry["session"], "completed": len(rows), "total": len(needed)}), flush=True)
            row, windows, used = evaluate_epoch(args, entry, traversals.loc[traversals.session == entry["session"]], p, hash_cache)
            if windows.empty:
                windows = pd.DataFrame(columns=["start_time_s", "end_time_s", "true_arm", "predicted_arm", "n_spikes"])
            windows.to_csv(window_path, index=False)
            packet = {"identity": identity, "row": row, "inputs": used, "windows_sha256": file_sha256(window_path)}
            legacy.atomic_json(path, packet)
        rows.append(packet["row"])
        inputs.extend(packet["inputs"])
        legacy.atomic_json(out / "progress.json", {"stage": "feasibility", "completed_epochs": len(rows), "total_epochs": len(needed), "last_result": packet["row"]})
        print(json.dumps(packet["row"]), flush=True)
    qc = pd.DataFrame(rows)
    cohort, animals, gates = cohort_after_qc(trials, qc, inventory.animal.unique(), p)
    passed = all(r["passed"] for r in gates)
    gates.append(
        {
            "gate": "feasibility_overall",
            "passed": passed,
            "observed": "validated_coverage_pass" if passed else "validated_coverage_fail",
            "criterion": "all fixed coverage floors",
            "gate_type": "feasibility",
        }
    )
    inventory["decoder_status"] = inventory.session.map(qc.set_index("session").status).fillna("not_needed_no_final_third_transition")
    inventory["decoder_qc_passed"] = inventory.session.map(qc.set_index("session").decoder_qc_passed)
    cohort["exclusion_reason_v2"] = np.where(
        ~cohort.final_third_eligible,
        cohort.v2_exclusion_reason.fillna(""),
        np.where(~cohort.decoder_qc_passed, "run_readout:" + cohort.decoder_status, np.where(~cohort.primary_cohort_eligible, "animal_lacks_both_outcomes_or_two_days", "")),
    )
    for name, data in [
        ("decoder_audit", qc),
        ("cohort_inventory", inventory),
        ("trial_inventory", cohort),
        ("by_animal", animals),
        ("gate_summary", gates),
        ("input_inventory", pd.DataFrame(inputs).drop_duplicates("path")),
        ("exclusions", cohort.groupby("exclusion_reason_v2", dropna=False).size().rename("n_transitions").reset_index()),
    ]:
        legacy.csv(out, name, data)
    decision = {
        "decision": "ready_for_frozen_calibration" if passed else "insufficient_v2_validated_coverage",
        "biological_association_tested": False,
        "run_readout_performed": True,
        "calibration_performed": False,
        "sequence_validation_performed": False,
        "evaluated_epochs": len(qc),
        "passing_epochs": int(qc.decoder_qc_passed.sum()),
    }
    legacy.atomic_json(out / (PREFIX + "decision.json"), decision)
    manifest = {
        "protocol": p,
        "stage": "feasibility",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "provenance": provenance,
        "source_directory": str(source),
        "source_manifest_sha256": file_sha256(source / (PREFIX + "manifest.json")),
        "development_deadline_utc": prior["development_deadline_utc"],
        "audit_passed": True,
        "feasibility_passed": passed,
        "calibration_passed": False,
        "biological_status": "not_tested",
        "next_action": decision["decision"],
        "output_sha256": {f.name: file_sha256(f) for f in sorted(out.glob("*.csv"))} | {PREFIX + "decision.json": file_sha256(out / (PREFIX + "decision.json"))},
        "checkpoint_sha256": {f.name: file_sha256(f) for f in sorted(checkpoints.glob("*"))},
    }
    legacy.atomic_json(out / (PREFIX + "manifest.json"), manifest)
    report_feasibility(out)


def report_feasibility(out):
    manifest = verify_manifest(out, "2.0")
    qc = pd.read_csv(out / (PREFIX + "decoder_audit.csv"))
    trials = pd.read_csv(out / (PREFIX + "trial_inventory.csv"))
    animals = pd.read_csv(out / (PREFIX + "by_animal.csv"))
    lines = [
        "# Post-error replay v2: causal RUN readout",
        "",
        f"Decision: `{manifest['next_action']}`.",
        "",
        "The first third alone trains; the middle third validates; final-third outcomes are not used in decoder fitting or selection.",
        "The fixed flat-prior decoder, temporal cutoffs and accuracy thresholds were not tuned.",
        "",
        f"Evaluated epochs: {len(qc)}. Decoder passes: {int(qc.decoder_qc_passed.sum())}.",
        f"Final-third behavioral transitions: {int(trials.final_third_eligible.sum())}. Decoder-qualified: {int(trials.decoder_qualified_transition.sum())}.",
        f"After both-outcomes/two-days animal requirement: {int(trials.primary_cohort_eligible.sum())} transitions across {int(animals.animal_supported.sum())} animals.",
        "",
        "| Animal | Decoder-qualified transitions | Corrections | Repetitions | Days | Animal supported |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    lines += [f"| {r.animal} | {r.transitions} | {r.corrections} | {r.repetitions} | {r.recording_days} | {r.animal_supported} |" for r in animals.itertuples()]
    lines += [
        "",
        "Zero-spike RUN windows are included. Shared-stem locations are not scored as an outer arm.",
        "Missing arm/traversal support is unavailable information, distinct from measured poor accuracy.",
        "The inventory includes all source epochs; neural fits are needed only for those with final-third transitions.",
        "",
        "No replay content, 1,000-replicate validation bank, behavioral regression or biological association has been scored.",
        "A failed coverage floor stops the study; it neither supports nor refutes corrective replay.",
        "V1 remains stopped and unchanged. No criterion was relaxed.",
        "",
        f"Producer: `{manifest['provenance']['code_commit']}`.",
        "",
    ]
    (out / (PREFIX + "report.md")).write_text("\n".join(lines))
