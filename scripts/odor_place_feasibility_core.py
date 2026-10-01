"""Source-gated inventory and verification; never infer a stimulus from an outcome."""

from __future__ import annotations

import csv
import hashlib
import json

import h5py
import numpy as np

from scripts._provenance import git_metadata
from scripts.odor_place_source_io import BoundedHTTPFile, atomic_json, digest

PREFIX = "odor_place_post_error_"


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, bytes):
        return value.decode()
    return value


def write_table(root, name, rows, columns=None):
    fields = list(columns or dict.fromkeys(k for row in rows for k in row))
    if not fields:
        raise ValueError("An empty table must still have an explicit schema")
    path = root / (PREFIX + name + ".csv")
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(clean(v), sort_keys=True) if isinstance(v, (list, dict)) else clean(v) for k, v in row.items()})
    temporary.replace(path)


def read_table(root, name):
    with (root / (PREFIX + name + ".csv")).open(newline="") as handle:
        return list(csv.DictReader(handle))


def mat_item(handle, node, index=0):
    if isinstance(node, h5py.Dataset) and h5py.check_dtype(ref=node.dtype):
        reference = node[()].ravel()[index]
        if not reference:
            raise ValueError("Empty MATLAB reference")
        return handle[reference]
    return node


def mat_string(handle, node):
    node = mat_item(handle, node)
    if isinstance(node, h5py.Group):
        raise ValueError("Expected a MATLAB character array")
    data = node[()].ravel()
    return "".join(chr(int(v)) for v in data)


def mat_strings(handle, node):
    if isinstance(node, h5py.Dataset) and h5py.check_dtype(ref=node.dtype):
        return [mat_string(handle, mat_item(handle, node, i)) for i in range(node.size)]
    return [mat_string(handle, node)]


def scalar(node):
    value = node[()].ravel()
    if value.size != 1:
        raise ValueError("Expected scalar source metadata")
    return clean(value[0])


def source_inventory(path):
    records, trials, units = [], [], []
    with h5py.File(path, "r") as handle:
        if "SuperRat" not in handle:
            raise ValueError("Pinned source must contain the actual SuperRat release, not a guessed ZIP adapter")
        root = handle["SuperRat"]
        for i in range(root["name"].size):
            def item(key):
                return mat_item(handle, root[key], i)
            animal, day = mat_string(handle, item("name")), int(scalar(item("daynum")))
            trial = item("trialdata")
            tracking = item("tracking")
            fields = mat_strings(handle, tracking["fields"])
            files = item("files")
            task_paths = mat_strings(handle, files["taskfile"])
            unit = item("units")
            epochs = item("RunEpochs")[()].ravel().astype(int).tolist()
            starts = trial["sniffstart"][()].ravel()
            record = {"source_record_index": i, "animal": animal, "source_day": day,
                      "declared_long_track": scalar(item("longTrack")), "source_run_epochs": epochs,
                      "n_source_trials": len(starts), "trial_fields": list(trial), "source_task_paths": task_paths,
                      "tracking_fields": fields, "tracking_shape": list(tracking["data"].shape),
                      "n_source_units": unit["ts"].size, "source_odor_trigger_arrays_present": False,
                      "cue_semantics_status": "compiled_side_field_not_verified_as_independent_stimulus",
                      "apparatus_status": "declared_long_track_not_independently_reconciled",
                      "recording_date_status": "source_day_available_calendar_date_unresolved"}
            records.append(record)
            arrays = {key: trial[key][()].ravel() for key in trial if key != "EpochInds"}
            record["source_trial_field_lengths"] = {key: len(v) for key, v in arrays.items()}
            record["nonaligned_trial_columns"] = {key: clean(v) for key, v in arrays.items() if len(v) != len(starts)}
            epoch_data = trial["EpochInds"][()].T
            if epoch_data.shape != (len(starts), 2):
                raise ValueError("Unexpected source EpochInds schema")
            for j in range(len(starts)):
                row = {"animal": animal, "source_day": day, "source_trial_index": j,
                       "source_epoch": int(epoch_data[j, 1]) if np.isfinite(epoch_data[j, 1]) else None,
                       "cue_verified": False, "cue_exclusion_reason": record["cue_semantics_status"]}
                row["source_unaligned_fields"] = list(record["nonaligned_trial_columns"])
                row.update({key: clean(values[j]) if len(values) == len(starts) else None for key, values in arrays.items()})
                trials.append(row)
            for j in range(unit["ts"].size):
                unit_row = {"animal": animal, "source_day": day, "source_unit_index": j,
                            "source_tetrode": scalar(mat_item(handle, unit["tet"], j)),
                            "source_cluster": scalar(mat_item(handle, unit["unitnum"], j)),
                            "source_area": mat_string(handle, mat_item(handle, unit["area"], j)),
                            "identity_status": "source_metadata_only_not_nwb_spike_verified"}
                units.append(unit_row)
    keys = [(r["animal"], r["source_day"]) for r in records]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate source recording days")
    return records, trials, units


def unique_clock_matches(source_times, target_times, tolerance):
    source, target = np.asarray(source_times, float), np.asarray(target_times, float)
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("Nonfinite clock timestamps")
    if len(np.unique(source)) != len(source) or len(np.unique(target)) != len(target):
        raise ValueError("Duplicate trial timestamps")
    result = []
    for value in target:
        matches = np.flatnonzero(np.abs(source - value) <= tolerance)
        result.append(int(matches[0]) if len(matches) == 1 else None)
    nonmissing = [i for i in result if i is not None]
    if len(set(nonmissing)) != len(nonmissing):
        raise ValueError("Ambiguous many-to-one trial crosswalk")
    return result


def source_examples(path, records, trials):
    examples = []
    with h5py.File(path, "r") as handle:
        root = handle["SuperRat"]
        for animal in sorted({r["animal"] for r in records}):
            errors = [t for t in trials if t["animal"] == animal and t["CorrIncorr10"] == 0]
            if not errors:
                continue
            first = min(errors, key=lambda t: (t["source_day"], t["sniffstart"]))
            record = next(r for r in records if r["animal"] == animal and r["source_day"] == first["source_day"])
            tracking = mat_item(handle, root["tracking"], record["source_record_index"])["data"][()].T
            fields = record["tracking_fields"]
            if fields != ["time", "x", "y", "dir", "vel", "epoch"]:
                raise ValueError("Source tracking columns have changed; no guessed position adapter")
            selected = tracking[(tracking[:, 0] >= first["sniffstart"] - .5) & (tracking[:, 0] <= first["sniffstart"] + 12)]
            if len(selected) > 500:
                raise ValueError("Example exceeds frozen bounded tracking excerpt")
            examples.append({"animal": animal, "source_day": first["source_day"], "source_trial_index": first["source_trial_index"],
                             "selection_rule": "earliest_source_annotated_error_unresolved_boundary_example",
                             "trial_annotations": first, "tracking_fields": fields, "tracking": clean(selected),
                             "not_an_eligible_cue_condition_example": True})
    return examples


def header_observations(handle):
    def text(path):
        value = handle[path][()]
        return clean(value)
    observations = {"subject": text("general/subject/subject_id"), "session_start_time": text("session_start_time"),
                    "session_description": text("session_description"), "top_level_groups": list(handle), "epochs": []}
    epoch_groups = [group for group in handle["intervals"].values() if isinstance(group, h5py.Group) and "epoch_type" in group]
    for group in epoch_groups:
        types = clean(group["epoch_type"][()])
        indices = group["epoch_type_index"][()] if "epoch_type_index" in group else np.arange(len(types)) + 1
        previous = 0
        for i, end in enumerate(indices):
            observations["epochs"].append({"epoch_index": i, "start_s": float(group["start_time"][i]), "end_s": float(group["stop_time"][i]),
                                            "epoch_types": types[previous:int(end)]})
            previous = int(end)
    trials = handle["intervals/trials"]
    observations["trial_fields"] = list(trials)
    observations["trial_column_descriptions"] = {k: clean(v.attrs.get("description", "")) for k, v in trials.items() if isinstance(v, h5py.Dataset)}
    observations["trial_arrays"] = {k: clean(trials[k][()]) for k in ["id", "start_time", "stop_time", "rewarded", "reward_start_time", "reward_end_time"]}
    position = handle["processing/behavior/Position/SpatialSeries"]
    observations["position"] = {"data_shape": list(position["data"].shape), "unit": clean(position["data"].attrs.get("unit", "")),
                               "comments": clean(position.attrs.get("comments", "")), "reference_frame": text(position.name + "/reference_frame")}
    units = handle["units"]
    observations["unit_ids"] = clean(units["id"][()])
    observations["unit_electrode_references"] = clean(units["electrodes"][()])
    observations["unit_electrode_index"] = clean(units["electrodes_index"][()]) if "electrodes_index" in units else None
    electrodes = handle["general/extracellular_ephys/electrodes"]
    observations["electrode_ids"] = clean(electrodes["id"][()])
    observations["electrode_locations"] = clean(electrodes["location"][()])
    observations["electrode_group_names"] = clean(electrodes["group_name"][()])
    observations["electrode_referenced"] = clean(electrodes["referenced"][()]) if "referenced" in electrodes else None
    observations["stimulus_presentation_fields"] = list(handle.get("stimulus/presentation", {}))
    return clean(observations)


def inventory(args, protocol):
    meta_dir, out = args.dataset_root / "metadata", args.output_dir
    verified = json.loads((meta_dir / "source_verified.json").read_text())
    if digest(args.source_archive) != verified["sha256"] or digest(args.source_archive, "md5") != verified["published_digest"]:
        raise ValueError("Source file differs from hash-verified acquisition")
    records, source_trials, source_units = source_inventory(args.source_archive)
    listing = json.loads((meta_dir / "dandi_assets.json").read_text())["results"]
    provenance = git_metadata()
    base_identity = {"code_commit": provenance["code_commit"], "protocol_sha256": digest(args.protocol), "source_sha256": verified["sha256"]}
    checkpoint_dir = out / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    asset_rows, nwb_trials, crosswalks, nwb_units = [], [], [], []
    for index, asset in enumerate(listing):
        metadata_path = meta_dir / (asset["asset_id"] + ".json")
        metadata = json.loads(metadata_path.read_text())
        identity = {**base_identity, "asset_metadata_sha256": digest(metadata_path)}
        checkpoint_path = checkpoint_dir / (asset["asset_id"] + ".json")
        if checkpoint_path.exists():
            saved = json.loads(checkpoint_path.read_text())
            if saved["identity"] != identity:
                raise ValueError("Checkpoint identity differs; use a new output directory")
            packet = saved["packet"]
            expected = hashlib.sha256(json.dumps(packet, sort_keys=True, allow_nan=False).encode()).hexdigest()
            if expected != saved["packet_sha256"]:
                raise ValueError("Header checkpoint checksum differs")
        else:
            urls = [u for u in metadata["contentUrl"] if u.startswith("https://dandiarchive.s3.amazonaws.com/")]
            if len(urls) != 1:
                raise ValueError("No unique public S3 asset URL")
            stream = BoundedHTTPFile(urls[0], metadata["contentSize"], protocol["remote_header_budget_bytes_per_asset"])
            try:
                with h5py.File(stream, "r") as handle:
                    observed = header_observations(handle)
                packet = {"status": "readable", "observations": observed, "transferred_bytes": stream.transferred_bytes, "failure_reason": ""}
            except Exception as exc:
                packet = {"status": "unreadable", "observations": {}, "transferred_bytes": stream.transferred_bytes,
                          "failure_reason": f"{type(exc).__name__}: {exc}"}
            packet = clean(packet)
            atomic_json(checkpoint_path, {"identity": identity, "packet": packet,
                                         "packet_sha256": hashlib.sha256(json.dumps(packet, sort_keys=True, allow_nan=False).encode()).hexdigest()})
        observed = packet["observations"]
        participants = [p["identifier"] for p in metadata["wasAttributedTo"] if p.get("schemaKey") == "Participant"]
        if len(participants) != 1:
            raise ValueError("Ambiguous DANDI animal identity")
        animal = participants[0].removeprefix("Symanski-")
        if observed and observed["subject"] != participants[0]:
            raise ValueError("NWB subject conflicts with pinned asset metadata")
        raw = observed.get("trial_arrays", {})
        starts = raw.get("start_time", [])
        candidates = []
        for day in [r["source_day"] for r in records if r["animal"] == animal]:
            prior = [t for t in source_trials if t["animal"] == animal and t["source_day"] == day]
            matched = unique_clock_matches([t["sniffstart"] for t in prior], starts, protocol["clock_match_tolerance_s"])
            candidates.append((sum(m is not None for m in matched), day, prior, matched))
        candidates.sort(key=lambda r: r[0], reverse=True)
        unique_day = bool(candidates and candidates[0][0] > 0 and (len(candidates) == 1 or candidates[0][0] > candidates[1][0]))
        best = candidates[0] if unique_day else (0, None, [], [None] * len(starts))
        row = {"asset_id": asset["asset_id"], "path": asset["path"], "animal": animal, "status": packet["status"], "failure_reason": packet["failure_reason"],
               "size_bytes": metadata["contentSize"], "published_sha256": metadata["digest"]["dandi:sha2-256"], "full_file_hash_verified": False,
               "header_transferred_bytes": packet["transferred_bytes"], "n_nwb_trials": len(starts), "n_nwb_errors": sum(v == 0 for v in raw.get("rewarded", [])),
               "candidate_source_day": best[1], "n_source_clock_matches": best[0],
               "chronology_status": "candidate_day_by_source_timestamps_not_fully_verified" if unique_day else "unresolved_source_day",
               "conversion_session_start_time": observed.get("session_start_time", ""), "used_conversion_date_for_chronology": False,
               "epochs": observed.get("epochs", []), "has_independent_cue_column": any(k in observed.get("trial_fields", []) for k in ["odor_identity", "odor_id", "cue_id"]),
               "stimulus_presentation_fields": observed.get("stimulus_presentation_fields", []), "n_nwb_units": len(observed.get("unit_ids", [])),
               "primary_eligible": False, "primary_exclusion_reason": "independent_cue_and_source_condition_not_verified"}
        asset_rows.append(row)
        for j in range(len(starts)):
            target = {"animal": animal, "asset_id": asset["asset_id"], "nwb_trial_id": raw["id"][j], "candidate_source_day": best[1],
                      "nosepoke_start_s": starts[j], "nosepoke_stop_s": raw["stop_time"][j], "correct_choice_annotation": raw["rewarded"][j],
                      "nwb_well_start_s": raw["reward_start_time"][j], "nwb_well_stop_s": raw["reward_end_time"][j],
                      "cue_verified": False, "cue_id": None, "eligible": False,
                      "exclusion_reason": "independent_stimulus_not_verified"}
            nwb_trials.append(target)
            source_index = best[3][j]
            source = best[2][source_index] if source_index is not None else {}
            crosswalks.append({"animal": animal, "asset_id": asset["asset_id"], "nwb_trial_id": raw["id"][j], "candidate_source_day": best[1],
                              "source_trial_index": source.get("source_trial_index"), "source_epoch": source.get("source_epoch"),
                              "clock_matched_within_5ms": source_index is not None,
                              "source_sniff_start_s": source.get("sniffstart"), "nwb_nosepoke_start_s": starts[j],
                              "onset_difference_s": starts[j] - source["sniffstart"] if source else None,
                              "source_sniff_stop_s": source.get("sniffend"), "nwb_nosepoke_stop_s": raw["stop_time"][j],
                              "source_well_start_s": source.get("rewardstart"), "source_well_stop_s": source.get("rewardend"),
                              "nwb_well_start_s": raw["reward_start_time"][j], "nwb_well_stop_s": raw["reward_end_time"][j],
                              "source_side_annotation": source.get("leftright10"), "source_correct_annotation": source.get("CorrIncorr10"),
                              "outcome_annotations_agree": source.get("CorrIncorr10") == raw["rewarded"][j] if source else None,
                              "cue_identity_status": "not_verified_no_outcome_based_reconstruction"})
        references = observed.get("unit_electrode_references", [])
        indices = observed.get("unit_electrode_index")
        for j, unit_id in enumerate(observed.get("unit_ids", [])):
            refs = references[int(indices[j-1]) if j else 0:int(indices[j])] if indices else [references[j]]
            nwb_units.append({"animal": animal, "asset_id": asset["asset_id"], "nwb_unit_id": unit_id, "nwb_electrode_reference_values": refs,
                              "source_tetrode": None, "source_cluster": None, "verified_ca1": False,
                              "identity_status": "not_attempted_source_prerequisite_failed_no_row_index_assumption"})
        print(f"header_inventory {index + 1}/{len(listing)} {animal}: {row['status']} trials={len(starts)} source_matches={best[0]}", flush=True)
        atomic_json(out / "progress.json", {"stage": "inventory", "assets_completed": index + 1, "assets_total": len(listing)})
    write_table(out, "source_inventory", records)
    write_table(out, "source_trial_inventory", source_trials)
    write_table(out, "source_unit_inventory", source_units)
    atomic_json(out / (PREFIX + "unresolved_examples.json"), source_examples(args.source_archive, records, source_trials))
    write_table(out, "asset_inventory", asset_rows)
    write_table(out, "trial_inventory", nwb_trials, list(nwb_trials[0]) if nwb_trials else ["animal", "asset_id", "nwb_trial_id", "eligible", "exclusion_reason"])
    write_table(out, "trial_crosswalk", crosswalks, list(crosswalks[0]) if crosswalks else ["animal", "asset_id", "nwb_trial_id", "clock_matched_within_5ms"])
    write_table(out, "unit_crosswalk", nwb_units, list(nwb_units[0]) if nwb_units else ["animal", "asset_id", "nwb_unit_id", "verified_ca1", "identity_status"])
    transitions = []
    for asset in asset_rows:
        rows = [t for t in nwb_trials if t["asset_id"] == asset["asset_id"]]
        for j, trial in enumerate(rows):
            if trial["correct_choice_annotation"] != 0:
                continue
            transitions.append({"animal": trial["animal"], "asset_id": trial["asset_id"], "error_trial_id": trial["nwb_trial_id"],
                                "next_nwb_trial_id": rows[j + 1]["nwb_trial_id"] if j + 1 < len(rows) else None,
                                "cue_condition": "unclassifiable", "next_choice_outcome": "unclassifiable", "eligible": False,
                                "exclusion_reason": "cue_source_condition_and_first_well_dwell_not_verified",
                                "ripple_opportunities": None, "run_qc_status": "not_run_source_prerequisite_failed"})
    write_table(out, "post_error_transition_inventory", transitions, ["animal", "asset_id", "error_trial_id", "next_nwb_trial_id", "cue_condition",
                                                                   "next_choice_outcome", "eligible", "exclusion_reason", "ripple_opportunities", "run_qc_status"])
    animal_rows = []
    for animal in protocol["animals"]:
        ar = [r for r in asset_rows if r["animal"] == animal]
        sr = [r for r in records if r["animal"] == animal]
        animal_rows.append({"animal": animal, "n_assets": len(ar), "n_readable_headers": sum(r["status"] == "readable" for r in ar),
                            "n_source_days": len(sr), "n_nwb_trials": sum(r["n_nwb_trials"] for r in ar), "n_nwb_errors": sum(r["n_nwb_errors"] for r in ar),
                            "n_independent_cue_verified_trials": 0, "n_primary_eligible_transitions": 0, "n_decoder_qualified_transitions": 0,
                            "supported_sequence_testing_opportunities": None, "status": "source_blocked"})
    write_table(out, "animal_summary", animal_rows)
    atomic_json(out / (PREFIX + "decision.json"), {"status": "inconclusive_feasibility", "ready_for_calibration": False,
                                                "source_prerequisites_passed": False, "reason": "independent_cues_condition_chronology_and_unit_crosswalk_unverified",
                                                "n_assets": len(asset_rows), "n_nwb_trials": len(nwb_trials), "n_nwb_errors": len(transitions),
                                                "n_source_days": len(records), "n_source_animals": len({r['animal'] for r in records}),
                                                "full_nwb_files_downloaded": 0, "association_fitted": False})
    outputs = {p.name: digest(p) for p in out.glob(PREFIX + "*.csv")}
    outputs[PREFIX + "unresolved_examples.json"] = digest(out / (PREFIX + "unresolved_examples.json"))
    atomic_json(out / (PREFIX + "inventory_identity.json"), {**base_identity, "output_sha256": outputs,
                                                           "metadata_sha256": {p.name: digest(p) for p in meta_dir.glob("*.json")}})
    return json.loads((out / (PREFIX + "decision.json")).read_text())


def blocked_run_qc(args, protocol):
    decision = json.loads((args.output_dir / (PREFIX + "decision.json")).read_text())
    if decision["source_prerequisites_passed"]:
        raise ValueError("This pinned-source adapter has no independently verified cues; do not bypass its source gate")
    rows = [{"asset_id": a["asset_id"], "animal": a["animal"], "status": "not_run_source_prerequisite_failed",
             "balanced_accuracy": None, "left_arm_recall": None, "right_arm_recall": None, "support_coverage": None,
             "n_verified_ca1_units": None, "failure_reason": decision["reason"]} for a in read_table(args.output_dir, "asset_inventory")]
    write_table(args.output_dir, "run_validation", rows)
    opportunity_rows = [{"asset_id": r["asset_id"], "animal": r["animal"], "status": r["status"], "n_ripples": None,
                         "n_sequence_testing_opportunities": None, "validated_replay": False} for r in rows]
    write_table(args.output_dir, "ripple_opportunities", opportunity_rows)
    return {"status": "not_run_source_prerequisite_failed", "full_nwb_downloads": 0, "reason": decision["reason"]}


def verify(args, protocol):
    out = args.output_dir
    identity = json.loads((out / (PREFIX + "inventory_identity.json")).read_text())
    if identity["protocol_sha256"] != digest(args.protocol) or identity["source_sha256"] != digest(args.source_archive):
        raise ValueError("Input identity changed")
    for name, expected in identity["output_sha256"].items():
        if digest(out / name) != expected:
            raise ValueError(f"Output hash differs: {name}")
    for name, expected in identity["metadata_sha256"].items():
        if digest(args.dataset_root / "metadata" / name) != expected:
            raise ValueError(f"Pinned metadata hash differs: {name}")
    checkpoint_rows, checkpoint_trials, checkpoint_errors, ids = 0, 0, 0, set()
    for path in sorted((out / "checkpoints").glob("*.json")):
        saved = json.loads(path.read_text())
        if any(saved["identity"][key] != identity[key] for key in ["code_commit", "protocol_sha256", "source_sha256"]):
            raise ValueError("Checkpoint input/code identity changed")
        packet = saved["packet"]
        if hashlib.sha256(json.dumps(packet, sort_keys=True, allow_nan=False).encode()).hexdigest() != saved["packet_sha256"]:
            raise ValueError("Checkpoint payload hash changed")
        checkpoint_rows += 1
        arrays = packet["observations"].get("trial_arrays", {})
        checkpoint_trials += len(arrays.get("id", []))
        checkpoint_errors += sum(v == 0 for v in arrays.get("rewarded", []))
        ids.update((path.stem, i) for i in arrays.get("id", []))
    assets, trials, transitions = read_table(out, "asset_inventory"), read_table(out, "trial_inventory"), read_table(out, "post_error_transition_inventory")
    actual_ids = {(r["asset_id"], int(r["nwb_trial_id"])) for r in trials}
    if checkpoint_rows != len(assets) or checkpoint_trials != len(trials) or checkpoint_errors != len(transitions) or ids != actual_ids:
        raise ValueError("Independent header/trial/error accounting differs")
    if len(actual_ids) != len(trials) or len({r["asset_id"] for r in assets}) != len(assets):
        raise ValueError("Duplicate trial or asset IDs")
    with h5py.File(args.source_archive, "r") as handle:
        root = handle["SuperRat"]
        independent_source_trials = sum(mat_item(handle, root["trialdata"], i)["sniffstart"].size for i in range(root["name"].size))
        if root["name"].size != len(read_table(out, "source_inventory")) or independent_source_trials != len(read_table(out, "source_trial_inventory")):
            raise ValueError("Independent source accounting differs")
    listing = json.loads((args.dataset_root / "metadata/dandi_assets.json").read_text())["results"]
    if len(listing) != len(assets) or {a["asset_id"] for a in listing} != {a["asset_id"] for a in assets}:
        raise ValueError("Pinned asset set differs")
    if {r["animal"] for r in assets} != set(protocol["animals"]):
        raise ValueError("Not all eight animals inventoried")
    if any(r["eligible"] != "False" or r["cue_id"] for r in trials):
        raise ValueError("Unverified cues entered primary cohort")
    result = {"status": "verified_source_stopgate", "independent_trial_accounting_passed": True,
              "n_assets": len(assets), "n_trials": len(trials), "n_errors": len(transitions), "n_animals": len(protocol["animals"]),
              "ready_for_calibration": False, "association_fitted": False}
    atomic_json(out / (PREFIX + "verification.json"), result)
    return result


def report(args, protocol):
    out = args.output_dir
    decision = json.loads((out / (PREFIX + "decision.json")).read_text())
    verification_path = out / (PREFIX + "verification.json")
    verified = verification_path.exists() and json.loads(verification_path.read_text()).get("independent_trial_accounting_passed") is True
    assets = read_table(out, "asset_inventory")
    source = read_table(out, "source_inventory")
    if decision["source_prerequisites_passed"] or decision["ready_for_calibration"]:
        raise ValueError("Source-stopgate reporter cannot reinterpret a passing amended protocol")
    gates = [
        ("pinned_source_hash_verified", bool(json.loads((args.dataset_root / "metadata/source_verified.json").read_text())["published_digest_verified"])),
        ("all_eight_animals_inventoried", len({a["animal"] for a in assets}) == 8),
        ("all_asset_headers_readable", bool(assets) and all(a["status"] == "readable" for a in assets)),
        ("independent_stimulus_records_verified", False), ("source_conditions_and_apparatus_verified", False),
        ("source_nwb_chronology_and_unit_crosswalk_verified", False), ("novelty_prerequisite", False),
        ("five_decoder_qualified_animals", False), ("100_decoder_qualified_transitions", False),
        ("both_cue_conditions_within_animals", False), ("20_corrections_and_repeated_errors_per_condition", False),
        ("independent_accounting", verified), ("ready_for_calibration", False),
    ]
    write_table(out, "gate_summary", [{"gate": name, "passed": passed, "status": "pass" if passed else "fail",
                                       "interpretation": "source_stopgate_not_biological_test"} for name, passed in gates])
    lines = ["# Odor-place post-error feasibility", "", "Status: **inconclusive_feasibility**. Not a negative biological result.", "",
             f"Pinned inputs: DANDI {protocol['dandi_id']} version {protocol['dandi_version']}; Figshare {protocol['figshare_article']} version {protocol['figshare_version']}.",
             f"Inventory: {len(assets)} NWB assets across {len({a['animal'] for a in assets})} animals; {len(source)} source recording days across {len({a['animal'] for a in source})} animals.",
             f"Observed NWB trial rows: {decision['n_nwb_trials']}; annotated incorrect trials: {decision['n_nwb_errors']}.", "",
             "## Source stopgate", "",
             "Figshare version 1 supplies ClaireDataShort-20-Apr-2022.mat, not Figure1-6.zip. The latter belongs to version 3 and was not substituted or downloaded.",
             "NWB trial columns contain nose-poke bounds, outcome annotations and well times but no independently recorded cue identity. Incorrect-trial well times can be absent.",
             "The compiled source leftright10 field is preserved as an annotation, not promoted to a verified stimulus. The archived ParseClaireBehavior code can populate that field from different sources, including reward records. Its aggregator explicitly notes unknown odor identities for some unrewarded trials. Historical file paths are not the missing records themselves.",
             "Compiled sniff boundaries and NWB onset/offset boundaries remain distinct. Candidate original-day matches use source timestamps, never conversion dates. No offsets were fitted.",
             "The original eight-animal source inventory is incomplete under version 1; CS41 and CS42 are absent. Declared longTrack flags are not a substitute for apparatus/task reconciliation.", "",
             "## What was not run", "",
             "Full NWB acquisition, unit spike-timing verification, RUN decoding, ripple redetection and replay-behavior associations were blocked before expensive processing. Not-run ripple counts and decoder metrics are missing, never measured zeros.",
             "No trial was assigned a cue from choice/outcome. No error was relabeled as reward omission from missing timestamps. All incorrect rows remain explicitly unclassifiable for the prospective contrast.",
             "The frozen screening floors are at least five decoder-qualified animals and 100 post-error transitions, with both cue conditions within animals and at least 20 corrections and 20 repeated errors in each condition. These are not a power certificate.", "",
             "## Verification and next decision", "",
             f"Independent raw-header/source trial accounting: {'passed' if verified else 'not verified'}.",
             "Do not start sequence/statistical calibration. A separately reviewed protocol amendment would be required to use Figshare version 3, whose original per-animal trigger and task tables may resolve these source gates. This report does not establish that version 3 will pass.",
             "Prior-art comparisons are documented separately; exact-contrast novelty remains provisional until source feasibility and the available supplementary/code audit are complete.",
             "Denovellis stopgate and existing manuscript claims remain unchanged.", ""]
    (out / (PREFIX + "go_no_go.md")).write_text("\n".join(lines))
    diagnostic_figures(out, protocol)
    return {"status": decision["status"], "ready_for_calibration": False, "independently_verified": verified, "non_rescoring": True}


def diagnostic_figures(out, protocol):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    animals = read_table(out, "animal_summary")
    crosswalk = read_table(out, "trial_crosswalk")
    trials = read_table(out, "trial_inventory")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    names = [a["animal"] for a in animals]
    match = []
    missing = []
    for animal in names:
        rows = [r for r in crosswalk if r["animal"] == animal]
        errors = [r for r in trials if r["animal"] == animal and float(r["correct_choice_annotation"]) == 0]
        match.append(sum(r["clock_matched_within_5ms"] == "True" for r in rows) / len(rows) if rows else 0)
        missing.append(sum(not r["nwb_well_start_s"] for r in errors))
    axes[0].bar(names, match, color="#277c87")
    axes[0].set(ylabel="Fraction matched", ylim=(0, 1), title="Compiled source onset matches within 5 ms")
    axes[1].bar(names, missing, color="#be6256")
    axes[1].set(ylabel="Incorrect trial rows", title="NWB incorrect trials missing well onset")
    name = PREFIX + "source_reconciliation.png"
    fig.savefig(out / name, dpi=150)
    plt.close(fig)
    manifests = [{"figure": name, "selection_rule": "all_eight_animals_in_protocol_order", "interpretation": "source_quality_not_replay_biology"}]
    examples = json.loads((out / (PREFIX + "unresolved_examples.json")).read_text())
    if examples:
        fig, axes = plt.subplots(len(examples), 1, figsize=(11, 2.3 * len(examples)), constrained_layout=True, squeeze=False)
        for axis, example in zip(axes[:, 0], examples, strict=True):
            track = np.array(example["tracking"], dtype=float)
            annotations = example["trial_annotations"]
            if len(track):
                breaks = np.r_[False, (np.diff(track[:, 0]) > protocol["maximum_tracking_gap_s"]) | (np.diff(track[:, 5]) != 0)]
                xy = track[:, 1:3].copy()
                xy[breaks] = np.nan
                relative = track[:, 0] - annotations["sniffstart"]
                axis.plot(relative, xy[:, 0], label="source x", color="#277c87")
                axis.plot(relative, xy[:, 1], label="source y", color="#79549b")
            for field, color in [("sniffend", "#be6256"), ("rewardstart", "#5a8e45"), ("rewardend", "#5a8e45")]:
                if annotations.get(field) is not None:
                    axis.axvline(annotations[field] - annotations["sniffstart"], color=color, linestyle="--", alpha=.7, label=field)
            axis.set(title=f"{example['animal']} source day {example['source_day']}, error {example['source_trial_index']}: unresolved",
                     xlabel="Seconds from compiled source sniff onset", ylabel="Source coordinate (cm)")
            axis.legend(loc="upper right", fontsize=8, ncol=5)
        name = PREFIX + "unresolved_boundary_examples.png"
        fig.savefig(out / name, dpi=150)
        plt.close(fig)
        manifests.append({"figure": name, "selection_rule": "earliest_source_error_per_animal_not_outcome_effect_selected",
                          "interpretation": "unresolved_boundaries_not_eligible_repeat_or_changed_cue_examples"})
    write_table(out, "figure_manifest", manifests)


def dispatch(args, protocol):
    return {"inventory": inventory, "run-qc": blocked_run_qc, "verify": verify, "report": report}[args.stage](args, protocol)
