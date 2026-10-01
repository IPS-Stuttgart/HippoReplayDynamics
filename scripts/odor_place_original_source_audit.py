"""Original cue and wrong-well source audit. No neural scoring or association fit."""

from __future__ import annotations

from collections import Counter
import hashlib
import io
import json
from pathlib import PurePosixPath
import re
import zipfile

import numpy as np
from scipy.io import loadmat

from scripts._provenance import git_metadata
from scripts.odor_place_feasibility_core import clean, write_table, read_table
from scripts.odor_place_source_io import atomic_json, digest

PREFIX = "odor_place_post_error_"
ROLES = {
    "task": "task", "DIO": "dio", "odorTriggers": "odorTriggers",
    "nosepokeWindow": "nosepokeWindow", "pos": "pos", "runTrialBounds": "runTrialBounds",
    "runTrajBounds": "runTrajBounds", "rewards": "rewards", "rewardTimes": "rewardTimes",
}


def unpack(value):
    while isinstance(value, np.ndarray) and value.dtype == object and value.size == 1:
        value = value.ravel()[0]
    return value


def struct(value):
    value = unpack(value)
    if not hasattr(value, "_fieldnames"):
        raise ValueError("Expected actual MATLAB source struct")
    return {key: getattr(value, key) for key in value._fieldnames}


def text(value):
    return "".join(np.asarray(value).ravel().tolist())


def epochs(value, day):
    if not isinstance(value, np.ndarray) or value.dtype != object or value.ndim != 2 or value.shape[0] != 1:
        raise ValueError("Source day cells must retain MATLAB row-vector indexing")
    if day > value.shape[1] or value[0, day - 1].size == 0:
        return {}
    inner = value[0, day - 1]
    if inner.dtype != object or inner.ndim != 2 or inner.shape[0] != 1:
        raise ValueError("Source epoch cells must retain MATLAB row-vector indexing")
    return {i + 1: inner[0, i] for i in range(inner.shape[1]) if inner[0, i].size}


def intervals(times, states):
    times, states = np.asarray(times, float).ravel(), np.asarray(states, float).ravel()
    if len(times) != len(states) or not np.isfinite(times).all() or not np.isfinite(states).all():
        raise ValueError("Missing or nonfinite digital sensor records")
    if np.any(np.diff(times) <= 0) or not np.isin(states, [0, 1]).all():
        raise ValueError("Duplicate/nonchronological timestamps or nonbinary sensor state")
    if len(states) > 1 and np.any(np.diff(states) == 0):
        raise ValueError("Repeated sensor states cannot be silently collapsed")
    output = []
    for i in np.flatnonzero(states == 1):
        output.append((float(times[i]), float(times[i + 1]) if i + 1 < len(times) else None))
    return output


def channel(dio, number):
    if not isinstance(dio, np.ndarray) or dio.dtype != object or dio.size < number:
        raise ValueError("Missing original digital-input channel")
    record = struct(dio.ravel()[number - 1])
    return intervals(record["time"], record["state"])


def matching(values, target, tolerance):
    values = np.asarray(values, float).ravel()
    return np.flatnonzero(np.isfinite(values) & (np.abs(values - target) <= tolerance))


def infer_cue(start, end, odors, tolerance):
    hits = [(side, onset) for side, windows in odors.items() for onset, _ in windows
            if start - tolerance <= onset <= end + tolerance]
    if len(hits) != 1:
        return None, None, "missing_or_multiple_independent_odor_onsets"
    return hits[0][0], hits[0][1], ""


def geometry(task):
    values = np.asarray(task["linearcoord"], dtype=object).ravel()
    if len(values) != 2:
        raise ValueError("Full T-maze source geometry needs two explicit outbound branches")
    points = []
    for value in values:
        array = np.asarray(value, float)
        if array.ndim != 3 or array.shape[:2] != (3, 2) or not np.isfinite(array).all():
            raise ValueError("Unknown source linearcoord schema")
        plane = array[:, :, 0]
        if not np.allclose(array, plane[:, :, None], rtol=0, atol=1e-8):
            raise ValueError("Source geometry planes vary; no guessed collapse")
        points.append(plane)
    if not np.allclose(points[0][:2], points[1][:2], atol=5):
        raise ValueError("Source branches do not share the same start and junction")
    if np.linalg.norm(points[0][-1] - points[1][-1]) < 20:
        raise ValueError("Spatially unresolved alternative wells")
    # Source branch endpoints are verified against independent well-sensor tracking.
    return {"left": points[0][-1], "right": points[1][-1]}


def tracking(record):
    if text(record["fields"]).split() != ["time", "x", "y", "dir", "vel"]:
        raise ValueError("Tracking schema not source-verified")
    data = np.asarray(record["data"], float)
    if data.ndim != 2 or data.shape[1] != 5 or len(data) < 2 or not np.isfinite(data[:, 0]).all() or np.any(np.diff(data[:, 0]) <= 0):
        raise ValueError("Invalid tracking timestamps")
    return data


def pause_window(data, center, sample_stop, well_onset, next_cue, protocol):
    t, xy, speed = data[:, 0], data[:, 1:3], data[:, 4]
    valid = np.isfinite(xy).all(axis=1) & np.isfinite(speed) & (speed >= 0)
    inside = valid & (np.linalg.norm(xy - center, axis=1) <= protocol["well_radius_cm"])
    right = np.searchsorted(t, well_onset)
    if right == 0 or right == len(t) or t[right] - t[right - 1] > protocol["maximum_tracking_gap_s"]:
        return {"usable_exposure_s": None, "pause_exclusion_reason": "well_sensor_tracking_gap"}
    weight = (well_onset - t[right - 1]) / (t[right] - t[right - 1])
    represented = (1 - weight) * xy[right - 1] + weight * xy[right]
    if not np.isfinite(represented).all() or np.linalg.norm(represented - center) > protocol["well_radius_cm"]:
        return {"usable_exposure_s": None, "pause_exclusion_reason": "well_sensor_not_at_source_graph_endpoint"}
    anchor = right if inside[right] else right - 1
    first = anchor
    while first > 0 and inside[first - 1] and t[first] - t[first - 1] <= protocol["maximum_tracking_gap_s"] and t[first - 1] > sample_stop:
        first -= 1
    if first == 0 or t[first] <= sample_stop or inside[first - 1]:
        return {"usable_exposure_s": None, "pause_exclusion_reason": "arrival_not_independently_observed"}
    if np.any(inside & (t > sample_stop) & (t < t[first])):
        return {"usable_exposure_s": None, "pause_exclusion_reason": "earlier_wrong_well_visit_before_sensor_choice"}
    departure = first
    while departure + 1 < len(t) and inside[departure + 1] and t[departure + 1] - t[departure] <= protocol["maximum_tracking_gap_s"]:
        departure += 1
    end_cap = min(t[first] + protocol["maximum_pause_s"], next_cue)
    reason = ""
    if departure + 1 == len(t):
        reason = "tracking_or_epoch_ends_before_departure"
    elif t[departure + 1] - t[departure] > protocol["maximum_tracking_gap_s"]:
        reason = "tracking_gap_before_departure"
    elif not valid[departure + 1]:
        reason = "invalid_tracking_before_departure"
    stop = min(end_cap, t[departure + 1] if departure + 1 < len(t) else t[departure])
    if stop <= t[first]:
        return {"usable_exposure_s": None, "pause_exclusion_reason": "no_well_exposure_before_next_cue"}
    exposure = 0.0
    segments = []
    for i in range(first, departure + 1):
        if i + 1 >= len(t):
            break
        a, b = max(t[first], t[i]), min(stop, t[i + 1])
        if b <= a:
            continue
        if t[i + 1] - t[i] <= protocol["maximum_tracking_gap_s"] and inside[i] and inside[i + 1] and max(speed[i], speed[i + 1]) <= protocol["immobility_max_speed_cm_s"]:
            exposure += b - a
            segments.append([float(a), float(b)])
    if end_cap <= t[departure]:
        reason = ""  # The cap is fully observed, even if a later departure is missing.
    if not reason and exposure < protocol["minimum_pause_exposure_s"]:
        reason = "insufficient_immobile_wrong_well_exposure"
    return {"pause_start_s": float(t[first]), "pause_stop_s": float(stop), "usable_exposure_s": float(exposure),
            "immobile_segments": segments, "pause_exclusion_reason": reason}


def annotated_error(record, start, tolerance):
    correct = len(matching(record["correctTriggers"], start, tolerance))
    incorrect = len(matching(record["incorrectTriggers"], start, tolerance))
    if correct + incorrect != 1:
        return None
    return bool(incorrect)


def empty_trial(index, start, end, annotation=None):
    return {"source_sample_index": index, "nosepoke_start_s": float(start), "nosepoke_stop_s": end,
            "cue": None, "chosen_arm": None, "task_correct": None, "source_error_annotation": annotation,
            "cue_verified": False, "trial_verified": False, "exclusion_reason": "",
            "well_onset_s": None, "well_offset_s": None, "pump_onset_s": None,
            "usable_exposure_s": None, "pause_exclusion_reason": "not_analyzed", "immobile_segments": []}


def unresolved_source_trials(record, reason):
    starts = np.asarray(record["allTriggers"], float).ravel()
    rows = []
    for i, start in enumerate(starts):
        row = empty_trial(-i - 1, start, None, annotated_error(record, start, 0.005))
        row["exclusion_reason"] = reason
        rows.append(row)
    return rows


def reconstruct_epoch(dio, odor_record, windows, data, centers, protocol):
    tol, channels = protocol["clock_match_tolerance_s"], protocol["dio_channels_one_based"]
    nose = channel(dio, channels["nose_poke"])
    odors = {side: channel(dio, channels[side + "_odor"]) for side in ["left", "right"]}
    wells = {side: channel(dio, channels[side + "_well"]) for side in ["left", "right"]}
    pumps = {side: channel(dio, channels[side + "_pump"]) for side in ["left", "right"]}
    source_triggers = np.asarray(odor_record["allTriggers"], float).ravel()
    if len(np.unique(source_triggers)) != len(source_triggers) or np.any(np.diff(source_triggers) <= 0):
        raise ValueError("Duplicate or nonchronological original trial triggers")
    windows = np.asarray(windows, float)
    if windows.ndim != 2 or windows.shape[1] != 2:
        raise ValueError("Unknown original nosepokeWindow schema")
    rows = []
    for index, (start, end) in enumerate(nose):
        row = empty_trial(index, start, end, annotated_error(odor_record, start, tol))
        if end is None:
            row["exclusion_reason"] = "missing_nosepoke_offset"
        elif end - start < protocol["minimum_odor_sample_s"]:
            row["exclusion_reason"] = "premature_odor_sample"
        else:
            cue, onset, failure = infer_cue(start, end, odors, tol)
            row.update(cue=cue, odor_onset_s=onset, exclusion_reason=failure)
            trigger_matches = matching(source_triggers, start, tol)
            bound_matches = np.flatnonzero(np.abs(windows[:, 0] - start) <= tol)
            if not failure and (len(trigger_matches) != 1 or len(bound_matches) != 1):
                row["exclusion_reason"] = "raw_nosepoke_not_uniquely_matched_to_original_trigger_and_window"
            elif not failure:
                bound = windows[bound_matches[0]]
                row["source_nosepoke_offset_difference_s"] = float(bound[1] - end)
                annotated_side = [side for side in ["left", "right"] if len(matching(odor_record[side + "Triggers"], start, tol)) == 1]
                annotations = [flag for flag, key in [(True, "correctTriggers"), (False, "incorrectTriggers")]
                               if len(matching(odor_record[key], start, tol)) == 1]
                if annotated_side != [cue] or len(annotations) != 1 or abs(bound[1] - end) > tol:
                    row["exclusion_reason"] = "conflicting_independent_cue_or_sample_annotations"
                else:
                    row["cue_verified"] = True
                    row["source_error_annotation"] = not annotations[0]
                    next_start = nose[index + 1][0] if index + 1 < len(nose) else float(data[-1, 0])
                    choices = sorted((on, side, off) for side, visits in wells.items() for on, off in visits if end <= on < next_start)
                    if not choices or (len(choices) > 1 and abs(choices[0][0] - choices[1][0]) <= tol):
                        row["exclusion_reason"] = "missing_or_ambiguous_first_well_choice"
                    else:
                        on, chosen, off = choices[0]
                        row.update(chosen_arm=chosen, well_onset_s=on, well_offset_s=off, task_correct=chosen == cue)
                        subsequent_pumps = [p for p, _ in pumps[chosen] if on - tol <= p < next_start]
                        row["pump_onset_s"] = subsequent_pumps[0] if subsequent_pumps else None
                        if row["task_correct"] != annotations[0]:
                            row["exclusion_reason"] = "raw_first_choice_conflicts_with_source_outcome_annotation"
                        else:
                            row["trial_verified"] = True
                            if chosen != cue and centers is not None:
                                row.update(pause_window(data, centers[chosen], end, on, next_start, protocol))
                            elif chosen != cue:
                                row["pause_exclusion_reason"] = "source_graph_geometry_unresolved"
        rows.append(row)
    # Keep stored trials even when no raw sample matches; otherwise unresolved errors disappear.
    for index, start in enumerate(source_triggers):
        if len(matching([s for s, _ in nose], start, tol)) != 1:
            row = empty_trial(-index - 1, start, None, annotated_error(odor_record, start, tol))
            row["exclusion_reason"] = "original_trigger_without_unique_raw_nosepoke"
            rows.append(row)
    return rows


def associate_nwb(rows, headers, tolerance):
    target = []
    for asset, observed in headers.items():
        animal = observed["subject"].removeprefix("Symanski-")
        arrays = observed["trial_arrays"]
        target.extend((animal, float(t), asset, int(identifier), bool(correct)) for t, identifier, correct
                      in zip(arrays["start_time"], arrays["id"], arrays["rewarded"], strict=True))
    by_animal = {animal: [entry for entry in target if entry[0] == animal] for animal in {r["animal"] for r in rows}}
    counts = Counter()
    for row in rows:
        matches = [entry for entry in by_animal[row["animal"]] if abs(entry[1] - row["nosepoke_start_s"]) <= tolerance]
        row.update(nwb_asset_id=None, nwb_trial_id=None, nwb_clock_status="unmatched", nwb_outcome_agrees=None)
        if len(matches) == 1:
            _, time, asset, identifier, correct = matches[0]
            row.update(nwb_asset_id=asset, nwb_trial_id=identifier, nwb_onset_difference_s=time - row["nosepoke_start_s"],
                       nwb_clock_status="unique_within_5ms", nwb_outcome_agrees=correct == row["task_correct"] if row["trial_verified"] else None)
            counts[(asset, identifier)] += 1
        elif matches:
            row["nwb_clock_status"] = "ambiguous_across_assets"
    for row in rows:
        if row["nwb_asset_id"] is not None and counts[(row["nwb_asset_id"], row["nwb_trial_id"])] > 1:
            row["nwb_clock_status"] = "ambiguous_across_source_samples"
    return target


def transitions(rows, protocol):
    output = []
    groups = {}
    for row in rows:
        groups.setdefault((row["animal"], row["source_day"], row["source_epoch"]), []).append(row)
    for key, samples in groups.items():
        samples.sort(key=lambda r: r["nosepoke_start_s"])
        for i, row in enumerate(samples):
            if row["source_error_annotation"] is not True:
                continue
            nxt = samples[i + 1] if i + 1 < len(samples) else None
            failure = row["exclusion_reason"] or row["pause_exclusion_reason"]
            if not row["primary_condition"]:
                failure = "not_verified_standard_full_maze_condition"
            elif not row["trial_verified"]:
                failure = failure or "unresolved_error_trial"
            elif row["nwb_clock_status"] != "unique_within_5ms" or row["nwb_outcome_agrees"] is not True:
                failure = "source_error_not_uniquely_reconciled_to_pinned_nwb"
            elif nxt is None:
                failure = "no_next_trial_before_epoch_end"
            elif not nxt["trial_verified"] or not nxt["cue_verified"]:
                failure = "intervening_unresolved_or_premature_sample"
            elif nxt["nwb_clock_status"] != "unique_within_5ms" or nxt["nwb_asset_id"] != row["nwb_asset_id"] or nxt["nwb_outcome_agrees"] is not True:
                failure = "next_trial_unresolved_or_different_nwb_file"
            if not failure and (row["usable_exposure_s"] is None or row["usable_exposure_s"] < protocol["minimum_pause_exposure_s"]):
                failure = "insufficient_immobile_wrong_well_exposure"
            condition = "repeat" if nxt and nxt["cue_verified"] and row["cue_verified"] and nxt["cue"] == row["cue"] else "changed" if nxt and nxt["cue_verified"] and row["cue_verified"] else "unclassifiable"
            outcome = None
            if nxt and nxt["trial_verified"] and row["cue_verified"]:
                outcome = "correction_to_prior_cue_arm" if nxt["chosen_arm"] == row["cue"] else "repeated_mistaken_arm"
            output.append({"animal": key[0], "source_day": key[1], "source_epoch": key[2], "error_sample_index": row["source_sample_index"],
                           "asset_id": row["nwb_asset_id"], "error_trial_id": row["nwb_trial_id"], "cue": row["cue"], "mistaken_arm": row["chosen_arm"],
                           "next_sample_index": nxt["source_sample_index"] if nxt else None, "next_cue": nxt["cue"] if nxt else None,
                           "next_chosen_arm": nxt["chosen_arm"] if nxt else None, "next_task_correct": nxt["task_correct"] if nxt else None,
                           "cue_condition": condition, "outcome_relative_to_prior_error_cue": outcome,
                           "pause_start_s": row.get("pause_start_s"), "pause_stop_s": row.get("pause_stop_s"), "usable_exposure_s": row["usable_exposure_s"],
                           "eligible_source_transition": not failure, "exclusion_reason": failure,
                           "decoder_qualified": False, "sequence_testing_opportunities": None})
    return output


def coverage_passed(animal_rows, eligible, cells, screen):
    animals = [r["animal"] for r in animal_rows if r["source_full_maze"] and r["both_conditions"]]
    outcomes = {"correction_to_prior_cue_arm": screen["minimum_corrections_per_cue_condition"],
                "repeated_mistaken_arm": screen["minimum_repeated_errors_per_cue_condition"]}
    passed = (len(animals) >= screen["minimum_animals"] and len(eligible) >= screen["minimum_post_error_transitions"]
              and len(cells) == 4 and all(r["n_source_transitions"] >= outcomes[r["outcome_relative_to_prior_error_cue"]] for r in cells))
    return passed, animals


def inventory(args, protocol):
    out = args.output_dir
    existing = out / (PREFIX + "inventory_identity.json")
    if existing.exists():
        saved = json.loads(existing.read_text())
        if saved["code_commit"] != git_metadata()["code_commit"]:
            raise ValueError("Resume requires the original committed producer; use a new output directory for changed code")
        verify(args, protocol)
        return json.loads((out / (PREFIX + "decision.json")).read_text())
    source_identity = json.loads((args.dataset_root / "metadata/source_verified.json").read_text())
    if digest(args.source_archive) != source_identity["sha256"] or source_identity["published_digest"] != protocol["source_published_md5"]:
        raise ValueError("Amended source acquisition identity changed")
    ref = args.reference_inventory / (PREFIX + "inventory_identity.json")
    if digest(ref) != protocol["v1_reference_inventory_sha256"]:
        raise ValueError("Immutable reference inventory differs")
    headers = {}
    for path in (args.reference_inventory / "checkpoints").glob("*.json"):
        saved = json.loads(path.read_text())
        packet = saved["packet"]
        if hashlib.sha256(json.dumps(packet, sort_keys=True, allow_nan=False).encode()).hexdigest() != saved["packet_sha256"] or packet["status"] != "readable":
            raise ValueError("Reference header checkpoint changed or is unreadable")
        headers[path.stem] = packet["observations"]
    if len(headers) != 38:
        raise ValueError("Pinned header cohort changed")
    archive_rows, records = [], {}
    with zipfile.ZipFile(args.source_archive) as archive:
        names = [entry.filename for entry in archive.infolist()]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate archive member names")
        for entry in archive.infolist():
            path = PurePosixPath(entry.filename)
            if path.is_absolute() or ".." in path.parts or entry.flag_bits & 1:
                raise ValueError("Unsafe or encrypted source member")
            row = {"member": entry.filename, "uncompressed_bytes": entry.file_size, "compressed_bytes": entry.compress_size,
                   "crc32": f"{entry.CRC:08x}", "selected": False, "member_sha256": None, "read_status": "not_selected", "failure_reason": ""}
            match = re.fullmatch(r"(CS\d+)(task|DIO|odorTriggers|nosepokeWindow|pos|runTrialBounds|runTrajBounds|rewards|rewardTimes)(\d+)\.mat", path.name)
            if match:
                animal, role, day_string = match.groups()
                day = int(day_string)
                if animal not in protocol["animals"] or entry.file_size > protocol["source_read_budget_bytes_per_member"]:
                    raise ValueError("Unexpected animal or source member exceeds frozen read budget")
                row["selected"] = True
                try:
                    content = archive.read(entry)
                    row["member_sha256"] = hashlib.sha256(content).hexdigest()
                    loaded = loadmat(io.BytesIO(content), struct_as_record=False, squeeze_me=False, variable_names=[ROLES[role]])
                    if ROLES[role] not in loaded:
                        raise ValueError("Selected member has no documented source variable")
                    key = (animal, day, role)
                    if key in records:
                        raise ValueError("Duplicate animal/day/role records")
                    records[key] = epochs(loaded[ROLES[role]], day)
                    row["read_status"] = "readable_crc_verified"
                except (ValueError, OSError, zipfile.BadZipFile, NotImplementedError) as exc:
                    row.update(read_status="unresolved", failure_reason=f"{type(exc).__name__}: {exc}")
            archive_rows.append(row)
            if row["selected"]:
                print(f"source_member {path.name}: {row['read_status']}", flush=True)
    write_table(out, "source_archive_inventory", archive_rows)
    trials, session_rows, packets, boundaries, track_records = [], [], [], [], {}
    for animal, day in sorted({(a, d) for a, d, role in records if role == "task"}):
        for epoch, value in records[(animal, day, "task")].items():
            task = struct(value)
            kind, environment = text(task.get("type", np.array([""]))), text(task.get("environment", np.array([""])))
            primary = animal in protocol["full_maze_animals_source_code"] and kind == "run" and environment == "odorplace"
            row = {"animal": animal, "source_day": day, "source_epoch": epoch, "task_type": kind, "environment": environment,
                   "apparatus_evidence": protocol["full_maze_identity_evidence"], "source_full_maze_animal": animal in protocol["full_maze_animals_source_code"],
                   "primary_condition": primary, "n_raw_samples": None, "n_verified_trials": None, "n_verified_errors": None,
                   "status": "not_standard_odor_run", "failure_reason": "", "geometry_verified": False}
            if kind == "run" and environment == "odorplace":
                reconstructed = []
                try:
                    required = {role: records[(animal, day, role)][epoch] for role in ["DIO", "odorTriggers", "nosepokeWindow", "pos"]}
                    data = tracking(struct(required["pos"]))
                    centers = geometry(task) if primary else None
                    row["geometry_verified"] = centers is not None
                    reconstructed = reconstruct_epoch(required["DIO"], struct(required["odorTriggers"]), required["nosepokeWindow"], data, centers, protocol)
                    track_records[(animal, day, epoch)] = (data, centers)
                    row.update(n_raw_samples=len(reconstructed), n_verified_trials=sum(t["trial_verified"] for t in reconstructed),
                               n_verified_errors=sum(t["trial_verified"] and t["task_correct"] is False for t in reconstructed), status="source_reconstructed")
                except (KeyError, ValueError, TypeError) as exc:
                    row.update(status="unresolved", failure_reason=f"{type(exc).__name__}: {exc}")
                    original = records.get((animal, day, "odorTriggers"), {}).get(epoch)
                    if original is not None:
                        reconstructed = unresolved_source_trials(struct(original), "unresolved_epoch: " + row["failure_reason"])
                for trial in reconstructed:
                    trial.update(animal=animal, source_day=day, source_epoch=epoch, primary_condition=primary)
                trials.extend(reconstructed)
                packets.append({"animal": animal, "day": day, "epoch": epoch, "source_sensor_trials": reconstructed})
                for role in ["runTrialBounds", "runTrajBounds"]:
                    value = records.get((animal, day, role), {}).get(epoch)
                    if value is not None:
                        try:
                            compiled = struct(value)
                            payload = {k: clean(v) for k, v in compiled.items() if k in ["data", "fields"]}
                        except ValueError:
                            payload = {"data": clean(value)}
                        boundaries.append({"animal": animal, "source_day": day, "source_epoch": epoch, "boundary_type": role,
                                           "original_values": payload, "interpretation": "separate_source_definition_no_fitted_clock_offset"})
                print(f"source_epoch {animal} day={day} epoch={epoch}: {row['status']} samples={len(reconstructed)}", flush=True)
            session_rows.append(row)
    target = associate_nwb(trials, headers, protocol["clock_match_tolerance_s"])
    transition_rows = transitions(trials, protocol)
    write_table(out, "source_session_inventory", session_rows)
    write_table(out, "source_trial_inventory", trials, None if trials else ["animal", "source_day", "source_epoch", "source_sample_index"])
    write_table(out, "post_error_transition_inventory", transition_rows, None if transition_rows else ["animal", "eligible_source_transition", "exclusion_reason"])
    write_table(out, "source_boundary_inventory", boundaries, None if boundaries else ["animal", "boundary_type", "original_values"])
    atomic_json(out / (PREFIX + "source_sensor_checkpoints.json"), clean(packets))
    animal_rows = []
    for animal in protocol["animals"]:
        eligible = [r for r in transition_rows if r["animal"] == animal and r["eligible_source_transition"]]
        conditions = Counter(r["cue_condition"] for r in eligible)
        animal_rows.append({"animal": animal, "source_full_maze": animal in protocol["full_maze_animals_source_code"],
                            "n_source_days": len({r["source_day"] for r in session_rows if r["animal"] == animal}),
                            "n_source_run_epochs": sum(r["animal"] == animal and r["environment"] == "odorplace" for r in session_rows),
                            "n_verified_trials": sum(r["animal"] == animal and r["trial_verified"] for r in trials),
                            "n_source_errors": sum(r["animal"] == animal for r in transition_rows), "n_eligible_source_transitions": len(eligible),
                            "n_repeat_cue": conditions["repeat"], "n_changed_cue": conditions["changed"],
                            "both_conditions": conditions["repeat"] > 0 and conditions["changed"] > 0,
                            "n_decoder_qualified_transitions": None, "n_supported_ripple_opportunities": None})
    write_table(out, "animal_summary", animal_rows)
    cells = []
    for condition in ["repeat", "changed"]:
        for outcome in ["correction_to_prior_cue_arm", "repeated_mistaken_arm"]:
            subset = [r for r in transition_rows if r["eligible_source_transition"] and r["cue_condition"] == condition and r["outcome_relative_to_prior_error_cue"] == outcome]
            cells.append({"cue_condition": condition, "outcome_relative_to_prior_error_cue": outcome, "n_source_transitions": len(subset),
                          "animals": sorted({r["animal"] for r in subset}), "interpretation": "coverage_only_not_content_association"})
    write_table(out, "source_coverage", cells)
    screen = protocol["screening"]
    eligible = [r for r in transition_rows if r["eligible_source_transition"]]
    passed, covered_animals = coverage_passed(animal_rows, eligible, cells, screen)
    decision = {"status": "source_ready_for_neural_feasibility" if passed else "inconclusive_source_feasibility",
                "source_screen_passed": passed, "ready_for_calibration": False, "n_source_trials": len(trials), "n_source_errors": len(transition_rows),
                "n_eligible_source_transitions": len(eligible), "animals_with_both_conditions": covered_animals,
                "n_pinned_nwb_trials": len(target), "n_nwb_uniquely_reconciled": sum(r["nwb_clock_status"] == "unique_within_5ms" for r in trials),
                "source_release_sha256": source_identity["sha256"], "association_fitted": False, "neural_processing_performed": False,
                "v1_stopgate_preserved": True, "manuscript_claims_changed": False,
                "recommended_next_action": "bounded_run_decoder_and_ripple_feasibility" if passed else "stop_primary_post_error_cue_branch"}
    atomic_json(out / (PREFIX + "decision.json"), decision)
    make_figures(out, transition_rows, track_records, protocol)
    identity = {"protocol_sha256": digest(args.protocol), "source_sha256": source_identity["sha256"], "reference_inventory_sha256": digest(ref),
                "code_commit": git_metadata()["code_commit"], "output_sha256": {p.name: digest(p) for p in out.glob(PREFIX + "*.csv")},
                "sensor_checkpoint_sha256": digest(out / (PREFIX + "source_sensor_checkpoints.json")),
                "decision_sha256": digest(out / (PREFIX + "decision.json")),
                "reference_checkpoint_sha256": {p.name: digest(p) for p in (args.reference_inventory / "checkpoints").glob("*.json")}}
    atomic_json(out / (PREFIX + "inventory_identity.json"), identity)
    return decision


def make_figures(out, rows, tracks, protocol):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    selected = []
    for animal in protocol["animals"]:
        for condition in ["repeat", "changed"]:
            subset = [r for r in rows if r["animal"] == animal and r["eligible_source_transition"] and r["cue_condition"] == condition]
            if subset:
                selected.append((min(subset, key=lambda r: (r["source_day"], r["source_epoch"], r["error_sample_index"])), "earliest_eligible_" + condition))
        unresolved = [r for r in rows if r["animal"] == animal and not r["eligible_source_transition"] and r["pause_start_s"] is not None]
        if unresolved:
            selected.append((min(unresolved, key=lambda r: (r["source_day"], r["source_epoch"], r["error_sample_index"])), "earliest_unresolved_with_tracking_pause"))
    manifest = []
    for row, rule in selected:
        data, centers = tracks[(row["animal"], row["source_day"], row["source_epoch"])]
        if centers is None:
            continue
        start, stop = row["pause_start_s"], row["pause_stop_s"]
        section = data[(data[:, 0] >= start - 1) & (data[:, 0] <= stop + 1)]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(data[::10, 1], data[::10, 2], color="0.85", linewidth=0.5)
        axes[0].plot(section[:, 1], section[:, 2], color="#007f75")
        for side, center in centers.items():
            axes[0].add_patch(plt.Circle(center, protocol["well_radius_cm"], fill=False, color="0.3"))
            axes[0].annotate(side, center)
        axes[0].set(xlabel="Source x (cm)", ylabel="Source y (cm)", aspect="equal")
        axes[1].plot(section[:, 0] - start, section[:, 4], color="#007f75")
        axes[1].axhline(protocol["immobility_max_speed_cm_s"], color="0.4", linestyle=":")
        axes[1].axvspan(0, stop - start, alpha=0.12, color="#007f75")
        axes[1].set(xlabel="Time from wrong-well arrival (s)", ylabel="Source speed (cm/s)")
        fig.suptitle(f"{row['animal']} day {row['source_day']} epoch {row['source_epoch']} sample {row['error_sample_index']}\n"
                     f"cue={row['cue']}; choice={row['mistaken_arm']}; next cue={row['next_cue']}; "
                     f"exposure={row['usable_exposure_s']:.2f}s")
        fig.tight_layout()
        name = f"source_pause_{row['animal']}_{row['source_day']}_{row['source_epoch']}_{row['error_sample_index']}.png"
        fig.savefig(out / name, dpi=150)
        plt.close(fig)
        manifest.append({"figure": name, "selection_rule": rule, "animal": row["animal"], "eligible": row["eligible_source_transition"],
                         "exclusion_reason": row["exclusion_reason"], "scope": "behavioral_source_check_not_replay"})
    write_table(out, "figure_manifest", manifest, None if manifest else ["figure", "selection_rule", "scope"])


def verify(args, protocol):
    out = args.output_dir
    identity = json.loads((out / (PREFIX + "inventory_identity.json")).read_text())
    if digest(args.protocol) != identity["protocol_sha256"] or digest(args.source_archive) != identity["source_sha256"]:
        raise ValueError("Amended audit input identity differs")
    for name, expected in identity["output_sha256"].items():
        if digest(out / name) != expected:
            raise ValueError("Source audit output differs: " + name)
    if digest(out / (PREFIX + "decision.json")) != identity["decision_sha256"]:
        raise ValueError("Source decision differs")
    for name, expected in identity["reference_checkpoint_sha256"].items():
        if digest(args.reference_inventory / "checkpoints" / name) != expected:
            raise ValueError("Reference header checkpoint differs")
    path = out / (PREFIX + "source_sensor_checkpoints.json")
    if digest(path) != identity["sensor_checkpoint_sha256"]:
        raise ValueError("Sensor checkpoint differs")
    packets = json.loads(path.read_text())
    independent = [r for p in packets for r in p["source_sensor_trials"]]
    keys = {(r["animal"], r["source_day"], r["source_epoch"], r["source_sample_index"]) for r in independent}
    csv_rows = read_table(out, "source_trial_inventory")
    actual = {(r["animal"], int(r["source_day"]), int(r["source_epoch"]), int(r["source_sample_index"])) for r in csv_rows}
    if keys != actual or len(keys) != len(independent) or len(actual) != len(csv_rows):
        raise ValueError("Independent source trial accounting differs")
    computed = transitions(independent, protocol)
    table = read_table(out, "post_error_transition_inventory")
    if len(computed) != len(table) or sum(r["eligible_source_transition"] for r in computed) != sum(r["eligible_source_transition"] == "True" for r in table):
        raise ValueError("Independent transition accounting differs")
    fields = ["animal", "source_day", "source_epoch", "error_sample_index", "cue_condition",
              "outcome_relative_to_prior_error_cue", "eligible_source_transition", "exclusion_reason"]
    expected = [tuple("" if r[k] is None else str(r[k]) for k in fields) for r in computed]
    observed = [tuple(r[k] for k in fields) for r in table]
    if expected != observed:
        raise ValueError("Independent transition labels differ")
    result = {"status": "verified_source_accounting", "n_source_trials": len(actual), "n_source_errors": len(computed),
              "n_eligible_source_transitions": sum(r["eligible_source_transition"] for r in computed),
              "scope": "independent saved sensor-to-trial accounting; not a second tracking reconstruction or neural validation",
              "ready_for_calibration": False, "association_fitted": False}
    atomic_json(out / (PREFIX + "verification.json"), result)
    return result


def report(args, protocol):
    out = args.output_dir
    decision = json.loads((out / (PREFIX + "decision.json")).read_text())
    verification = json.loads((out / (PREFIX + "verification.json")).read_text())
    if verification["n_eligible_source_transitions"] != decision["n_eligible_source_transitions"]:
        raise ValueError("Source verification and decision disagree")
    counts = Counter(r["exclusion_reason"] for r in read_table(out, "post_error_transition_inventory") if r["eligible_source_transition"] == "False")
    write_table(out, "exclusion_summary", [{"reason": key, "n_source_errors": value} for key, value in sorted(counts.items())], ["reason", "n_source_errors"])
    animals = read_table(out, "animal_summary")
    coverage = read_table(out, "source_coverage")
    gates = [
        ("published_source_checksum_verified", True), ("all_eight_animals_source_inventoried", len({r["animal"] for r in animals if int(r["n_source_days"])}) == 8),
        ("five_full_maze_animals_with_both_conditions", len(decision["animals_with_both_conditions"]) >= protocol["screening"]["minimum_animals"]),
        ("100_source_eligible_transitions", decision["n_eligible_source_transitions"] >= protocol["screening"]["minimum_post_error_transitions"]),
        ("20_each_outcome_each_cue_condition", all(int(r["n_source_transitions"]) >= protocol["screening"]["minimum_corrections_per_cue_condition" if r["outcome_relative_to_prior_error_cue"] == "correction_to_prior_cue_arm" else "minimum_repeated_errors_per_cue_condition"] for r in coverage)),
        ("independent_source_accounting", verification["status"] == "verified_source_accounting"),
        ("source_screen", decision["source_screen_passed"]), ("ready_for_calibration", False),
    ]
    write_table(out, "gate_summary", [{"gate": name, "passed": passed, "status": "pass" if passed else "fail", "scope": "source_feasibility_only"} for name, passed in gates])
    lines = ["# Odor-place original-source audit", "", f"Status: **{decision['status']}**.",
             "No neural processing or replay-content association. Version-1 stopgate and manuscript claims unchanged.", "",
             f"Figshare 19620783 version 3; DANDI 001539 version {protocol['dandi_version']}. Original ZIP was checksum-verified; no full NWB downloaded or entire archive extracted.",
             f"Source samples inventoried: {decision['n_source_trials']}; source-annotated errors: {decision['n_source_errors']}; source-eligible transitions: {decision['n_eligible_source_transitions']}.",
             f"Full-maze animals with both cue conditions: {', '.join(decision['animals_with_both_conditions']) or 'none'}.", "",
             "Independent cue readout uses odor-solenoid digital channels, separately from well choice and outcome. Raw nose-poke windows must uniquely match stored triggers and windows within 5 ms. No offsets fitted.",
             "First well choice uses digital well sensors; wrong-well pauses use source graph endpoints and tracking. <=4 cm/s, 10 cm radius, >=0.5 s exposure, 10 s cap; no bridging >100 ms tracking gaps or epochs.",
             "Full-maze animal identity follows the published five-animal spatial cohort and archived author spatial-analysis code; session condition uses source task records, not filenames.",
             "Aborted/unresolved samples between trials are not skipped. Next choice is evaluated relative to the preceding error cue; next-trial task correctness is stored separately, particularly when cues change.",
             "Missing pump events are not interpreted as reward omission. Original processed position may already contain interpolation; the gap rule operates on the supplied timestamp series.", "",
             "## Coverage (not associations)", "", "| Cue condition | Outcome relative to prior cue | Source transitions |", "| --- | --- | --- |"]
    lines.extend(f"| {r['cue_condition']} | {r['outcome_relative_to_prior_error_cue']} | {r['n_source_transitions']} |" for r in coverage)
    lines.extend(["", "## Exclusions", ""] + [f"- {key}: {value}" for key, value in sorted(counts.items())])
    lines.extend(["", "## Next decision", "", f"Recommended: `{decision['recommended_next_action']}`.",
                  "A source pass would allow bounded RUN/ripple feasibility only, not sequence calibration or a biological claim. CA1 spike identity, RUN decoding and supported ripple opportunities remain unmeasured. Novelty remains provisional.", ""])
    (out / (PREFIX + "go_no_go.md")).write_text("\n".join(lines))
    return {**decision, "non_rescoring": True, "independently_accounted": True}


def dispatch(args, protocol):
    if args.stage == "run-qc":
        raise ValueError("This amended delivery is source-only; neural processing requires a separately reviewed stage")
    return {"inventory": inventory, "verify": verify, "report": report}[args.stage](args, protocol)
