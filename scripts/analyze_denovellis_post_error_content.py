#!/usr/bin/env python3
"""Gated Denovellis post-error route-content study; report never rescores."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import importlib.metadata
import json
import platform
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.io import loadmat

try:
    from scripts._provenance import build_script_provenance, file_sha256
    from scripts.denovellis_post_error_core import build_transitions, contains_event, day_epochs, field, reconstruct_visits, score_visits, valid_intervals
except ModuleNotFoundError:
    from _provenance import build_script_provenance, file_sha256
    from denovellis_post_error_core import build_transitions, contains_event, day_epochs, field, reconstruct_visits, score_visits, valid_intervals

ROOT = Path(__file__).resolve().parents[1]
ANIMALS = dict(bon="Bond", cha="Chapati", con="Conley", cor="Corriander", dav="Dave", dud="Dudley", egy="Egypt", fra="Frank", gov="Government", remy="Remy")
PREFIX = "denovellis_post_error_"
PRIOR_ART = [
    {"study": "Denovellis 2021", "doi": "10.7554/eLife.64505", "source": "https://elifesciences.org/articles/64505", "comparison": "Dynamics and conventional sequence detectors, not post-error behavioral correction", "exact_contrast_established": False},
    {"study": "Shin 2019", "doi": "10.1016/j.neuron.2019.09.012", "source": "https://doi.org/10.1016/j.neuron.2019.09.012", "comparison": "Past/future paths and incorrect upcoming choices; Figs. 3,5,S6, replay prediction methods, not conditioning on a preceding outbound error", "exact_contrast_established": False},
    {"study": "Gillespie 2021", "doi": "10.1016/j.neuron.2021.07.029", "source": "https://doi.org/10.1016/j.neuron.2021.07.029", "comparison": "Replay-arm content predicts correct/error repeat-phase performance in a changing-goal task, not the specified post-error W-track contrast", "exact_contrast_established": False},
]


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False, default=_json_default) + "\n")
    tmp.replace(path)


def _json_default(x):
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    raise TypeError(type(x).__name__)


def csv(out, name, rows, columns=None):
    path = out / (PREFIX + name + ".csv")
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows, columns=columns)
    if not len(frame.columns):
        frame = pd.DataFrame(columns=["status", "reason"])
    frame.to_csv(path, index=False)
    return path


def unique_file(folder, pattern):
    matches = sorted(folder.glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"Expected one {pattern}, found {len(matches)}")
    return matches[0]


def mat_epoch(path, key, day, epoch):
    a = day_epochs(loadmat(path, squeeze_me=True, struct_as_record=False)[key], day)
    if not 1 <= epoch <= len(a):
        raise ValueError("Epoch absent from MATLAB cell wrapper")
    return a[epoch - 1]


def native_events(path):
    frame = pd.read_csv(path, low_memory=False).rename(columns={"Animal ID": "animal"})
    required = {"animal", "day", "epoch", "ripple_number", "start_time", "end_time", "actual_speed"}
    if not required.issubset(frame):
        raise ValueError(f"Native event schema missing {sorted(required - set(frame))}")
    frame["animal"] = frame.animal.str.lower()
    if frame.duplicated(["animal", "day", "epoch", "ripple_number"]).any():
        raise ValueError("Duplicate native event identifiers; no silent deduplication")
    for column in ("start_time", "end_time"):
        frame[column + "_s"] = pd.to_timedelta(frame[column], errors="raise").dt.total_seconds()
    if not np.isfinite(frame[["start_time_s", "end_time_s"]]).all().all() or (frame.end_time_s <= frame.start_time_s).any():
        raise ValueError("Invalid native event timestamps")
    return frame


def task_inventory(root):
    rows = []
    for animal, name in ANIMALS.items():
        for path in sorted((root / name).glob("*task[0-9][0-9].mat")):
            day = int(re.search(r"task(\d+)\.mat$", path.name)[1])
            try:
                epochs = day_epochs(loadmat(path, squeeze_me=True, struct_as_record=False)["task"], day)
                for epoch, task in enumerate(epochs, 1):
                    if str(field(task, "type", "")).lower() == "run":
                        rows.append({"animal": animal, "day": day, "epoch": epoch, "task_path": str(path), "task_inventory_status": "run_verified"})
            except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
                rows.append({"animal": animal, "day": day, "epoch": 0, "task_path": str(path), "task_inventory_status": f"unresolved: {exc}"})
    return pd.DataFrame(rows)


def marks_for_day(folder, day):
    return sorted(path for path in folder.rglob("*marks*.mat") if re.search(rf"marks{day:02d}(?:-\d+)?\.mat$", path.name))


def epoch_data(folder, day, epoch, p):
    paths = {k: unique_file(folder, f"*{k}{day:02d}.mat") for k in ("linpos", "pos", "task")}
    lin = mat_epoch(paths["linpos"], "linpos", day, epoch)
    pos = mat_epoch(paths["pos"], "pos", day, epoch)
    task = mat_epoch(paths["task"], "task", day, epoch)
    if str(field(task, "type", "")).lower() != "run":
        raise ValueError("Task metadata does not identify a RUN epoch")
    a = np.asarray(field(pos, "data", []), float)
    if a.ndim != 2 or a.shape[1] < 5:
        raise ValueError("Missing documented position columns")
    time = a[:, 0]
    xy = a[:, 5:7] if a.shape[1] >= 9 else a[:, 1:3]
    speed = a[:, 8] if a.shape[1] >= 9 else a[:, 4]
    sm = field(lin, "statematrix")
    lt = np.asarray(field(sm, "time", []), float).reshape(-1)
    if len(lt) != len(time) or np.max(np.abs(lt-time), initial=0) > p["position_alignment_tolerance_s"]:
        raise ValueError("Position and linear annotation clocks do not align")
    wells = np.asarray(field(field(lin, "wellSegmentInfo"), "wellCoord", []), float)
    coords = np.asarray(field(field(lin, "segmentInfo"), "segmentCoords", []), float)
    traj = np.asarray(field(lin, "trajwells", []), int)
    if traj.shape != (2, 2) or len(set(traj[:, 0])) != 1 or len(set(traj[:, 1])) != 2:
        raise ValueError("Published center/outer trajectory metadata unavailable")
    center, outers = int(traj[0, 0]), tuple(sorted(traj[:, 1].astype(int)))
    if set((center, *outers)) != {1, 2, 3} or wells.shape != (3, 2) or coords.shape != (5, 4):
        raise ValueError("Unexpected W-track well/segment geometry")
    return time, xy, speed, wells, coords, center, outers, sm, task, paths


def dio_audit(folder, day, epoch, visits):
    matches = list((folder / "DIO").glob(f"*DIO{day:02d}.mat"))
    if len(matches) != 1:
        return [], "absent_or_nonunique_dio", []
    path = matches[0]
    try:
        pins = mat_epoch(path, "DIO", day, epoch)
    except (ValueError, KeyError, IndexError, TypeError):
        return [], "unreadable_dio_epoch", [path]
    rows = []
    for pin, x in enumerate(np.asarray(pins, dtype=object).reshape(-1), 1):
        pulses = np.asarray(field(x, "pulsetimes", []), float)
        if pulses.size == 2:
            pulses = pulses.reshape(1, 2)
        if pulses.ndim != 2 or pulses.shape[1] != 2 or len(pulses) < 3:
            continue
        duration = pulses[:, 1]-pulses[:, 0]
        median = float(np.median(duration))
        if not np.isfinite(pulses).all() or median < 100 or median > 10000:
            continue
        stable = float(np.mean(np.abs(duration-median) <= max(3, 0.05*median)))
        if stable < 0.6:
            continue
        # Pulse association is a diagnostic, not proof of pin identity or reward delivery.
        onset = pulses[:, 0] / 10000
        hits = [(t, v) for t in onset for v in visits if v["arrival_s"] <= t < v["departure_s"]]
        counts = {w: sum(v["well"] == w for _, v in hits) for w in (1, 2, 3)}
        total = sum(counts.values())
        well = max(counts, key=counts.get)
        matched = [v for _, v in hits if v["well"] == well and v["task_correct"] is not None]
        rows.append({"pin": pin, "n_pulses": len(pulses), "median_width_ticks": median, "stable_width_fraction": stable,
                     "n_visit_hits": total, "associated_well": well if total else None,
                     "well_association_purity": counts[well]/total if total else None,
                     "pulse_hits_on_task_incorrect_visits": sum(not v["task_correct"] for v in matched),
                     "hardware_reward_pin_verified": False, "clock_scale_status": "10000_ticks_per_second_candidate_not_independently_verified",
                     "reward_observed": None, "dio_path": str(path)})
    return rows, "candidate_pulse_associations_only" if rows else "no_candidate_pulses", [path]


def audit_epoch(root, key, events, p):
    animal, day, epoch = key
    folder = root / ANIMALS[animal]
    session = f"{animal}-{day:02d}-{epoch:02d}"
    identity = dict(animal=animal, day=day, epoch=epoch, session=session)
    t, xy, speed, wells, coords, center, outers, sm, task, paths = epoch_data(folder, day, epoch, p)
    visits, labels, distances = reconstruct_visits(t, xy, wells, radius=p["well_radius_cm"], max_gap=p["max_tracking_gap_s"])
    visits = score_visits(visits, center, outers)
    good = (labels == center) & np.isfinite(speed) & (speed >= 0) & (speed <= p["max_immobile_speed_cm_s"])
    intervals = valid_intervals(t, good, p["max_tracking_gap_s"])
    transitions = build_transitions(visits, center=center, outers=outers, immobile_intervals=intervals,
                                    max_window=p["max_window_s"], min_exposure=p["min_exposure_s"])
    native_distance = np.asarray(field(sm, "linearDistanceToWells", []), float)
    graph_labels = np.full(len(t), -1)
    if native_distance.shape == (len(t), 3):
        near = native_distance <= p["well_radius_cm"]
        graph_labels = np.where(near.sum(axis=1) == 1, near.argmax(axis=1) + 1, 0)
    comparison = (labels > 0) & (graph_labels > 0)
    annotation_agreement = float(np.mean(labels[comparison] == graph_labels[comparison])) if comparison.any() else None
    assignments = []
    for i, row in enumerate(transitions, 1):
        row.update(identity, trial_id=f"{session}-posterror-{i}", trial_progress=row["error_visit_index"] / max(1, len(visits)),
                   target_side=int(row["correct_alternative_well"] == max(outers)), native_candidate_count=0,
                   sequence_duration_capable_candidates=0, reward_observed=None, reward_status="not_verified")
        if row["eligible"]:
            matched = events[(events.start_time_s >= row["window_start_s"]) & (events.end_time_s <= row["window_end_s"])]
            for e in matched.itertuples(index=False):
                contained = contains_event(row["immobile_intervals"], e.start_time_s, e.end_time_s)
                if not contained:
                    continue
                duration = e.end_time_s-e.start_time_s
                assignments.append({**identity, "trial_id": row["trial_id"], "ripple_number": int(e.ripple_number), "start_time_s": e.start_time_s,
                                    "end_time_s": e.end_time_s, "duration_s": duration, "actual_speed_cm_s_native": float(e.actual_speed),
                                    "wholly_contained_in_immobility": True, "sequence_duration_capable": duration >= p["time_bin_s"]*p["min_spike_supported_bins"],
                                    "sequence_validated": None, "correct_route_content": None, "mistaken_route_content": None})
                row["native_candidate_count"] += 1
                row["sequence_duration_capable_candidates"] += duration >= p["time_bin_s"]*p["min_spike_supported_bins"]
        row["immobile_intervals_json"] = json.dumps(row.pop("immobile_intervals"))
    reward, dio_status, dio_paths = dio_audit(folder, day, epoch, visits)
    mark_files = marks_for_day(folder, day)
    tet_files = sorted(folder.glob("*tetinfo.mat"))
    inputs = [{"path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": file_sha256(path), "role": k}
              for k, path in [*paths.items(), *[("dio", v) for v in dio_paths]]]
    availability = [{"path": str(path), "size_bytes": path.stat().st_size, "sha256": None, "role": "unconsumed_mark_availability"} for path in mark_files]
    inputs += availability
    inputs += [{"path": str(v), "size_bytes": v.stat().st_size, "sha256": file_sha256(v), "role": "tetrode_metadata_available_not_yet_parsed"} for v in tet_files]
    inventory = {**identity, "status": "parsed", "failure_reason": "", "environment": str(field(task, "environment", field(task, "description", ""))),
                 "n_position_samples": len(t), "n_well_visits": len(visits), "n_history_segments": len(set(v["history_segment"] for v in visits)),
                 "n_post_error_transitions": len(transitions), "n_eligible_transitions": sum(v["eligible"] for v in transitions),
                 "n_corrections": sum(v["eligible"] and v["next_outcome"] == "correction" for v in transitions),
                 "n_repeated_errors": sum(v["eligible"] and v["next_outcome"] == "repeated_error" for v in transitions),
                 "n_native_candidates_linked": len(assignments), "position_annotation_comparison_frames": int(comparison.sum()),
                 "position_annotation_agreement": annotation_agreement, "has_mark_files": bool(mark_files), "has_tetrode_metadata": len(tet_files) == 1,
                 "dio_status": dio_status, "source_cmperpixel_not_reapplied": True}
    cohort_ok = bool(annotation_agreement is not None and annotation_agreement >= p["min_position_annotation_agreement"] and len(events) and mark_files and len(tet_files) == 1)
    inventory["cohort_eligible"] = cohort_ok
    inventory["native_table_epoch_covered"] = bool(len(events))
    for row in transitions:
        row["behavior_eligible"] = row["eligible"]
        row["cohort_eligible"] = cohort_ok
        if not cohort_ok:
            row["eligible"] = False
            row["exclusion_reason"] = "epoch_failed_frozen_cohort_eligibility"
    inventory["n_eligible_transitions"] = sum(row["eligible"] for row in transitions)
    inventory["n_corrections"] = sum(row["eligible"] and row["next_outcome"] == "correction" for row in transitions)
    inventory["n_repeated_errors"] = sum(row["eligible"] and row["next_outcome"] == "repeated_error" for row in transitions)
    for event in assignments:
        event["cohort_eligible"] = cohort_ok
    for v in visits:
        v.update(identity)
    for v in reward:
        v.update(identity)
    return {"inventory": inventory, "visits": visits, "transitions": transitions, "assignments": assignments,
            "reward_audit": reward, "inputs": inputs, "geometry": {**identity, "wells": wells.tolist(), "segments": coords.tolist()},
            "figure_data": {"time": t.tolist(), "xy": np.where(np.isfinite(xy), xy, None).tolist()}}


def gates_for(inventory, trials, assignments, p):
    eligible = trials[trials.eligible.astype(bool)] if not trials.empty else trials
    n = len(eligible)
    animals = eligible.animal.nunique() if n else 0
    corrections = int((eligible.next_outcome == "correction").sum()) if n else 0
    repeats = int((eligible.next_outcome == "repeated_error").sum()) if n else 0
    parsed = inventory[inventory.status == "parsed"]
    retained = parsed[parsed.cohort_eligible.astype(bool)] if not parsed.empty else parsed
    aligned = not retained.empty and retained.position_annotation_comparison_frames.gt(0).all() and retained.position_annotation_agreement.ge(p["min_position_annotation_agreement"]).all()
    rows = [
        ("novelty_branch_open", not any(x["exact_contrast_established"] for x in PRIOR_ART), "exact contrast not established in inspected sources", "bounded audit; not exhaustive novelty proof"),
        ("epoch_accounting_present", not inventory.empty and inventory.status.isin(["parsed", "failed"]).all(), f"{len(parsed)}/{len(inventory)} parsed", "all RUN epochs inventoried; failed epochs excluded before association"),
        ("position_annotation_agreement", bool(aligned), float(retained.position_annotation_agreement.min()) if len(retained) else 0.0, ">=95% at near-well frames in each retained epoch"),
        ("animals_represented", animals >= p["min_animals"], animals, p["min_animals"]),
        ("eligible_transitions", n >= p["min_transitions"], n, p["min_transitions"]),
        ("corrections_present", corrections >= p["min_corrections"], corrections, p["min_corrections"]),
        ("repeated_errors_present", repeats >= p["min_repeated_errors"], repeats, p["min_repeated_errors"]),
        ("native_candidates_present", not assignments.empty, len(assignments), ">0; not sequence validation"),
        ("raw_neural_inputs_available", not retained.empty and retained.has_mark_files.all() and retained.has_tetrode_metadata.all(),
         int((retained.has_mark_files & retained.has_tetrode_metadata).sum()) if len(retained) else 0, "marks and tetrode metadata available in each retained epoch; not readout QC"),
    ]
    result = [{"gate": k, "passed": bool(b), "observed": o, "criterion": c, "gate_type": "feasibility"} for k, b, o, c in rows]
    result.append({"gate": "feasibility_overall", "passed": all(x["passed"] for x in result), "observed": n, "criterion": "all feasibility gates", "gate_type": "feasibility"})
    return result


def route_figures(out, packets):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    manifest = []
    seen = set()
    for packet in packets:
        for row in packet["transitions"]:
            key = (row["animal"], row["next_outcome"])
            if key in seen or row["window_start_s"] is None:
                continue
            seen.add(key)
            t = np.asarray(packet["figure_data"]["time"])
            xy = np.asarray(packet["figure_data"]["xy"], float)
            use = (t >= row["error_arrival_s"]-5) & (t <= row.get("next_choice_arrival_s", row["window_end_s"])+1)
            if not use.any():
                continue
            fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
            for segment in np.asarray(packet["geometry"]["segments"]):
                axes[0].plot(segment[[0, 2]], segment[[1, 3]], color="0.7", lw=3)
            axes[0].scatter(xy[use, 0], xy[use, 1], c=t[use]-row["error_arrival_s"], s=5, cmap="viridis")
            for i, well in enumerate(packet["geometry"]["wells"], 1):
                axes[0].text(*well, f"well {i}")
            axes[0].set_aspect("equal"); axes[0].set(xlabel="x (cm)", ylabel="y (cm)", title=row["trial_id"])
            v = [v for v in packet["visits"] if row["error_arrival_s"]-5 <= v["arrival_s"] <= row.get("next_choice_arrival_s", row["window_end_s"])+1]
            axes[1].step([x["arrival_s"]-row["error_arrival_s"] for x in v], [x["well"] for x in v], where="post")
            axes[1].axvspan(row["window_start_s"]-row["error_arrival_s"], row["window_end_s"]-row["error_arrival_s"], color="0.8", alpha=.5)
            axes[1].set(xlabel="seconds after error arrival", ylabel="well", title=f"{row['next_outcome']}; exposure {row['usable_exposure_s']:.2f} s")
            name = f"route_{row['trial_id']}.png"
            fig.savefig(out / name, dpi=120); plt.close(fig)
            manifest.append({"figure": name, "trial_id": row["trial_id"], "selection_rule": "chronologically first per animal/outcome; no neural result used"})
    csv(out, "figures_manifest", manifest)


def feasibility(args, p):
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    native = native_events(args.native_events)
    provenance = build_script_provenance(input_paths={"protocol": args.protocol, "native_events": args.native_events, "dataset_root": args.dataset_root}, cwd=ROOT)
    checkpoints = out / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    checkpoint_identity = {"protocol_sha256": file_sha256(args.protocol), "native_sha256": file_sha256(args.native_events), "code_commit": provenance["code_commit"]}
    packets = []
    task_rows = task_inventory(args.dataset_root)
    csv(out, "task_inventory", task_rows)
    keys = task_rows[["animal", "day", "epoch"]].drop_duplicates().sort_values(["animal", "day", "epoch"])
    for animal, day, epoch in keys.itertuples(index=False, name=None):
        identity = dict(animal=str(animal), day=int(day), epoch=int(epoch), session=f"{animal}-{day:02d}-{epoch:02d}")
        checkpoint = checkpoints / (identity["session"] + ".json")
        if checkpoint.exists():
            packet = json.loads(checkpoint.read_text())
            if packet["checkpoint_identity"] != checkpoint_identity:
                raise ValueError("Checkpoint code/protocol/native input mismatch; use a new output directory")
        else:
            try:
                group = native[(native.animal == animal) & (native.day == day) & (native.epoch == epoch)]
                packet = audit_epoch(args.dataset_root, (animal, int(day), int(epoch)), group, p)
            except (KeyError, IndexError, ValueError, TypeError, OSError) as exc:
                packet = {"inventory": {**identity, "status": "failed", "cohort_eligible": False, "failure_reason": f"{type(exc).__name__}: {exc}", "position_annotation_agreement": None},
                          "visits": [], "transitions": [], "assignments": [], "inputs": [], "reward_audit": []}
            packet["checkpoint_identity"] = checkpoint_identity
            atomic_json(checkpoint, packet)
        packets.append(packet)
        print(json.dumps(packet["inventory"]), flush=True)
        atomic_json(out / "progress.json", {"stage": "feasibility", "completed_epochs": len(packets), "total_epochs": len(keys), "last_session": identity["session"]})
    inventory = pd.DataFrame([x["inventory"] for x in packets])
    trials = pd.DataFrame([r for x in packets for r in x["transitions"]])
    events = pd.DataFrame([r for x in packets for r in x["assignments"]])
    gates = gates_for(inventory, trials, events, p)
    for name, rows in [("cohort_inventory", inventory), ("trial_inventory", trials), ("native_event_assignments", events),
                       ("all_well_visits", [r for x in packets for r in x["visits"]]), ("reward_pulse_audit", [r for x in packets for r in x["reward_audit"]]),
                       ("input_inventory", [r for x in packets for r in x["inputs"]]), ("prior_art", PRIOR_ART), ("gate_summary", gates)]:
        csv(out, name, rows)
    csv(out, "by_animal", trials[trials.eligible.astype(bool)].groupby("animal").size().rename("n_eligible_transitions").reset_index() if not trials.empty else [])
    route_figures(out, [x for x in packets if x["inventory"]["status"] == "parsed"])
    overall = gates[-1]["passed"]
    atomic_json(out / (PREFIX + "manifest.json"), {"created_at_utc": datetime.now(UTC).isoformat(), "stage": "feasibility", "protocol": p,
                "provenance": provenance, "environment": {"python": platform.python_version(), **{k: importlib.metadata.version(k) for k in ("numpy", "scipy", "pandas")}},
                "feasibility_passed": overall, "biological_status": "not_tested", "next_action": "preceding_run_and_sequence_calibration" if overall else "stop_inconclusive_feasibility",
                "neural_inputs": "availability only; no mark decoding or content/outcome association",
                "output_sha256": {v.name: file_sha256(v) for v in sorted(out.glob("*.csv"))}})
    report(out)


def report(out):
    manifest = json.loads((out / (PREFIX + "manifest.json")).read_text())
    gate = pd.read_csv(out / (PREFIX + "gate_summary.csv"))
    lines = ["# Denovellis post-error route-content study", "", f"Stage: {manifest['stage']}", f"Biological status: {manifest['biological_status']}",
             f"Next action: {manifest['next_action']}", "", "## Gates", "", "| Gate | Passed | Observed |", "| --- | --- | --- |"]
    lines.extend(f"| {r.gate} | {r.passed} | {r.observed} |" for r in gate.itertuples())
    lines += ["", "## Interpretation", "", "This is a prospective protocol applied to previously inspected public data, not an independent replication.",
              "Visits are reconstructed from raw position and published well geometry. Native event assignments are candidates, not validated sequences.",
              "No scalar replay summary is substituted for route-resolved content. Eligible zero-event trials remain in the inventory.",
              "Missing reward metadata are unknown, not omissions; pulse/well associations do not establish hardware identity.",
              "Failed feasibility or calibration is inconclusive, not absence of a biological association.",
              "Calibration and biological models must remain withheld until their actual prerequisites pass.", "",
              f"Code commit: `{manifest['provenance']['code_commit']}`", f"Frozen seed: `{manifest['protocol']['seed']}`", ""]
    (out / (PREFIX + "report.md")).write_text("\n".join(lines))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["feasibility", "calibration", "analysis", "report"], required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--native-events", type=Path)
    parser.add_argument("--protocol", type=Path, default=ROOT / "docs" / "denovellis_post_error_protocol.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prerequisite-dir", type=Path)
    parser.add_argument("--seed", type=int, default=20261001)
    args = parser.parse_args(argv)
    p = json.loads(args.protocol.read_text())
    if args.seed != p["seed"]:
        parser.error("Seed differs from frozen protocol")
    if args.stage == "report":
        report(args.output_dir)
        return
    if args.stage == "feasibility":
        if args.dataset_root is None or args.native_events is None:
            parser.error("Feasibility requires dataset root and native events")
        feasibility(args, p)
        return
    if args.prerequisite_dir is None:
        parser.error("Calibration/analysis require a completed prerequisite directory")
    prerequisite = json.loads((args.prerequisite_dir / (PREFIX + "manifest.json")).read_text())
    if prerequisite["protocol"] != p or not prerequisite.get("feasibility_passed", False):
        parser.error("Feasibility failed or protocol differs; downstream experiment is blocked")
    parser.error("Actual preceding-RUN readout/calibration artifacts are required; downstream stages are not yet implemented. No biological scores were produced.")


if __name__ == "__main__":
    main()
