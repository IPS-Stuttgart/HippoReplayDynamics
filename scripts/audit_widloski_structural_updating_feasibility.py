#!/usr/bin/env python3
"""Bounded Widloski/Foster inventory and behavioral opportunity audit, without decoding."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.io import loadmat, whosmat

try:
    from scripts import widloski_structural_updating_core as core
    from scripts._provenance import build_script_provenance
except ModuleNotFoundError:
    import widloski_structural_updating_core as core
    from _provenance import build_script_provenance

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "widloski_structural_updating_"
DEFAULT_ROOT = Path("/mnt/lexar4tb/datasets/widloski-foster-2025/archive")
DEFAULT_PROTOCOL = ROOT / "docs/widloski_structural_updating_protocol.json"
DEFAULT_SOURCES = ROOT / "docs/widloski_structural_updating_source_manifest.json"


def atomic_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def table(out, name, rows, columns):
    frame = pd.DataFrame(rows, columns=columns)
    frame.to_csv(Path(out) / (PREFIX + name + ".csv"), index=False)


def protocol_check(p):
    core.require(p.get("schema_version") == 1, "Unsupported protocol")
    core.require(p.get("scope") == "feasibility_only", "Biological scoring is not supported")
    core.require(p.get("allow_joint_goal_changes") is True, "Frozen protocol allows joint goal changes with a design gate")
    core.require(p.get("tracking_gap_s") == 0.25 and p.get("immobility_speed_cm_s") == 5.0, "Changed frozen behavior thresholds")
    core.require(p.get("minimum_episodes_per_animal") == 2 and p.get("minimum_days_per_animal") == 2, "Changed engineering floor")
    core.require(p.get("animals") and len(p["animals"]) == len(set(p["animals"])), "Missing/duplicate expected animals")
    core.require(p.get("route_tolerance_cm", 0) > 0, "Invalid numerical route tolerance")


def hdf_scalar(handle, dataset, row, text=False):
    ref = dataset[row, 0]
    core.require(h5py.check_dtype(ref=dataset.dtype) is not None, "Expected documented MATLAB struct reference")
    values = np.asarray(handle[ref][()])
    if text:
        core.require(values.dtype.kind in "ui", "MATLAB character field has unexpected type")
        return "".join(chr(int(v)) for v in values.ravel())
    core.require(values.size == 1 and np.isfinite(values).all(), "Session label must be a finite scalar")
    val = float(values.ravel()[0])
    core.require(val == int(val) and val >= 1, "Session label is not a positive integer")
    return int(val)


def session_identity(animal, date, number):
    return f"{animal}/{date}/run{number}"


def read_session_inventory(path, animal_order):
    """Read ONLY the five author-assigned identity fields, not Data.replays/ratLocs."""
    rows = []
    with h5py.File(path, "r") as handle:
        data = handle["Data"]
        fields = ["ratName", "file", "rat", "day", "sessionNum"]
        core.require(set(fields) <= set(data), "Missing documented Data identity fields")
        shape = data["ratName"].shape
        core.require(len(shape) == 2 and shape[1] == 1 and all(data[k].shape == shape for k in fields), "Inconsistent MATLAB identity records")
        for i in range(shape[0]):
            animal = hdf_scalar(handle, data["ratName"], i, True)
            date = hdf_scalar(handle, data["file"], i, True)
            index = hdf_scalar(handle, data["rat"], i)
            core.require(index <= len(animal_order) and animal_order[index - 1] == animal, "Conflicting author animal identifiers")
            core.require(len(date) == 8 and date.isdigit(), "Undocumented recording-date label")
            day = hdf_scalar(handle, data["day"], i)
            number = hdf_scalar(handle, data["sessionNum"], i)
            rows.append(
                {
                    "animal": animal,
                    "date": date,
                    "session_id": session_identity(animal, date, number),
                    "day_index": day,
                    "session_number": number,
                    "source_row": i + 1,
                    "published_events": None,
                    "identity_status": "author_code_verified",
                }
            )
    core.require(len({r["session_id"] for r in rows}) == len(rows), "Duplicate source session identities")
    core.require(len({(r["animal"], r["day_index"], r["session_number"]) for r in rows}) == len(rows), "Conflicting day/session labels")
    return rows


def read_event_identities(path, animal_order):
    """Only author-code columns 43:46: source event index, rat, day, session."""
    a = loadmat(path, variable_names=["replays_stats"])["replays_stats"]
    core.require(a.ndim == 2 and a.shape[1] == 46, "Unexpected documented event-statistics schema")
    identities = a[:, 42:46]
    finite = np.isfinite(identities).all(axis=1)
    core.require(np.all(identities[finite] == np.floor(identities[finite])) and np.all(identities[finite] >= 1), "Invalid native event identity")
    core.require(len(np.unique(identities[finite], axis=0)) == int(finite.sum()), "Duplicate native event identity")
    result = []
    for i, values in enumerate(identities):
        if not finite[i]:
            result.append(
                {
                    "source_row": i + 1,
                    "animal": None,
                    "day_index": None,
                    "session_number": None,
                    "source_event_index": None,
                    "identity_status": "unresolved_missing_identity",
                    "session_id": None,
                }
            )
            continue
        event, rat, day, session = values.astype(int)
        core.require(rat <= len(animal_order), "Unknown event animal index")
        result.append(
            {
                "source_row": i + 1,
                "animal": animal_order[rat - 1],
                "day_index": int(day),
                "session_number": int(session),
                "source_event_index": int(event),
                "identity_status": "author_code_verified",
            }
        )
    return result


def inspect_file(path):
    if h5py.is_hdf5(path):
        with h5py.File(path, "r") as handle:
            result = []
            for key, obj in handle.items():
                if key == "#refs#":
                    continue
                result.append(
                    {
                        "variable": key,
                        "shape": list(obj.shape) if isinstance(obj, h5py.Dataset) else None,
                        "kind": "group" if isinstance(obj, h5py.Group) else str(obj.dtype),
                        "fields": sorted(obj.keys()) if isinstance(obj, h5py.Group) else [],
                    }
                )
            return result
    return [{"variable": k, "shape": list(shape), "kind": kind, "fields": []} for k, shape, kind in whosmat(path)]


def controls_fingerprint(args):
    inputs = {"source_manifest": args.source_manifest.resolve(), "protocol": args.protocol.resolve()}
    if args.verified_metadata:
        inputs["verified_metadata"] = args.verified_metadata.resolve()
    hashes = {k: core.file_hash(v) for k, v in inputs.items()}
    return inputs, {"dataset_root": str(args.dataset_root.resolve()), "control_hashes": hashes}


def begin_stage(out, fingerprint, check_inputs=True):
    out.mkdir(parents=True, exist_ok=True)
    state = out / "checkpoint.json"
    if state.exists():
        core.require(read_json(state)["fingerprint"] == fingerprint, "Refusing to mix stages with changed inputs; use a new output directory")
        if (out / "manifest.json").exists():
            manifest = verify_stage(out)
            if check_inputs:
                verify_input_files(manifest)
            return False
    else:
        core.require(not any(out.iterdir()), "Refusing nonempty unmanaged stage directory")
        atomic_json(state, {"fingerprint": fingerprint, "status": "running", "completed_files": []})
    return True


def finish_stage(out, inputs, fingerprint, stage, extra=None):
    checkpoint = read_json(out / "checkpoint.json")
    checkpoint["status"] = "completed"
    atomic_json(out / "checkpoint.json", checkpoint)
    outputs = {p.name: core.file_hash(p) for p in sorted(out.iterdir()) if p.is_file() and p.name != "manifest.json"}
    provenance = build_script_provenance(input_paths=inputs, cwd=ROOT)
    atomic_json(
        out / "manifest.json", dict(stage=stage, fingerprint=fingerprint, created_at_utc=datetime.now(UTC).isoformat(), **provenance, output_sha256=outputs, **(extra or {}))
    )


def verify_stage(out):
    manifest = read_json(out / "manifest.json")
    for name, sha in manifest["output_sha256"].items():
        core.require(Path(name).name == name, "Invalid artifact path")
        core.require((out / name).is_file() and core.file_hash(out / name) == sha, f"Artifact hash mismatch: {out / name}")
    return manifest


def verify_input_files(manifest):
    for key, path in manifest["input_file_paths"].items():
        core.require(Path(path).is_file() and core.file_hash(path) == manifest["input_file_sha256"][key], f"Upstream input changed or disappeared: {path}")


def verify_inventory_inputs(out):
    verify_input_files(verify_stage(out))
    for row in pd.read_csv(out / (PREFIX + "file_inventory.csv"), keep_default_na=False).to_dict("records"):
        path = Path(row["path"])
        if row["sha256"]:
            core.require(path.is_file() and core.file_hash(path) == row["sha256"], f"Inventory source changed: {path}")
        else:
            core.require(not path.exists(), f"Previously absent source appeared; use a new inventory: {path}")


def source_check(source):
    core.require(source.get("schema_version") == 1 and source.get("files"), "Missing source schema or data file list")
    core.require(source.get("animal_order"), "Missing author animal mapping")
    for f in source["files"]:
        core.require(Path(f["filename"]).name == f["filename"], "Source files must be filenames within dataset root")
        core.require(f.get("documentation") and f.get("source_url"), "Undocumented source field/adapter")
        core.require(f.get("reader") in {"headers", "event_identities_v1", "session_inventory_v1"}, "Unsupported source adapter")


def inventory(args):
    inputs, fingerprint = controls_fingerprint(args)
    p, source = read_json(args.protocol), read_json(args.source_manifest)
    protocol_check(p)
    source_check(source)
    out = args.output_dir / "inventory"
    if not begin_stage(out, fingerprint):
        verify_inventory_inputs(out)
        return read_json(out / "inventory_summary.json")
    atomic_json(out / "protocol.json", p)
    atomic_json(out / "source_manifest.json", source)
    sessions, events, files, issues = [], [], [], []
    sessions_available, events_available = False, False
    for spec in source["files"]:
        path = args.dataset_root / spec["filename"]
        row = {
            "filename": spec["filename"],
            "path": str(path.resolve()),
            "status": "absent",
            "size_bytes": None,
            "sha256": "",
            "md5": "",
            "source_url": spec["source_url"],
            "reason": "File absent",
        }
        if path.is_file():
            row.update(status="verified", size_bytes=path.stat().st_size, sha256=core.file_hash(path), reason=spec["documentation"])
            md5 = hashlib.md5()  # Published release integrity, not a security primitive.
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    md5.update(chunk)
            row["md5"] = md5.hexdigest()
            try:
                if spec.get("expected_md5"):
                    core.require(row["md5"] == spec["expected_md5"], "Published file checksum mismatch")
                if spec.get("expected_sha256"):
                    core.require(row["sha256"] == spec["expected_sha256"], "Source SHA256 mismatch")
                row["variables_json"] = json.dumps(inspect_file(path), sort_keys=True)
                if spec["reader"] == "session_inventory_v1":
                    core.require(not sessions, "Multiple session-inventory sources require explicit reconciliation")
                    sessions = read_session_inventory(path, source["animal_order"])
                    sessions_available = True
                elif spec["reader"] == "event_identities_v1":
                    core.require(not events, "Multiple event sources require explicit reconciliation")
                    events = read_event_identities(path, source["animal_order"])
                    events_available = True
                inputs[f"dataset_{spec['filename']}"] = path.resolve()
            except (ValueError, KeyError, OSError, TypeError) as exc:
                row.update(status="unresolved", reason=str(exc))
                issues.append(str(exc))
        files.append(row)
        atomic_json(out / "file_inventory_checkpoint.json", files)
        state = read_json(out / "checkpoint.json")
        state["completed_files"] = [f["filename"] for f in files]
        atomic_json(out / "checkpoint.json", state)
    by_key = {(s["animal"], s["day_index"], s["session_number"]): s for s in sessions}
    for s in sessions:
        s["published_events"] = 0 if events_available else None
    for e in events:
        if e["identity_status"] != "author_code_verified":
            continue
        key = (e["animal"], e["day_index"], e["session_number"])
        if key not in by_key:
            issues.append(f"Native event references an absent session: {key}")
            continue
        e["session_id"] = by_key[key]["session_id"]
        by_key[key]["published_events"] += 1
    unresolved = sum(e["identity_status"] != "author_code_verified" for e in events)
    if unresolved:
        issues.append(f"{unresolved} event-statistic rows have unresolved identities; no row-order assignment is permitted")
    metadata = None
    metadata_valid = False
    if args.verified_metadata:
        metadata = read_json(args.verified_metadata)
        try:
            supplied = core.validate_metadata(metadata)
            core.require(
                {s["session_id"] for s in supplied} == {s["session_id"] for s in sessions},
                "Verified metadata must account for every source session; do not silently select a subset",
            )
            source_by_id = {s["session_id"]: s for s in sessions}
            for s in supplied:
                original = source_by_id[s["session_id"]]
                core.require(s["animal"] == original["animal"] and s["date"].replace("-", "") == original["date"], "Verified metadata conflict with source identity")
            availability = core.capability_rows(metadata, args.verified_metadata.parent)
            metadata_valid = True
        except (core.MetadataError, KeyError, ValueError) as exc:
            issues.append(f"Metadata: {exc}")
            availability = []
        atomic_json(out / "verified_metadata.json", metadata)
    else:
        availability = []
    if not metadata_valid:
        for s in sessions:
            for cap in core.CAPABILITIES:
                default = source["unprovided_capabilities"][cap]
                availability.append(
                    {
                        "animal": s["animal"],
                        "session_id": s["session_id"],
                        "capability": cap,
                        "status": default["status"],
                        "reason": default["reason"],
                        "evidence_path": "",
                        "evidence_sha256": "",
                    }
                )
    for r in availability:
        if r["evidence_path"]:
            inputs[f"evidence_{len(inputs)}"] = Path(r["evidence_path"])
    valid = sessions_available and events_available and all(f["status"] == "verified" for f in files) and not issues
    rows = [
        {
            "animal": a,
            "sessions": sum(s["animal"] == a for s in sessions) if sessions_available else None,
            "published_events": sum(e["animal"] == a for e in events) if events_available else None,
            "opportunity_events": None,
            "status": "inventory_only_not_eligibility",
        }
        for a in p["animals"]
    ]
    table(out, "file_inventory", files, ["filename", "path", "status", "size_bytes", "sha256", "md5", "source_url", "reason", "variables_json"])
    table(out, "session_inventory", sessions, core.SESSION_COLUMNS)
    table(out, "native_event_identity_inventory", events, ["source_row", "animal", "session_id", "day_index", "session_number", "source_event_index", "identity_status"])
    table(out, "availability", availability, ["animal", "session_id", "capability", "status", "reason", "evidence_path", "evidence_sha256"])
    table(out, "inventory_by_animal", rows, ["animal", "sessions", "published_events", "opportunity_events", "status"])
    table(out, "prior_art", source["prior_art"], ["study", "url", "materials_reviewed", "existing_result", "exact_contrast_status", "remaining_question"])
    summary = {
        "source_integrity_valid": valid,
        "metadata_valid": metadata_valid,
        "sessions": len(sessions) if sessions_available else None,
        "published_events": len(events) if events_available else None,
        "resolved_event_rows": len(events) - unresolved if events_available else None,
        "unresolved_event_rows": unresolved if events_available else None,
        "verified_metadata_base": str(args.verified_metadata.resolve().parent) if args.verified_metadata else None,
        "metadata_path": str(args.verified_metadata.resolve()) if args.verified_metadata else None,
        "novelty_resolution": source["novelty_resolution"],
        "issues": issues,
        "eligible_events": None,
        "interpretation": "Published event count is not an eligible opportunity count.",
    }
    atomic_json(out / "inventory_summary.json", summary)
    author_request(out, sessions, availability)
    finish_stage(out, inputs, fingerprint, "inventory", {"replay_content_read": False, "neural_scoring_performed": False})
    return summary


def author_request(out, sessions, availability):
    missing = sorted({a["capability"] for a in availability if a["status"] != "verified"})
    text = [
        "# UNSENT DRAFT: Widloski/Foster structural-updating feasibility",
        "",
        "To: John Widloski and David Foster (addresses to verify before sending)",
        "Subject: Data/metadata question about replay after barrier reconfiguration",
        "",
        "Dear John and David,",
        "",
        "We are assessing a narrow extension of your barrier-maze replay work: whether, after a configuration change, replay reflects altered route costs at locally unchanged junctions before the animal traverses those outgoing transitions again. We recognize that rapid barrier conformity and goal-directed replay are already established by your work.",
        "",
        f"We have inventoried {len(sessions)} sessions in the public processed release. The attached session table identifies the files and within-day session labels. We have not decoded new replay or tested this biological contrast.",
        "",
        "Could you advise whether this specific experience-conditioned analysis has already been performed or is underway, and whether the following inputs can be shared or located?",
        "",
        "- Continuous synchronized position/speed and full-session sorted spike times, including units tracked across adjacent configurations and the clock/unit documentation.",
        "- Native replay start/end times, event IDs and validation flags linked to the published event tables.",
        "- How to interpret statistics rows whose event/rat/day/session identifiers are missing, and whether the cell-shuffle table refers to the same event set (the public tables have different row counts). We will not join them by row position.",
        "- Barrier/wall geometry, configuration IDs, complete session order, animal entry/exit times and any unrecorded exposure between recordings.",
        "- Home/Random goal identities, trial/reward timing and a behaviorally verified connectivity/traversal representation, if available.",
        "- Whether additional recordings support this contrast and whether pre-change RUN coverage can support an independent encoder.",
        "",
        "We allow simultaneous changes in goals and barriers in the inventory, but will not claim a topology-specific association unless the design supports separating them. Not retraversed does not mean unknown to the animal, since the barriers were visible.",
        "",
        "Thank you,",
        "Florian",
        "",
        "## Audit attachment checklist",
        "",
        f"Session inventory: {PREFIX}session_inventory.csv",
        "Unverified capabilities: " + ", ".join(missing),
        "",
        "Status: unsent. Sending requires Florian's separate approval; no raw recordings or result claims are attached.",
    ]
    (out / "author_request_unsent.md").write_text("\n".join(text) + "\n")


def opportunities(args):
    inputs, fingerprint = controls_fingerprint(args)
    inv_dir = args.output_dir / "inventory"
    inv_manifest = verify_stage(inv_dir)
    verify_inventory_inputs(inv_dir)
    core.require(inv_manifest["fingerprint"] == fingerprint, "Upstream inventory control hashes do not match")
    inputs["inventory_manifest"] = inv_dir / "manifest.json"
    fingerprint = {**fingerprint, "inventory_manifest_sha256": core.file_hash(inv_dir / "manifest.json")}
    out = args.output_dir / "opportunities"
    if not begin_stage(out, fingerprint):
        return read_json(out / "decision.json")
    p = read_json(inv_dir / "protocol.json")
    inv = read_json(inv_dir / "inventory_summary.json")
    availability = pd.read_csv(inv_dir / (PREFIX + "availability.csv"), keep_default_na=False).to_dict("records")
    result = {"opportunities": [], "memberships": [], "design": [], "exclusions": [], "episodes": [], "timelines": []}
    available, metadata, reason = False, None, "Verified metadata and continuous behavioral inputs are unavailable"
    if inv["metadata_valid"] and inv["source_integrity_valid"]:
        metadata = read_json(inv_dir / "verified_metadata.json")
        current = core.capability_rows(metadata, inv["verified_metadata_base"])
        # Verification must be current: a stale capability ledger cannot authorize reads.
        core.require(current == availability, "Capability evidence changed after inventory")
        if current and all(a["status"] == "verified" for a in current if a["capability"] in core.OPPORTUNITY_CAPABILITIES):
            try:
                result = core.build_opportunities(metadata, inv["verified_metadata_base"], p)
                available = result["measurement_complete"]
                reason = (
                    "Behavioral opportunity construction completed"
                    if available
                    else "Some declared behavioral inputs are invalid; partial ledgers are retained but coverage is unavailable"
                )
                for s in metadata["sessions"]:
                    for kind in ("position", "events", "traversals"):
                        if kind in s:
                            inputs[f"{s['session_id']}_{kind}"] = core.checked_file(s[kind], inv["verified_metadata_base"], s["clock_id"])
            except (core.MetadataError, KeyError, ValueError, OSError) as exc:
                reason = f"Unresolved opportunity inputs: {exc}"
    if not available:
        result["exclusions"].append({"animal": "", "session_id": "", "junction": "", "destination": "", "event_id": "", "reason": reason})
    matched = {r["opportunity_id"] for r in result["opportunities"] if r["match_status"] == "matched_geometry_only"}
    confounding, support = core.confounding_audit([d for d in result["design"] if d["opportunity_id"] in matched])
    by_animal = core.coverage(result["opportunities"], result["memberships"], p, available)
    gates, decision = core.gates_and_decision(availability, by_animal, inv["novelty_resolution"], confounding, available, inv["source_integrity_valid"])
    table(out, "opportunities", result["opportunities"], core.OPPORTUNITY_COLUMNS)
    table(out, "event_opportunity_memberships", result["memberships"], core.MEMBERSHIP_COLUMNS)
    table(out, "exclusions", result["exclusions"], core.EXCLUSION_COLUMNS)
    table(out, "design", result["design"], core.DESIGN_COLUMNS)
    table(
        out,
        "episodes",
        result["episodes"],
        [
            "animal",
            "date",
            "episode_id",
            "session_id",
            "old_configuration",
            "new_configuration",
            "old_goal",
            "new_goal",
            "goal_changed",
            "history_end_s",
            "history_end_reason",
            "native_events",
            "validated_native_events",
        ],
    )
    table(out, "goal_support", support, ["goal_pair", "episodes", "animals", "configuration_pairs", "structural_range_cm", "supported"])
    table(out, "confounding", [confounding], list(confounding))
    table(out, "animal_coverage", by_animal, ["animal", "eligible_episodes", "eligible_days", "unique_events", "event_opportunity_memberships", "status"])
    table(out, "gate_summary", gates, ["gate", "status", "reason"])
    table(
        out, "goal_configuration_crossing", result["episodes"], ["animal", "date", "episode_id", "old_configuration", "new_configuration", "old_goal", "new_goal", "goal_changed"]
    )
    counts = []
    if available:
        for changed in (False, True):
            rows = [o for o in result["opportunities"] if o["goal_changed"] == changed]
            counts.append(
                {
                    "goal_changed": changed,
                    "opportunities": len(rows),
                    "exposure_s": sum(o["exposure_s"] for o in rows),
                    "median_history_duration_s": float(np.median([o["end_time_s"] - o["start_time_s"] for o in rows])) if rows else None,
                }
            )
    table(out, "goal_change_exposure", counts, ["goal_changed", "opportunities", "exposure_s", "median_history_duration_s"])
    atomic_json(
        out / "decision.json",
        {
            **decision,
            "opportunity_measurement_available": available,
            "eligible_unique_events": len({(m["animal"], m["session_id"], m["event_id"]) for m in result["memberships"]}) if available else None,
            "opportunity_rows": len(result["opportunities"]) if available else None,
            "reason": reason,
        },
    )
    atomic_json(out / "timelines.json", result["timelines"])
    plot_geometry_timelines(out, result, metadata if available else None)
    finish_stage(out, inputs, fingerprint, "opportunities", {"replay_content_read": False, "neural_scoring_performed": False})
    return read_json(out / "decision.json")


def plot_geometry_timelines(out, result, metadata):
    plots = []
    if metadata and result["opportunities"]:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        nodes, edges, configs, _ = core.load_graph(metadata["graph"])
        # Deterministic representatives, selected without replay-content values.
        representatives = {}
        for row in sorted(result["opportunities"], key=lambda r: r["opportunity_id"]):
            representatives.setdefault(row["animal"], row)
        timing = {t["opportunity_id"]: t for t in result["timelines"]}
        for i, row in enumerate(representatives.values()):
            fig, axes = plt.subplots(1, 3, figsize=(11, 3.7), layout="constrained")
            for ax, config, title in zip(axes[:2], [row["old_configuration"], row["new_configuration"]], ["Previous connectivity", "Current connectivity"], strict=True):
                for eid in sorted(configs[config]):
                    edge = edges[eid]
                    a, b = nodes[edge["u"]], nodes[edge["v"]]
                    ax.plot([a["x_cm"], b["x_cm"]], [a["y_cm"], b["y_cm"]], color="0.55", linewidth=1)
                for nid, node in nodes.items():
                    ax.scatter(node["x_cm"], node["y_cm"], c="#b42342" if nid == row["junction"] else "#247c72", s=20)
                    ax.annotate(nid, (node["x_cm"], node["y_cm"]), xytext=(4, 4), textcoords="offset points", fontsize=8)
                ax.set(title=title, xlabel="x (cm)", ylabel="y (cm)", aspect="equal")
            t = timing[row["opportunity_id"]]
            origin = row["start_time_s"]
            ax = axes[2]
            for lo, hi in t["immobile_intervals"]:
                ax.plot([lo - origin, hi - origin], [1, 1], linewidth=6, color="#247c72")
            for lo, hi in t["event_intervals"]:
                ax.plot([lo - origin, hi - origin], [0, 0], linewidth=5, color="#b42342")
            ax.axvline(row["end_time_s"] - origin, color="0.3", linestyle="--")
            ax.set(
                xlabel="Seconds after session entry",
                yticks=[0, 1],
                yticklabels=["Native events", "Immobile support"],
                ylim=(-0.5, 1.5),
                title=row["censor_reason"].replace("_", " "),
            )
            fig.suptitle(f"{row['animal']} | {row['session_id']} | geometry and timing only", fontsize=11)
            name = f"geometry_timeline_{i + 1}.png"
            fig.savefig(out / name, dpi=150)
            plt.close(fig)
            plots.append({"file": name, "status": "written", "reason": "No replay content plotted", "opportunity_id": row["opportunity_id"]})
    if not plots:
        plots.append({"file": "", "status": "unavailable", "reason": "No verified geometry/behavior opportunities; no fabricated panel", "opportunity_id": ""})
    table(out, "figure_manifest", plots, ["file", "status", "reason", "opportunity_id"])


def report(output_dir):
    inv, opp = output_dir / "inventory", output_dir / "opportunities"
    im, om = verify_stage(inv), verify_stage(opp)
    core.require(om["fingerprint"]["inventory_manifest_sha256"] == core.file_hash(inv / "manifest.json"), "Mixed upstream inventories")
    fingerprint = {"inventory_manifest_sha256": core.file_hash(inv / "manifest.json"), "opportunities_manifest_sha256": core.file_hash(opp / "manifest.json")}
    out = output_dir / "report"
    if not begin_stage(out, fingerprint, check_inputs=False):
        return read_json(out / "decision.json")
    summary, decision = read_json(inv / "inventory_summary.json"), read_json(opp / "decision.json")
    animals = pd.read_csv(inv / (PREFIX + "inventory_by_animal.csv"))
    gates = pd.read_csv(opp / (PREFIX + "gate_summary.csv"))
    availability = pd.read_csv(inv / (PREFIX + "availability.csv"), keep_default_na=False)

    def count(value):
        return "unavailable" if pd.isna(value) else f"{int(value):,}"

    lines = [
        "# Replay structural updating: feasibility only",
        "",
        f"Decision: **{decision['decision']}**.",
        "",
        "Question: does replay reflect changed connectivity at locally unchanged junctions before renewed physical traversal?",
        "",
        "No new replay decoding, content-effect calculation, biological association, or calibration was performed.",
        "",
        f"Published event-statistic rows: {count(summary['published_events'])}; session records: {count(summary['sessions'])}.",
        "",
        "| Animal | Sessions | Identified event rows |",
        "| --- | ---: | ---: |",
    ]
    for row in animals.to_dict("records"):
        lines.append(f"| {row['animal']} | {count(row['sessions'])} | {count(row['published_events'])} |")
    lines += [
        "",
        f"Identified event rows: {count(summary['resolved_event_rows'])}; unresolved identity rows: {count(summary['unresolved_event_rows'])}. Unresolved rows are retained in the ledger and are not assigned to sessions by row order.",
        "",
        "## Opportunity coverage",
        "",
        decision["reason"],
        "",
    ]
    if not decision["opportunity_measurement_available"]:
        lines += [
            "Eligible opportunity/event counts are **unavailable, not zero**. Empty ledgers mean construction was blocked; they do not demonstrate an absence of replay or suitable behavioral periods.",
            "",
        ]
    else:
        lines += [
            f"Constructed opportunity rows: {decision['opportunity_rows']}. Unique eligible native events: {decision['eligible_unique_events']}. Repeated event-opportunity memberships are reported separately.",
            "",
        ]
    lines += ["## Availability", "", "| Capability | Status |", "| --- | --- |"]
    for key, group in availability.groupby("capability", sort=True):
        lines.append(f"| {key} | {', '.join(sorted(set(group.status)))} |")
    lines += ["", "## Gates", "", "| Gate | Status |", "| --- | --- |"]
    for g in gates.to_dict("records"):
        lines.append(f"| {g['gate']} | {g['status']} |")
    lines += [
        "",
        "## Interpretation and next action",
        "",
        "Rapid barrier conformity is already published. The narrower before-retraversal contrast remains a candidate extension, not an established novel result. See the prior-art ledger for the checked papers and code.",
        "",
        "Goal-changing episodes are permitted, but topology/goal separability must be evaluated, not assumed. Full-rank design and coverage floors would authorize only a subsequent readout assessment, not causal or population-level claims.",
        "",
        "The author request is prepared and UNSENT. Obtain or locate the specified metadata/full-session inputs before attempting further analysis. Do not infer continuous tracking from per-event rat locations, or native event boundaries from elapsed-time summaries.",
        "",
        "Not retraversed does not mean unknown to the animal: barriers were visible. Three animals would constitute a small-cohort discovery study. Power is not established.",
        "",
        "## Provenance",
        "",
        f"Inventory commit: `{im['code_commit']}`.",
        f"Opportunity commit: `{om['code_commit']}`.",
        "This report verified upstream artifact hashes and did not reopen recordings or refit anything.",
    ]
    (out / (PREFIX + "report.md")).write_text("\n".join(lines) + "\n")
    atomic_json(out / "decision.json", decision)
    finish_stage(out, {"inventory_manifest": inv / "manifest.json", "opportunities_manifest": opp / "manifest.json"}, fingerprint, "report", {"non_rescoring": True})
    return decision


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    result.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCES)
    result.add_argument("--verified-metadata", type=Path)
    result.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--stage", choices=["inventory", "opportunities", "report"], required=True)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if args.stage == "inventory":
        result = inventory(args)
    elif args.stage == "opportunities":
        result = opportunities(args)
    else:
        result = report(args.output_dir)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (core.MetadataError, OSError, KeyError, ValueError) as exc:
        print(f"Technical input/artifact error: {exc}", file=sys.stderr)
        sys.exit(2)
