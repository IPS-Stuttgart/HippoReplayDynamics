#!/usr/bin/env python3
"""Audit the public Igata release before any obsolete-route group contrast.

This first stage deliberately does not fit the biological model. An audit route
label is not a verified longitudinal history or a causal inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import socket
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts._provenance import build_script_provenance, file_sha256
from scripts.igata_route_audit import adjacent_transitions, classify_route, file_identity, field_string, grid_labels, inspect_log, modified_levenshtein, stimulation_alignment


def record(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def write_table(path, rows, columns=()):
    pd.DataFrame(rows, columns=None if rows else columns).to_csv(path, index=False)


def log_digest(log):
    x = np.ascontiguousarray(log, dtype="<f8")
    return hashlib.sha256(str(x.shape).encode() + x.tobytes()).hexdigest()


def scan_file(path, dataset, protocol):
    identity = file_identity(path)
    parts = path.relative_to(dataset).parts
    row = dict(**identity, released_group=parts[0], animal=parts[1], record_kind=parts[2], relative_path=path.relative_to(dataset).as_posix(), file_sha256=file_sha256(path))
    rules = protocol["route_rules"]
    try:
        with np.load(path, allow_pickle=False) as d:
            log = d["log"]
            row["log_sha256"] = log_digest(log)
            row.update(inspect_log(log, rules["max_tracking_gap_ms"]))
            if row["log_status"] != "ok":
                raise ValueError(row["log_status"])
            native = "".join(d["locus_string_list"].tolist())
            stay = d["locus_stay_frame_list"]
            row.update(
                native_locus_string=native,
                native_locus_dwell_frames=int(stay.sum()),
                native_dwell_covers_log=bool(int(stay.sum()) == len(log)),
                native_string_matches_full_coordinates=field_string(native) == row["full_coordinate_field_string"],
                n_recorded_units=int(d["spike_hist"].shape[0]),
                lfp_samples=int(d["lfp"].size),
                lfp_duration_ms=float(d["lfp"].size / 2),
                expected_trial_duration_ms=float(d["stim_mat"].size) if "stim_mat" in d.files else float(d["lfp"].size / 2),
                read_status="ok",
                read_failure_reason="",
            )
            if "trial_stim" in d.files and "stim_mat" in d.files:
                row.update(stimulation_alignment(d["trial_stim"], d["stim_mat"], row["expected_trial_duration_ms"]))
            else:
                row.update(n_stimulations=0, stimulation_raster_aligned=False, stimulation_clock="not_released", online_trigger_latency_verified=False)
            row.update(classify_route(native, row, total_limit=rules["optimized_total_string_length_strictly_below"], segment_limit=rules["segment_string_length_strictly_below"]))
            if not row["native_dwell_covers_log"] or not row["native_string_matches_full_coordinates"]:
                row.update(route_label="unclassifiable", route_reason="native_route_tracking_disagreement", optimized_new_success=False)
            sensitivity = []
            for offset in rules["boundary_sensitivity_offsets_mm"]:
                labels = grid_labels(log[:, 1:3], offset)
                # Preserve external box boundaries for this field-only diagnostic.
                labels[(log[:, 1] < 0) | (log[:, 1] > 1000) | (log[:, 2] < 0) | (log[:, 2] > 1000)] = "-"
                s = field_string(labels)
                alt = classify_route(s, row, total_limit=rules["optimized_total_string_length_strictly_below"], segment_limit=rules["segment_string_length_strictly_below"])
                sensitivity.append([offset, alt["route_label"]])
            row["boundary_sensitivity_labels"] = json.dumps(sensitivity)
            row["boundary_route_label_stable"] = all(label == row["route_label"] for _, label in sensitivity)
            gap_labels = []
            for gap in rules["tracking_gap_sensitivity_ms"]:
                info = inspect_log(log, gap)
                label = classify_route(native, info, total_limit=rules["optimized_total_string_length_strictly_below"], segment_limit=rules["segment_string_length_strictly_below"])["route_label"]
                gap_labels.append([gap, label])
            row["tracking_gap_sensitivity_labels"] = json.dumps(gap_labels)
    except (ValueError, KeyError, OSError, zipfile.BadZipFile) as exc:
        row.update(read_status="failed", read_failure_reason=str(exc), route_label="unclassifiable", route_reason="read_or_schema_failure", optimized_new_success=False)
    return row


def block_inventory(rows):
    blocks = defaultdict(list)
    for r in rows:
        if r["record_kind"] == "trial_data":
            blocks[(r["released_group"], r["animal"], r["date"], r["recording_block"])].append(r)
    output = []
    for key, trials in sorted(blocks.items()):
        trials.sort(key=lambda r: r["trial_number"])
        ids = [r["trial_number"] for r in trials]
        missing = sorted(set(range(1, max(ids) + 1)) - set(ids))
        phases = Counter(r.get("active_checkpoint_phase", "unavailable") for r in trials)
        good = [r for r in trials if r.get("read_status") == "ok"]
        overlap = sum(a["end_time_ms"] >= b["start_time_ms"] for a, b in zip(good, good[1:]))
        output.append(
            dict(
                zip(("released_group", "animal", "date", "recording_block"), key),
                n_trials=len(trials),
                min_trial_number=min(ids),
                max_trial_number=max(ids),
                missing_trial_numbers=json.dumps(missing),
                n_missing_trial_numbers=len(missing),
                duplicate_trial_numbers=len(ids) - len(set(ids)),
                within_block_clock_overlaps=overlap,
                checkpoint_phase_counts=json.dumps(dict(phases), sort_keys=True),
                n_new_checkpoint_trials=phases["new_checkpoint_active"],
                n_old_checkpoint_trials=phases["old_checkpoint_active"],
                cross_block_order_verified=False,
                relocation_timestamp_verified=False,
                complete_prior_experience_verified=False,
            )
        )
    return output


def duplicate_inventory(rows):
    by_digest = defaultdict(list)
    for row in rows:
        if row.get("log_sha256"):
            by_digest[row["log_sha256"]].append(row)
    return [
        {"log_sha256": digest, "n_records": len(rr), "animals": "|".join(sorted({r["animal"] for r in rr})), "relative_paths": json.dumps([r["relative_path"] for r in rr]), "cross_animal_duplicate": len({r["animal"] for r in rr}) > 1}
        for digest, rr in sorted(by_digest.items()) if len(rr) > 1
    ]


def instability_rows(rows):
    """Reproduce the distance algorithm on observable adjacent trials, not the published group result."""
    blocks = defaultdict(list)
    for r in rows:
        if r["record_kind"] == "trial_data":
            blocks[(r["animal"], r["date"], r["recording_block"])].append(r)
    out = []
    for _, rr in sorted(blocks.items()):
        rr.sort(key=lambda r: r["trial_number"])
        first = next((r["trial_number"] for r in rr if r.get("optimized_new_success")), None)
        for a, b in zip(rr, rr[1:]):
            if first is None or a["trial_number"] < first or b["trial_number"] != a["trial_number"] + 1:
                continue
            if a.get("read_status") != "ok" or b.get("read_status") != "ok" or b["start_time_ms"] <= a["end_time_ms"]:
                continue
            if a["route_label"] == "unclassifiable" or b["route_label"] == "unclassifiable":
                continue
            distance = modified_levenshtein(a["field_string"], b["field_string"])
            out.append({"animal": a["animal"], "date": a["date"], "recording_block": a["recording_block"], "previous_trial_number": a["trial_number"], "next_trial_number": b["trial_number"], "modified_levenshtein_distance": distance, "behavior_change_distance_ge_2": distance >= 2, "scope": "algorithm_check_not_published_cohort_reproduction"})
    return out


def plot_examples(rows, dataset, output, limit=2):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    candidates = [r for r in rows if r["record_kind"] == "trial_data" and r.get("active_checkpoint_phase") == "new_checkpoint_active"]
    examples = []
    # First release-path examples per label, not strongest effects or favorable animals.
    for label in ("new", "obsolete", "other", "unclassifiable"):
        rr = sorted((r for r in candidates if r["route_label"] == label), key=lambda r: r["relative_path"])[:limit]
        for n, row in enumerate(rr, 1):
            path = output / f"route_{label}_{n}.png"
            with np.load(dataset / row["relative_path"], allow_pickle=False) as d:
                log = d["log"]
            fig, ax = plt.subplots(figsize=(6, 6))
            for v in range(0, 1001, 200):
                ax.axvline(v, color=".8", linewidth=.6)
                ax.axhline(v, color=".8", linewidth=.6)
            valid = (log[:, 3] > 0) & np.isfinite(log[:, 1:3]).all(axis=1)
            xy = log[:, 1:3].copy()
            xy[~valid] = np.nan
            ax.plot(xy[:, 0], xy[:, 1], color=".25", linewidth=1)
            for name, letter in SPECIAL_POINTS.items():
                index = ord(letter) - 65
                x, y = 100 + index % 5 * 200, 100 + index // 5 * 200
                ax.scatter(x, y, s=45)
                ax.text(x + 30, y + 30, name, fontsize=9)
            ax.set(xlim=(-300, 1300), ylim=(-50, 1100), xlabel="Release x (mm)", ylabel="Release y (mm)", title=f"{row['animal']} / {row['recording_block']} / trial {row['trial_number']}\n{label}: {row['route_reason']}")
            ax.set_aspect("equal")
            fig.tight_layout()
            fig.savefig(path, dpi=130)
            plt.close(fig)
            examples.append({"relative_path": row["relative_path"], "animal": row["animal"], "route_label": label, "figure": path.name, "selection_rule": "first_lexical_release_path_per_label", "figure_sha256": file_sha256(path)})
    return examples


SPECIAL_POINTS = {"Start": "U", "Goal": "E", "C1 old": "S", "C2 new": "G"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", type=Path, required=True, help="Extracted release containing readme.txt and group directories")
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--source-main-xml", type=Path, required=True)
    p.add_argument("--source-si-pdf", type=Path, required=True)
    p.add_argument("--source-si-text", type=Path, required=True)
    p.add_argument("--protocol", type=Path, default=ROOT / "docs/igata_obsolete_route_protocol.json")
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    dataset, output = a.dataset_root.resolve(), a.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        raise SystemExit("Completed output exists; choose a new run directory.")
    protocol = json.loads(a.protocol.read_text())
    inputs = {"protocol": a.protocol, "archive": a.archive, "readme": dataset / "readme.txt", "geometry": dataset / "virtual_maze_field.py", "main_xml": a.source_main_xml, "supplement_pdf": a.source_si_pdf, "supplement_text": a.source_si_text}
    if any(not path.is_file() for path in inputs.values()):
        raise SystemExit("Missing mandatory release/source input")
    provenance = build_script_provenance(input_paths=inputs, cwd=ROOT)
    # Use the actual checkout, not an inherited CI environment SHA.
    import subprocess

    provenance["code_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if provenance["git_dirty"]:
        raise SystemExit("Commit the isolated checkout before running.")
    for key, expected in (("readme", "release_readme_sha256"), ("geometry", "release_geometry_sha256"), ("supplement_pdf", "supplement_pdf_sha256"), ("supplement_text", "supplement_text_sha256")):
        if provenance["input_file_sha256"][key] != protocol[expected]:
            raise SystemExit(f"Source fingerprint mismatch: {key}")
    checkpoint = output / "checkpoint.json"
    state = json.loads(checkpoint.read_text()) if checkpoint.exists() else {"code_commit": provenance["code_commit"], "input_hashes": provenance["input_file_sha256"], "rows": []}
    if state["code_commit"] != provenance["code_commit"] or state["input_hashes"] != provenance["input_file_sha256"]:
        raise SystemExit("Checkpoint inputs/code changed")
    done = {r["relative_path"]: r for r in state["rows"]}
    paths = sorted(dataset.glob("*/*/*_data/*.npz"))
    if not paths:
        raise SystemExit("No trial/return files found")
    for i, path in enumerate(paths, 1):
        relative = path.relative_to(dataset).as_posix()
        if relative in done:
            if file_sha256(path) != done[relative]["file_sha256"]:
                raise SystemExit(f"Checkpoint file changed: {relative}")
            continue
        row = scan_file(path, dataset, protocol)
        state["rows"].append(row)
        if i % 50 == 0 or i == len(paths):
            # Infinity represents missing support internally; JSON records explicit null instead.
            for r in state["rows"]:
                for k, v in list(r.items()):
                    if isinstance(v, float) and not np.isfinite(v):
                        r[k] = None
            record(checkpoint, state)
            record(output / "progress.json", {"status": "auditing", "processed_files": len(state["rows"]), "total_files": len(paths), "updated_at_utc": datetime.now(UTC).isoformat()})
            print(f"Audited {len(state['rows'])}/{len(paths)} files", flush=True)
    rows = state["rows"]
    with zipfile.ZipFile(a.archive) as z:
        members = {n for n in z.namelist() if n.endswith(".npz")}
    extracted = {r["relative_path"] for r in rows}
    archive_match = members == extracted and all(r["file_sha256"] for r in rows)
    write_table(output / "trial_inventory.csv", rows)
    blocks = block_inventory(rows)
    write_table(output / "block_inventory.csv", blocks)
    duplicates = duplicate_inventory(rows)
    write_table(output / "duplicate_record_audit.csv", duplicates, ["log_sha256", "n_records", "animals", "relative_paths", "cross_animal_duplicate"])
    trials = [r for r in rows if r["record_kind"] == "trial_data"]
    write_table(output / "trial_route_labels_audit.csv", trials)
    cohort = []
    for group, expected in protocol["expected_animals"].items():
        for animal in expected:
            rr = [r for r in trials if r["released_group"] == group and r["animal"] == animal]
            bb = [b for b in blocks if b["released_group"] == group and b["animal"] == animal]
            cohort.append({"released_group": group, "animal": animal, "n_trials": len(rr), "n_blocks": len(bb), "n_new_checkpoint_blocks": sum(b["n_new_checkpoint_trials"] > 0 for b in bb), "missing_trial_numbers": sum(b["n_missing_trial_numbers"] for b in bb), "n_read_failures": sum(r["read_status"] != "ok" for r in rr), "n_unclassifiable_trials": sum(r["route_label"] == "unclassifiable" for r in rr), "n_stimulations": sum(r.get("n_stimulations", 0) for r in rr), "recording_order_verified": False, "primary_eligible": False, "exclusion_reason": "complete_trial_history_and_recording_order_unverified"})
    write_table(output / "cohort_inventory.csv", cohort)
    candidate_trials = [r for r in trials if r["released_group"] in protocol["expected_animals"]]
    transitions = adjacent_transitions(candidate_trials)
    write_table(output / "observable_transition_audit.csv", transitions, ["animal", "released_group", "date", "recording_block", "previous_trial_number", "next_trial_number", "previous_relative_path", "next_relative_path", "next_route_label", "next_route_reason", "primary_eligible", "eligibility_reason"])
    distances = instability_rows(candidate_trials)
    write_table(output / "published_distance_algorithm_check.csv", distances, ["animal", "date", "recording_block", "previous_trial_number", "next_trial_number", "modified_levenshtein_distance", "behavior_change_distance_ge_2", "scope"])
    examples = plot_examples(candidate_trials, dataset, output)
    write_table(output / "route_examples_manifest.csv", examples, ["relative_path", "animal", "route_label", "figure", "selection_rule", "figure_sha256"])
    review = protocol["source_review"]
    gates = [
        ("archive_members_match_extracted_inventory", archive_match, f"{len(members)} archive / {len(extracted)} extracted files; raw-content hashes recorded separately"),
        ("twelve_feedback_animals_present", all(c["n_trials"] > 0 for c in cohort), "6 immediate-folder and 6 delayed-folder animals; released labels, not latency measurements"),
        ("source_defined_geometry_and_route_rule", True, "SI p6: U/E/S/G; segment length <8; optimized total string length <12"),
        ("selective_relapse_analysis_not_found_in_reviewed_sources", not review["existing_selective_obsolete_vs_other_test_found"], "Bounded paper/SI/release review, not proof of worldwide novelty"),
        ("complete_trial_numbers", all(b["n_missing_trial_numbers"] == 0 for b in blocks if b["released_group"] in protocol["expected_animals"]), "Never bridge missing trial indices"),
        ("recording_block_order_verified", review["recording_block_order_documented"], "Clock resets and additional blocks cannot be concatenated from filenames"),
        ("relocation_boundary_verified", review["reward_relocation_trial_documented"], "Checkpoint flags identify current task lattice, not a dated C1-to-C2 switch"),
        ("stimulation_raster_alignment", all(r.get("stimulation_raster_aligned", False) and r.get("stimulation_within_trial_support", False) for r in candidate_trials), "trial_stim must agree with 1ms stim_mat; online ripple trigger latency is separate"),
        ("online_stimulation_latency_verified", review["online_ripple_trigger_timestamps_released"], "No native online-trigger timestamp field documented; delivered pulse times alone cannot measure 250ms delay"),
        ("route_labels_identifiable_for_every_feedback_trial", all(r["route_label"] != "unclassifiable" for r in candidate_trials), "Missing/contradictory tracking stays unclassifiable, not other"),
        ("published_cohort_comparison_reproduced", False, "Distance algorithm reproduced only; Fig5 grouping/chronology not independently reproduced"),
    ]
    go = all(g[1] for g in gates)
    gates.append(("go_for_model_validation", go, "No new group contrast until all upstream gates pass"))
    write_table(output / "gate_summary.csv", [{"gate": name, "status": "pass" if ok else "fail", "detail": detail} for name, ok, detail in gates])
    decision = {"decision": "go_to_frozen_design_validation" if go else "stop_unverified_public_trial_design", "biological_contrast_run": False, "validation_replicates_run": 0, "published_group_comparison_reproduced": False, "observable_conditional_transitions": len(transitions), "n_primary_eligible_animals": 0, "failed_gates": [name for name, ok, _ in gates if not ok], "interpretation": "Feasibility stop, not a negative relapse result. No data exclusion or endpoint substitution rescues the proposed primary test."}
    record(output / "decision.json", decision)
    summary = [
        "# Igata obsolete-route relapse feasibility", "", f"Protocol: `{protocol['protocol_id']}`. Decision: **{decision['decision']}**.", "",
        f"Release: {len(rows)} NPZ files, {len(trials)} task trials, {len(cohort)} expected feedback animals. {len(transitions)} observable within-block post-new-route transitions are audit observations, not a verified primary cohort.", "",
        "## Source and novelty audit", "The paper and full supplement were reviewed. Fig5/S15 already report trajectory instability after first optimized-route use. The reviewed material does not report the proposed obsolete-versus-other conditional transition test. This is a candidate extension, not a global novelty certification.",
        "", "Sources: [article](https://doi.org/10.1073/pnas.2011266118), [public release](https://doi.org/10.17632/4xk5w69yr5.1), [supplement package](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC7817193/supplementaryFiles).",
        "", "## Observable route rules", "The release uses uppercase U (start), E (goal), S (old C1), G (new C2). Do not confuse lowercase external-box labels with SI lowercase lattice labels. New routes require the source's short S-C2-G criterion, checkpoint arrival and later goal-port contact. Obsolete routes require a short start-old-checkpoint-goal attempt before new-checkpoint correction. Mixed old/new without an early goal is other. Tracking/log failures are unclassifiable. These audit labels do not directly measure reward delivery, memory state, or replay content.",
        "", "## Go/no-go gates", *[f"- {name}: {'pass' if ok else 'FAIL'}. {detail}" for name, ok, detail in gates],
        "", "## What was and was not executed", "Input fingerprints, trial/block/animal inventories, missing/duplicate-trial audits, pulse-raster alignment, route labels, boundary/gap sensitivity, route figures, and the published distance algorithm were executed. The published group-level instability comparison was NOT reproduced. The hierarchical multinomial model, 4 x 1,000 actual-design validation simulations, native primary contrast, and leave-one-animal-out estimates were NOT run: their required design is unresolved. Empty tables are not described as successful validation.",
        "", "## Claim boundary", "No conclusion about selective relapse, nonspecific instability, a null treatment effect, replay-sequence causation, or Bayesian smoothing follows from this audit. Conditional post-success effects would not be total causal treatment effects even with complete data. Public/existing data only; no author request or new recordings.",
        "", "## Required to reopen", "A publicly documented mapping of recording blocks, reset clocks, relocation boundary, missing trials, and task contingencies; independently checkable stimulation-trigger timing; validated route accounting; then frozen actual-design calibration before a single native contrast. Do not infer any of these from outcome patterns or favorable group results.", "",
    ]
    (output / "go_no_go_report.md").write_text("\n".join(summary))
    outputs = {p.name: file_sha256(p) for p in sorted(output.iterdir()) if p.is_file() and p.name not in {"manifest.json", "checkpoint.json", "progress.json"}}
    record(output / "manifest.json", {**provenance, "created_at_utc": datetime.now(UTC).isoformat(), "host": socket.gethostname(), "protocol_id": protocol["protocol_id"], "n_input_npz": len(rows), "input_npz_inventory": "trial_inventory.csv", "archive_member_inventory_matches": archive_match, "source_review": review, "decision": decision, "output_file_sha256": outputs, "raw_data_copied": False})
    record(output / "progress.json", {"status": "completed", "decision": decision["decision"], "processed_files": len(rows), "finished_at_utc": datetime.now(UTC).isoformat()})
    print(json.dumps(decision), flush=True)


if __name__ == "__main__":
    main()
