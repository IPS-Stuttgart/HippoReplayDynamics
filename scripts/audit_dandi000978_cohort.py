#!/usr/bin/env python3
"""Inventory verified DANDI000978 data without resolving ambiguous unit labels."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import socket
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from acquire_dandi000978 import destination, write_json
from preflight_dandi000978 import inspect, text


def verify_acquisition_record(root, manifest, status):
    """Reuse acquisition hashes, explicitly not a new read/hash of all raw bytes."""
    if status.get("status") != "complete_verified":
        raise ValueError("Acquisition is not complete and verified")
    if manifest.get("dataset") != "000978" or manifest.get("version") != "0.240511.0307":
        raise ValueError("Wrong dataset or version")
    records = status.get("verified", [])
    by_path = {row["path"]: row for row in records}
    if len(records) != len(by_path):
        raise ValueError("Duplicate verification records")
    assets = manifest["assets"]
    if len({a["path"] for a in assets}) != len(assets) or set(by_path) != {a["path"] for a in assets}:
        raise ValueError("Manifest and verification record coverage differ")
    for asset in assets:
        row = by_path[asset["path"]]
        if row["sha256"] != asset["digest"]["dandi:sha2-256"] or row["size"] != asset["size"]:
            raise ValueError("Recorded checksum/size disagrees with frozen manifest")
        if destination(root, asset).stat().st_size != asset["size"]:
            raise ValueError("Current local size differs from verified asset")


def unit_epoch_counts(arrays, epochs):
    rows = []
    starts = np.r_[0, arrays["spike_ends"][:-1]]
    for i, (a, b) in enumerate(zip(starts, arrays["spike_ends"], strict=True)):
        spikes = arrays["spikes"][a:b]
        for ep in epochs.itertuples():
            count = int(np.searchsorted(spikes, ep.stop_time_s, side="left") - np.searchsorted(spikes, ep.start_time_s, side="left"))
            rows.append({"unit_id": int(arrays["unit_ids"][i]), "epoch_index": ep.epoch_index, "n_spikes": count, "role_from_trial_presence": ep.role_from_trial_presence})
    return pd.DataFrame(rows)


def file_relationships(records):
    columns = [
        "animal",
        "first_file",
        "second_file",
        "clock_order",
        "gap_s",
        "same_identifier",
        "same_unit_ids",
        "same_unit_electrode_references",
        "n_common_unit_ids",
        "n_common_ids_with_different_references",
        "cross_file_unit_identity_verified",
    ]
    rows = []
    for a, b in itertools.combinations(records, 2):
        if a["animal"] != b["animal"]:
            continue
        if a["start"] > b["start"]:
            a, b = b, a
        amap, bmap = dict(zip(a["ids"], a["refs"], strict=True)), dict(zip(b["ids"], b["refs"], strict=True))
        common = sorted(set(amap) & set(bmap))
        rows.append(
            {
                "animal": a["animal"],
                "first_file": a["file"],
                "second_file": b["file"],
                "clock_order": "nonoverlapping_successive_intervals" if a["stop"] <= b["start"] else "overlapping_intervals",
                "gap_s": b["start"] - a["stop"],
                "same_identifier": a["identifier"] == b["identifier"],
                "same_unit_ids": a["ids"] == b["ids"],
                "same_unit_electrode_references": a["refs"] == b["refs"],
                "n_common_unit_ids": len(common),
                "n_common_ids_with_different_references": sum(amap[i] != bmap[i] for i in common),
                # Equal IDs or tetrode references alone do not identify sorted clusters.
                "cross_file_unit_identity_verified": False,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def metadata_fingerprint(file):
    h = hashlib.sha256()
    for path in (
        "units/id",
        "units/electrodes",
        "units/spike_times_index",
        "intervals/epoch intervals/start_time",
        "intervals/epoch intervals/stop_time",
        "intervals/trials/id",
        "intervals/trials/start_time",
        "intervals/trials/stop_time",
    ):
        value = np.asarray(file[path][()])
        h.update(path.encode())
        h.update(str(value.dtype).encode())
        h.update(str(value.shape).encode())
        h.update(value.tobytes())
    return h.hexdigest()


def audit_cohort(root, out):
    manifest_path = root / "metadata/asset_manifest.json"
    status_path = root / "download_status.json"
    manifest = json.loads(manifest_path.read_text())
    status = json.loads(status_path.read_text())
    verify_acquisition_record(root, manifest, status)
    out.mkdir(parents=True, exist_ok=False)
    inventories, epochs_all, trials_all, units_all, activity_all, identities = [], [], [], [], [], []
    for asset in manifest["assets"]:
        path = destination(root, asset)
        print(f"Inspecting {path.name}", flush=True)
        with h5py.File(path, "r") as f:
            report, epochs, trials, units, arrays = inspect(f)
            report["file"] = path.name
            report["identifier"] = text(f["identifier"][()])
            report["session_start_time_unvalidated_calendar_date"] = text(f["session_start_time"][()])
            report["expected_sha256"] = asset["digest"]["dandi:sha2-256"]
            report["inspected_metadata_sha256"] = metadata_fingerprint(f)
            report["local_size_bytes"] = path.stat().st_size
            report["literal_ca1_units"] = int((units.nwb_row_region == "CA1").sum())
            report["literal_pfc_units"] = int((units.nwb_row_region == "PFC").sum())
            alt = units.alternative_tetrode_id_region_not_author_confirmed
            report["unconfirmed_tetrode_id_ca1_units"] = int((alt == "CA1").sum())
            report["unconfirmed_tetrode_id_pfc_units"] = int((alt == "PFC").sum())
            report["anatomical_identity_confirmed"] = False
            report["region_specific_RUN_scoring_allowed"] = False
            report["full_asset_rehashed_in_this_audit"] = False
            report["checksum_basis"] = "completed_acquisition_record_with_current_size_rechecked"
            report["nwb_row_region_counts"] = json.dumps(report["nwb_row_region_counts"], sort_keys=True)
            report["position_shape"] = json.dumps(report["position_shape"])
            activity = unit_epoch_counts(arrays, epochs)
            for frame in (epochs, trials, units, activity):
                frame.insert(0, "file", path.name)
                frame.insert(0, "animal", report["animal"])
            run_activity = activity[activity.role_from_trial_presence == "RUN"]
            run_epochs = epochs[epochs.role_from_trial_presence == "RUN"]
            report["n_units_spiking_in_all_RUN_epochs"] = int((run_activity.groupby("unit_id").n_spikes.apply(lambda x: bool((x > 0).all()))).sum())
            report["trial_count_is_RUN_support_not_sleep_staging"] = True
            report["n_RUN_epochs_with_position_and_spikes"] = int(((run_epochs.n_position_samples > 0) & (run_epochs.n_spikes > 0)).sum())
            identities.append(
                {
                    "animal": report["animal"],
                    "file": path.name,
                    "identifier": report["identifier"],
                    "start": float(epochs.start_time_s.min()),
                    "stop": float(epochs.stop_time_s.max()),
                    "ids": arrays["unit_ids"].tolist(),
                    "refs": f["units/electrodes"][()].tolist(),
                }
            )
        inventories.append(report)
        epochs_all.append(epochs)
        trials_all.append(trials)
        units_all.append(units)
        activity_all.append(activity)
    inventory = pd.DataFrame(inventories)
    relationships = file_relationships(identities)
    tables = {
        "cohort_inventory": inventory,
        "epoch_inventory": pd.concat(epochs_all),
        "trial_inventory": pd.concat(trials_all),
        "unit_reference_audit": pd.concat(units_all),
        "unit_epoch_coverage": pd.concat(activity_all),
        "same_subject_file_relationships": relationships,
    }
    for name, frame in tables.items():
        frame.to_csv(out / f"{name}.csv", index=False)
    source = Path(__file__).resolve()
    commit = subprocess.check_output(["git", "-C", str(source.parents[1]), "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(subprocess.check_output(["git", "-C", str(source.parents[1]), "status", "--porcelain"], text=True).strip())
    provenance = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "host": socket.gethostname(),
        "dataset": manifest["dataset"],
        "version": manifest["version"],
        "dataset_root": str(root),
        "command_line": sys.argv,
        "code_commit": commit,
        "git_dirty": dirty,
        "code_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "preflight_helper_sha256": hashlib.sha256(source.with_name("preflight_dandi000978.py").read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "acquisition_record_sha256": hashlib.sha256(status_path.read_bytes()).hexdigest(),
        "files": len(inventory),
        "animals": int(inventory.animal.nunique()),
        "units_are_file_local_and_must_not_be_summed_as_unique_across_ZT2": True,
        "full_asset_rehashed_in_this_audit": False,
        "anatomical_identity_confirmed": False,
        "replay_scored": False,
        "artifacts_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.glob("*.csv"))},
    }
    write_json(out / "cohort_audit_manifest.json", provenance)
    lines = [
        "# DANDI000978 cohort metadata audit",
        "",
        "## Result",
        "",
        f"{len(inventory)} verified local NWB assets from {inventory.animal.nunique()} rats were inspected. Acquisition SHA-256 records were reused; current sizes were checked. This audit did not rehash all raw bytes.",
        "",
        "| Subject/file | Units | Trials | RUN epochs | References with differing regional interpretations |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in inventory.itertuples():
        lines.append(f"| {row.file} | {row.unit_count} | {row.trial_count} | {row.run_epoch_count_from_trials} | {row.unit_reference_hypotheses_disagree_count} |")
    lines.extend(
        [
            "",
            "## Interpretation boundaries",
            "",
            "- The NWB standard defines DynamicTableRegion values as zero-based table-row references. A second, explicitly unconfirmed interpretation treats these values as tetrode IDs. Their disagreement is a warning, not proof that either correction is valid.",
            "- No anatomy is inferred from firing rate, spatial tuning, decoder performance, or replay outcomes. No regional replay or RUN decoder has been fitted here.",
            "- ZT2 is one animal with two assets. Successive clocks and a shared identifier do not prove sorted-unit identity across those assets. Do not concatenate spike trains by row number.",
            "- Trial presence identifies RUN support. The complement is a rest candidate, not validated NREM staging. Calendar session dates have not been validated against the recording dates.",
            "- A unit firing in every RUN epoch is an availability check, not proof of sorting stability.",
            "- Source exporter/author confirmation is required before the planned CA1/PFC experiment. Raw NWB files remain unchanged.",
            "",
            "## Planned scientific test",
            "",
            "Test whether recording-coverage reductions that remove a CA1 event's trajectory label also remove content-specific evidence in a separately recorded cortical population. Independent RUN decoding must pass first, and replay selection and controls must be frozen before viewing replay outcomes. Cortex is not ground truth.",
            "",
            "## Related published result",
            "",
            "Shin and Jadhav already report that CA1 replay associated with coordinated CA1/PFC ripples is less sequential, and that place-template shuffling degrades it more. Our potential contribution must therefore concern controlled recording coverage and independently tested content, not discovering cross-area reactivation or lower sequentiality: https://pmc.ncbi.nlm.nih.gov/articles/PMC11233241/",
            "",
        ]
    )
    (out / "cohort_metadata_audit.md").write_text("\n".join(lines))
    question = [
        "# Draft clarification request (not sent)",
        "",
        "Subject: Unit-to-tetrode mapping in DANDI000978 version 0.240511.0307",
        "",
        "We are reusing Single Day W-Track Learning and want to confirm the CA1/PFC assignment without inferring anatomy from the neural responses.",
        "",
        "In JS14, /general/extracellular_ephys/electrodes has 62 rows (Ref OFF/Ref ON entries per tetrode), while /units/electrodes contains values 4 through 32 and declares a DynamicTableRegion into that table. Reading these as zero-based row indices gives 31 CA1 and 41 PFC units. Reading them as one-based tetrode IDs gives 52 CA1 and 20 PFC units; 35 of 72 assignments differ. The same ambiguity affects the other eight assets.",
        "",
        "1. Are the stored values electrode-table row indices, tetrode IDs, or another index? Could you share the original NWB export script or a unit-ID-to-tetrode/region crosswalk?",
        "2. ZT2's two files have 233 and 240 units and successive recording-clock intervals. Are unit IDs preserved between the files, or is a separate sorted-cluster crosswalk needed?",
        "3. Are validated NREM intervals and ripple event tables available separately? The NWB interval groups we found contain epochs and trials only.",
        "",
        "We have not changed the labels or run regional replay decoding. A short confirmation or conversion-code pointer would let us preserve the correct anatomical and cell-identity interpretation.",
        "",
    ]
    (out / "author_clarification_draft.md").write_text("\n".join(question))
    print(json.dumps(provenance, indent=2), flush=True)
    return inventory, relationships


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    audit_cohort(args.dataset_root.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()
