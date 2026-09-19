#!/usr/bin/env python3
"""Inventory published DANDI 001695 navigation files without scoring replay."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import socket
import sys
import time
import urllib.request
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _provenance import build_script_provenance

VERSION = "0.260319.2023"
API = f"https://api.dandiarchive.org/api/dandisets/001695/versions/{VERSION}"
CATALOG_SHA256 = "d5b66e12bfbc2ff6eea508948cc728da755cafde8685d647ff09d5da1a4fdf22"
POSITION = "processing/behavior/AnimalPosition/Position"
LFP = "processing/ecephys/LFP/Best_Ripple_channel_LFP_CA1"


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def json_value(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    if isinstance(value, (np.ndarray, np.generic)):
        return json_value(value.tolist())
    if isinstance(value, (list, tuple)):
        return [json_value(x) for x in value]
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def write_json(path, value):
    Path(path).write_text(json.dumps(json_value(value), indent=2, allow_nan=False) + "\n")


def safe_path(root, relative):
    rel = PurePosixPath(relative)
    if rel.is_absolute() or not rel.parts or ".." in rel.parts or "\\" in relative or ":" in relative:
        raise ValueError("unsafe asset path")
    result = (Path(root) / str(rel)).resolve()
    if not result.is_relative_to(Path(root).resolve()):
        raise ValueError("asset path escapes root")
    return result


def verify_file(path, size, digest):
    if not re.fullmatch(r"[0-9a-f]{64}", digest) or size <= 0:
        raise ValueError("invalid published checksum/size")
    if Path(path).stat().st_size != size or sha256(path) != digest:
        raise ValueError(f"published size/SHA256 mismatch: {path}")


def fetch_asset(asset, metadata, root, download=False):
    if metadata["path"] != asset["path"] or metadata["identifier"] != asset["asset_id"]:
        raise ValueError("published metadata does not match catalog identity")
    size = int(metadata["contentSize"])
    if size != int(asset["size"]):
        raise ValueError("catalog size differs from published metadata")
    digest = metadata["digest"]["dandi:sha2-256"]
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("published SHA256 required")
    path = safe_path(root, asset["path"])
    if not path.exists():
        if not download:
            raise FileNotFoundError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(path.name + f".part-{os.getpid()}")
        url = f"https://api.dandiarchive.org/api/assets/{asset['asset_id']}/download/"
        try:
            with urllib.request.urlopen(url, timeout=120) as response, temp.open("xb") as target:
                while block := response.read(1024 * 1024):
                    target.write(block)
                    if target.tell() > size:
                        raise ValueError("download exceeds published size")
            verify_file(temp, size, digest)
            os.link(temp, path)
        finally:
            if temp.exists():
                temp.unlink()
    verify_file(path, size, digest)
    return path


def validate_spikes(ids, indices, times):
    ids, indices, times = np.asarray(ids), np.asarray(indices), np.asarray(times)
    if ids.ndim != 1 or not len(ids) or len(np.unique(ids)) != len(ids):
        raise ValueError("unit IDs must be unique and nonempty")
    if indices.shape != ids.shape or indices.dtype.kind not in "iu":
        raise ValueError("invalid ragged spike index")
    stops = indices.astype(np.int64)
    if (stops < 0).any() or (np.diff(stops) < 0).any() or stops[-1] != len(times):
        raise ValueError("ragged spike index does not cover timestamps")
    if times.ndim != 1 or not len(times) or not np.isfinite(times).all():
        raise ValueError("missing/nonfinite spike timestamps")
    start = 0
    for end in stops:
        if (np.diff(times[start:end]) < 0).any():
            raise ValueError("spike timestamps are not sorted within unit")
        start = end


def inventory(path):
    row, schema, intervals = {}, [], []
    with h5py.File(path, "r") as file:
        row["animal"] = json_value(file["general/subject/subject_id"][()])
        row["session_start_time"] = json_value(file["session_start_time"][()])
        row["session_description"] = json_value(file["session_description"][()])
        ids = file["units/id"][()]
        area = file["units/cell_area"].asstr()[()]
        cell_type = file["units/cell_type"].asstr()[()]
        if len(area) != len(ids) or len(cell_type) != len(ids):
            raise ValueError("unit annotation length mismatch")
        spikes = file["units/spike_times"][()]
        validate_spikes(ids, file["units/spike_times_index"][()], spikes)
        row.update(n_units=len(ids), n_spikes=len(spikes), spike_start_s=float(spikes.min()), spike_end_s=float(spikes.max()))
        unit_rows = [
            {"region": a, "cell_type": c, "n_units": n}
            for (a, c), n in sorted(Counter(zip(area, cell_type, strict=True)).items())
        ]
        for region in ("CA1", "CA3", "RSC"):
            row[f"n_{region}_units"] = int((area == region).sum())
            row[f"n_{region}_pyramidal"] = int(((area == region) & (cell_type == "Pyramidal Cell")).sum())
        clock = np.asarray(file[f"{POSITION}/timestamps"][()], float)
        position = np.asarray(file[f"{POSITION}/data"][()], float)
        if clock.ndim != 1 or len(clock) < 2 or position.shape != (len(clock), 2):
            raise ValueError("position/clock shape mismatch")
        if not np.isfinite(clock).all() or (np.diff(clock) <= 0).any():
            raise ValueError("invalid position clock")
        if not np.isfinite(position).all(axis=1).any():
            raise ValueError("no finite position")
        attrs = file[f"{POSITION}/data"].attrs
        row.update(
            n_position_samples=len(clock), position_start_s=float(clock[0]), position_end_s=float(clock[-1]),
            position_valid_fraction=float(np.isfinite(position).all(axis=1).mean()),
            position_max_timestamp_gap_s=float(np.diff(clock).max()),
            position_gaps_over_one_second=int((np.diff(clock) > 1).sum()),
            position_declared_unit=json_value(attrs.get("unit", "missing")),
            position_conversion=float(attrs.get("conversion", 1)), position_offset=float(attrs.get("offset", 0)),
            position_raw_x_span=float(np.nanmax(position[:, 0]) - np.nanmin(position[:, 0])),
            position_raw_y_span=float(np.nanmax(position[:, 1]) - np.nanmin(position[:, 1])),
        )
        row["position_scaling_requires_review"] = row["position_declared_unit"] in ("cm", "centimeters") and row["position_conversion"] != 1
        row["speed_position_clocks_equal"] = bool(np.array_equal(file["processing/behavior/Speed/timestamps"][()], clock))
        lfp_start = float(file[f"{LFP}/starting_time"][()])
        lfp_rate = float(file[f"{LFP}/starting_time"].attrs["rate"])
        if not np.isfinite([lfp_start, lfp_rate]).all() or lfp_rate <= 0:
            raise ValueError("invalid LFP clock")
        lfp_stop = lfp_start + len(file[f"{LFP}/data"]) / lfp_rate
        row.update(lfp_start_s=lfp_start, lfp_stop_s=lfp_stop, lfp_rate_hz=lfp_rate)
        row["position_inside_lfp_clock"] = bool(clock[0] >= lfp_start and clock[-1] < lfp_stop)
        row["spikes_inside_lfp_clock"] = bool(spikes.min() >= lfp_start and spikes.max() < lfp_stop)
        interval_paths = []

        def visit(name, obj):
            if name.startswith("specifications"):
                return
            if isinstance(obj, h5py.Dataset):
                item = {"path": name, "shape": list(obj.shape), "dtype": str(obj.dtype)}
                if name.startswith(("intervals", "epochs", "processing/behavior")):
                    item["attributes"] = json_value(dict(obj.attrs))
                if obj.shape == () and h5py.check_string_dtype(obj.dtype) is not None:
                    item["value"] = json_value(obj[()])
                schema.append(item)
            elif json_value(obj.attrs.get("neurodata_type")) == "TimeIntervals":
                interval_paths.append(name)
                starts, stops = np.asarray(obj["start_time"][()]), np.asarray(obj["stop_time"][()])
                valid = starts.shape == stops.shape and starts.ndim == 1
                valid = valid and bool(np.isfinite(starts).all() and np.isfinite(stops).all() and (stops > starts).all())
                labels = {}
                for key, value in obj.items():
                    if isinstance(value, h5py.Dataset) and h5py.check_string_dtype(value.dtype) is not None:
                        labels[key] = dict(Counter(np.asarray(value.asstr()[()]).reshape(-1).tolist()))
                intervals.append({
                    "path": name, "n_intervals": len(starts), "valid_intervals": valid,
                    "inside_lfp_clock": bool(valid and (starts >= lfp_start).all() and (stops <= lfp_stop).all()),
                    "description": json_value(obj.attrs.get("description", "")), "label_counts": labels,
                })

        file.visititems(visit)
        row["interval_table_paths"] = ";".join(sorted(interval_paths))
        row["non_sleep_interval_table_paths"] = ";".join(sorted(p for p in interval_paths if p != "intervals/SleepStates"))
        row["has_epochs_group"] = "epochs" in file
        row["has_trials_table"] = "intervals/trials" in file
        sleep = next((x for x in intervals if x["path"] == "intervals/SleepStates"), None)
        for label in ("WAKE", "NREM", "REM", "Ripple"):
            row[f"n_native_{label}_intervals"] = (sleep or {}).get("label_counts", {}).get("state", {}).get(label, 0)
        row["intervals_valid_and_inside_lfp"] = bool(intervals and all(x["valid_intervals"] and x["inside_lfp_clock"] for x in intervals))
        row["basic_clock_checks_passed"] = all(row[k] for k in (
            "speed_position_clocks_equal", "position_inside_lfp_clock", "spikes_inside_lfp_clock", "intervals_valid_and_inside_lfp",
        ))
        # Gaps and dates are NOT ground-truth maze labels.
        row["context_validation_status"] = "requires_author_confirmed_epoch_assignment"
    return row, {"schema": schema, "unit_counts": unit_rows, "interval_tables": intervals}


def run(args):
    if sha256(args.catalog_json) != CATALOG_SHA256:
        raise ValueError("catalog differs from the frozen published response")
    catalog = json.loads(args.catalog_json.read_text())
    if catalog["next"] is not None or catalog["count"] != 22 or len(catalog["results"]) != 22:
        raise ValueError("incomplete published catalog")
    assets = sorted((a for a in catalog["results"] if "behavior+ecephys" in a["path"]), key=lambda a: a["path"])
    if len(assets) != 15 or len({a["asset_id"] for a in assets}) != 15:
        raise ValueError("navigation catalog coverage changed")
    provenance = build_script_provenance(input_paths={"published_catalog": args.catalog_json}, cwd=ROOT)
    if provenance["git_dirty"] or not re.fullmatch(r"[0-9a-f]{40}", provenance["code_commit"]):
        raise ValueError("a clean committed producer is required")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.output_dir / "manifest.json", {
        **provenance, "created_at_utc": datetime.now(UTC).isoformat(), "hostname": socket.gethostname(),
        "dandiset": "001695", "version": VERSION, "assets": assets,
        "analysis_scope": "metadata_and_structural_integrity_only_no_replay_or_context_scoring",
        "clock_caveat": "range overlap does not prove fine-scale synchronization",
        "unit_caveat": "region labels do not establish across-epoch cell stability",
    })
    rows, unit_rows = [], []
    for asset in assets:
        start = time.monotonic()
        row = {"asset_id": asset["asset_id"], "path": asset["path"], "status": "failed", "failure_reason": ""}
        try:
            with urllib.request.urlopen(f"{API}/assets/{asset['asset_id']}/", timeout=90) as response:
                metadata = json.load(response)
            write_json(args.output_dir / f"{asset['asset_id']}_published_metadata.json", metadata)
            path = fetch_asset(asset, metadata, args.dataset_root, download=args.download)
            details, schema = inventory(path)
            row.update(details, status="inspected", raw_sha256=metadata["digest"]["dandi:sha2-256"], raw_bytes=path.stat().st_size)
            write_json(args.output_dir / f"{asset['asset_id']}_inventory.json", schema)
            unit_rows.extend({"asset_id": asset["asset_id"], "animal": row["animal"], **u} for u in schema["unit_counts"])
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
            row["failure_reason"] = f"{type(exc).__name__}: {exc}"
        row["runtime_s"] = time.monotonic() - start
        rows.append(row)
        pd.DataFrame(rows).to_csv(args.output_dir / "session_inventory.csv", index=False)
        pd.DataFrame(unit_rows).to_csv(args.output_dir / "unit_inventory.csv", index=False)
        print(json.dumps({k: row.get(k) for k in ("path", "status", "failure_reason", "basic_clock_checks_passed")}), flush=True)
    complete = len(rows) == 15 and all(x["status"] == "inspected" for x in rows)
    gates = [
        {"gate": "all_15_navigation_files_checksum_verified_and_inspected", "passed": complete},
        {"gate": "all_basic_clock_checks_passed", "passed": complete and all(x.get("basic_clock_checks_passed", False) for x in rows)},
        {"gate": "context_decoder_ready", "passed": False, "reason": "requires independently documented maze epochs and position scaling review"},
    ]
    pd.DataFrame(gates).to_csv(args.output_dir / "gate_summary.csv", index=False)
    outputs = {p.name: sha256(p) for p in sorted(args.output_dir.iterdir()) if p.is_file()}
    write_json(args.output_dir / "terminal_status.json", {"returncode": 0 if complete else 2, "completed_at_utc": datetime.now(UTC).isoformat(), "output_sha256": outputs})
    return 0 if complete else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog-json", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
