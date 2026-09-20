#!/usr/bin/env python3
"""Bounded, read-only DANDI000978 schema/clock preflight; no replay scoring."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import socket
import subprocess
import sys
import urllib.request
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from acquire_dandi000978 import destination, verify, write_json


class RangeReader(io.RawIOBase):
    """Fail closed if the server ignores Range or the metadata budget is exceeded."""

    def __init__(self, url, size, budget=128 * 1024**2):
        self.url, self.size, self.budget = url, size, budget
        self.pos, self.downloaded, self.block_size = 0, 0, 512 * 1024
        self.blocks = {}

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        if whence not in (0, 1, 2):
            raise ValueError("invalid seek origin")
        pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        if pos < 0:
            raise ValueError("negative seek")
        self.pos = pos
        return pos

    def read(self, size=-1):
        if size < 0:
            size = self.size - self.pos
        if size > 32 * 1024**2:
            raise ValueError("Refusing large single payload read")
        end = min(self.size, self.pos + size)
        chunks = []
        while self.pos < end:
            index = self.pos // self.block_size
            if index not in self.blocks:
                start = index * self.block_size
                stop = min(self.size, start + self.block_size) - 1
                length = stop - start + 1
                if self.downloaded + length > self.budget:
                    raise RuntimeError("Range-read budget exceeded")
                request = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{stop}"})
                with urllib.request.urlopen(request, timeout=60) as response:
                    if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{stop}/{self.size}":
                        raise RuntimeError("Server did not honor exact byte range")
                    data = response.read(length + 1)
                if len(data) != length:
                    raise RuntimeError("Unexpected range length")
                self.blocks[index] = data
                self.downloaded += len(data)
            offset = self.pos - index * self.block_size
            n = min(end - self.pos, len(self.blocks[index]) - offset)
            chunks.append(self.blocks[index][offset : offset + n])
            self.pos += n
        return b"".join(chunks)

    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)


def text(value):
    return value.decode() if isinstance(value, bytes) else str(value)


def assign_epochs(start, stop, epoch_start, epoch_stop):
    start, stop = np.asarray(start), np.asarray(stop)
    epoch_start, epoch_stop = np.asarray(epoch_start), np.asarray(epoch_stop)
    if not len(epoch_start) or not np.isfinite(np.r_[epoch_start, epoch_stop]).all():
        raise ValueError("Missing/nonfinite epochs")
    if (epoch_stop <= epoch_start).any() or (epoch_start[1:] < epoch_stop[:-1]).any():
        raise ValueError("Epochs must be ordered, disjoint and positive-duration")
    index = np.searchsorted(epoch_start, start, side="right") - 1
    clipped = np.clip(index, 0, len(epoch_start) - 1)
    valid = (index >= 0) & (stop > start) & (stop <= epoch_stop[clipped] + 1e-6)
    return np.where(valid, index, -1)


def reference_audit(unit_ids, references, locations, group_names):
    references = np.asarray(references)
    if references.shape != np.asarray(unit_ids).shape or references.dtype.kind not in "iu":
        raise ValueError("Unit-to-electrode reference shape/type invalid")
    if ((references < 0) | (references >= len(locations))).any():
        raise ValueError("Out-of-range NWB electrode-table reference")
    by_tetrode = {}
    for name, location in zip(group_names, locations, strict=True):
        match = re.fullmatch(r"tetrode(\d+)", name, re.IGNORECASE)
        if match:
            by_tetrode.setdefault(int(match.group(1)), set()).add(location)
    rows = []
    for unit, ref in zip(unit_ids, references, strict=True):
        alternatives = by_tetrode.get(int(ref), set())
        alternative = next(iter(alternatives)) if len(alternatives) == 1 else "unresolved"
        rows.append(
            {
                "unit_id": int(unit),
                "exported_electrode_reference": int(ref),
                "nwb_row_region": locations[ref],
                "nwb_row_group": group_names[ref],
                "alternative_tetrode_id_region_not_author_confirmed": alternative,
                "mapping_hypotheses_disagree": alternative not in ("unresolved", locations[ref]),
            }
        )
    return pd.DataFrame(rows)


def inspect(file):
    epochs = file["intervals/epoch intervals"]
    start, stop = epochs["start_time"][()], epochs["stop_time"][()]
    trials = pd.DataFrame({k: file[f"intervals/trials/{k}"][()] for k in ("id", "start_time", "stop_time", "correct", "start_well", "end_well", "trajectory_type")})
    trials["epoch_index"] = assign_epochs(trials.start_time, trials.stop_time, start, stop)
    if (trials.epoch_index < 0).any():
        raise ValueError("Trial spans outside one declared epoch")
    pos = file["processing/behavior/Position/SpatialSeries"]
    clock = pos["timestamps"][()]
    data = pos["data"][()]
    if data.shape != (len(clock), 3) or not np.isfinite(clock).all() or (np.diff(clock) <= 0).any():
        raise ValueError("Position shape or timestamp ordering invalid")
    ids, ends = file["units/id"][()], file["units/spike_times_index"][()].astype(np.int64)
    spikes = file["units/spike_times"][()]
    if len(ids) != len(set(ids)) or ends.shape != ids.shape or ends[-1] != len(spikes) or (np.diff(ends) < 0).any():
        raise ValueError("Ragged spike index or identity invalid")
    if not np.isfinite(spikes).all():
        raise ValueError("Nonfinite spikes")
    for a, b in zip(np.r_[0, ends[:-1]], ends, strict=True):
        if (np.diff(spikes[a:b]) < 0).any():
            raise ValueError("Unsorted unit spike times")
    table = file[file["units/electrodes"].attrs["table"]]
    locations = table["location"].asstr()[()]
    groups = table["group_name"].asstr()[()]
    units = reference_audit(ids, file["units/electrodes"][()], locations, groups)
    units["n_spikes"] = np.diff(np.r_[0, ends])
    epoch_rows = []
    for i, (a, b) in enumerate(zip(start, stop, strict=True)):
        n_trials = int((trials.epoch_index == i).sum())
        inside = (clock >= a) & (clock <= b)
        epoch_rows.append(
            {
                "epoch_index": i,
                "start_time_s": float(a),
                "stop_time_s": float(b),
                "n_trials": n_trials,
                "role_from_trial_presence": "RUN" if n_trials else "no_trials_rest_candidate",
                "n_position_samples": int(inside.sum()),
                "n_spikes": int(((spikes >= a) & (spikes < b)).sum()),
                "median_native_speed_cm_s": float(np.nanmedian(data[inside, 2])) if inside.any() else None,
            }
        )
    lfp = file["processing/ecephys/LFP/ElectricalSeries"]
    lfp_clock = lfp["timestamps"]
    first, last = float(lfp_clock[0]), float(lfp_clock[-1])
    head = lfp_clock[:1000]
    report = {
        "animal": text(file["general/subject/subject_id"][()]),
        "epoch_count": len(start),
        "trial_count": len(trials),
        "unit_count": len(ids),
        "run_epoch_count_from_trials": int(trials.epoch_index.nunique()),
        "rest_epoch_roles_not_sleep_staging": True,
        "nwb_row_region_counts": dict(Counter(units.nwb_row_region)),
        "unit_reference_hypotheses_disagree_count": int(units.mapping_hypotheses_disagree.sum()),
        "unit_reference_mapping_status": "requires_source_confirmation" if units.mapping_hypotheses_disagree.any() else "no_candidate_disagreement_found",
        "electrode_table_reference_target": table.name,
        "position_shape": list(data.shape),
        "position_unit": text(pos["data"].attrs.get("unit", "missing")),
        "position_conversion": float(pos["data"].attrs.get("conversion", 1)),
        "position_valid_fraction": float(np.isfinite(data).all(axis=1).mean()),
        "position_start_s": float(clock[0]),
        "position_stop_s": float(clock[-1]),
        "spike_start_s": float(spikes.min()),
        "spike_stop_s": float(spikes.max()),
        "lfp_start_s": first,
        "lfp_stop_s": last,
        "lfp_channels": int(lfp["data"].shape[1]),
        "lfp_nominal_rate_hz_first_1000_samples": float(1 / np.median(np.diff(head))),
        "lfp_clock_length_matches_data": len(lfp_clock) == len(lfp["data"]),
        "position_in_lfp_clock_support": bool(first <= clock[0] <= clock[-1] <= last),
        "spikes_in_lfp_clock_support": bool(first <= spikes.min() <= spikes.max() <= last),
        "clock_support_is_not_independent_synchronization_proof": True,
        "native_replay_labels_used": False,
        "replay_scored": False,
        "region_specific_RUN_scoring_allowed": not units.mapping_hypotheses_disagree.any(),
    }
    arrays = {"clock": clock, "position_and_speed": data, "spikes": spikes, "spike_ends": ends, "unit_ids": ids}
    return report, pd.DataFrame(epoch_rows), trials, units, arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--allow-stream", action="store_true")
    args = parser.parse_args()
    root, out = args.dataset_root.resolve(), args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((root / "metadata/asset_manifest.json").read_text())
    asset = next(a for a in manifest["assets"] if "sub-JDS-SingleDay-JS14/" in a["path"])
    path = destination(root, asset)
    reader, verified = None, False
    if path.exists():
        verify(path, asset)
        verified = True
    elif args.allow_stream:
        reader = RangeReader(asset["url"], asset["size"])
    else:
        raise FileNotFoundError("Verified local JS14 asset not ready; no partial-file reads")
    with h5py.File(path if reader is None else reader, "r") as file:
        report, epochs, trials, units, arrays = inspect(file)
    report.update(
        dataset="000978",
        version=manifest["version"],
        asset_id=asset["asset_id"],
        expected_asset_sha256=asset["digest"]["dandi:sha2-256"],
        full_local_asset_verified=verified,
        source_mode="verified_local_file" if verified else "bounded_remote_ranges_not_full_asset_verified",
        range_bytes_fetched=reader.downloaded if reader else 0,
        created_at_utc=datetime.now(UTC).isoformat(),
        host=socket.gethostname(),
        command_line=sys.argv,
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    try:
        report["code_commit"] = subprocess.check_output(["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"], text=True).strip()
        report["git_dirty"] = bool(subprocess.check_output(["git", "-C", str(Path(__file__).resolve().parents[1]), "status", "--porcelain"], text=True).strip())
    except (OSError, subprocess.CalledProcessError) as exc:
        report.update(code_commit="unavailable", git_error=str(exc))
    epochs.to_csv(out / "epoch_inventory.csv", index=False)
    trials.to_csv(out / "trial_inventory.csv", index=False)
    units.to_csv(out / "unit_reference_audit.csv", index=False)
    np.savez_compressed(out / "bounded_run_input_arrays.npz", **arrays)
    report["input_arrays_sha256"] = hashlib.sha256((out / "bounded_run_input_arrays.npz").read_bytes()).hexdigest()
    write_json(out / "preflight_report.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
