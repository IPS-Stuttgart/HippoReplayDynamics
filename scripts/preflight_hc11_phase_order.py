#!/usr/bin/env python3
"""Read-only hc-11 PRE/POST metadata audit; no event selection or scoring."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from _hc11_native_encoding import as_intervals, mat_struct, times_in_intervals
from _provenance import build_script_provenance, file_sha256
from preflight_hc11_coverage_forecasts import SESSIONS, intersect_nrem


def inspect(folder):
    session = folder.name
    position = mat_struct(folder / f"{session}.position.behavior.mat", "position")
    spikes = mat_struct(folder / f"{session}.spikes.cellinfo.mat", "spikes")
    sleep = mat_struct(folder / f"{session}.SleepState.states.mat", "SleepState")
    epochs = [as_intervals(getattr(position.Epochs, name)) for name in ("PREEpoch", "MazeEpoch", "POSTEpoch")]
    if any(not len(x) for x in epochs):
        raise ValueError("missing epoch")
    if epochs[0][:, 1].max() > epochs[1][:, 0].min() or epochs[1][:, 1].max() > epochs[2][:, 0].min():
        raise ValueError("unordered or overlapping PRE/MAZE/POST")
    domains = [intersect_nrem(sleep.ints.NREMstate, epochs[i]) for i in (0, 2)]
    if any(not len(x) for x in domains):
        raise ValueError("empty guarded PRE or POST NREM")
    ids = np.atleast_1d(spikes.UID).astype(int)
    times = np.asarray(spikes.times, object).ravel()
    regions = np.asarray(spikes.region, object).ravel()
    if len(set(ids)) != len(ids) or len(ids) != len(times) or len(ids) != len(regions):
        raise ValueError("duplicate or unaligned unit metadata")
    selected = [i for i, r in enumerate(regions) if "CA1" in str(r).upper()]
    counts = []
    for i in selected:
        x = np.asarray(times[i], float).ravel()
        if not np.isfinite(x).all() or (np.diff(x) < 0).any():
            raise ValueError("invalid raw CA1 timestamps")
        counts.append([int(times_in_intervals(x, d).sum()) for d in domains])
    counts = np.asarray(counts).reshape(-1, 2)
    return {
        "animal": folder.parent.name, "session": session,
        "pre_nrem_seconds": float(np.diff(domains[0], axis=1).sum()),
        "post_nrem_seconds": float(np.diff(domains[1], axis=1).sum()),
        "ca1_units": len(selected), "ca1_units_active_both": int((counts.min(axis=1) > 0).sum()),
        "pre_nrem_spikes": int(counts[:, 0].sum()), "post_nrem_spikes": int(counts[:, 1].sum()),
        "same_recording_unit_namespace": True,
        "raw_waveform_field_present": "rawWaveform" in spikes._fieldnames,
        "unit_drift_validated": False,
        "status": "metadata_feasible_not_biologically_validated",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    folders, inputs = {}, {}
    for session in SESSIONS:
        matches = list(args.dataset_root.glob(f"*/{session}"))
        if len(matches) != 1:
            raise ValueError(f"missing or ambiguous {session}")
        folders[session] = matches[0]
        for suffix in ("position.behavior", "spikes.cellinfo", "SleepState.states"):
            inputs[f"{session}_{suffix}"] = matches[0] / f"{session}.{suffix}.mat"
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = build_script_provenance(input_paths=inputs)
    rows = []
    for session, folder in folders.items():
        try:
            rows.append(inspect(folder))
        except (ValueError, KeyError, AttributeError, OSError) as exc:
            rows.append({"animal": folder.parent.name, "session": session, "status": "failed", "error": repr(exc)})
    table = pd.DataFrame(rows)
    table.to_csv(output / "hc11_pre_post_metadata.csv", index=False)
    unchanged = all(file_sha256(p) == manifest["input_file_sha256"][name] for name, p in inputs.items())
    manifest.update(status="complete", real_data_scored=False, inputs_unchanged=unchanged,
                    output_sha256={"hc11_pre_post_metadata.csv": file_sha256(output / "hc11_pre_post_metadata.csv")})
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return 0 if unchanged and not table.status.eq("failed").any() else 2


if __name__ == "__main__":
    raise SystemExit(main())
