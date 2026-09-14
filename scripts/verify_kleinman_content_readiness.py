#!/usr/bin/env python3
"""Check readiness tables against raw MATLAB data without the producer reader."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.io import loadmat

from scripts._provenance import build_script_provenance, file_sha256


def direct_counts(spikes, selected_keys, start, end):
    within = spikes[(spikes[:, 0] >= start) & (spikes[:, 0] < end)]
    counter = Counter((int(tetrode), int(cluster)) for _, cluster, tetrode in within)
    values = [counter[key] for key in selected_keys]
    return sum(values), sum(value > 0 for value in values)


def direct_immobile(velocity, start, end):
    t = velocity[:, 0]
    before, after = np.where(t <= start)[0], np.where(t >= end)[0]
    if not len(before) or not len(after):
        return False
    lo, hi = before[-1], after[0]
    if hi <= lo:
        return False
    return bool(np.all(np.diff(t[lo : hi + 1]) <= 0.1) and np.all(np.abs(velocity[lo : hi + 1, 1]) < 5))


def verify(root, inventory, output):
    manifest_path = inventory / "manifest.json"
    source = json.loads(manifest_path.read_text())
    if not source["inputs_unchanged"]:
        raise ValueError("producer source inputs were not stable")
    for name, path in source["input_file_paths"].items():
        if file_sha256(path) != source["source_sha256_before"][name]:
            raise ValueError("source hash changed: " + name)
    for name, digest in source["output_sha256"].items():
        if file_sha256(inventory / name) != digest:
            raise ValueError("output hash changed: " + name)
    sessions = pd.read_csv(inventory / "sessions.csv")
    events = pd.read_csv(inventory / "native_event_support.csv.gz")
    support = pd.read_csv(inventory / "support_by_session.csv")
    native = sorted(root.glob("Experiment_*/*/*/session_info.mat"))
    assert set(sessions.session_path) == {str(p.parent) for p in native}
    assert source["native_spiking_sessions"] == sum((p.parent / "spike_data.mat").is_file() for p in native)
    checks = []
    clocks, no_clock = 0, 0
    for row in sessions.itertuples():
        folder = Path(row.session_path)
        relative = folder.relative_to(root)
        info = loadmat(folder / "session_info.mat", simplify_cells=True)["session_info"]
        v = np.asarray(info["velocity"], float).reshape(-1, 2)
        valid_clock = len(v) >= 2 and np.isfinite(v).all() and np.all(np.diff(v[:, 0]) > 0)
        if row.status == "failed":
            assert not valid_clock and "timestamps" in row.failure_reason
            no_clock += 1
            continue
        assert valid_clock
        clocks += 1
        file = folder / "spike_data.mat"
        spikes = np.asarray(loadmat(file, simplify_cells=True)["spike_data"], float).reshape(-1, 3) if file.is_file() else np.empty((0, 3))
        keys = sorted({(int(t), int(c)) for _, c, t in spikes})
        assert len(keys) == row.n_units_raw and len(spikes) == row.n_spikes_raw
        rows = events.loc[events.recording_group.eq(row.recording_group) & events.session.eq(row.session)]
        for event_type in ("sdes", "ripple_events"):
            path = folder / (event_type + ".mat")
            if not path.is_file():
                assert rows.loc[rows.event_type.eq(event_type)].empty
                continue
            arr = np.asarray(loadmat(path, simplify_cells=True)[event_type], float).reshape(-1, 4)
            local = rows.loc[rows.event_type.eq(event_type)]
            assert len(local) == len(arr) * 6
            for (phase, split), case in local.groupby(["phase", "split"]):
                state = int.from_bytes(hashlib.sha256(f"{source['seed']}|{relative}|{split}".encode()).digest()[:8], "little")
                order = np.random.default_rng(state).permutation(len(keys))
                half = len(keys) // 2
                a = {keys[i] for i in order[:half]}
                b = {keys[i] for i in order[half : 2 * half]}
                assert not a.intersection(b)
                chosen = case.sort_values("event_index").iloc[np.unique(np.linspace(0, len(case) - 1, min(3, len(case)), dtype=int))]
                for event in chosen.itertuples():
                    start, end, peak, _ = arr[int(event.event_index)]
                    ws, we = (end - 0.02, end) if phase == "native_endpoint" else (peak - 0.01, peak + 0.01)
                    np.testing.assert_allclose([event.start_s, event.end_s], [ws, we], rtol=0, atol=1e-10)
                    valid = end - start >= 0.02 and start <= peak <= end
                    if phase == "native_peak":
                        valid = valid and ws >= start and we <= end
                    assert bool(event.valid_20ms_window) == valid
                    assert bool(event.immobile) == direct_immobile(v, ws, we)
                    ca, aa = direct_counts(spikes, a, ws, we)
                    cb, ab = direct_counts(spikes, b, ws, we)
                    assert (ca, cb, aa, ab) == (event.n_spikes_a, event.n_spikes_b, event.n_active_a, event.n_active_b)
                    assert bool(event.both_halves_have_3_spikes_2_units) == (min(ca, cb) >= 3 and min(aa, ab) >= 2)
                    checks.append({"session": str(relative), "event_type": event_type, "phase": phase, "split": split, "event_index": event.event_index, "pass": True})
    columns = ["recording_group", "session", "event_type", "phase", "split"]
    assert not support.duplicated(columns).any()
    for values, local in events.groupby(columns, dropna=False):
        mask = np.ones(len(support), bool)
        for name, value in zip(columns, values, strict=True):
            mask &= support[name].eq(value)
        assert int(mask.sum()) == 1
        summary = support.loc[mask].iloc[0]
        eligible = local.valid_20ms_window & local.immobile
        count = int(eligible.sum())
        ok = int((eligible & local.both_halves_have_3_spikes_2_units).sum())
        assert summary.native_events == len(local) and summary.valid_immobile_windows == count and summary.both_halves_supported == ok
        if count:
            np.testing.assert_allclose(summary.support_fraction, ok / count)
        else:
            assert np.isnan(summary.support_fraction)
    output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(checks).to_csv(output / "raw_window_checks.csv", index=False)
    result = build_script_provenance(input_paths={"source": manifest_path, "script": Path(__file__)}, cwd=ROOT)
    result.update(
        status="passed",
        native_sessions=len(native),
        readable_clocks=clocks,
        rejected_nonmonotonic_clocks=no_clock,
        raw_window_checks=len(checks),
        summary_groups=len(support),
        decoder_ready=False,
        scientific_goal_achieved=False,
    )
    (output / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "native_sessions", "raw_window_checks", "summary_groups")}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--inventory", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    verify(args.dataset_root, args.inventory, args.output_dir)


if __name__ == "__main__":
    main()
