#!/usr/bin/env python3
"""Independent raw chronology and sampled count checks; no outcome scoring."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter1d

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from verify_kleinman_ripple_run_opportunities import raw_runs, sha


def raw_count_check(folder, row):
    info = loadmat(folder / "session_info.mat", simplify_cells=True)["session_info"]
    raw = np.atleast_2d(loadmat(folder / "spike_data.mat", simplify_cells=True)["spike_data"])
    ripple = np.asarray(loadmat(folder / "ripple_events.mat", simplify_cells=True)["ripple_events"]).reshape(-1, 4)
    t, x, visits, runs = raw_runs(info)
    lookup = {r["traversal"]: r for r in runs}
    baseline, target, past = (lookup[int(row[name + "_traversal"])] for name in ("baseline", "target", "past"))
    ref = [lookup[i] for i in json.loads(row.reference_traversals)]
    v = np.asarray(info["velocity"])[:, 1]
    dt, dx = np.diff(t), np.diff(x)
    valid = (dt >= 0.001) & (dt <= 0.1) & (np.maximum(v[:-1], v[1:]) <= 200) & (np.minimum(v[:-1], v[1:]) >= 0) & (np.abs(dx) / dt <= 200)
    nb = int(np.ceil(x.max() / 2) - np.floor(x.min() / 2))
    edges = np.arange(nb + 1) * 2 + np.floor(x.min() / 2) * 2
    spatial = np.clip(np.searchsorted(edges, (x[:-1] + x[1:]) / 2, side="right") - 1, 0, nb - 1)
    good_units = raw[np.all(raw[:, 1:] > 0, axis=1)]
    keys = np.unique(good_units[:, [2, 1]], axis=0)
    trains = [np.sort(raw[(raw[:, 2] == tt) & (raw[:, 1] == cc), 0]) for tt, cc in keys]
    counts = np.column_stack([np.histogram(s, bins=t)[0] for s in trains])
    # numpy.histogram includes the final endpoint; the producer uses half-open time.
    for j, s in enumerate(trains):
        counts[-1, j] -= np.count_nonzero(s == t[-1])

    def run_mask(rs):
        result = np.zeros(len(dt), bool)
        for r in rs:
            result |= (t[:-1] >= r["start_s"]) & (t[1:] <= r["end_s"]) & (dx * (2 * r["direction"] - 1) > 0)
        return result & valid

    train = run_mask(ref) & ((v[:-1] + v[1:]) / 2 > 8)
    occ = np.bincount(spatial[train], weights=dt[train], minlength=nb)
    hist = np.zeros((nb, len(trains)))
    for j in range(len(trains)):
        hist[:, j] = np.bincount(spatial[train], weights=counts[train, j], minlength=nb)
    smooth_occ = gaussian_filter1d(occ, 2, mode="constant", truncate=4)
    rate = gaussian_filter1d(hist, 2, axis=0, mode="constant", truncate=4) / np.maximum(smooth_occ[:, None], 1e-12)
    included = (hist.sum(axis=0) >= 10) & (rate.max(axis=0) >= 1)
    assert int(included.sum()) == row.n_reference_units
    assert int(hist[:, included].sum()) == row.reference_spikes
    active = {}
    for name, run in (("past", past), ("baseline", baseline), ("target", target)):
        mask = run_mask([run]) & ((v[:-1] + v[1:]) / 2 > 20) & ((x[:-1] + x[1:]) / 2 > info["reward_ends"][0] + 20) & ((x[:-1] + x[1:]) / 2 < info["reward_ends"][1] - 20)
        cellcounts = counts[mask][:, included].sum(axis=0)
        assert int(cellcounts.sum()) == row[name + "_spikes"]
        active[name] = cellcounts > 0
        assert np.isclose(dt[mask].sum(), row[name + "_readout_s"], atol=1e-10, rtol=0)
    eligible, clipped = [], []
    for a, b, _ in visits:
        lo, hi = max(a, baseline["end_s"]), min(b, a + 10, target["start_s"])
        if hi <= lo:
            continue
        indices = np.flatnonzero(valid & (np.maximum(v[:-1], v[1:]) <= 8) & (t[:-1] < hi) & (t[1:] > lo))
        eligible.extend((max(t[i], lo), min(t[i + 1], hi)) for i in indices)

    def union(rows):
        out = []
        for a, b in sorted(rows):
            if out and a <= out[-1][1]:
                out[-1][1] = max(out[-1][1], b)
            else:
                out.append([a, b])
        return out

    eligible = union(eligible)
    native_hits = 0
    for a, b, _, _ in ripple:
        hit = False
        for lo, hi in eligible:
            if max(a, lo) < min(b, hi):
                clipped.append([max(a, lo), min(b, hi)])
                hit = True
        native_hits += hit
    clipped = union(clipped)
    assert np.isclose(sum(b - a for a, b in clipped), row.eligible_ripple_s, atol=1e-10, rtol=0)
    assert np.isclose(sum(b - a for a, b in eligible), (row.eligible_ripple_s + row.eligible_background_s), atol=1e-10, rtol=0)

    def count_union(s, intervals):
        keep = np.zeros(len(s), bool)
        for a, b in intervals:
            keep |= (s >= a) & (s < b)
        return int(keep.sum())

    e = np.array([count_union(s, eligible) for s in trains])[included]
    r = np.array([count_union(s, clipped) for s in trains])[included]
    assert int(r.sum()) == row.ripple_reference_spikes
    assert int((e - r).sum()) == row.background_reference_spikes


def verify(root):
    m = json.loads((root / "manifest.json").read_text())
    assert not m["git_dirty"] and not m["drug_effect_scored"] and not m["outcome_scored"]
    for name, digest in m["outputs"].items():
        assert sha(root / name) == digest, name
    for name, path in m["input_file_paths"].items():
        assert sha(Path(path)) == m["input_file_sha256"][name], name
    source_path = next(Path(p) for k, p in m["input_file_paths"].items() if k.endswith("spike_data.mat"))
    source = source_path.parents[3]
    sessions = pd.read_csv(root / "sessions.csv", float_precision="round_trip")
    packets = pd.read_csv(root / "packets.csv", float_precision="round_trip")
    assert not packets.duplicated(["animal", "session", "packet_id"]).any()
    checked = 0
    for session in sessions.itertuples():
        actual = packets.loc[packets.animal.eq(session.animal) & packets.session.eq(session.session)]
        if not session.run_pass:
            assert actual.empty
            continue
        assert session.status == "audited"
        info = loadmat(source / "Experiment_1" / session.animal / session.session / "session_info.mat", simplify_cells=True)["session_info"]
        _, _, _, runs = raw_runs(info)
        expected = {}
        for target in runs:
            earlier = [r for r in runs if r["epoch"] == target["epoch"] and r["direction"] == target["direction"] and r["start_s"] < target["start_s"]]
            if len(earlier) >= 5:
                chain = earlier[-5:] + [target]
                key = f"e{target['epoch']}_d{target['direction']}_t{target['traversal']}"
                expected[key] = chain
        assert set(actual.packet_id) == set(expected)
        chronological = sorted(expected, key=lambda key: (expected[key][-1]["end_s"], expected[key][0]["start_s"], expected[key][-1]["direction"], key))
        selected = []
        for key in chronological:
            if not selected or expected[key][0]["start_s"] >= expected[selected[-1]][-1]["end_s"]:
                selected.append(key)
        assert set(actual.loc[actual.selected, "packet_id"]) == set(selected)
        for row in actual.itertuples():
            chain = expected[row.packet_id]
            assert json.loads(row.reference_traversals) == [r["traversal"] for r in chain[:3]]
            assert all(a["end_s"] < b["start_s"] for a, b in itertools.pairwise(chain))
            assert row.span_start_s == chain[0]["start_s"] and row.span_end_s == chain[-1]["end_s"]
            for name, r in zip(("past", "baseline", "target"), chain[3:], strict=True):
                for field in ("traversal", "start_s", "end_s"):
                    assert getattr(row, name + "_" + field) == r[field]
            checked += 1
    selected = packets.loc[packets.selected]
    sample = selected.sort_values(["session", "packet_id"]).groupby(["animal", "drug", "novel"], sort=True).head(1)
    for _, row in sample.iterrows():
        raw_count_check(source / "Experiment_1" / row.animal / row.session, row)
    inventory = pd.read_csv(root / "metadata_inventory.csv")
    for row in inventory.itertuples():
        info = loadmat(source / row.experiment / row.animal / row.session / "session_info.mat", simplify_cells=True)["session_info"]
        assert json.loads(row.fields) == sorted(info)
        assert not any("track" in k.lower() or "environment" in k.lower() for k in info)
    result = {
        "status": "pass",
        "manifest_sha256": sha(root / "manifest.json"),
        "candidate_chronologies_checked": checked,
        "selected_packets": int(packets.selected.sum()),
        "sampled_raw_count_packets": len(sample),
        "metadata_sessions_checked": len(inventory),
        "outcome_scored": False,
        "verifier_sha256": sha(Path(__file__)),
        "helper_sha256": sha(ROOT / "scripts/verify_kleinman_ripple_run_opportunities.py"),
    }
    (root / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    verify(parser.parse_args().results)
