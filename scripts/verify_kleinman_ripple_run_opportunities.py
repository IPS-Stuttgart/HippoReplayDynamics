#!/usr/bin/env python3
"""Independently check chronology and sampled raw counts for the opportunity audit."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter1d


def sha(path):
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def raw_runs(info):
    clock = np.asarray(info["velocity"])[:, 0]
    pos = np.asarray(info["position"]).reshape(-1)[1:]
    visits = []
    for side, key in enumerate(("left_visit", "right_visit")):
        for a, b in np.asarray(info[key]).reshape(-1, 2):
            visits.append((clock[int(a) - 2], clock[int(b) - 2], side))
    visits.sort()
    transitions = np.asarray(info["epoch_change"], int).reshape(-1, 2) - 1
    starts = np.r_[clock[0], clock[transitions[:, 1]]]
    stops = np.r_[clock[transitions[:, 0]], clock[-1]]
    runs = []
    for a, b in itertools.pairwise(visits):
        epoch = np.flatnonzero((starts <= a[1]) & (stops >= b[0]))
        if a[2] != b[2] and b[0] > a[1] and len(epoch) == 1:
            runs.append({"traversal": len(runs), "start_s": a[1], "end_s": b[0], "direction": b[2], "epoch": int(epoch[0]) + 1})
    return clock, pos, visits, runs


def expected_pairs(runs):
    result = {}
    for target in runs:
        before = [r for r in runs if r["direction"] == target["direction"] and r["epoch"] == target["epoch"] and r["start_s"] < target["start_s"]]
        if not before:
            continue
        baseline, ref = before[-1], before[max(0, len(before) - 4) : -1]
        key = f"e{target['epoch']}_d{target['direction']}_t{target['traversal']}"
        result[key] = (baseline, target, ref)
    return result


def raw_count_check(folder, row):
    info = loadmat(folder / "session_info.mat", simplify_cells=True)["session_info"]
    raw = np.atleast_2d(loadmat(folder / "spike_data.mat", simplify_cells=True)["spike_data"])
    ripple = np.asarray(loadmat(folder / "ripple_events.mat", simplify_cells=True)["ripple_events"]).reshape(-1, 4)
    t, x, visits, runs = raw_runs(info)
    baseline, target, ref = expected_pairs(runs)[row.opportunity_id]
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
    assert int(included.sum()) == row.n_encoding_units
    assert int(hist[:, included].sum()) == row.reference_encoding_spikes
    active = {}
    for name, run in (("baseline", baseline), ("target", target)):
        mask = run_mask([run]) & ((v[:-1] + v[1:]) / 2 > 20) & ((x[:-1] + x[1:]) / 2 > info["reward_ends"][0] + 20) & ((x[:-1] + x[1:]) / 2 < info["reward_ends"][1] - 20)
        cellcounts = counts[mask][:, included].sum(axis=0)
        assert int(cellcounts.sum()) == row[name + "_encoding_spikes"]
        active[name] = cellcounts > 0
        assert int(active[name].sum()) == row["n_encoding_units_" + name + "_active"]
        assert np.isclose(dt[mask].sum(), row[name + "_readout_s"], atol=1e-10, rtol=0)
        support = dt[mask][occ[spatial[mask]] >= 0.1].sum() / dt[mask].sum()
        assert np.isclose(support, row[name + "_reference_support_fraction"], atol=1e-10, rtol=0)
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
    assert native_hits == row.n_native_ripples_intersecting
    assert np.isclose(sum(b - a for a, b in clipped), row.eligible_ripple_s, atol=1e-10, rtol=0)
    assert np.isclose(sum(b - a for a, b in eligible), row.eligible_immobile_s, atol=1e-10, rtol=0)

    def count_union(s, intervals):
        keep = np.zeros(len(s), bool)
        for a, b in intervals:
            keep |= (s >= a) & (s < b)
        return int(keep.sum())

    e = np.array([count_union(s, eligible) for s in trains])[included]
    r = np.array([count_union(s, clipped) for s in trains])[included]
    assert int(r.sum()) == row.ripple_encoding_spikes
    assert int((e - r).sum()) == row.background_encoding_spikes
    assert int((r > 0).sum()) == row.n_encoding_units_ripple_active
    both = active["baseline"] & active["target"]
    assert int(both.sum()) == row.n_encoding_units_both_runs_active
    assert int((both & (r > 0)).sum()) == row.n_encoding_units_both_runs_and_ripple_active


def verify(root):
    m = json.loads((root / "manifest.json").read_text())
    assert not m["git_dirty"] and not m["drug_effect_scored"]
    assert not m["future_spatial_outcome_scored"]
    for name, digest in m["outputs"].items():
        assert sha(root / name) == digest, name
    for name, path in m["input_file_paths"].items():
        assert sha(Path(path)) == m["input_file_sha256"][name], name
    source = Path(m["input_file_paths"]["Experiment_1/Con_1/20210524_run1/spike_data.mat"]).parents[3]
    sessions = pd.read_csv(root / "sessions.csv", float_precision="round_trip")
    rows = pd.read_csv(root / "opportunities.csv", float_precision="round_trip")
    assert not rows.duplicated(["animal", "session", "opportunity_id"]).any()
    chronological = 0
    for session in sessions.itertuples():
        actual = rows.loc[rows.animal.eq(session.animal) & rows.session.eq(session.session)]
        if session.status == "failed":
            assert actual.empty
            continue
        info = loadmat(source / "Experiment_1" / session.animal / session.session / "session_info.mat", simplify_cells=True)["session_info"]
        _, _, _, runs = raw_runs(info)
        pairs = expected_pairs(runs)
        assert set(actual.opportunity_id) == set(pairs)
        for row in actual.itertuples():
            base, target, ref = pairs[row.opportunity_id]
            assert json.loads(row.reference_traversals) == [r["traversal"] for r in ref]
            assert row.baseline_traversal == base["traversal"]
            assert row.target_traversal == target["traversal"]
            assert row.baseline_start_s == base["start_s"] and row.baseline_end_s == base["end_s"]
            assert row.target_start_s == target["start_s"] and row.target_end_s == target["end_s"]
            assert row.status == ("audited" if len(ref) == 3 else "insufficient_prior_reference_runs")
            chronological += 1
        assert int(session.n_opportunities) == len(actual)
        assert int(session.n_prior_reference_available) == actual.status.eq("audited").sum()
        assert int(session.n_reference_min_units) == actual.minimum_units_descriptor.sum()
    sampled = []
    good = rows.loc[rows.original_run_pass & rows.minimum_units_descriptor]
    for animal, sub in good.groupby("animal"):
        sub = sub.sort_values(["session", "opportunity_id"])
        picks = [sub.loc[sub.eligible_ripple_s.eq(0)].iloc[0]]
        positive = sub.loc[sub.eligible_ripple_s.gt(0)]
        picks.extend([positive.iloc[0], positive.iloc[len(positive) // 2]])
        for row in picks:
            raw_count_check(source / "Experiment_1" / animal / row.session, row)
            sampled.append([animal, row.session, row.opportunity_id])
    coverage = pd.read_csv(root / "coverage.csv")
    primary = rows.loc[rows.original_run_pass]
    for c in coverage.itertuples():
        g = primary.loc[primary.animal.eq(c.animal) & primary.drug.eq(c.drug) & primary.novel.eq(c.novel)]
        assert len(g) == c.n_opportunities
        assert g.session.nunique() == c.n_sessions
        assert g.status.eq("audited").sum() == c.n_reference_available
        assert g.minimum_units_descriptor.sum() == c.n_reference_min_units
        assert (g.minimum_units_descriptor & g.eligible_ripple_s.gt(0)).sum() == c.n_reference_min_units_with_ripple
        assert (g.minimum_units_descriptor & g.eligible_ripple_s.eq(0)).sum() == c.n_reference_min_units_without_ripple
        assert g.n_encoding_units_both_runs_and_ripple_active.sum() == c.n_units_both_runs_and_ripple
    result = {
        "status": "passed",
        "manifest_sha256": sha(root / "manifest.json"),
        "producer_commit": m["code_commit"],
        "verifier_sha256": sha(Path(__file__)),
        "n_opportunity_chronologies_checked": chronological,
        "raw_count_checks": sampled,
        "n_raw_count_checks": len(sampled),
        "coverage_rows_checked": len(coverage),
        "not_an_outcome_or_treatment_effect_validation": True,
    }
    with (root / "verification.json").open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--result-dir", type=Path, required=True)
    verify(p.parse_args().result_dir)
