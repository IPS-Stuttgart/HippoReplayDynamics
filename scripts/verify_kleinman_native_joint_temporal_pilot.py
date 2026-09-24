#!/usr/bin/env python3
"""Rebuild native three-period histograms and check both scores independently."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import score_kleinman_native_joint_temporal_pilot as producer
from verify_kleinman_conditional_coupling import independent_anchor, independent_score, raw_predictors
from verify_kleinman_ripple_run_opportunities import raw_runs
from verify_kleinman_spatial_expression import sha, smooth


def independent_reference(counts, exposure):
    rate = smooth(counts) / np.maximum(smooth(exposure), 1e-12)
    return np.maximum(rate, 1e-5), bool(counts.sum() >= 10 and rate.max() >= 1)


def independent_data(folder, a):
    info = loadmat(folder / "session_info.mat", simplify_cells=True)["session_info"]
    t, x, _, runs = raw_runs(info)
    lookup = {r["traversal"]: r for r in runs}
    groups = [[lookup[i] for i in json.loads(a.reference_traversals)], *[[lookup[getattr(a, n + "_traversal")]] for n in ("past", "baseline", "target")]]
    v = np.asarray(info["velocity"])[:, 1]
    dt, dx = np.diff(t), np.diff(x)
    valid = (dt >= 0.001) & (dt <= 0.1) & (np.minimum(v[:-1], v[1:]) >= 0) & (np.maximum(v[:-1], v[1:]) <= 200) & (np.abs(dx) / dt <= 200)
    edges = np.arange(np.floor(x.min() / 2) * 2, np.ceil(x.max() / 2) * 2 + 2, 2)
    bins = np.clip(np.searchsorted(edges, (x[:-1] + x[1:]) / 2, side="right") - 1, 0, len(edges) - 2)
    raw = np.atleast_2d(loadmat(folder / "spike_data.mat", simplify_cells=True)["spike_data"])
    raw = raw[np.all(raw[:, 1:] > 0, axis=1)]
    keys = np.unique(raw[:, [2, 1]].astype(int), axis=0)
    counts = []
    for tt, cc in keys:
        spikes = raw[(raw[:, 2] == tt) & (raw[:, 1] == cc), 0]
        h = np.histogram(spikes, bins=t)[0]
        h[-1] -= np.count_nonzero(spikes == t[-1])
        counts.append(h)
    counts = np.asarray(counts)
    occ, hist = [], []
    for k, rs in enumerate(groups):
        mask = np.zeros(len(dt), bool)
        for r in rs:
            mask |= (t[:-1] >= r["start_s"]) & (t[1:] <= r["end_s"]) & (dx * (2 * r["direction"] - 1) > 0)
        mask &= valid & ((v[:-1] + v[1:]) / 2 > (8 if k == 0 else 20))
        if k:
            mask &= ((x[:-1] + x[1:]) / 2 > info["reward_ends"][0] + 20) & ((x[:-1] + x[1:]) / 2 < info["reward_ends"][1] - 20)
        occ.append(np.bincount(bins[mask], weights=dt[mask], minlength=len(edges) - 1))
        hist.append([np.bincount(bins[mask], weights=c[mask], minlength=len(edges) - 1) for c in counts])
    rec = producer.recruitment(folder, a, keys)
    return keys, np.asarray(occ), np.asarray(hist), rec


def verify(result, dataset):
    m = json.loads((result / "manifest.json").read_text())
    assert not m["git_dirty"] and m["native_outcomes_scored"] and not m["drug_effect_scored"]
    for name, digest in m["outputs"].items():
        assert sha(result / name) == digest
    for name, path in m["input_file_paths"].items():
        assert sha(Path(path)) == m["input_file_sha256"][name]
    selected = pd.read_csv(m["input_file_paths"]["selected_packets.csv"], float_precision="round_trip")
    packets = pd.read_csv(result / "packet_scores.csv", float_precision="round_trip")
    cells = pd.read_csv(result / "cell_scores.csv.gz", float_precision="round_trip")
    assert len(selected) == len(packets) == 48
    assert not cells.duplicated(["animal", "session", "packet_id", "unit_id"]).any()
    producer.packet_data = independent_data
    producer.estimate_reference = independent_reference
    producer.spatial_score = independent_score
    producer.anchor_score = independent_anchor
    maximum = 0.0
    for a in selected.itertuples():
        folder = dataset / "Experiment_1" / a.animal / a.session
        c, s = producer.score_packet(folder, a)
        actual = cells.loc[cells.animal.eq(a.animal) & cells.session.eq(a.session) & cells.packet_id.eq(a.packet_id)].set_index("unit_id")
        assert set(c.unit_id) == set(actual.index)
        for row in c.itertuples():
            saved = actual.loc[row.unit_id]
            for k in ("reference_included", "reference_spikes", "past_spikes", "baseline_spikes", "target_spikes", "ripple_spikes", "background_spikes"):
                assert getattr(row, k) == saved[k]
            for lag in ("preceding", "prospective"):
                for kind in ("score", "information"):
                    k = lag + "_" + kind
                    value = getattr(row, k)
                    if np.isfinite(value):
                        maximum = max(maximum, abs(value - saved[k]))
                        np.testing.assert_allclose(value, saved[k], atol=2e-4, rtol=1e-5)
                    else:
                        assert np.isnan(saved[k])
        pred = c.copy()
        pred["ripple_s"] = a.eligible_ripple_s
        pred["background_s"] = a.eligible_background_s
        raw_predictors(folder, SimpleNamespace(**a._asdict(), opportunity_id=a.packet_id), pred)
        saved = packets.loc[packets.animal.eq(a.animal) & packets.session.eq(a.session) & packets.packet_id.eq(a.packet_id)].iloc[0]
        for lag in ("preceding", "prospective"):
            for kind in ("score", "information"):
                k = lag + "_" + kind
                np.testing.assert_allclose(s[k], saved[k], atol=2e-4, rtol=1e-5)
    animal_rows = pd.read_csv(result / "by_animal.csv")
    differences = []
    for animal, g in packets.loc[packets.preceding_information.gt(1e-10) & packets.prospective_information.gt(1e-10)].groupby("animal"):
        prior = g.preceding_score.sum() / g.preceding_information.sum()
        future = g.prospective_score.sum() / g.prospective_information.sum()
        a = animal_rows.loc[animal_rows.animal.eq(animal)].iloc[0]
        np.testing.assert_allclose([prior, future, future - prior], [a.preceding_effect, a.prospective_effect, a.contrast], atol=1e-12, rtol=0)
        differences.append(future - prior)
    assert len(differences) == 6
    estimate = pd.read_csv(result / "primary_temporal_contrast.csv").iloc[0]
    mean = np.mean(differences)
    half = student_t.ppf(0.975, 5) * np.std(differences, ddof=1) / np.sqrt(6)
    np.testing.assert_allclose([mean, mean - half, mean + half], [estimate.mean_contrast, estimate.ci_low, estimate.ci_high], atol=1e-12, rtol=0)
    assert estimate.positive_flag == (mean - half > 0) and estimate.negative_flag == (mean + half < 0)
    previous = pd.read_csv(m["input_file_paths"]["previous_native_selection"])
    prior = set(zip(previous.animal, previous.session, previous.target_traversal, strict=True))
    overlap = sum((a.animal, a.session, a.target_traversal) in prior for a in selected.itertuples())
    assert overlap == m["overlapping_previous_targets"]
    record = {
        "status": "passed",
        "manifest_sha256": sha(result / "manifest.json"),
        "verifier_sha256": sha(Path(__file__)),
        "native_packets_checked": 48,
        "raw_cell_rows_checked": len(cells),
        "maximum_moment_difference": maximum,
        "overlapping_previous_targets": overlap,
        "drug_effect_validated": False,
        "helper_sha256": {
            name: sha(ROOT / "scripts" / name)
            for name in ("verify_kleinman_conditional_coupling.py", "verify_kleinman_spatial_expression.py", "verify_kleinman_ripple_run_opportunities.py")
        },
    }
    with (result / "verification.json").open("x") as f:
        json.dump(record, f, indent=2)
        f.write("\n")
    print(json.dumps(record))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--result-dir", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, required=True)
    a = p.parse_args()
    verify(a.result_dir, a.dataset_root)
