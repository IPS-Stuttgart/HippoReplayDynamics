#!/usr/bin/env python3
"""Rebuild native RUN histograms and conditional scores independently."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import t as student_t
from verify_kleinman_conditional_coupling import independent_anchor, independent_score, raw_predictors
from verify_kleinman_ripple_run_opportunities import expected_pairs, raw_runs
from verify_kleinman_spatial_expression import smooth


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_anchor(folder, anchor, cells, summary):
    info = loadmat(folder / "session_info.mat", simplify_cells=True)["session_info"]
    t, x, _, runs = raw_runs(info)
    baseline, target, reference = expected_pairs(runs)[anchor.opportunity_id]
    assert len(reference) == 3
    speed = np.asarray(info["velocity"])[:, 1]
    dt, dx = np.diff(t), np.diff(x)
    valid = (dt >= 0.001) & (dt <= 0.1) & (np.minimum(speed[:-1], speed[1:]) >= 0) & (np.maximum(speed[:-1], speed[1:]) <= 200) & (np.abs(dx) / dt <= 200)
    bins = np.clip(
        np.searchsorted(np.arange(np.floor(x.min() / 2) * 2, np.ceil(x.max() / 2) * 2 + 2, 2), (x[:-1] + x[1:]) / 2, side="right") - 1,
        0,
        int(np.ceil(x.max() / 2) - np.floor(x.min() / 2)) - 1,
    )
    nb = int(np.ceil(x.max() / 2) - np.floor(x.min() / 2))

    def mask(rs, readout=False):
        m = np.zeros(len(dt), bool)
        for r in rs:
            m |= (t[:-1] >= r["start_s"]) & (t[1:] <= r["end_s"]) & (dx * (2 * r["direction"] - 1) > 0)
        m &= valid & ((speed[:-1] + speed[1:]) / 2 > (20 if readout else 8))
        if readout:
            m &= ((x[:-1] + x[1:]) / 2 > info["reward_ends"][0] + 20) & ((x[:-1] + x[1:]) / 2 < info["reward_ends"][1] - 20)
        return m

    masks = [mask(reference), mask([baseline], True), mask([target], True)]
    occupancies = [np.bincount(bins[m], weights=dt[m], minlength=nb) for m in masks]
    common = (occupancies[1] > 0) & (occupancies[2] > 0)
    raw = np.atleast_2d(loadmat(folder / "spike_data.mat", simplify_cells=True)["spike_data"])
    keys = np.unique(raw[np.all(raw[:, 1:] > 0, axis=1)][:, [2, 1]].astype(int), axis=0)
    assert list(cells.unit_id) == ["_".join(map(str, key)) for key in keys]
    scores, infos, predictors = [], [], []
    checked = 0
    maxdiff = 0.0
    for row, (tt, cc) in zip(cells.itertuples(), keys, strict=True):
        spikes = raw[(raw[:, 2] == tt) & (raw[:, 1] == cc), 0]
        counts = np.histogram(spikes, bins=t)[0]
        counts[-1] -= np.count_nonzero(spikes == t[-1])
        h = [np.bincount(bins[m], weights=counts[m], minlength=nb) for m in masks]
        fitted = smooth(h[0]) / np.maximum(smooth(occupancies[0]), 1e-12)
        included = h[0].sum() >= 10 and fitted.max() >= 1
        assert bool(row.reference_included) == included
        np.testing.assert_array_equal([row.reference_spikes, row.baseline_spikes, row.target_spikes], [a.sum() for a in h])
        assert np.isclose(row.reference_peak_hz, max(fitted.max(), 1e-5), atol=1e-9, rtol=0)
        if included:
            f = np.log(np.maximum(fitted, 1e-5))
            sd = f[common].std() if np.any(common) else 0.0
            feature = (f - f[common].mean()) / sd if sd >= 1e-10 else np.zeros_like(f)
            result = independent_score(h[1], h[2], occupancies[1], occupancies[2], feature)
            for k in ("before_common_spikes", "after_common_spikes", "excluded_spikes"):
                assert result[k] == getattr(row, k)
            for k in ("score", "information"):
                diff = abs(result[k] - getattr(row, k))
                maxdiff = max(maxdiff, diff)
                assert np.isclose(result[k], getattr(row, k), atol=2e-4, rtol=1e-5), (row.unit_id, k, diff)
            scores.append(result["score"])
            infos.append(result["information"])
            predictors.append(row.log_rate_enrichment)
            checked += 1
        else:
            assert np.isnan(row.score) and row.information == 0
    table = cells.assign(ripple_s=summary.ripple_s, background_s=summary.background_s)
    raw_predictors(folder, anchor, table)
    x = np.asarray(predictors)
    usable = len(x) >= 2 and np.isfinite(x).all() and x.std() > 1e-10
    u, v = 0.0, 0.0
    if usable:
        z = (x - x.mean()) / x.std()
        u, v = independent_anchor(z, scores, infos)
    np.testing.assert_allclose([u, v], [summary.score, summary.information], atol=2e-4, rtol=1e-5)
    return checked, maxdiff


def verify(result, dataset):
    m = json.loads((result / "manifest.json").read_text())
    assert not m["git_dirty"] and m["real_future_outcome_scored"] and not m["drug_effect_scored"]
    for name, digest in m["outputs"].items():
        assert sha(result / name) == digest, name
    for name, path in m["input_file_paths"].items():
        assert sha(Path(path)) == m["input_file_sha256"][name], name
    original = pd.read_csv(m["input_file_paths"]["selected_anchors"], float_precision="round_trip")
    cells = pd.read_csv(result / "cell_scores.csv.gz", float_precision="round_trip")
    anchors = pd.read_csv(result / "anchor_scores.csv", float_precision="round_trip")
    animals = pd.read_csv(result / "by_animal.csv", float_precision="round_trip")
    primary = pd.read_csv(result / "primary_association.csv", float_precision="round_trip").iloc[0]
    keys = ["animal", "session", "opportunity_id"]
    assert len(anchors) == len(original) == 48 and not cells.duplicated(keys + ["unit_id"]).any()
    assert set(map(tuple, anchors[keys].values)) == set(map(tuple, original[keys].values))
    checked = 0
    maxdiff = 0.0
    for row in original.itertuples():
        mask = anchors.animal.eq(row.animal) & anchors.session.eq(row.session) & anchors.opportunity_id.eq(row.opportunity_id)
        actual = anchors.loc[mask].iloc[0]
        cf = cells.loc[cells.animal.eq(row.animal) & cells.session.eq(row.session) & cells.opportunity_id.eq(row.opportunity_id)]
        n, d = check_anchor(dataset / "Experiment_1" / row.animal / row.session, row, cf, actual)
        checked += n
        maxdiff = max(maxdiff, d)
    effects = []
    for animal, g in anchors.loc[anchors.information > 1e-10].groupby("animal"):
        effect = g.score.sum() / g.information.sum()
        a = animals.loc[animals.animal.eq(animal)].iloc[0]
        np.testing.assert_allclose([effect, g.score.sum(), g.information.sum()], [a.effect, a.score, a.information], atol=1e-12, rtol=0)
        effects.append(effect)
    assert len(effects) == primary.n_animals == 6
    mean = float(np.mean(effects))
    half = float(student_t.ppf(0.975, 5) * np.std(effects, ddof=1) / np.sqrt(6))
    np.testing.assert_allclose([mean, mean - half, mean + half], [primary.mean_effect, primary.ci_low, primary.ci_high], atol=1e-12, rtol=0)
    record = {
        "status": "passed",
        "producer_commit": m["code_commit"],
        "manifest_sha256": sha(result / "manifest.json"),
        "verifier_sha256": sha(Path(__file__)),
        "raw_anchors_checked": len(anchors),
        "raw_cell_rows_checked": len(cells),
        "independent_spatial_scores_checked": checked,
        "maximum_score_or_information_difference": maxdiff,
        "causal_plasticity_or_drug_effect_validated": False,
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
