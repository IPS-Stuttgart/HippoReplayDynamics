#!/usr/bin/env python3
"""Verify raw simulation banks and independently recalculate conditional moments."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from verify_kleinman_ripple_run_opportunities import expected_pairs, raw_runs


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def smooth(values):
    offsets = np.arange(-8, 9)
    kernel = np.exp(-(offsets**2) / 8)
    kernel /= kernel.sum()
    return np.convolve(values, kernel, mode="same")


def reference_bank(folder, opportunity, side):
    info = loadmat(folder / "session_info.mat", simplify_cells=True)["session_info"]
    t, x, _, runs = raw_runs(info)
    base, target, refs = expected_pairs(runs)[opportunity]
    v = np.asarray(info["velocity"])[:, 1]
    dt, dx = np.diff(t), np.diff(x)
    valid = (dt >= 0.001) & (dt <= 0.1) & (np.maximum(v[:-1], v[1:]) <= 200) & (np.minimum(v[:-1], v[1:]) >= 0) & (np.abs(dx) / dt <= 200)
    d = np.full(len(dt), -1)

    def mask(rs):
        keep = np.zeros(len(dt), bool)
        for r in rs:
            keep |= (t[:-1] >= r["start_s"]) & (t[1:] <= r["end_s"])
        return keep

    for r in runs:
        d[mask([r])] = r["direction"]
    train = valid & (d >= 0) & ((v[:-1] + v[1:]) / 2 > 8) & (dx * (2 * d - 1) > 0)
    edges = np.arange(np.floor(x.min() / 2) * 2, np.ceil(x.max() / 2) * 2 + 2, 2)
    nb = len(edges) - 1
    bins = np.clip(np.searchsorted(edges, (x[:-1] + x[1:]) / 2, side="right") - 1, 0, nb - 1)
    raw = np.atleast_2d(loadmat(folder / "spike_data.mat", simplify_cells=True)["spike_data"])
    raw = raw[np.all(raw[:, 1:] > 0, axis=1)]
    keys = np.unique(raw[:, [2, 1]].astype(int), axis=0)
    c = np.zeros((len(dt), len(keys)))
    for u, (tt, cc) in enumerate(keys):
        s = raw[(raw[:, 2] == tt) & (raw[:, 1] == cc), 0]
        c[:, u] = np.histogram(s, bins=t)[0]
        c[-1, u] -= np.count_nonzero(s == t[-1])
    rates = np.zeros((2, nb, len(keys)))
    totals = np.zeros(len(keys))
    for direction in (0, 1):
        m = train & (d == direction)
        occ = np.bincount(bins[m], weights=dt[m], minlength=nb)
        for u in range(len(keys)):
            n = np.bincount(bins[m], weights=c[m, u], minlength=nb)
            rates[direction, :, u] = smooth(n) / np.maximum(smooth(occ), 1e-12)
            totals[u] += n.sum()
    keep = (totals >= 10) & (rates.max(axis=(0, 1)) >= 1)
    readout = train & ((v[:-1] + v[1:]) / 2 > 20) & ((x[:-1] + x[1:]) / 2 > info["reward_ends"][0] + 20) & ((x[:-1] + x[1:]) / 2 < info["reward_ends"][1] - 20)
    result = {"edges": edges, "unit_keys": keys[keep], "generating_rates": np.maximum(rates[side, :, keep], 1e-5)}
    for key, m in (("reference", train & mask(refs)), ("baseline", readout & mask([base])), ("target", readout & mask([target]))):
        result[key + "_exposure"] = np.bincount(bins[m], weights=dt[m], minlength=nb)
    return result


def scalar_moment(truth, model, occupancy):
    normal_true = sum(float(r) * float(t) for r, t in zip(truth, occupancy, strict=True))
    normal_ref = sum(float(r) * float(t) for r, t in zip(model, occupancy, strict=True))
    probabilities = [float(r) * float(t) / normal_true for r, t in zip(truth, occupancy, strict=True)]
    logs = [float(np.log(r)) for r in model]
    mean = sum(p * l for p, l in zip(probabilities, logs, strict=True))
    centered = mean - sum(float(t) * float(r) * l / normal_ref for t, r, l in zip(occupancy, model, logs, strict=True))
    var = sum(p * (l - mean) ** 2 for p, l in zip(probabilities, logs, strict=True))
    return centered, var


def verify(root, source):
    m = json.loads((root / "manifest.json").read_text())
    assert not m["git_dirty"] and not m["real_future_outcome_scored"] and not m["drug_effect_scored"]
    for name, digest in m["outputs"].items():
        assert sha(root / name) == digest, name
    for name, path in m["input_file_paths"].items():
        assert sha(Path(path)) == m["input_file_sha256"][name], name
    anchors = pd.read_csv(root / "selected_anchors.csv", float_precision="round_trip")
    inv = pd.read_csv(root / "bank_inventory.csv", float_precision="round_trip")
    rows = pd.read_csv(root / "calibration_rows.csv.gz", float_precision="round_trip")
    fits = pd.read_csv(root / "reference_fit_records.csv.gz", float_precision="round_trip")
    seed = m["parameters"]["seed"]
    original = pd.read_csv(m["input_file_paths"]["opportunities"], float_precision="round_trip")
    eligible = original.loc[original.original_run_pass & original.minimum_units_descriptor & original.status.eq("audited")]
    for _, g in eligible.groupby(["animal", "drug", "novel", "direction"]):
        expected = min(hashlib.sha256(f"{seed}|{r.animal}|{r.session}|{r.opportunity_id}".encode()).hexdigest() for r in g.itertuples())
        assert expected in set(anchors.selection_key)
    assert len(anchors) == len(inv) == 48
    grouping = ["anchor", "unit_id", "reference_gain", "replicate"]
    assert rows.groupby(grouping).size().eq(12).all()
    assert not rows.duplicated(grouping + ["arm", "case"]).any()
    assert set(rows.arm) == {"known_map", "estimated_map"}
    assert set(rows.case) == set(m["cases"])
    assert len(fits) == len(rows) // 12
    checked = 0
    for index, a in anchors.iterrows():
        bank = dict(np.load(root / f"banks/{index:03}.npz"))
        raw = reference_bank(source / "Experiment_1" / a.animal / a.session, a.opportunity_id, a.direction)
        for name in raw:
            np.testing.assert_allclose(bank[name], raw[name], atol=1e-9, rtol=1e-10, err_msg=name)
        identity = f"{a.animal}/{a.session}/{a.opportunity_id}"
        sub = rows.loc[rows.anchor.eq(identity)].set_index(["unit_id", "reference_gain", "replicate", "arm", "case"])
        for unit, truth in list(zip(bank["unit_keys"], bank["generating_rates"], strict=True))[:2]:
            uid = "_".join(map(str, unit.astype(int)))
            for gain in (0.25, 1.0, 4.0):
                for rep in (0, 15):
                    digest = hashlib.sha256(f"{seed}|{identity}|{uid}|{gain}|{rep}".encode()).digest()
                    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
                    counts = rng.poisson(bank["reference_exposure"] * truth * gain)
                    model = smooth(counts) / np.maximum(smooth(bank["reference_exposure"]), 1e-12)
                    included = bool(counts.sum() >= 10 and model.max() >= 1)
                    model = np.maximum(model, 1e-5)
                    for arm, ref in (("known_map", truth), ("estimated_map", model)):
                        for case in m["cases"]:
                            record = sub.loc[(uid, gain, rep, arm, case)]
                            assert bool(record.included) == included
                            if case == "unchanged":
                                target = truth
                            elif case == "gain_only_4x":
                                target = truth * 4
                            elif case == "sharpen_1p5":
                                target = (truth / truth.max()) ** 1.5
                            elif case == "broaden_0p5":
                                target = np.sqrt(truth / truth.max())
                            elif case == "uniform":
                                target = np.ones_like(truth)
                            else:
                                sign_hash = hashlib.sha256(f"{seed}|{identity}|{uid}|shift".encode()).digest()
                                sign = 1 if int.from_bytes(sign_hash[:8], "little") % 2 else -1
                                shift = sign * 10
                                target = np.full(len(truth), 1e-5)
                                for b in range(len(truth)):
                                    if 0 <= b - shift < len(truth):
                                        target[b] = truth[b - shift]
                            bm, bv = scalar_moment(truth, ref, bank["baseline_exposure"])
                            tm, tv = scalar_moment(target, ref, bank["target_exposure"])
                            np.testing.assert_allclose([record.expected_change_nats_per_spike, record.variance_at_one_spike_per_readout], [tm - bm, tv + bv], atol=1e-8, rtol=1e-10)
                            for n in (5, 20, 80):
                                assert np.isclose(record[f"sd_at_{n}_spikes"], np.sqrt((tv + bv) / n), atol=1e-9, rtol=1e-10)
                            checked += 1
    summary = pd.read_csv(root / "calibration_summary.csv", float_precision="round_trip")
    keys = ["animal", "drug", "novel", "reference_gain", "arm", "case"]
    groups = dict(tuple(rows.loc[rows.included].groupby(keys)))
    for _, row in summary.iterrows():
        g = groups[tuple(row[key] for key in keys)]
        assert len(g) == row.n_rows and g.anchor.nunique() == row.n_anchors
        x = g.expected_change_nats_per_spike.to_numpy()
        np.testing.assert_allclose(
            [row.mean_expected_change, row.median_expected_change, row.p10_expected_change, row.p90_expected_change],
            [np.mean(x), np.median(x), np.quantile(x, 0.1), np.quantile(x, 0.9)],
            atol=1e-10,
            rtol=1e-10,
        )
    result = {
        "status": "passed",
        "producer_commit": m["code_commit"],
        "manifest_sha256": sha(root / "manifest.json"),
        "verifier_sha256": sha(Path(__file__)),
        "raw_banks_rebuilt": len(anchors),
        "independent_scalar_moments": checked,
        "summary_rows_checked": len(summary),
        "real_coupling_or_drug_effect_validated": False,
    }
    with (root / "verification.json").open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--result-dir", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, required=True)
    args = p.parse_args()
    verify(args.result_dir, args.dataset_root)
