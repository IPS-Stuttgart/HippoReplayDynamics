#!/usr/bin/env python3
"""Independent raw-predictor and log-partition checks of conditional calibration."""

import argparse
import hashlib
import json
from pathlib import Path

import calibrate_kleinman_conditional_coupling as producer
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.special import gammaln, ndtri
from scipy.stats import t as student_t
from verify_kleinman_ripple_run_opportunities import expected_pairs, raw_runs


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def independent_score(before, after, t0, t1, feature):
    common = (t0 > 0) & (t1 > 0)
    c = (before + after)[common].astype(int)
    a = feature[common]
    m = int(after[common].sum())
    n = int(c.sum())
    observed = float(after[common] @ a)
    odds = np.log(t1[common] / t0[common])
    if m == 0 or m == n or np.ptp(a[c > 0]) < 1e-10:
        u, v = 0.0, 0.0
    else:

        def partition(beta):
            dp = np.full(m + 1, -np.inf)
            dp[0] = 0
            for count, weight, value in zip(c, odds, a, strict=True):
                if count == 0:
                    continue
                new = np.full(m + 1, -np.inf)
                for j in range(min(count, m) + 1):
                    l = gammaln(count + 1) - gammaln(j + 1) - gammaln(count - j + 1) + j * (weight + beta * value)
                    new[j:] = np.logaddexp(new[j:], dp[: m + 1 - j] + l)
                dp = new
            return dp[m]

        h = 0.003
        mm, minus, zero, plus, pp = [partition(b) for b in (-2 * h, -h, 0, h, 2 * h)]
        mean = (mm - 8 * minus + 8 * plus - pp) / (12 * h)
        var = (-pp + 16 * plus - 30 * zero + 16 * minus - mm) / (12 * h * h)
        u, v = observed - mean, max(0.0, var)
    return {
        "score": u,
        "information": v,
        "informative": bool(v > 1e-10),
        "before_common_spikes": int(before[common].sum()),
        "after_common_spikes": m,
        "excluded_spikes": int((before + after)[~common].sum()),
    }


def independent_anchor(x, u, v):
    x, u, v = [np.asarray(a) for a in (x, u, v)]
    if v.sum() <= 1e-10:
        return 0.0, 0.0
    return float(np.dot(x, u) - np.dot(x, v) * u.sum() / v.sum()), float(np.dot(x * x, v) - np.dot(x, v) ** 2 / v.sum())


def raw_predictors(folder, anchor, rows):
    info = loadmat(folder / "session_info.mat", simplify_cells=True)["session_info"]
    t, x, visits, runs = raw_runs(info)
    base, target, _ = expected_pairs(runs)[anchor.opportunity_id]
    velocity = np.asarray(info["velocity"])[:, 1]
    dt, dx = np.diff(t), np.diff(x)
    valid = (dt >= 0.001) & (dt <= 0.1) & (velocity[:-1] >= 0) & (velocity[1:] >= 0) & (velocity[:-1] <= 8) & (velocity[1:] <= 8) & (np.abs(dx) / dt <= 200)
    pieces = []
    for lo, hi, _ in visits:
        lo, hi = max(lo, base["end_s"]), min(hi, lo + 10, target["start_s"])
        if hi <= lo:
            continue
        for i in np.flatnonzero(valid & (t[:-1] < hi) & (t[1:] > lo)):
            pieces.append((max(lo, t[i]), min(hi, t[i + 1])))

    def union(pieces):
        result = []
        for a, b in sorted(pieces):
            if result and a <= result[-1][1]:
                result[-1][1] = max(b, result[-1][1])
            else:
                result.append([a, b])
        return result

    pieces = union(pieces)
    native = np.asarray(loadmat(folder / "ripple_events.mat", simplify_cells=True)["ripple_events"]).reshape(-1, 4)
    clipped = union([(max(a, lo), min(b, hi)) for a, b, _, _ in native for lo, hi in pieces if max(a, lo) < min(b, hi)])
    rt = sum(b - a for a, b in clipped)
    bt = sum(b - a for a, b in pieces) - rt
    raw = np.atleast_2d(loadmat(folder / "spike_data.mat", simplify_cells=True)["spike_data"])
    for row in rows.itertuples():
        tt, cc = map(int, row.unit_id.split("_"))
        s = raw[(raw[:, 2] == tt) & (raw[:, 1] == cc), 0]
        rc = sum(np.count_nonzero((s >= a) & (s < b)) for a, b in clipped)
        bc = sum(np.count_nonzero((s >= a) & (s < b)) for a, b in pieces) - rc
        assert rc == row.ripple_spikes and bc == row.background_spikes
        np.testing.assert_allclose([rt, bt], [row.ripple_s, row.background_s], atol=1e-9, rtol=0)
        if rt > 0 and bt > 0:
            assert np.isclose(np.log((rc + 0.5) * bt / ((bc + 0.5) * rt)), row.log_rate_enrichment, atol=1e-12, rtol=0)
        else:
            assert np.isnan(row.log_rate_enrichment)
    return pieces, clipped


def verify(result, dataset):
    m = json.loads((result / "manifest.json").read_text())
    assert not m["git_dirty"] and not m["real_future_outcome_scored"] and not m["drug_effect_scored"]
    for name, digest in m["outputs"].items():
        assert sha(result / name) == digest, name
    for name, path in m["input_file_paths"].items():
        assert sha(Path(path)) == m["input_file_sha256"][name], name
    anchors = pd.read_csv(m["input_file_paths"]["selected_anchors.csv"], float_precision="round_trip")
    preds = pd.read_csv(result / "predictors.csv", float_precision="round_trip")
    stored_intervals = {r["identity"]: r for r in json.loads((result / "exposure_intervals.json").read_text())}
    assert len(anchors) == 48 and not preds.duplicated(["identity", "unit_id"]).any()
    banks = []
    for idx, a in anchors.iterrows():
        identity = f"{a.animal}/{a.session}/{a.opportunity_id}"
        rows = preds.loc[preds.identity.eq(identity)]
        exp, rip = raw_predictors(dataset / "Experiment_1" / a.animal / a.session, a, rows)
        np.testing.assert_allclose(exp, stored_intervals[identity]["eligible"], atol=1e-10, rtol=0)
        np.testing.assert_allclose(rip, stored_intervals[identity]["ripples"], atol=1e-10, rtol=0)
        bank = dict(np.load(m["input_file_paths"][f"bank_{idx}"]))
        assert list(rows.unit_id) == ["_".join(map(str, k.astype(int))) for k in bank["unit_keys"]]
        bank.update({k: getattr(a, k) for k in ("animal", "session", "drug", "novel", "direction")})
        bank.update(identity=identity, predictor=rows.log_rate_enrichment.to_numpy())
        banks.append(bank)
    ar = pd.read_csv(result / "anchor_scores.csv.gz", float_precision="round_trip")
    animals = pd.read_csv(result / "animal_scores.csv.gz", float_precision="round_trip")
    estimates = pd.read_csv(result / "replicate_estimates.csv", float_precision="round_trip")
    summary = pd.read_csv(result / "calibration_summary.csv", float_precision="round_trip")
    assert len(ar) == 48 * 128 * 7 and not ar.duplicated(["identity", "case", "replicate"]).any()
    assert len(estimates) == 128 * 7 and len(summary) == 7
    for (rep, case), g in ar.groupby(["replicate", "case"]):
        effects = []
        for animal, a in g.loc[g.information > 1e-10].groupby("animal"):
            effect = a.score.sum() / a.information.sum()
            row = animals.loc[animals.replicate.eq(rep) & animals.case.eq(case) & animals.animal.eq(animal)].iloc[0]
            assert np.isclose(effect, row.effect, atol=1e-12, rtol=0)
            effects.append(effect)
        assert len(effects) == 6
        mean = float(np.mean(effects))
        half = float(student_t.ppf(0.975, 5) * np.std(effects, ddof=1) / np.sqrt(6))
        row = estimates.loc[estimates.replicate.eq(rep) & estimates.case.eq(case)].iloc[0]
        np.testing.assert_allclose([mean, mean - half, mean + half], [row.mean_effect, row.ci_low, row.ci_high], atol=1e-12, rtol=0)
        assert bool(row.flag) == (abs(mean) > half)
    for row in summary.itertuples():
        g = estimates.loc[estimates.case.eq(row.case)]
        k = int(g.correct_flag.sum() if row.planted else g.flag.sum())
        assert k == row.n_flags
        z = ndtri(0.975)
        p = k / len(g)
        upper = (p + z * z / (2 * len(g)) + z * np.sqrt(p * (1 - p) / len(g) + z * z / (4 * len(g) ** 2))) / (1 + z * z / len(g))
        assert np.isclose(upper, row.wilson_high, atol=1e-12, rtol=0)
        passed = k / len(g) >= 0.8 if row.planted else k / len(g) <= 0.1 and upper <= 0.15
        assert bool(row.engineering_screen_passed) == passed
    selected = []
    for animal in sorted({b["animal"] for b in banks}):
        selected.extend([b for b in banks if b["animal"] == animal and np.isfinite(b["predictor"]).all()][:2])
    producer.init_worker(selected)
    producer.spatial_score = independent_score
    producer.anchor_score = independent_anchor
    checked = 0
    max_difference = 0.0
    for rep in (0, 127):
        independent, _, _ = producer.simulate_replica(rep)
        for row in independent:
            actual = ar.loc[ar.replicate.eq(rep) & ar.case.eq(row["case"]) & ar.identity.eq(row["identity"])].iloc[0]
            for key in ("score", "information"):
                delta = abs(actual[key] - row[key])
                max_difference = max(max_difference, delta)
                assert np.isclose(actual[key], row[key], atol=2e-4, rtol=1e-5), (rep, row["identity"], row["case"], key, delta)
            for key in ("before_common_spikes", "after_common_spikes", "excluded_spikes", "n_reference_included"):
                assert row[key] == actual[key]
            checked += 1
    record = {
        "status": "passed",
        "manifest_sha256": sha(result / "manifest.json"),
        "producer_commit": m["code_commit"],
        "verifier_sha256": sha(Path(__file__)),
        "raw_predictor_rows_checked": len(preds),
        "raw_opportunity_windows_checked": 48,
        "replicate_intervals_recomputed": len(estimates),
        "independent_partition_anchor_scores": checked,
        "maximum_partition_moment_difference": max_difference,
        "real_biological_association_validated": False,
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
