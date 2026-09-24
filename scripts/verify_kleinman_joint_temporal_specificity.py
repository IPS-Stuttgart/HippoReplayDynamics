#!/usr/bin/env python3
"""Verify raw joint banks, paired summaries and independent conditional moments."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.special import ndtri
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import calibrate_kleinman_joint_temporal_specificity as producer
from verify_kleinman_conditional_coupling import independent_anchor, independent_score, raw_predictors
from verify_kleinman_ripple_run_opportunities import raw_runs
from verify_kleinman_spatial_expression import reference_bank, sha


def raw_exposures(folder, anchor):
    info = loadmat(folder / "session_info.mat", simplify_cells=True)["session_info"]
    t, x, _, runs = raw_runs(info)
    v = np.asarray(info["velocity"])[:, 1]
    dt, dx = np.diff(t), np.diff(x)
    valid = (dt >= 0.001) & (dt <= 0.1) & (np.minimum(v[:-1], v[1:]) >= 0) & (np.maximum(v[:-1], v[1:]) <= 200) & (np.abs(dx) / dt <= 200)
    lookup = {r["traversal"]: r for r in runs}
    refs = [lookup[i] for i in json.loads(anchor.reference_traversals)]
    readouts = [lookup[getattr(anchor, name + "_traversal")] for name in ("past", "baseline", "target")]
    edges = np.arange(np.floor(x.min() / 2) * 2, np.ceil(x.max() / 2) * 2 + 2, 2)
    bins = np.clip(np.searchsorted(edges, (x[:-1] + x[1:]) / 2, side="right") - 1, 0, len(edges) - 2)
    out = []
    for i, rs in enumerate([refs, *[[r] for r in readouts]]):
        mask = np.zeros(len(dt), bool)
        for r in rs:
            mask |= (t[:-1] >= r["start_s"]) & (t[1:] <= r["end_s"]) & (dx * (2 * r["direction"] - 1) > 0)
        mask &= valid & ((v[:-1] + v[1:]) / 2 > (8 if i == 0 else 20))
        if i:
            mask &= ((x[:-1] + x[1:]) / 2 > info["reward_ends"][0] + 20) & ((x[:-1] + x[1:]) / 2 < info["reward_ends"][1] - 20)
        out.append(np.bincount(bins[mask], weights=dt[mask], minlength=len(edges) - 1))
    return np.asarray(out)


def verify(root, dataset):
    m = json.loads((root / "manifest.json").read_text())
    assert not m["git_dirty"] and not m["native_outcomes_scored"] and not m["drug_effect_scored"]
    for name, digest in m["outputs"].items():
        assert sha(root / name) == digest
    for name, path in m["input_file_paths"].items():
        assert sha(Path(path)) == m["input_file_sha256"][name]
    original = pd.read_csv(m["input_file_paths"]["packets.csv"], float_precision="round_trip")
    selected = pd.read_csv(root / "selected_packets.csv", float_precision="round_trip")
    eligible = original.loc[original.selected & original.availability_descriptor]
    expected = set()
    for _, g in eligible.groupby(["animal", "drug", "novel"]):
        h = sorted(hashlib.sha256(f"{m['parameters']['seed']}|{r.animal}|{r.session}|{r.packet_id}".encode()).hexdigest() for r in g.itertuples())
        expected.update(h[:2])
    assert set(selected.selection_key) == expected and len(selected) == 48
    banks, n_predictors = [], 0
    for i, a in enumerate(selected.itertuples()):
        bank = dict(np.load(root / f"banks/{i:03}.npz"))
        folder = dataset / "Experiment_1" / a.animal / a.session
        raw = reference_bank(folder, a.packet_id, a.direction)
        for key in ("generating_rates", "edges", "unit_keys"):
            np.testing.assert_allclose(bank[key], raw[key], atol=1e-9, rtol=1e-10)
        exposures = raw_exposures(folder, a)
        np.testing.assert_allclose(bank["reference_exposure"], exposures[0], atol=1e-10, rtol=0)
        np.testing.assert_allclose(bank["readout_exposure"], exposures[1:], atol=1e-10, rtol=0)
        pred = pd.DataFrame(
            {
                "unit_id": ["_".join(map(str, u.astype(int))) for u in bank["unit_keys"]],
                "ripple_spikes": bank["ripple_counts"],
                "background_spikes": bank["background_counts"],
                "ripple_s": a.eligible_ripple_s,
                "background_s": a.eligible_background_s,
                "log_rate_enrichment": bank["predictor"],
            }
        )
        raw_predictors(folder, SimpleNamespace(**a._asdict(), opportunity_id=a.packet_id), pred)
        n_predictors += len(pred)
        bank.update({k: getattr(a, k) for k in ("animal", "session", "drug", "novel", "direction")})
        bank["identity"] = f"{a.animal}/{a.session}/{a.packet_id}"
        banks.append(bank)
    scores = pd.read_csv(root / "packet_scores.csv.gz", float_precision="round_trip")
    animals = pd.read_csv(root / "animal_scores.csv.gz", float_precision="round_trip")
    estimates = pd.read_csv(root / "replicate_estimates.csv", float_precision="round_trip")
    summary = pd.read_csv(root / "calibration_summary.csv", float_precision="round_trip")
    assert len(scores) == 48 * 8 * 128 and not scores.duplicated(["identity", "case", "replicate"]).any()
    assert len(estimates) == 8 * 128 and len(summary) == 8
    for (rep, case), g in scores.groupby(["replicate", "case"]):
        differences = []
        g = g.loc[g.preceding_information.gt(1e-10) & g.prospective_information.gt(1e-10)]
        for animal, a in g.groupby("animal"):
            preceding = a.preceding_score.sum() / a.preceding_information.sum()
            prospective = a.prospective_score.sum() / a.prospective_information.sum()
            row = animals.loc[animals.replicate.eq(rep) & animals.case.eq(case) & animals.animal.eq(animal)].iloc[0]
            np.testing.assert_allclose([row.preceding_effect, row.prospective_effect, row.contrast], [preceding, prospective, prospective - preceding], atol=1e-12, rtol=0)
            differences.append(prospective - preceding)
        result = estimates.loc[estimates.replicate.eq(rep) & estimates.case.eq(case)].iloc[0]
        assert result.n_animals == len(differences)
        if len(differences) == 6:
            mean, sd = np.mean(differences), np.std(differences, ddof=1)
            half = student_t.ppf(0.975, 5) * sd / np.sqrt(6)
            np.testing.assert_allclose([result.mean_contrast, result.ci_low, result.ci_high], [mean, mean - half, mean + half], atol=1e-12, rtol=0)
            assert result.positive_flag == (mean - half > 0) and result.negative_flag == (mean + half < 0)
        else:
            assert result.status == "missing_information" and not result.flag
    passes = []
    for row in summary.itertuples():
        g = estimates.loc[estimates.case.eq(row.case)]
        flags = g.positive_flag if row.case == "future_only" else g.negative_flag if row.case == "past_only" else g.flag
        k, n, z = int(flags.sum()), len(g), ndtri(0.975)
        p = k / n
        upper = (p + z * z / (2 * n) + z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / (1 + z * z / n)
        assert row.n_flags == k and row.n_replicates == n
        assert np.isclose(row.wilson_high, upper, atol=1e-12, rtol=0)
        complete = g.n_animals.eq(6).all() and g.status.eq("scored").all()
        passed = bool(complete and (p >= 0.8 if row.case in ("future_only", "past_only") else p <= 0.1 and upper <= 0.15))
        assert bool(row.engineering_screen_passed) == passed
        passes.append(passed)
    assert bool(m["engineering_screen_passed"]) == all(passes)
    sample = [next(b for b in banks if b["animal"] == a) for a in sorted({b["animal"] for b in banks})]
    producer.init_worker(sample)
    producer.spatial_score, producer.anchor_score = independent_score, independent_anchor
    checked, max_difference = 0, 0.0
    for rep in (0, 127):
        rows, _, _ = producer.simulate_replica(rep)
        for r in rows:
            actual = scores.loc[scores.replicate.eq(rep) & scores.case.eq(r["case"]) & scores.identity.eq(r["identity"])].iloc[0]
            for lag in ("preceding", "prospective"):
                for what in ("score", "information"):
                    k = lag + "_" + what
                    difference = abs(r[k] - actual[k])
                    max_difference = max(max_difference, difference)
                    np.testing.assert_allclose(r[k], actual[k], atol=2e-4, rtol=1e-5)
            checked += 1
    record = {
        "status": "passed",
        "manifest_sha256": sha(root / "manifest.json"),
        "verifier_sha256": sha(Path(__file__)),
        "raw_banks_rebuilt": len(banks),
        "raw_predictor_rows_checked": n_predictors,
        "paired_intervals_checked": len(estimates),
        "independent_packet_pairs_checked": checked,
        "maximum_moment_difference": max_difference,
        "engineering_screen_passed": all(passes),
        "native_effect_validated": False,
        "helper_sha256": {
            name: sha(ROOT / "scripts" / name)
            for name in ("verify_kleinman_conditional_coupling.py", "verify_kleinman_spatial_expression.py", "verify_kleinman_ripple_run_opportunities.py")
        },
    }
    with (root / "verification.json").open("x") as f:
        json.dump(record, f, indent=2)
        f.write("\n")
    print(json.dumps(record))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--result-dir", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, required=True)
    a = p.parse_args()
    verify(a.result_dir, a.dataset_root)
