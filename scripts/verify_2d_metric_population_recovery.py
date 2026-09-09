#!/usr/bin/env python3
"""Independent reconstruction, RNG audit and convex optimality certificates."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from scripts._provenance import build_script_provenance, file_sha256
from scripts.verify_2d_metric_finite_recovery import assert_table, multinomial_log
from scripts.verify_2d_metric_oracle_recovery import backward_evidence
from scripts.verify_2d_physical_neural_metric import cost, kernel_from_parameters

IDS = ["dataset", "animal", "session"]
MODELS = ["physical", "neural", "stationary", "iid"]
CONDITIONS = ["exact", "train_geometry", "gain_drift"]
SCENARIOS = [0.25, 0.5, 0.75]
LIMIT = 3.841458820694124


def independent_rng(*parts):
    b = "|".join(map(str, (20260909, "population_v1", *parts))).encode()
    return np.random.default_rng(int.from_bytes(hashlib.sha256(b).digest()[:16], "little"))


def audit_record(item, source, metric, run):
    from hmmlearn import _hmmc

    tag = item["tag"]
    with np.load(source / f"{tag}_cache.npz") as z:
        rates, centers, train, held = [z[k] for k in ("rates", "centers", "train_0", "held_0")]
        profiles = {}
        for key in sorted(k for k in z.files if k.startswith("counts_")):
            event = int(key.split("_")[1])
            widths = np.diff(z[f"edges_{event}"])
            x = z[key]
            assert (widths > 0).all() and (widths <= 0.02 + 1e-8).all() and np.allclose(widths[:-1], 0.02, atol=1e-8, rtol=0)
            if abs(widths[-1] - 0.02) > 1e-8:
                x = x[:-1]
            if len(x) >= 3:
                profiles[event] = x
    with np.load(metric / f"{tag}_kernel_parameters.npz") as z:
        physical = kernel_from_parameters(cost(centers=centers), z, "physical")
        neural = kernel_from_parameters(cost(rates=rates), z, "full_neural")
        estimated = kernel_from_parameters(cost(rates=rates[train]), z, "train_neural_0")
    teachers = [physical, neural, np.eye(len(neural)), np.ones_like(neural) / len(neural)]
    with np.load(run / f"{tag}_parameters.npz") as z:
        gains = z["gains"]
    np.testing.assert_array_equal(gains, np.exp(0.35 * independent_rng(tag, "gains").standard_normal(len(rates)) - 0.35**2 / 2))
    maximum, n_scores, n_arrays, n_paths, n_library = 0.0, 0, 0, 0, 0
    for repeat in range(50):
        prefix = run / f"{tag}_r{repeat:03d}"
        meta = pd.read_csv(str(prefix) + "_events.csv")
        scores = pd.read_csv(str(prefix) + "_scores.csv.gz")
        assert len(meta) == 384 and len(scores) == 1152
        assert not scores.duplicated(["simulation_index", "condition"]).any()
        assert meta.simulation_index.tolist() == list(range(384))
        assert set(scores.condition) == set(CONDITIONS)
        for key in IDS:
            assert meta[key].eq(item[key]).all() and scores[key].eq(item[key]).all()
        for key in meta.columns:
            np.testing.assert_array_equal(scores[key], scores.simulation_index.map(meta.set_index("simulation_index")[key]))
        chosen = independent_rng(tag, repeat, "profiles").choice(sorted(profiles), 128, replace=True)
        with np.load(str(prefix) + "_observations.npz") as z:
            states, offsets = z["states"], z["offsets"]
            assert len(offsets) == 385 and offsets[0] == 0 and offsets[-1] == len(states)
            expected_totals = []
            for index, phi in enumerate(SCENARIOS):
                sub = meta.iloc[index * 128 : (index + 1) * 128]
                assert sub.scenario.eq(phi).all() and sub.repeat.eq(repeat).all()
                assert sub.event_in_recording.tolist() == list(range(128))
                np.testing.assert_array_equal(sub.template_event, chosen)
                labels = independent_rng(tag, repeat, phi, "labels").choice(MODELS, 128, p=[0.6 * (1 - phi), 0.6 * phi, 0.2, 0.2])
                np.testing.assert_array_equal(sub.generator, labels)
                for i, row in enumerate(sub.itertuples()):
                    profile = profiles[row.template_event]
                    assert row.n_bins == len(profile) and row.n_spikes == profile.sum()
                    sl = slice(offsets[row.simulation_index], offsets[row.simulation_index + 1])
                    path = states[sl]
                    assert len(path) == len(profile)
                    r = independent_rng(tag, repeat, phi, i, "path")
                    expected = [r.integers(len(neural))]
                    a = teachers[MODELS.index(row.generator)]
                    for _ in range(1, len(path)):
                        p = a[expected[-1]]
                        expected.append(r.choice(len(p), p=p / p.sum()))
                    np.testing.assert_array_equal(path, expected)
                    expected_totals.append(profile)
                    n_paths += 1
            counts = np.concatenate(expected_totals)
            for emitter in ("base", "gain"):
                truth = rates if emitter == "base" else rates * gains[:, None]
                arrays = []
                for name, cells in (("train", train), ("held", held)):
                    x = z[emitter + "_" + name]
                    totals = counts[:, cells].sum(axis=1)
                    np.testing.assert_array_equal(x.sum(axis=1), totals)
                    p = truth[cells] / truth[cells].sum(axis=0, keepdims=True)
                    r = independent_rng(tag, repeat, emitter, name)
                    regenerated = np.array([r.multinomial(int(n), p[:, s]) for n, s in zip(totals, states, strict=True)])
                    np.testing.assert_array_equal(x, regenerated)
                    arrays.append(x)
                    n_arrays += 1
                ll = multinomial_log(arrays[0], rates[train]) + multinomial_log(arrays[1], rates[held])
                references = {"physical": backward_evidence(ll, offsets, physical), "neural": backward_evidence(ll, offsets, neural), "stationary": [], "iid": []}
                for i in range(384):
                    block = ll[offsets[i] : offsets[i + 1]]
                    references["stationary"].append(logsumexp(block.sum(axis=0)) - np.log(len(neural)))
                    references["iid"].append((logsumexp(block, axis=1) - np.log(len(neural))).sum())
                candidates = [("exact" if emitter == "base" else "gain_drift", references)]
                if emitter == "base":
                    candidates.append(("train_geometry", references | {"neural": backward_evidence(ll, offsets, estimated)}))
                for condition, reference in candidates:
                    actual = scores[scores.condition.eq(condition)].sort_values("simulation_index")
                    assert actual.simulation_index.tolist() == list(range(384))
                    for m in MODELS:
                        difference = np.max(abs(actual["score_" + m].to_numpy() - np.asarray(reference[m])))
                        maximum = max(maximum, float(difference))
                        np.testing.assert_allclose(actual["score_" + m], reference[m], atol=1e-8, rtol=0)
                        n_scores += 384
                    if repeat in (0, 49):
                        for index in (0, 128, 256):
                            a = estimated if condition == "train_geometry" else neural
                            value, _ = _hmmc.forward_log(np.ones(len(neural)) / len(neural), a, ll[offsets[index] : offsets[index + 1]])
                            assert abs(value - reference["neural"][index]) < 1e-8
                            n_library += 1
    return {"tag": tag, "scores": n_scores, "count_arrays": n_arrays, "paths": n_paths, "library_checks": n_library, "max_error": maximum}


def certify(log_scores, weights):
    ll, w = np.asarray(log_scores, float), np.asarray(weights, float)
    assert np.isfinite(ll).all() and np.isfinite(w).all() and abs(w.sum() - 1) < 1e-8 and (w >= 0.5e-9).all()
    mixture = logsumexp(ll + np.log(w)[None, :], axis=1)
    gradient = np.exp(ll - mixture[:, None]).mean(axis=0)
    free = w > 1e-8
    level = gradient[free].mean()
    error = max(np.max(abs(gradient[free] - level), initial=0), np.max(gradient[~free] - level, initial=0))
    assert error <= 2.2e-5, error
    return float(mixture.sum())


def verify_fit(ll, row):
    ll = ll - ll.max(axis=1, keepdims=True)
    w = np.array([row["weight_" + m] for m in MODELS])
    value = certify(ll, w)
    assert abs(value - row.relative_log_likelihood) < 1e-5
    assert abs(row.phi_hat - w[1] / (w[0] + w[1])) < 1e-7
    assert abs(row.moving_weight - w[:2].sum()) < 1e-7
    assert 0 <= row.phi_low <= row.phi_hat + 1e-6 <= row.phi_high + 1e-6 <= 1 + 1e-6
    for label, phi in (("low", row.phi_low), ("high", row.phi_high), ("null", 0.5)):
        with np.errstate(divide="ignore"):
            combined = np.logaddexp(ll[:, 0] + np.log1p(-phi), ll[:, 1] + np.log(phi))
        prof = np.column_stack([combined, ll[:, 2:]])
        weights = [row[label + "_weight_" + m] for m in ("moving", "stationary", "iid")]
        pv = certify(prof, weights)
        assert abs(pv - row[label + "_profile_log_likelihood"]) < 1e-5
        lr = max(0.0, 2 * (value - pv))
        if label == "null":
            assert abs(lr - row.null_lr) < 1e-5
        elif phi in (0, 1):
            assert lr <= LIMIT + 1e-4
        else:
            assert abs(lr - LIMIT) < 5e-4
    direction = "neural" if row.phi_low > 0.5 else "physical" if row.phi_high < 0.5 else "undetermined"
    assert row.direction == direction


def audit_fit_job(dataset, repeat, items, run, fits):
    items = [x for x in items if x["dataset"] == dataset]
    data = pd.concat([pd.read_csv(run / f"{x['tag']}_r{repeat:03d}_scores.csv.gz") for x in items])
    selected = fits[(fits.dataset == dataset) & (fits.repeat == repeat)]
    assert len(selected) == 18
    for _, row in selected.iterrows():
        events = data[(data.condition == row.condition) & (data.scenario == row.scenario) & (data.event_in_recording < row.events_per_recording)]
        assert len(events) == row.n_events == row.events_per_recording * len(items)
        assert row.n_recordings == len(items) and row.n_animals == events.animal.nunique()
        assert len(events[IDS].drop_duplicates()) == len(items)
        verify_fit(events[["score_" + m for m in MODELS]].to_numpy(), row)
    return len(selected)


def reference_summary(fits):
    keys = ["dataset", "scenario", "repeat", "events_per_recording", "condition"]
    expected = set(product(("pfeiffer_foster", "tanni2022"), SCENARIOS, range(50), (32, 128), CONDITIONS))
    assert len(fits) == len(expected) and not fits.duplicated(keys).any()
    assert set(fits[keys].itertuples(index=False, name=None)) == expected
    work = fits.copy()
    work["error"] = work.phi_hat - work.scenario
    work["covered"] = (work.phi_low <= work.scenario + 1e-8) & (work.phi_high >= work.scenario - 1e-8)
    work["directional_claim"] = (work.phi_low > 0.5) | (work.phi_high < 0.5)
    work["correct_direction"] = ((work.scenario > 0.5) & (work.phi_low > 0.5)) | ((work.scenario < 0.5) & (work.phi_high < 0.5))
    summary = []
    for key, group in work.groupby(["dataset", "condition", "events_per_recording", "scenario"]):
        row = dict(zip(["dataset", "condition", "events_per_recording", "scenario"], key, strict=True))
        row.update(
            n_replicates=50,
            mean_estimate=group.phi_hat.mean(),
            bias=group.error.mean(),
            rmse=np.sqrt(np.mean(group.error**2)),
            median_interval_width=(group.phi_high - group.phi_low).median(),
        )
        for col in ("covered", "directional_claim", "correct_direction"):
            p, z, n = group[col].mean(), 1.959963984540054, len(group)
            lo = (p + z * z / (2 * n) - z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / (1 + z * z / n)
            hi = (p + z * z / (2 * n) + z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / (1 + z * z / n)
            row.update({col + "_fraction": p, col + "_mc_low": lo, col + "_mc_high": hi})
        summary.append(row)
    summary = pd.DataFrame(summary)
    gates = []
    for key, group in summary.groupby(["dataset", "condition", "events_per_recording"]):
        bias = group.bias.abs().max()
        coverage = group.covered_fraction.min()
        null = group.loc[group.scenario.eq(0.5), "directional_claim_fraction"].item()
        power = group.loc[group.scenario.ne(0.5), "correct_direction_fraction"].min()
        gates.append(
            dict(zip(["dataset", "condition", "events_per_recording"], key, strict=True))
            | {
                "min_direction_power": power,
                "min_coverage": coverage,
                "max_absolute_bias": bias,
                "null_false_direction_fraction": null,
                "practical_pass": bool(power >= 0.8 and coverage >= 0.9 and bias <= 0.1 and null <= 0.05),
            }
        )
    return work, summary, pd.DataFrame(gates)


def audit(run, output, workers):
    mp = run / "metric_population_manifest.json"
    m = json.loads(mp.read_text())
    assert m["status"] == "complete" and len(m["completed"]) == 33 and m["fit_rows"] == 1800 and m["rows"] == 1900800
    assert not m["git_dirty"] and not m["real_events_rescored"] and not m["biological_mechanism_established"]
    for key, path in m["input_file_paths"].items():
        assert file_sha256(Path(path)) == m["input_file_sha256"][key]
    for name, digest in m["output_sha256"].items():
        assert file_sha256(run / name) == digest
    source, metric = Path(m["source_dir"]), Path(m["metric_dir"])
    sm, km = [json.loads(Path(m["input_file_paths"][name]).read_text()) for name in ("source_manifest", "metric_manifest")]
    for item in m["completed"]:
        for directory, name, hashes in ((source, item["tag"] + "_cache.npz", sm["output_sha256"]), (metric, item["tag"] + "_kernel_parameters.npz", km["output_sha256"])):
            assert file_sha256(directory / name) == hashes[name]
    details = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(audit_record, x, source, metric, run) for x in m["completed"]]
        for future in as_completed(futures):
            details.append(future.result())
            print(json.dumps(details[-1]), flush=True)
    fits = pd.read_csv(run / "metric_population_fits.csv")
    checked = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(audit_fit_job, d, r, m["completed"], run, fits) for d in ("pfeiffer_foster", "tanni2022") for r in range(50)]
        for future in as_completed(futures):
            checked += future.result()
    rows, summary, gates = reference_summary(fits)
    assert_table(fits, rows, ["dataset", "scenario", "repeat", "events_per_recording", "condition"])
    assert_table(pd.read_csv(run / "metric_population_summary.csv"), summary, ["dataset", "condition", "events_per_recording", "scenario"])
    assert_table(pd.read_csv(run / "metric_population_gates.csv"), gates, ["dataset", "condition", "events_per_recording"])
    decision = pd.read_csv(run / "metric_population_decision.csv")
    assert len(decision) == 1
    primary = gates[gates.events_per_recording.eq(128)]
    assert decision.exact_population_recovery_pass.item() == primary[primary.condition.eq("exact")].practical_pass.all()
    assert decision.robust_population_recovery_pass.item() == primary.practical_pass.all()
    assert not decision[["real_events_rescored", "biological_mechanism_established", "new_real_scoring_authorized"]].any().any()
    p = build_script_provenance(input_paths={"run_manifest": mp, "verifier": Path(__file__)}, cwd=ROOT)
    p.update(
        status="pass",
        likelihoods_checked=sum(x["scores"] for x in details),
        fresh_paths_regenerated=sum(x["paths"] for x in details),
        count_arrays_regenerated=sum(x["count_arrays"] for x in details),
        library_checks=sum(x["library_checks"] for x in details),
        max_score_error=max(x["max_error"] for x in details),
        mixture_fits_checked=checked,
        scope="All generated labels/paths/counts, backward likelihood reconstruction, all simplex KKT certificates/profile limits, summaries and hashes. Source RUN maps not refitted; no biological scoring.",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(p, indent=2) + "\n")
    print(json.dumps(p), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    audit(args.run_dir, args.output, args.workers)
