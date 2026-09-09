#!/usr/bin/env python3
"""Independent reconstruction of finite metric simulations and decisions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import pairwise, product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np
import pandas as pd
from scipy.special import gammaln, logsumexp

from scripts._provenance import build_script_provenance, file_sha256
from scripts.verify_2d_physical_neural_metric import cost, kernel_from_parameters

IDS = ["dataset", "animal", "session"]
ARMS = ["condition", "support", "origin"]
MODELS = ["physical", "neural", "stationary", "iid"]
CONDITIONS = ["matched", "gain_drift", "map_error"]
KEYS = IDS + ["generator", "trial"] + ARMS


def seeded(*parts):
    data = "|".join(str(x) for x in (20260909, *parts)).encode()
    return np.random.default_rng(int.from_bytes(hashlib.sha256(data).digest()[:16], "little"))


def multinomial_log(counts, rates):
    p = rates / rates.sum(axis=0)
    return counts @ np.log(p) + gammaln(counts.sum(axis=1) + 1)[:, None] - gammaln(counts + 1).sum(axis=1)[:, None]


def reference_scores(x, y, maps_train, maps_held, states, offsets, kernels):
    origins = np.concatenate([np.arange(offsets[i], offsets[i + 1] - 2) for i in range(len(offsets) - 1)])
    target = origins + 2
    ll = multinomial_log(x[origins], maps_train)
    neutral = np.exp(ll - logsumexp(ll, axis=1)[:, None])
    oracle = np.zeros_like(neutral)
    oracle[np.arange(len(origins)), states[origins]] = 1
    held_ll = multinomial_log(y[target], maps_held)
    totals = y[target].sum(axis=1)
    slices = np.r_[0, np.cumsum(np.diff(offsets) - 2)]
    result = []
    for arm, q in (("decoded", neutral), ("known", oracle)):
        scores = {}
        for model in MODELS:
            if model == "stationary":
                f = q
            elif model == "iid":
                f = np.full_like(q, 1 / q.shape[1])
            else:
                f = q @ kernels[model]
            assert np.isfinite(f).all() and (f >= 0).all()
            np.testing.assert_allclose(f.sum(axis=1), 1, atol=1e-9)
            log_f = np.full_like(f, -np.inf)
            np.log(f, out=log_f, where=f > 0)
            s = logsumexp(log_f + held_ll, axis=1)
            s[totals == 0] = 0
            scores[model] = s
        for i in range(len(offsets) - 1):
            sl = slice(slices[i], slices[i + 1])
            result.append(
                {
                    "simulation_index": i,
                    "origin": arm,
                    "n_target_bins": sl.stop - sl.start,
                    "n_train_origin_spikes": int(x[origins[sl]].sum()),
                    "n_held_target_spikes": int(y[target[sl]].sum()),
                    **{"score_" + m: float(scores[m][sl].sum()) for m in MODELS},
                }
            )
    return pd.DataFrame(result)


def verify_record(item, source, parent, run):
    tag = item["tag"]
    with np.load(source / f"{tag}_cache.npz") as z:
        rates, centers = z["rates"], z["centers"]
        splits = [(z[f"train_{s}"], z[f"held_{s}"]) for s in range(5)]
        profiles, discarded = {}, 0
        keys = sorted(k for k in z.files if k.startswith("counts_"))
        for key in keys:
            event = int(key.split("_")[1])
            x, edges = z[key], z[f"edges_{event}"]
            widths = np.diff(edges)
            assert len(widths) == len(x) and (widths > 0).all()
            np.testing.assert_allclose(widths[:-1], 0.02, atol=1e-8, rtol=0)
            assert widths[-1] <= 0.02 + 1e-8
            if abs(widths[-1] - 0.02) > 1e-8:
                discarded += int(x[-1].sum())
                x = x[:-1]
            if len(x) >= 3:
                profiles[event] = x.astype(np.int32)
    chosen = seeded(tag, "templates").choice(sorted(profiles), 64, replace=True)
    assert item["profiles"] == len(keys) and item["eligible_profiles"] == len(profiles)
    assert item["unique_sampled_profiles"] == len(set(chosen)) and item["discarded_partial_spikes"] == discarded
    with np.load(parent / f"{tag}_kernel_parameters.npz") as z:
        physical = kernel_from_parameters(cost(centers=centers), z, "physical")
        full = kernel_from_parameters(cost(rates=rates), z, "full_neural")
        train_k = [kernel_from_parameters(cost(rates=rates[t]), z, f"train_neural_{s}") for s, (t, _) in enumerate(splits)]
    with np.load(run / f"{tag}_perturbation_parameters.npz") as z:
        gains, error_maps = z["gains"], z["error_maps"]
        error_k = [kernel_from_parameters(cost(rates=error_maps[t]), z, f"train_neural_{s}") for s, (t, _) in enumerate(splits)]
    random = seeded(tag, "observation_stress")
    np.testing.assert_array_equal(gains, np.exp(0.35 * random.standard_normal(len(rates)) - 0.35**2 / 2))
    smooth = np.exp(-cost(centers=centers) / 512.0)
    smooth /= smooth.sum(axis=1)[:, None]
    noise = random.standard_normal(rates.shape) @ smooth.T / np.sqrt(np.square(smooth).sum(axis=1))[None, :]
    estimated = rates * np.exp(0.35 * noise - 0.35**2 / 2)
    estimated *= rates.mean(axis=1)[:, None] / estimated.mean(axis=1)[:, None]
    np.testing.assert_allclose(estimated, error_maps, rtol=1e-12, atol=1e-12)
    metadata = pd.read_csv(run / f"{tag}_simulation_trials.csv")
    assert len(metadata) == 256 and metadata.simulation_index.tolist() == list(range(256))
    teachers = [physical, full, np.eye(len(physical)), np.ones_like(physical) / len(physical)]
    states, source_counts, offsets = [], [], [0]
    for generator, kernel in zip(MODELS, teachers, strict=True):
        for trial, event in enumerate(chosen):
            x = profiles[event]
            rng = seeded(tag, generator, trial, "path")
            path = [int(rng.integers(len(kernel)))]
            for _ in range(1, len(x)):
                p = kernel[path[-1]]
                path.append(int(rng.choice(len(p), p=p / p.sum())))
            row = metadata.iloc[len(states)]
            assert row.generator == generator and row.trial == trial and row.template_event == event
            assert row.phase == ("calibration" if trial < 32 else "evaluation") and row.n_full_bins == len(x)
            states.append(np.array(path))
            source_counts.append(x)
            offsets.append(offsets[-1] + len(x))
    states, source_counts, offsets = np.concatenate(states), np.concatenate(source_counts), np.array(offsets)
    scores = pd.read_csv(run / f"{tag}_recovery_scores.csv.gz")
    assert len(scores) == 15360
    for field in IDS:
        assert scores[field].eq(item[field]).all()
    for field in ("generator", "trial", "phase", "template_event", "n_full_bins"):
        expected = scores.simulation_index.map(metadata.set_index("simulation_index")[field])
        np.testing.assert_array_equal(scores[field], expected)
    checked, max_error, arrays = 0, 0.0, 0
    physical2 = physical @ physical
    with np.load(run / f"{tag}_observations.npz") as saved:
        assert len(saved.files) == 46
        np.testing.assert_array_equal(states, saved["states"])
        np.testing.assert_array_equal(offsets, saved["offsets"])
        np.testing.assert_array_equal(chosen, saved["template_ids"])
        origins = np.concatenate([np.arange(a, b - 2) for a, b in pairwise(offsets)])
        np.testing.assert_array_equal(origins, saved["origins"])
        np.testing.assert_array_equal(origins + 2, saved["targets"])
        np.testing.assert_array_equal(np.r_[0, np.cumsum(np.diff(offsets) - 2)], saved["target_offsets"])
        for split, (train, held) in enumerate(splits):
            assert not set(train) & set(held)
            assert sorted([*train, *held]) == list(range(len(rates)))
            moving = {"matched": train_k[split] @ train_k[split], "map_error": error_k[split] @ error_k[split]}
            moving["gain_drift"] = moving["matched"]
            for support in (1, 4):
                samples = {}
                for emitter in ("base", "gain"):
                    maps = rates if emitter == "base" else rates * gains[:, None]
                    for name, cells in (("train", train), ("held", held)):
                        n = support * source_counts[:, cells].sum(axis=1)
                        p = maps[cells] / maps[cells].sum(axis=0)
                        rng = seeded(tag, split, support, emitter, name)
                        expected = np.array([rng.multinomial(int(total), p[:, state]) for total, state in zip(n, states, strict=True)])
                        actual = saved[f"s{split}_u{support}_{emitter}_{name}"]
                        np.testing.assert_array_equal(expected, actual)
                        np.testing.assert_array_equal(n, actual.sum(axis=1))
                        arrays += 1
                        samples[emitter, name] = actual
                for condition in CONDITIONS:
                    emitter = "gain" if condition == "gain_drift" else "base"
                    maps = error_maps if condition == "map_error" else rates
                    reference = reference_scores(
                        samples[emitter, "train"], samples[emitter, "held"], maps[train], maps[held], states, offsets, {"physical": physical2, "neural": moving[condition]}
                    )
                    selected = scores[scores.split.eq(split) & scores.support.eq(support) & scores.condition.eq(condition)]
                    merged = selected.merge(reference, on=["simulation_index", "origin"], validate="one_to_one", suffixes=("", "_reference"))
                    assert len(merged) == 512
                    for field in ["n_target_bins", "n_train_origin_spikes", "n_held_target_spikes"]:
                        np.testing.assert_array_equal(merged[field], merged[field + "_reference"])
                    for model in MODELS:
                        field = "score_" + model
                        max_error = max(max_error, float(np.max(abs(merged[field] - merged[field + "_reference"]))))
                        np.testing.assert_allclose(merged[field], merged[field + "_reference"], rtol=0, atol=1e-8)
                        checked += len(merged)
    return {"tag": tag, "scores_checked": checked, "observation_arrays_checked": arrays, "max_absolute_score_error": max_error, "kernels_checked": 12}


def assert_table(actual, expected, keys):
    assert len(actual) == len(expected) and set(actual.columns) == set(expected.columns)
    assert not actual.duplicated(keys).any()
    a, b = [x.sort_values(keys).reset_index(drop=True) for x in (actual, expected)]
    # CSV preserves labels, but not the pandas column-axis name from a pivot.
    a.columns.name = b.columns.name = None
    pd.testing.assert_frame_equal(a[expected.columns], b, check_dtype=False, atol=1e-10, rtol=1e-10)


def reference_reductions(scores):
    assert not scores.duplicated(KEYS + ["split"]).any()
    required = set(product(MODELS, range(64), CONDITIONS, (1, 4), ("decoded", "known"), range(5)))
    for _, group in scores.groupby(IDS):
        assert set(group[["generator", "trial"] + ARMS + ["split"]].itertuples(index=False, name=None)) == required
    assert scores[IDS].drop_duplicates().shape[0] == 33 and scores.animal.nunique() == 9
    contrasts = scores[KEYS + ["n_held_target_spikes", "n_train_origin_spikes"]].copy()
    for m in MODELS:
        contrasts["margin_" + m] = scores["score_" + m] - scores.score_physical
    trials = contrasts.groupby(KEYS, as_index=False).median()
    for m in MODELS:
        name = "margin_" + m
        trials.loc[trials[name].abs() <= 1e-9, name] = 0.0
    trials["phase"] = np.where(trials.trial < 32, "calibration", "evaluation")
    trials["informative"] = (trials.n_held_target_spikes > 0) & ((trials.origin == "known") | (trials.n_train_origin_spikes > 0))
    trials["structure_statistic"] = np.maximum(trials.margin_physical, trials.margin_neural) - np.maximum(trials.margin_stationary, trials.margin_iid)
    trials.loc[~trials.informative, "structure_statistic"] = -np.inf
    trials["binary_accuracy"] = np.nan
    for g in MODELS[:2]:
        mask = trials.generator.eq(g)
        d = trials.loc[mask, "margin_neural"]
        trials.loc[mask, "binary_accuracy"] = np.where(d == 0, 0.5, ((d > 0) if g == "neural" else (d < 0)).astype(float))
    values = trials[["margin_" + m for m in MODELS]].to_numpy()
    ties = (np.abs(values - values.max(axis=1)[:, None]) <= 1e-9).sum(axis=1)
    trials["winner"] = np.where(ties == 1, np.array(MODELS)[values.argmax(axis=1)], "ambiguous")
    thresholds = []
    for key, group in trials.groupby(IDS + ARMS):
        nulls = []
        for g in ("stationary", "iid"):
            values = group[group.generator.eq(g) & group.trial.lt(32)].structure_statistic.to_numpy()
            assert len(values) == 32
            nulls.append(np.sort(values)[math.ceil(33 * 0.95) - 1])
        limit = max(nulls)
        thresholds.append(dict(zip(IDS + ARMS, key, strict=True)) | {"threshold": limit, "finite_calibration": np.isfinite(limit), "n_calibration_per_null": 32})
    thresholds = pd.DataFrame(thresholds)
    trials = trials.merge(thresholds[IDS + ARMS + ["threshold"]], on=IDS + ARMS, validate="many_to_one")
    trials["structured_detected"] = trials.informative & np.isfinite(trials.threshold) & (trials.structure_statistic > trials.threshold + 1e-9)
    return trials, thresholds


def bootstrap_reference(frame, metric, key):
    rng = seeded(*key, metric, "bootstrap")
    estimates, draws = [], []
    for _, animal in frame.groupby("animal"):
        local, sampled = [], []
        for _, session in animal.groupby("session"):
            matrix = session.pivot(index="trial", columns="generator", values=metric).sort_index().to_numpy(dtype=float)
            assert matrix.shape[0] == 32 and np.isfinite(matrix).all()
            local.append(matrix.mean())
            samples = rng.integers(matrix.shape[0], size=(2000, matrix.shape[0]))
            sampled.append(matrix[samples].mean(axis=(1, 2)))
        estimates.append(np.mean(local))
        draws.append(np.mean(sampled, axis=0))
    lo, hi = np.quantile(np.mean(draws, axis=0), [0.025, 0.975])
    return {"mean": np.mean(estimates), "ci_low": lo, "ci_high": hi, "animals": len(estimates), "min_animal": min(estimates)}


def verify_reductions(run, scores):
    def read(name):
        return pd.read_csv(run / f"metric_finite_recovery_{name}.csv.gz")

    trials, thresholds = reference_reductions(scores)
    assert_table(read("trials"), trials, KEYS)
    assert_table(read("thresholds"), thresholds, IDS + ARMS)
    evaluation = trials[trials.trial >= 32]
    session = evaluation.groupby(IDS + ARMS + ["generator"], as_index=False).agg(
        binary_accuracy=("binary_accuracy", "mean"),
        structured_detection=("structured_detected", "mean"),
        mean_neural_minus_physical=("margin_neural", "mean"),
        n_trials=("trial", "size"),
    )
    animal = session.groupby(["dataset", "animal"] + ARMS + ["generator"], as_index=False)[["binary_accuracy", "structured_detection", "mean_neural_minus_physical"]].mean()
    assert_table(read("sessions"), session, IDS + ARMS + ["generator"])
    assert_table(read("animals"), animal, ["dataset", "animal"] + ARMS + ["generator"])
    rows = []
    for key, group in evaluation.groupby(["dataset"] + ARMS):
        base = dict(zip(["dataset"] + ARMS, key, strict=True))
        moving = group[group.generator.isin(MODELS[:2])]
        rows.append(base | {"metric": "balanced_metric_accuracy"} | bootstrap_reference(moving, "binary_accuracy", key))
        for g in MODELS:
            rows.append(base | {"metric": g + "_structured_detection"} | bootstrap_reference(group[group.generator.eq(g)], "structured_detected", (*key, g)))
    summary = pd.DataFrame(rows)
    assert_table(read("summary"), summary, ["dataset"] + ARMS + ["metric"])
    checks = []
    for (d, c), group in summary[(summary.support == 1) & (summary.origin == "decoded")].groupby(["dataset", "condition"]):
        endpoint = group.set_index("metric")
        accuracy = endpoint.loc["balanced_metric_accuracy"]
        power = min(endpoint.loc[g + "_structured_detection", "mean"] for g in MODELS[:2])
        fpr = max(endpoint.loc[g + "_structured_detection", "mean"] for g in MODELS[2:])
        sub = thresholds[(thresholds.dataset == d) & (thresholds.condition == c) & (thresholds.support == 1) & (thresholds.origin == "decoded")]
        finite = len(sub) > 0 and sub.finite_calibration.all()
        checks.append(
            {
                "dataset": d,
                "condition": c,
                "accuracy": accuracy["mean"],
                "accuracy_ci_low": accuracy.ci_low,
                "min_animal_accuracy": accuracy.min_animal,
                "min_moving_power": power,
                "max_null_fpr": fpr,
                "finite_calibration": finite,
                "ready": accuracy.ci_low > 0.5 and accuracy.min_animal > 0.5 and power >= 0.5 and fpr <= 0.05 and finite,
            }
        )
    checks = pd.DataFrame(checks)
    assert_table(read("checks"), checks, ["dataset", "condition"])
    decision = read("decision")
    assert len(checks) == 6 and len(decision) == 1
    assert decision.native_matched_ready.item() == checks[checks.condition == "matched"].ready.all()
    assert decision.robust_native_ready.item() == checks.ready.all()
    assert not decision.real_events_rescored.item() and not decision.biological_mechanism_established.item()
    winners = evaluation.groupby(IDS + ARMS + ["generator", "winner"]).size().unstack("winner", fill_value=0).reindex(columns=MODELS + ["ambiguous"], fill_value=0)
    winners = winners.div(32).add_prefix("winner_fraction_").reset_index()
    assert_table(read("winners"), winners, IDS + ARMS + ["generator"])
    return len(trials)


def run_audit(run, output, workers):
    mp = run / "metric_finite_recovery_manifest.json"
    m = json.loads(mp.read_text())
    assert m["status"] == "complete" and len(m["completed"]) == 33 and m["rows"] == 506880
    assert not m["git_dirty"] and not m["real_events_rescored"]
    for name, digest in m["output_sha256"].items():
        assert file_sha256(run / name) == digest, name
    for name, path in m["input_file_paths"].items():
        assert file_sha256(Path(path)) == m["input_file_sha256"][name], name
    source, parent = Path(m["source_dir"]), Path(m["parent_dir"])
    sm = json.loads((source / "conditional_2d_manifest.json").read_text())
    pm = json.loads((parent / "physical_neural_metric_manifest.json").read_text())
    for item in m["completed"]:
        name = item["tag"] + "_cache.npz"
        assert file_sha256(source / name) == sm["output_sha256"][name]
    for name, digest in pm["output_sha256"].items():
        assert file_sha256(parent / name) == digest
    details = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(verify_record, x, source, parent, run) for x in m["completed"]]
        for future in as_completed(futures):
            row = future.result()
            details.append(row)
            print(json.dumps(row), flush=True)
    scores = pd.concat([pd.read_csv(run / f"{x['tag']}_recovery_scores.csv.gz") for x in m["completed"]], ignore_index=True)
    trials = verify_reductions(run, scores)
    p = build_script_provenance(input_paths={"run_manifest": mp, "verifier": Path(__file__)}, cwd=ROOT)
    p.update(
        status="pass",
        recordings=len(details),
        predictive_scores_checked=sum(x["scores_checked"] for x in details),
        observation_arrays_checked=sum(x["observation_arrays_checked"] for x in details),
        kernels_checked=sum(x["kernels_checked"] for x in details),
        max_absolute_score_error=max(x["max_absolute_score_error"] for x in details),
        trial_reductions_checked=trials,
        scope="All source/output hashes, all simulation paths and observation draws/count totals, matched/stressed kernels, all predictive scores, all event medians/calibration/bootstrap summaries and decisions. RUN map estimation and raw recording timestamps not repeated.",
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
    run_audit(args.run_dir, args.output, args.workers)
