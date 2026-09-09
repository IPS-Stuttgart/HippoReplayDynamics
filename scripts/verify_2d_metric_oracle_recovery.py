#!/usr/bin/env python3
"""Independent backward recursion and reconstruction of oracle recovery."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import pairwise, product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from scripts._provenance import build_script_provenance, file_sha256
from scripts.verify_2d_metric_finite_recovery import assert_table, bootstrap_reference, multinomial_log, reference_scores
from scripts.verify_2d_physical_neural_metric import cost, kernel_from_parameters

IDS = ["dataset", "animal", "session"]
ARMS = ["method", "support"]
KEYS = IDS + ["generator", "trial"] + ARMS
MODELS = ["physical", "neural", "stationary", "iid"]
OBSERVED = ["whole_exact", "whole_train_geometry", "lagged_full_geometry", "lagged_train_geometry"]


def backward_evidence(ll, offsets, a):
    lengths = np.diff(offsets)
    n = ll.shape[1]
    message = np.ones((len(lengths), n))
    accumulated = np.zeros(len(lengths))
    for depth in range(max(lengths)):
        active = np.flatnonzero(lengths > depth)
        future = message[active] @ a.T if depth else message[active]
        likelihood = ll[offsets[active + 1] - 1 - depth]
        shift = likelihood.max(axis=1)
        product_ = np.exp(likelihood - shift[:, None]) * future
        scale = product_.sum(axis=1)
        assert np.isfinite(scale).all() and (scale > 0).all()
        message[active] = product_ / scale[:, None]
        accumulated[active] += shift + np.log(scale)
    return accumulated - np.log(n)


def verify_record(item, source, metric, parent, run):
    from hmmlearn import _hmmc

    tag = item["tag"]
    with np.load(source / f"{tag}_cache.npz") as z:
        rates, centers = z["rates"], z["centers"]
        splits = [(z[f"train_{s}"], z[f"held_{s}"]) for s in range(5)]
    with np.load(metric / f"{tag}_kernel_parameters.npz") as z:
        physical = kernel_from_parameters(cost(centers=centers), z, "physical")
        full = kernel_from_parameters(cost(rates=rates), z, "full_neural")
        partial = [kernel_from_parameters(cost(rates=rates[t]), z, f"train_neural_{s}") for s, (t, _) in enumerate(splits)]
    metadata = pd.read_csv(parent / f"{tag}_simulation_trials.csv").sort_values("simulation_index").reset_index(drop=True)
    scores = pd.read_csv(run / f"{tag}_oracle_scores.csv.gz")
    assert len(scores) == 10496
    for field in IDS:
        assert scores[field].eq(item[field]).all()
    for field in ("generator", "trial", "phase", "template_event", "n_full_bins"):
        np.testing.assert_array_equal(scores[field], scores.simulation_index.map(metadata.set_index("simulation_index")[field]))
    maximum, checked, copied, library = 0.0, 0, 0, 0

    def compare(reference, method, support, split, informative):
        nonlocal maximum, checked, copied
        actual = scores[scores.method.eq(method) & scores.support.eq(support) & scores.split.eq(split)].sort_values("simulation_index")
        assert actual.simulation_index.tolist() == list(range(256))
        np.testing.assert_array_equal(actual.informative, np.broadcast_to(informative, (256,)))
        for model in MODELS:
            a, b = actual["score_" + model].to_numpy(), np.asarray(reference[model])
            np.testing.assert_array_equal(np.isneginf(a), np.isneginf(b))
            finite = np.isfinite(a) & np.isfinite(b)
            maximum = max(maximum, float(np.max(abs(a[finite] - b[finite]), initial=0)))
            np.testing.assert_allclose(a, b, rtol=0, atol=1e-8)
            if method == "lagged_train_geometry":
                copied += len(a)
            else:
                checked += len(a)

    baseline = pd.read_csv(parent / f"{tag}_recovery_scores.csv.gz")
    baseline = baseline[baseline.condition.eq("matched") & baseline.origin.eq("decoded")]
    with np.load(parent / f"{tag}_observations.npz") as z:
        states, offsets = z["states"], z["offsets"]
        n = len(physical)
        latent = {m: [] for m in MODELS}
        for i in range(256):
            path = states[offsets[i] : offsets[i + 1]]
            for m, a in (("physical", physical), ("neural", full)):
                latent[m].append(-np.log(n) + sum(np.log(a[left, right]) for left, right in pairwise(path)))
            latent["stationary"].append(-np.log(n) if len(set(path)) == 1 else -np.inf)
            latent["iid"].append(-len(path) * np.log(n))
        compare(latent, "latent_path", 0, -1, True)
        squared = {"physical": physical @ physical, "neural": full @ full}
        for split, (train, held) in enumerate(splits):
            for support in (1, 4):
                x, y = [z[f"s{split}_u{support}_base_{name}"] for name in ("train", "held")]
                ll = multinomial_log(x, rates[train]) + multinomial_log(y, rates[held])
                static, iid, info = [], [], []
                for i in range(256):
                    sl = slice(offsets[i], offsets[i + 1])
                    static.append(logsumexp(ll[sl].sum(axis=0)) - np.log(n))
                    iid.append((logsumexp(ll[sl], axis=1) - np.log(n)).sum())
                    info.append(x[sl].sum() + y[sl].sum() > 0)
                exact = {"physical": backward_evidence(ll, offsets, physical), "neural": backward_evidence(ll, offsets, full), "stationary": static, "iid": iid}
                estimate = exact | {"neural": backward_evidence(ll, offsets, partial[split])}
                compare(exact, "whole_exact", support, split, info)
                compare(estimate, "whole_train_geometry", support, split, info)
                if split == 0:
                    for index in (0, 32, 64, 96, 128, 160, 192, 224):
                        block = ll[offsets[index] : offsets[index + 1]]
                        for a, expected in ((physical, exact["physical"]), (full, exact["neural"]), (partial[split], estimate["neural"])):
                            value, _ = _hmmc.forward_log(np.ones(n) / n, a, block)
                            assert abs(value - expected[index]) < 1e-8
                            library += 1
                forecast = reference_scores(x, y, rates[train], rates[held], np.zeros(len(x), int), offsets, squared)
                forecast = forecast[forecast.origin.eq("decoded")].sort_values("simulation_index")
                available = forecast.n_train_origin_spikes.gt(0) & forecast.n_held_target_spikes.gt(0)
                compare({m: forecast["score_" + m].to_numpy() for m in MODELS}, "lagged_full_geometry", support, split, available.to_numpy())
                old = baseline[baseline.split.eq(split) & baseline.support.eq(support)].sort_values("simulation_index")
                compare({m: old["score_" + m].to_numpy() for m in MODELS}, "lagged_train_geometry", support, split, available.to_numpy())
    return {"tag": tag, "independent_scores": checked, "audited_parent_copies": copied, "hmmlearn_log_checks": library, "max_error": maximum}


def reference_classification(scores):
    expected = {(g, trial, "latent_path", 0, -1) for g in MODELS for trial in range(64)}
    expected |= set(product(MODELS, range(64), OBSERVED, (1, 4), range(5)))
    assert not scores.duplicated(KEYS + ["split"]).any()
    assert scores[IDS].drop_duplicates().shape[0] == 33 and scores.animal.nunique() == 9
    for _, group in scores.groupby(IDS):
        assert len(group) == len(expected) and set(group[["generator", "trial"] + ARMS + ["split"]].itertuples(index=False, name=None)) == expected
    rows = scores.copy()
    delta = rows.score_neural - rows.score_physical
    rows["delta_neural_minus_physical"] = delta.where(delta.abs() > 1e-9, 0.0)
    rows["binary_accuracy"] = np.nan
    for generator in MODELS[:2]:
        mask = rows.generator.eq(generator)
        d = rows.loc[mask, "delta_neural_minus_physical"]
        correct = (d > 0) if generator == "neural" else (d < 0)
        rows.loc[mask, "binary_accuracy"] = np.where(d == 0, 0.5, correct.astype(float))
    rows["structure_statistic"] = np.maximum(rows.score_physical, rows.score_neural) - np.maximum(rows.score_stationary, rows.score_iid)
    rows.loc[~rows.informative, "structure_statistic"] = -np.inf
    values = rows[["score_" + m for m in MODELS]].to_numpy()
    best = values.max(axis=1)
    ties = (np.abs(values - best[:, None]) <= 1e-9).sum(axis=1)
    rows["winner"] = np.where(ties == 1, np.array(MODELS)[values.argmax(axis=1)], "ambiguous")
    thresholds = []
    for key, group in rows.groupby(IDS + ARMS + ["split"]):
        limits = []
        for null in MODELS[2:]:
            cal = group[group.generator.eq(null) & group.trial.lt(32)]
            assert len(cal) == 32
            # ceil(33 * .95) is 32: the largest of the 32 calibration scores.
            limits.append(cal.structure_statistic.max())
        limit = max(limits)
        thresholds.append(dict(zip(IDS + ARMS + ["split"], key, strict=True)) | {"threshold": limit, "finite_calibration": np.isfinite(limit)})
    thresholds = pd.DataFrame(thresholds)
    rows = rows.merge(thresholds.drop(columns="finite_calibration"), on=IDS + ARMS + ["split"], validate="many_to_one")
    rows["structured_detected"] = rows.informative & np.isfinite(rows.threshold) & (rows.structure_statistic > rows.threshold + 1e-9)
    for m in MODELS + ["ambiguous"]:
        rows["winner_fraction_" + m] = (rows.winner == m).astype(float)
    cols = ["binary_accuracy", "structured_detected", "delta_neural_minus_physical", "informative", *("winner_fraction_" + m for m in MODELS + ["ambiguous"])]
    trials = rows.groupby(KEYS, as_index=False)[cols].mean()
    counts = rows.groupby(KEYS, as_index=False).size().rename(columns={"size": "n_realizations"})
    trials = trials.merge(counts, on=KEYS, validate="one_to_one")
    trials["phase"] = np.where(trials.trial < 32, "calibration", "evaluation")
    return rows, thresholds, trials


def verify_reductions(run, scores):
    def read(name):
        return pd.read_csv(run / f"metric_oracle_{name}.csv.gz")

    rows, thresholds, trials = reference_classification(scores)
    assert_table(read("decisions"), rows, KEYS + ["split"])
    assert_table(read("thresholds"), thresholds, IDS + ARMS + ["split"])
    assert_table(read("trials"), trials, KEYS)
    evaluation = trials[trials.phase.eq("evaluation")]
    cols = ["binary_accuracy", "structured_detected", "delta_neural_minus_physical", *("winner_fraction_" + m for m in MODELS + ["ambiguous"])]
    sessions = evaluation.groupby(IDS + ARMS + ["generator"], as_index=False)[cols].mean()
    animals = sessions.groupby(["dataset", "animal"] + ARMS + ["generator"], as_index=False)[cols].mean()
    assert_table(read("sessions"), sessions, IDS + ARMS + ["generator"])
    assert_table(read("animals"), animals, ["dataset", "animal"] + ARMS + ["generator"])
    summary = []
    for key, group in evaluation.groupby(["dataset"] + ARMS):
        base = dict(zip(["dataset"] + ARMS, key, strict=True))
        moving = group[group.generator.isin(MODELS[:2])]
        summary.append(base | {"metric": "balanced_accuracy"} | bootstrap_reference(moving, "binary_accuracy", ("oracle", *key)))
        for g in MODELS:
            summary.append(base | {"metric": g + "_structured_detection"} | bootstrap_reference(group[group.generator.eq(g)], "structured_detected", ("oracle", *key, g)))
    summary = pd.DataFrame(summary)
    assert_table(read("summary"), summary, ["dataset"] + ARMS + ["metric"])
    moving = evaluation[evaluation.generator.isin(MODELS[:2])]
    paired = []
    for left, right, label in (
        ("whole_exact", "whole_train_geometry", "whole_geometry_increment"),
        ("lagged_full_geometry", "lagged_train_geometry", "lagged_geometry_increment"),
        ("whole_exact", "lagged_full_geometry", "whole_information_inference_increment"),
        ("latent_path", "whole_exact", "latent_information_gap"),
    ):
        for support in (1, 4):
            a = moving[(moving.method == left) & (moving.support == (0 if left == "latent_path" else support))]
            b = moving[(moving.method == right) & (moving.support == support)]
            merged = a.merge(b, on=IDS + ["generator", "trial"], validate="one_to_one", suffixes=("_a", "_b"))
            assert len(merged) == len(a) == len(b)
            merged["value"] = merged.binary_accuracy_a - merged.binary_accuracy_b
            for dataset, group in merged.groupby("dataset"):
                paired.append({"dataset": dataset, "contrast": label, "support": support} | bootstrap_reference(group, "value", ("oracle", label, support, dataset)))
    assert_table(read("paired"), pd.DataFrame(paired), ["dataset", "contrast", "support"])
    checks = []
    for key, group in summary.groupby(["dataset"] + ARMS):
        d, method, support = key
        e = group.set_index("metric")
        acc = e.loc["balanced_accuracy"]
        power = min(e.loc[g + "_structured_detection", "mean"] for g in MODELS[:2])
        fpr = max(e.loc[g + "_structured_detection", "mean"] for g in MODELS[2:])
        t = thresholds[(thresholds.dataset == d) & (thresholds.method == method) & (thresholds.support == support)]
        finite = len(t) > 0 and t.finite_calibration.all()
        checks.append(
            dict(zip(["dataset"] + ARMS, key, strict=True))
            | {
                "accuracy": acc["mean"],
                "accuracy_ci_low": acc.ci_low,
                "min_animal_accuracy": acc.min_animal,
                "min_moving_power": power,
                "max_null_fpr": fpr,
                "finite_calibration": finite,
                "operating_pass": bool(acc.ci_low > 0.5 and acc.min_animal > 0.5 and power >= 0.5 and fpr <= 0.05 and finite),
            }
        )
    checks = pd.DataFrame(checks)
    assert len(checks) == 18
    assert_table(read("checks"), checks, ["dataset"] + ARMS)
    decision = read("decision")
    assert len(decision) == 1
    assert decision.whole_event_oracle_native_operating_pass.item() == checks[(checks.method == "whole_exact") & (checks.support == 1)].operating_pass.all()
    assert decision.latent_path_operating_pass.item() == checks[checks.method == "latent_path"].operating_pass.all()
    assert not decision[["real_events_rescored", "biological_mechanism_established", "new_real_scoring_authorized"]].any().any()
    return len(trials)


def audit_run(run, output, workers):
    mp = run / "metric_oracle_manifest.json"
    m = json.loads(mp.read_text())
    assert m["status"] == "complete" and m["rows"] == 346368 and len(m["completed"]) == 33
    assert not m["git_dirty"] and not m["real_events_rescored"]
    for k, path in m["input_file_paths"].items():
        assert file_sha256(Path(path)) == m["input_file_sha256"][k]
    for name, digest in m["output_sha256"].items():
        assert file_sha256(run / name) == digest
    parent, source, metric = [Path(m[x + "_dir"]) for x in ("parent", "source", "metric")]
    for path, manifest in ((parent, "metric_finite_recovery_manifest.json"), (metric, "physical_neural_metric_manifest.json")):
        previous = json.loads((path / manifest).read_text())
        for name, digest in previous["output_sha256"].items():
            assert file_sha256(path / name) == digest
    original = json.loads((source / "conditional_2d_manifest.json").read_text())
    for item in m["completed"]:
        name = item["tag"] + "_cache.npz"
        assert file_sha256(source / name) == original["output_sha256"][name]
    details = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(verify_record, x, source, metric, parent, run) for x in m["completed"]]
        for future in as_completed(futures):
            detail = future.result()
            details.append(detail)
            print(json.dumps(detail), flush=True)
    scores = pd.concat([pd.read_csv(run / f"{x['tag']}_oracle_scores.csv.gz") for x in m["completed"]], ignore_index=True)
    trials = verify_reductions(run, scores)
    p = build_script_provenance(input_paths={"run_manifest": mp, "verifier": Path(__file__)}, cwd=ROOT)
    p.update(
        status="pass",
        independently_reconstructed_scores=sum(x["independent_scores"] for x in details),
        verified_parent_copies=sum(x["audited_parent_copies"] for x in details),
        hmmlearn_log_crosschecks=sum(x["hmmlearn_log_checks"] for x in details),
        max_absolute_score_error=max(x["max_error"] for x in details),
        trial_reductions_checked=trials,
        scope="All source/output hashes; all new scores by independent backward recursion, path probability or independent predictive mixtures; all baseline copies against independently audited parent; all reductions/calibration/bootstrap/decisions. Additional log-domain hmmlearn checks in every recording. No raw RUN map refitting or biological event scoring.",
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
    audit_run(args.run_dir, args.output, args.workers)
