#!/usr/bin/env python3
"""Frozen finite-spike geometry recovery, not real replay scoring."""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np
import pandas as pd

from hipporeplayimm.lagged_neural_prediction import full_count_bins
from hipporeplayimm.metric_finite_recovery import (
    CONDITIONS,
    GENERATORS,
    MODELS,
    ORIGINS,
    SUPPORTS,
    TRIALS,
    null_threshold,
    observation_stress,
    origin_target_indices,
    reconstruct_kernel,
    rng_for,
    sample_identities,
    sample_path,
    score_batch,
)
from hipporeplayimm.physical_neural_metric import matched_kernel, neural_cost, physical_cost
from scripts._provenance import build_script_provenance, file_sha256

PARENT_SHA = "71882248599186e0e790b408d1524a98d713d4c3220225004e00aafc05fde937"
ID = ["dataset", "animal", "session"]
ARM = ["condition", "support", "origin"]
KEY = ID + ["generator", "trial"] + ARM


def task(item, source, parent, out):
    started = time.monotonic()
    tag = item["tag"]
    with np.load(source / f"{tag}_cache.npz", allow_pickle=False) as z:
        rates, centers = z["rates"], z["centers"]
        splits = [(z[f"train_{s}"], z[f"held_{s}"]) for s in range(5)]
        profiles, discarded = {}, 0
        keys = sorted(k for k in z.files if k.startswith("counts_"))
        for key in keys:
            event = int(key.split("_")[1])
            x, partial = full_count_bins(z[key], z[f"edges_{event}"])
            discarded += partial
            if len(x) >= 3:
                profiles[event] = x.astype(np.int32)
    if not profiles:
        raise ValueError("recording without eligible count profiles")
    chosen = rng_for(tag, "templates").choice(sorted(profiles), TRIALS, replace=True)
    with np.load(parent / f"{tag}_kernel_parameters.npz", allow_pickle=False) as z:
        a = reconstruct_kernel(physical_cost(centers), z, "physical")
        full = reconstruct_kernel(neural_cost(rates), z, "full_neural")
        original = [reconstruct_kernel(neural_cost(rates[train]), z, f"train_neural_{s}") for s, (train, _) in enumerate(splits)]
    gains, error_maps = observation_stress(rates, centers, tag)
    params = {"gains": gains, "error_maps": error_maps}
    perturbed = []
    for s, (train, held) in enumerate(splits):
        if not np.array_equal(np.sort(np.r_[train, held]), np.arange(len(rates))):
            raise ValueError("invalid source cell partition")
        k = matched_kernel(neural_cost(error_maps[train]))
        perturbed.append(k.matrix)
        for name in ("beta", "cost_scale", "log_scale", "entropy", "target_entropy"):
            params[f"train_neural_{s}__{name}"] = getattr(k, name)
    teachers = {"physical": a, "neural": full, "stationary": np.eye(len(a)), "iid": np.full_like(a, 1 / len(a))}
    details, states, totals, offsets = [], [], [], [0]
    for generator in GENERATORS:
        for trial, event in enumerate(chosen):
            profile = profiles[int(event)]
            path = sample_path(teachers[generator], len(profile), rng_for(tag, generator, trial, "path"))
            details.append(
                {
                    "simulation_index": len(details),
                    "generator": generator,
                    "trial": trial,
                    "phase": "calibration" if trial < TRIALS // 2 else "evaluation",
                    "template_event": int(event),
                    "n_full_bins": len(profile),
                }
            )
            states.append(path)
            totals.append(profile)
            offsets.append(offsets[-1] + len(profile))
    states, counts = np.concatenate(states), np.concatenate(totals)
    offsets = np.asarray(offsets)
    origins, targets, groups = origin_target_indices(offsets)
    observations = {"states": states, "offsets": offsets, "template_ids": chosen, "origins": origins, "targets": targets, "target_offsets": groups}
    metadata = pd.DataFrame(details)
    metadata.to_csv(out / f"{tag}_simulation_trials.csv", index=False)
    rows = []
    physical2 = a @ a
    for split, (train, held) in enumerate(splits):
        neural2 = {"matched": original[split] @ original[split], "map_error": perturbed[split] @ perturbed[split]}
        neural2["gain_drift"] = neural2["matched"]
        for support in SUPPORTS:
            samples = {}
            for emitter in ("base", "gain"):
                truth = rates if emitter == "base" else rates * gains[:, None]
                x = sample_identities(truth, train, states, support * counts[:, train].sum(axis=1), rng_for(tag, split, support, emitter, "train"))
                y = sample_identities(truth, held, states, support * counts[:, held].sum(axis=1), rng_for(tag, split, support, emitter, "held"))
                samples[emitter] = x, y
                observations[f"s{split}_u{support}_{emitter}_train"] = x
                observations[f"s{split}_u{support}_{emitter}_held"] = y
            for condition in CONDITIONS:
                x, y = samples["gain" if condition == "gain_drift" else "base"]
                decoder = error_maps if condition == "map_error" else rates
                scores = pd.DataFrame(score_batch(x, y, decoder[train], decoder[held], states, offsets, {"physical": physical2, "neural": neural2[condition]}))
                scores = scores.merge(metadata, on="simulation_index", validate="many_to_one")
                for key in ID:
                    scores[key] = item[key]
                scores["split"], scores["support"], scores["condition"] = split, support, condition
                rows.append(scores)
    table = pd.concat(rows, ignore_index=True)
    table.to_csv(out / f"{tag}_recovery_scores.csv.gz", index=False)
    np.savez_compressed(out / f"{tag}_observations.npz", **observations)
    np.savez_compressed(out / f"{tag}_perturbation_parameters.npz", **params)
    return {
        **{k: item[k] for k in ID},
        "tag": tag,
        "profiles": len(keys),
        "eligible_profiles": len(profiles),
        "unique_sampled_profiles": len(set(chosen)),
        "discarded_partial_spikes": discarded,
        "simulations": len(details),
        "rows": len(table),
        "runtime_s": time.monotonic() - started,
    }


def validate_factors(scores):
    if scores.empty or scores.duplicated(KEY + ["split"]).any():
        raise ValueError("nonempty unique simulation factors required")
    if scores[KEY + ["split", "phase"]].isna().any().any():
        raise ValueError("missing simulation factor metadata")
    expected = set(product(GENERATORS, range(TRIALS), CONDITIONS, SUPPORTS, ORIGINS, range(5)))
    for _, frame in scores.groupby(ID):
        actual = set(frame[["generator", "trial"] + ARM + ["split"]].itertuples(index=False, name=None))
        if actual != expected or len(frame) != len(expected):
            raise ValueError("incomplete simulation factor coverage")
    phase = np.where(scores.trial.lt(TRIALS // 2), "calibration", "evaluation")
    if not np.array_equal(scores.phase, phase):
        raise ValueError("changed calibration/evaluation assignment")
    counts = scores[["n_held_target_spikes", "n_train_origin_spikes"]].to_numpy()
    if not np.isfinite(counts).all() or np.any(counts < 0) or not np.array_equal(counts, np.round(counts)):
        raise ValueError("invalid simulated spike totals")


def reduce_trials(scores):
    validate_factors(scores)
    columns = ["score_" + m for m in MODELS]
    if not np.isfinite(scores[columns].to_numpy()).all() or not scores[columns].le(1e-8).all().all():
        raise ValueError("improper or missing log scores")
    work = scores.copy()
    for model in MODELS:
        work["margin_" + model] = work["score_" + model] - work.score_physical
    margins = ["margin_" + m for m in MODELS]
    result = work.groupby(KEY, as_index=False)[margins + ["n_held_target_spikes", "n_train_origin_spikes"]].median()
    for m in margins:
        result.loc[result[m].abs().le(1e-9), m] = 0
    result["phase"] = np.where(result.trial.lt(TRIALS // 2), "calibration", "evaluation")
    result["informative"] = result.n_held_target_spikes.gt(0) & (result.origin.eq("known") | result.n_train_origin_spikes.gt(0))
    result["structure_statistic"] = result[["margin_physical", "margin_neural"]].max(axis=1) - result[["margin_stationary", "margin_iid"]].max(axis=1)
    result.loc[~result.informative, "structure_statistic"] = -np.inf
    delta = result.margin_neural
    result["binary_accuracy"] = np.nan
    for generator, sign in (("physical", -1), ("neural", 1)):
        mask = result.generator.eq(generator)
        result.loc[mask, "binary_accuracy"] = np.where(delta[mask].eq(0), 0.5, (sign * delta[mask] > 0).astype(float))
    values = result[margins].to_numpy()
    unique = (np.abs(values - values.max(axis=1, keepdims=True)) <= 1e-9).sum(axis=1) == 1
    result["winner"] = np.where(unique, np.asarray(MODELS)[values.argmax(axis=1)], "ambiguous")
    return result


def calibrate(trials):
    tables, thresholds = [], []
    for key, group in trials.groupby(ID + ARM):
        expected = {(g, t) for g in GENERATORS for t in range(TRIALS)}
        if set(group[["generator", "trial"]].itertuples(index=False, name=None)) != expected or len(group) != len(expected):
            raise ValueError("missing simulation/control trials")
        values = []
        for null in ("stationary", "iid"):
            cal = group[group.generator.eq(null) & group.phase.eq("calibration")]
            values.append(null_threshold(cal.structure_statistic.to_numpy()))
        threshold = max(values)
        entry = dict(zip(ID + ARM, key, strict=True)) | {"threshold": threshold, "finite_calibration": bool(np.isfinite(threshold)), "n_calibration_per_null": TRIALS // 2}
        thresholds.append(entry)
        frame = group.copy()
        frame["threshold"] = threshold
        frame["structured_detected"] = frame.informative & np.isfinite(threshold) & frame.structure_statistic.gt(threshold + 1e-9)
        tables.append(frame)
    return pd.concat(tables, ignore_index=True), pd.DataFrame(thresholds)


def mc_interval(frame, metric, parts):
    rng = rng_for(*parts, metric, "bootstrap")
    animal_values, animal_draws = [], []
    for _, animal in frame.groupby("animal"):
        session_values, session_draws = [], []
        for _, session in animal.groupby("session"):
            x = session.pivot(index="trial", columns="generator", values=metric).sort_index().to_numpy(dtype=float)
            if not np.isfinite(x).all() or not len(x):
                raise ValueError("invalid Monte Carlo observations")
            # Generators share count-profile draws; resample their trial index together.
            session_values.append(x.mean())
            session_draws.append(x[rng.integers(len(x), size=(2000, len(x)))].mean(axis=(1, 2)))
        animal_values.append(np.mean(session_values))
        animal_draws.append(np.mean(session_draws, axis=0))
    low, high = np.quantile(np.mean(animal_draws, axis=0), [0.025, 0.975])
    return {"mean": float(np.mean(animal_values)), "ci_low": low, "ci_high": high, "animals": len(animal_values), "min_animal": min(animal_values)}


def aggregate(scores):
    trials, thresholds = calibrate(reduce_trials(scores))
    evaluation = trials[trials.phase.eq("evaluation")]
    rows = []
    for key, group in evaluation.groupby(["dataset"] + ARM):
        fields = dict(zip(["dataset"] + ARM, key, strict=True))
        metric = group[group.generator.isin(("physical", "neural"))]
        rows.append(fields | {"metric": "balanced_metric_accuracy"} | mc_interval(metric, "binary_accuracy", key))
        for generator in GENERATORS:
            rows.append(fields | {"metric": generator + "_structured_detection"} | mc_interval(group[group.generator.eq(generator)], "structured_detected", (*key, generator)))
    summary = pd.DataFrame(rows)
    session = evaluation.groupby(ID + ARM + ["generator"], as_index=False).agg(
        binary_accuracy=("binary_accuracy", "mean"),
        structured_detection=("structured_detected", "mean"),
        mean_neural_minus_physical=("margin_neural", "mean"),
        n_trials=("trial", "size"),
    )
    animal = session.groupby(["dataset", "animal"] + ARM + ["generator"], as_index=False)[["binary_accuracy", "structured_detection", "mean_neural_minus_physical"]].mean()
    winners = evaluation.groupby(ID + ARM + ["generator", "winner"]).size().unstack("winner", fill_value=0).reindex(columns=[*MODELS, "ambiguous"], fill_value=0)
    winners = winners.div(winners.sum(axis=1), axis=0).add_prefix("winner_fraction_").reset_index()
    checks = []
    for (dataset, condition), group in summary[summary.support.eq(1) & summary.origin.eq("decoded")].groupby(["dataset", "condition"]):
        m = group.set_index("metric")
        if set(m.index) != {"balanced_metric_accuracy", *(g + "_structured_detection" for g in GENERATORS)}:
            raise ValueError("incomplete recovery endpoints")
        acc = m.loc["balanced_metric_accuracy"]
        min_power = m.loc[["physical_structured_detection", "neural_structured_detection"], "mean"].min()
        max_fpr = m.loc[["stationary_structured_detection", "iid_structured_detection"], "mean"].max()
        finite = thresholds[
            thresholds.dataset.eq(dataset) & thresholds.condition.eq(condition) & thresholds.support.eq(1) & thresholds.origin.eq("decoded")
        ].finite_calibration.all()
        checks.append(
            {
                "dataset": dataset,
                "condition": condition,
                "accuracy": acc["mean"],
                "accuracy_ci_low": acc.ci_low,
                "min_animal_accuracy": acc.min_animal,
                "min_moving_power": min_power,
                "max_null_fpr": max_fpr,
                "finite_calibration": finite,
                "ready": bool(acc.ci_low > 0.5 and acc.min_animal > 0.5 and min_power >= 0.5 and max_fpr <= 0.05 and finite),
            }
        )
    checks = pd.DataFrame(checks)
    if len(checks) != 6 or checks.dataset.nunique() != 2:
        raise ValueError("complete two-dataset three-condition primary required")
    decision = pd.DataFrame(
        [
            {
                "native_matched_ready": bool(checks[checks.condition.eq("matched")].ready.all()),
                "robust_native_ready": bool(checks.ready.all()),
                "real_events_rescored": False,
                "biological_mechanism_established": False,
            }
        ]
    )
    return trials, thresholds, session, animal, summary, checks, decision, winners


def run(parent, audit_path, out, workers):
    mp = parent / "physical_neural_metric_manifest.json"
    m, audit = json.loads(mp.read_text()), json.loads(audit_path.read_text())
    if file_sha256(mp) != PARENT_SHA or m["status"] != "complete" or audit["status"] != "pass" or audit["input_file_sha256"]["run_manifest"] != PARENT_SHA:
        raise ValueError("frozen passing parent required")
    for name, digest in m["output_sha256"].items():
        if file_sha256(parent / name) != digest:
            raise ValueError("changed parent output " + name)
    source = Path(m["source_dir"])
    smp = source / "conditional_2d_manifest.json"
    if file_sha256(smp) != m["input_file_sha256"]["source_manifest"]:
        raise ValueError("changed source manifest")
    sm = json.loads(smp.read_text())
    items = sorted(sm["completed"], key=lambda x: x["tag"])
    if len(items) != 33:
        raise ValueError("all 33 recordings required")
    for item in items:
        name = item["tag"] + "_cache.npz"
        if file_sha256(source / name) != sm["output_sha256"][name]:
            raise ValueError("changed original cache " + name)
    p = build_script_provenance(
        input_paths={
            "parent": mp,
            "parent_audit": audit_path,
            "source": smp,
            "protocol": ROOT / "docs/metric_finite_recovery_protocol.md",
            "producer": Path(__file__),
            "kernel": ROOT / "src/hipporeplayimm/metric_finite_recovery.py",
        },
        cwd=ROOT,
    )
    if p["git_dirty"] or p["code_commit"] == "unavailable":
        raise ValueError("commit a clean protocol before simulations")
    out.mkdir(parents=True, exist_ok=False)
    manifest = out / "metric_finite_recovery_manifest.json"
    p.update(status="running", completed=[], source_dir=str(source), parent_dir=str(parent), real_events_rescored=False)
    manifest.write_text(json.dumps(p, indent=2) + "\n")
    start = time.monotonic()
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(task, item, source, parent, out) for item in items]
            for future in as_completed(futures):
                result = future.result()
                p["completed"].append(result)
                manifest.write_text(json.dumps(p, indent=2) + "\n")
                print(json.dumps(result), flush=True)
        scores = pd.concat([pd.read_csv(out / f"{x['tag']}_recovery_scores.csv.gz") for x in items], ignore_index=True)
        names = ("trials", "thresholds", "sessions", "animals", "summary", "checks", "decision", "winners")
        for name, frame in zip(names, aggregate(scores), strict=True):
            frame.to_csv(out / f"metric_finite_recovery_{name}.csv.gz", index=False)
        p.update(status="complete", rows=len(scores), runtime_s=time.monotonic() - start)
        p["output_sha256"] = {f.name: file_sha256(f) for f in sorted(out.iterdir()) if f != manifest}
    except Exception as exc:
        p.update(status="failed", error=repr(exc), runtime_s=time.monotonic() - start)
        raise
    finally:
        manifest.write_text(json.dumps(p, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-dir", type=Path, required=True)
    parser.add_argument("--parent-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    run(args.parent_dir, args.parent_audit, args.output_dir, args.workers)
