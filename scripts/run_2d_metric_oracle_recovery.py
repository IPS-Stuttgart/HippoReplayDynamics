#!/usr/bin/env python3
"""Classify existing synthetic observations with exact generating models."""

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

from hipporeplayimm.conditional_spatial_prediction import identity_likelihood
from hipporeplayimm.metric_finite_recovery import MODELS, null_threshold, reconstruct_kernel
from hipporeplayimm.metric_oracle_recovery import lagged_evidence, path_evidence, whole_evidence
from hipporeplayimm.physical_neural_metric import neural_cost, physical_cost
from scripts._provenance import build_script_provenance, file_sha256
from scripts.run_2d_metric_finite_recovery import mc_interval

PARENT_SHA = "ad202b19f9b187f85f83cb5700f46860b43ecd679764d6278cc5cabb0c5099f8"
ID = ["dataset", "animal", "session"]
ARM = ["method", "support"]
KEY = ID + ["generator", "trial"] + ARM
OBSERVED = ("whole_exact", "whole_train_geometry", "lagged_full_geometry", "lagged_train_geometry")


def pack(values, metadata, item, method, support, split, informative):
    frame = metadata.copy()
    for name in ID:
        frame[name] = item[name]
    for name in MODELS:
        frame["score_" + name] = values[name]
    frame["method"], frame["support"], frame["split"] = method, support, split
    frame["informative"] = informative
    frame["score_kind"] = (
        "latent_path_log_probability" if method == "latent_path" else "joint_event_log_probability" if method.startswith("whole_") else "sum_40ms_predictive_log_scores"
    )
    return frame


def task(item, source, metric, parent, output):
    start = time.monotonic()
    tag = item["tag"]
    with np.load(source / f"{tag}_cache.npz") as z:
        rates, centers = z["rates"], z["centers"]
        splits = [(z[f"train_{s}"], z[f"held_{s}"]) for s in range(5)]
    with np.load(metric / f"{tag}_kernel_parameters.npz") as z:
        physical = reconstruct_kernel(physical_cost(centers), z, "physical")
        full = reconstruct_kernel(neural_cost(rates), z, "full_neural")
        partial = [reconstruct_kernel(neural_cost(rates[t]), z, f"train_neural_{s}") for s, (t, _) in enumerate(splits)]
    kernels = {"physical": physical, "neural": full}
    squared = {k: a @ a for k, a in kernels.items()}
    metadata = pd.read_csv(parent / f"{tag}_simulation_trials.csv").sort_values("simulation_index").reset_index(drop=True)
    baseline = pd.read_csv(parent / f"{tag}_recovery_scores.csv.gz")
    baseline = baseline[baseline.condition.eq("matched") & baseline.origin.eq("decoded")]
    rows = []
    with np.load(parent / f"{tag}_observations.npz") as z:
        offsets, states = z["offsets"], z["states"]
        values = path_evidence(states, offsets, kernels)
        rows.append(pack(values, metadata, item, "latent_path", 0, -1, True))
        for split, (train, held) in enumerate(splits):
            for support in (1, 4):
                x, y = [z[f"s{split}_u{support}_base_{name}"] for name in ("train", "held")]
                ll = identity_likelihood(x, rates[train]) + identity_likelihood(y, rates[held])
                info = np.add.reduceat(x.sum(axis=1) + y.sum(axis=1), offsets[:-1]) > 0
                exact = whole_evidence(ll, offsets, kernels)
                rows.append(pack(exact, metadata, item, "whole_exact", support, split, info))
                estimated = exact | whole_evidence(ll, offsets, {"neural": partial[split]})
                rows.append(pack(estimated, metadata, item, "whole_train_geometry", support, split, info))
                forecast, info = lagged_evidence(x, y, rates[train], rates[held], offsets, squared)
                rows.append(pack(forecast, metadata, item, "lagged_full_geometry", support, split, info))
                old = baseline[baseline.split.eq(split) & baseline.support.eq(support)].sort_values("simulation_index")
                if old.simulation_index.tolist() != metadata.simulation_index.tolist():
                    raise ValueError("misaligned frozen baseline")
                for model in ("physical", "stationary", "iid"):
                    np.testing.assert_allclose(forecast[model], old["score_" + model], rtol=0, atol=1e-8)
                info_old = old.n_held_target_spikes.gt(0) & old.n_train_origin_spikes.gt(0)
                np.testing.assert_array_equal(info, info_old)
                rows.append(pack({m: old["score_" + m].to_numpy() for m in MODELS}, metadata, item, "lagged_train_geometry", support, split, info))
    scores = pd.concat(rows, ignore_index=True)
    validate(scores)
    scores.to_csv(output / f"{tag}_oracle_scores.csv.gz", index=False)
    return {**{k: item[k] for k in ID}, "tag": tag, "rows": len(scores), "runtime_s": time.monotonic() - start}


def validate(scores):
    factors = [("latent_path", 0, -1)] + list(product(OBSERVED, (1, 4), range(5)))
    expected = {(g, t, *f) for g in MODELS for t in range(64) for f in factors}
    if scores.empty or scores.duplicated(KEY + ["split"]).any() or scores[KEY + ["split", "phase", "informative", "score_kind"]].isna().any().any():
        raise ValueError("unique nonempty complete score metadata required")
    for _, group in scores.groupby(ID):
        if len(group) != len(expected) or set(group[["generator", "trial"] + ARM + ["split"]].itertuples(index=False, name=None)) != expected:
            raise ValueError("incomplete oracle factor coverage")
    if not np.array_equal(scores.phase, np.where(scores.trial < 32, "calibration", "evaluation")):
        raise ValueError("changed calibration assignment")
    kinds = np.where(
        scores.method.eq("latent_path"),
        "latent_path_log_probability",
        np.where(scores.method.str.startswith("whole_"), "joint_event_log_probability", "sum_40ms_predictive_log_scores"),
    )
    if not np.array_equal(scores.score_kind, kinds) or not scores.informative.isin([True, False]).all():
        raise ValueError("invalid score interpretation metadata")
    for m in MODELS:
        v = scores["score_" + m].to_numpy()
        allowed = scores.method.eq("latent_path").to_numpy() if m == "stationary" else np.zeros(len(v), bool)
        if np.isnan(v).any() or (v > 1e-8).any() or ((~np.isfinite(v)) & ~(allowed & np.isneginf(v))).any():
            raise ValueError("invalid model probability")


def classify(scores):
    validate(scores)
    rows = scores.astype({"score_" + m: float for m in MODELS}).copy()
    delta = rows.score_neural - rows.score_physical
    delta = delta.mask(delta.abs() <= 1e-9, 0.0)
    rows["delta_neural_minus_physical"] = delta
    rows["binary_accuracy"] = np.nan
    for g, sign in (("physical", -1), ("neural", 1)):
        mask = rows.generator.eq(g)
        rows.loc[mask, "binary_accuracy"] = np.where(delta[mask].eq(0), 0.5, (sign * delta[mask] > 0).astype(float))
    rows["structure_statistic"] = rows[["score_physical", "score_neural"]].max(axis=1) - rows[["score_stationary", "score_iid"]].max(axis=1)
    rows.loc[~rows.informative, "structure_statistic"] = -np.inf
    values = rows[["score_" + m for m in MODELS]].to_numpy()
    ties = (np.abs(values - values.max(axis=1)[:, None]) <= 1e-9).sum(axis=1)
    rows["winner"] = np.where(ties == 1, np.asarray(MODELS)[values.argmax(axis=1)], "ambiguous")
    frames, thresholds = [], []
    for key, group in rows.groupby(ID + ARM + ["split"]):
        limits = []
        for null in ("stationary", "iid"):
            cal = group[group.generator.eq(null) & group.phase.eq("calibration")]
            if len(cal) != 32:
                raise ValueError("complete null calibration required")
            limits.append(null_threshold(cal.structure_statistic))
        limit = max(limits)
        thresholds.append(dict(zip(ID + ARM + ["split"], key, strict=True)) | {"threshold": limit, "finite_calibration": np.isfinite(limit)})
        frame = group.copy()
        frame["threshold"] = limit
        frame["structured_detected"] = frame.informative & np.isfinite(limit) & frame.structure_statistic.gt(limit + 1e-9)
        frames.append(frame)
    rows = pd.concat(frames, ignore_index=True)
    for m in (*MODELS, "ambiguous"):
        rows["winner_fraction_" + m] = rows.winner.eq(m).astype(float)
    cols = ["binary_accuracy", "structured_detected", "delta_neural_minus_physical", "informative", *("winner_fraction_" + m for m in (*MODELS, "ambiguous"))]
    trials = rows.groupby(KEY, as_index=False)[cols].mean()
    counts = rows.groupby(KEY, as_index=False).size().rename(columns={"size": "n_realizations"})
    trials = trials.merge(counts, on=KEY, validate="one_to_one")
    trials["phase"] = np.where(trials.trial < 32, "calibration", "evaluation")
    return rows, pd.DataFrame(thresholds), trials


def aggregate(scores):
    rows, thresholds, trials = classify(scores)
    evaluation = trials[trials.phase.eq("evaluation")]
    results = []
    for key, group in evaluation.groupby(["dataset"] + ARM):
        base = dict(zip(["dataset"] + ARM, key, strict=True))
        moving = group[group.generator.isin(("physical", "neural"))]
        results.append(base | {"metric": "balanced_accuracy"} | mc_interval(moving, "binary_accuracy", ("oracle", *key)))
        for g in MODELS:
            results.append(base | {"metric": g + "_structured_detection"} | mc_interval(group[group.generator.eq(g)], "structured_detected", ("oracle", *key, g)))
    summary = pd.DataFrame(results)
    cols = ["binary_accuracy", "structured_detected", "delta_neural_minus_physical", *("winner_fraction_" + m for m in (*MODELS, "ambiguous"))]
    sessions = evaluation.groupby(ID + ARM + ["generator"], as_index=False)[cols].mean()
    animals = sessions.groupby(["dataset", "animal"] + ARM + ["generator"], as_index=False)[cols].mean()
    paired = []
    moving = evaluation[evaluation.generator.isin(("physical", "neural"))]
    for left, right, label in (
        ("whole_exact", "whole_train_geometry", "whole_geometry_increment"),
        ("lagged_full_geometry", "lagged_train_geometry", "lagged_geometry_increment"),
        ("whole_exact", "lagged_full_geometry", "whole_information_inference_increment"),
        ("latent_path", "whole_exact", "latent_information_gap"),
    ):
        for support in (1, 4):
            a = moving[moving.method.eq(left) & moving.support.eq(0 if left == "latent_path" else support)]
            b = moving[moving.method.eq(right) & moving.support.eq(support)]
            joined = a.merge(b, on=ID + ["generator", "trial"], validate="one_to_one", suffixes=("_left", "_right"))
            if len(joined) != len(a) or len(joined) != len(b):
                raise ValueError("incomplete paired method comparison")
            joined["value"] = joined.binary_accuracy_left - joined.binary_accuracy_right
            for dataset, group in joined.groupby("dataset"):
                paired.append({"dataset": dataset, "contrast": label, "support": support} | mc_interval(group, "value", ("oracle", label, support, dataset)))
    checks = []
    for key, group in summary.groupby(["dataset"] + ARM):
        metrics = group.set_index("metric")
        acc = metrics.loc["balanced_accuracy"]
        power = metrics.loc[[g + "_structured_detection" for g in MODELS[:2]], "mean"].min()
        fpr = metrics.loc[[g + "_structured_detection" for g in MODELS[2:]], "mean"].max()
        d, method, support = key
        sub = thresholds[thresholds.dataset.eq(d) & thresholds.method.eq(method) & thresholds.support.eq(support)]
        finite = len(sub) > 0 and sub.finite_calibration.all()
        checks.append(
            dict(zip(["dataset"] + ARM, key, strict=True))
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
    if len(checks) != 18 or checks.dataset.nunique() != 2:
        raise ValueError("both datasets and all diagnostic arms required")
    decision = pd.DataFrame(
        [
            {
                "whole_event_oracle_native_operating_pass": checks[checks.method.eq("whole_exact") & checks.support.eq(1)].operating_pass.all(),
                "latent_path_operating_pass": checks[checks.method.eq("latent_path")].operating_pass.all(),
                "real_events_rescored": False,
                "biological_mechanism_established": False,
                "new_real_scoring_authorized": False,
            }
        ]
    )
    return rows, thresholds, trials, sessions, animals, summary, pd.DataFrame(paired), checks, decision


def run(parent, audit_path, output, workers):
    mp = parent / "metric_finite_recovery_manifest.json"
    m, audit = json.loads(mp.read_text()), json.loads(audit_path.read_text())
    if file_sha256(mp) != PARENT_SHA or m["status"] != "complete" or audit["status"] != "pass" or audit["input_file_sha256"]["run_manifest"] != PARENT_SHA:
        raise ValueError("frozen independently verified parent required")
    for name, digest in m["output_sha256"].items():
        if file_sha256(parent / name) != digest:
            raise ValueError("changed parent output " + name)
    source, metric = Path(m["source_dir"]), Path(m["parent_dir"])
    smp, kmp = source / "conditional_2d_manifest.json", metric / "physical_neural_metric_manifest.json"
    if file_sha256(smp) != m["input_file_sha256"]["source"] or file_sha256(kmp) != m["input_file_sha256"]["parent"]:
        raise ValueError("changed source manifest")
    sm, km = json.loads(smp.read_text()), json.loads(kmp.read_text())
    for item in m["completed"]:
        name = item["tag"] + "_cache.npz"
        if file_sha256(source / name) != sm["output_sha256"][name]:
            raise ValueError("changed map cache")
    for name, digest in km["output_sha256"].items():
        if file_sha256(metric / name) != digest:
            raise ValueError("changed metric parameters")
    p = build_script_provenance(
        input_paths={
            "parent": mp,
            "parent_audit": audit_path,
            "source": smp,
            "metric": kmp,
            "producer": Path(__file__),
            "module": ROOT / "src/hipporeplayimm/metric_oracle_recovery.py",
            "protocol": ROOT / "docs/metric_oracle_recovery_protocol.md",
        },
        cwd=ROOT,
    )
    if p["git_dirty"] or p["code_commit"] == "unavailable":
        raise ValueError("freeze a clean committed design first")
    if len(m["completed"]) != 33:
        raise ValueError("all 33 recordings required")
    output.mkdir(parents=True, exist_ok=False)
    manifest = output / "metric_oracle_manifest.json"
    p.update(status="running", completed=[], parent_dir=str(parent), source_dir=str(source), metric_dir=str(metric), real_events_rescored=False)
    manifest.write_text(json.dumps(p, indent=2) + "\n")
    start = time.monotonic()
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(task, x, source, metric, parent, output) for x in sorted(m["completed"], key=lambda x: x["tag"])]
            for future in as_completed(futures):
                item = future.result()
                p["completed"].append(item)
                manifest.write_text(json.dumps(p, indent=2) + "\n")
                print(json.dumps(item), flush=True)
        scores = pd.concat([pd.read_csv(output / f"{x['tag']}_oracle_scores.csv.gz") for x in p["completed"]], ignore_index=True)
        names = ("decisions", "thresholds", "trials", "sessions", "animals", "summary", "paired", "checks", "decision")
        for name, frame in zip(names, aggregate(scores), strict=True):
            frame.to_csv(output / f"metric_oracle_{name}.csv.gz", index=False)
        p.update(status="complete", rows=len(scores), runtime_s=time.monotonic() - start)
        p["output_sha256"] = {f.name: file_sha256(f) for f in sorted(output.iterdir()) if f != manifest}
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
