#!/usr/bin/env python3
"""Exact synthetic identifiability screen; never reads candidate spike counts."""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np
import pandas as pd

from hipporeplayimm.physical_neural_metric import STAY, matched_kernel, neural_cost, oracle_screen, physical_cost
from scripts._provenance import build_script_provenance, file_sha256

SOURCE_SHA = "d16542d0bf935929a05a335f84cd2060167c76dd8e255378beb202fb84e97386"
VALUE = "neural_true_train_neural_minus_physical"
METRICS = ["physical_true_physical_minus_train_neural", VALUE, "neural_true_oracle_neural_minus_physical", "neural_true_oracle_minus_train_neural"]


def task(item, source, out):
    start = time.monotonic()
    tag = item["tag"]
    with np.load(source / f"{tag}_cache.npz", allow_pickle=False) as z:
        rates, centers = z["rates"], z["centers"]
        splits = [(z[f"train_{i}"], z[f"held_{i}"]) for i in range(5)]
    physical = physical_cost(centers)
    full_code = neural_cost(rates)
    kernels, parameters, diagnostics = {}, {}, []

    def build(name, cost, cells):
        k = matched_kernel(cost)
        kernels[name] = k.matrix
        for key in ("beta", "cost_scale", "log_scale", "entropy", "target_entropy"):
            parameters[f"{name}__{key}"] = getattr(k, key)
        diagnostics.append(
            {
                "dataset": item["dataset"],
                "animal": item["animal"],
                "session": item["session"],
                "kernel": name,
                "states": len(cost),
                "cells": cells,
                "beta": k.beta,
                "cost_scale": k.cost_scale,
                "entropy": k.entropy,
                "target_entropy": k.target_entropy,
                "evaluations": k.evaluations,
                "max_balance_iterations": k.max_balance_iterations,
                "row_residual": float(np.max(np.abs(k.matrix.sum(axis=1) - 1))),
                "symmetry_residual": float(np.max(np.abs(k.matrix - k.matrix.T))),
                "dwell_residual": float(np.max(np.abs(np.diag(k.matrix) - STAY))),
                "physical_rms_step_cm": float(np.sqrt(np.sum(k.matrix * physical) / len(cost))),
            }
        )
        return k.matrix

    a = build("physical", physical, rates.shape[0])
    n = build("full_neural", full_code, rates.shape[0])
    rows = []
    for split, (train, held) in enumerate(splits):
        if train.dtype.kind not in "iu" or held.dtype.kind not in "iu" or min(len(train), len(held)) < 2:
            raise ValueError("integer cell index partitions required")
        joined = np.sort(np.r_[train, held])
        if not np.array_equal(joined, np.arange(len(rates))):
            raise ValueError("cell partition incomplete, repeated or overlapping")
        k = build(f"train_neural_{split}", neural_cost(rates[train]), len(train))
        for row in oracle_screen(rates, held, a, n, k):
            rows.append(
                dict(dataset=item["dataset"], animal=item["animal"], session=item["session"], split=split, n_states=rates.shape[1], n_train=len(train), n_held=len(held), **row)
            )
        del kernels[f"train_neural_{split}"]
    np.savez_compressed(out / f"{tag}_kernel_parameters.npz", **parameters)
    pd.DataFrame(rows).to_csv(out / f"{tag}_oracle_scores.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(out / f"{tag}_kernel_diagnostics.csv", index=False)
    return {"tag": tag, "rows": len(rows), "kernels": len(diagnostics), "runtime_s": time.monotonic() - start}


def summarize(scores, expected):
    key = ["dataset", "animal", "session", "split", "horizon"]
    wanted = {(r["dataset"], r["animal"], r["session"], split, h) for r in expected for split in range(5) for h in (1, 2, 4)}
    actual = set(scores[key].itertuples(index=False, name=None))
    if not wanted or actual != wanted or scores.duplicated(key).any() or not np.isfinite(scores[METRICS].to_numpy()).all():
        raise ValueError("missing, duplicate, nonfinite or unexpected screening rows")
    sessions = scores.groupby(["dataset", "animal", "session", "horizon"], as_index=False)[METRICS].median()
    animals = sessions.groupby(["dataset", "animal", "horizon"], as_index=False)[METRICS].mean()
    dataset = animals.groupby(["dataset", "horizon"], as_index=False)[METRICS].mean()
    counts = animals.assign(positive=animals[VALUE].gt(0)).groupby(["dataset", "horizon"], as_index=False).agg(animals=("animal", "size"), positive_animals=("positive", "sum"))
    dataset = dataset.merge(counts, on=["dataset", "horizon"], validate="one_to_one")
    primary = dataset[dataset.horizon.eq(2)]
    screen_pass = len(primary) == 2 and set(primary.dataset) == {"pfeiffer_foster", "tanni2022"} and bool(primary.positive_animals.eq(primary.animals).all())
    decision = pd.DataFrame(
        [
            {
                "horizon_ms": 40,
                "necessary_oracle_screen_pass": screen_pass,
                "real_event_scoring_performed": False,
                "real_replay_mechanism_established": False,
                "high_importance_discovery_established": False,
            }
        ]
    )
    return sessions, animals, dataset, decision


def run(source, out, workers):
    source_manifest = source / "conditional_2d_manifest.json"
    if file_sha256(source_manifest) != SOURCE_SHA:
        raise ValueError("source manifest does not match frozen protocol")
    m = json.loads(source_manifest.read_text())
    items = sorted(m["completed"], key=lambda x: x["tag"])
    if m["status"] != "complete" or len(items) != 33 or len({x["animal"] for x in items}) != 9:
        raise ValueError("complete frozen 33-recording cohort required")
    for item in items:
        name = item["tag"] + "_cache.npz"
        if file_sha256(source / name) != m["output_sha256"][name]:
            raise ValueError("changed source map cache " + name)
    p = build_script_provenance(
        input_paths={
            "source_manifest": source_manifest,
            "protocol": ROOT / "docs/physical_neural_metric_protocol.md",
            "producer": Path(__file__),
            "kernel": ROOT / "src/hipporeplayimm/physical_neural_metric.py",
        },
        cwd=ROOT,
    )
    if p["git_dirty"] or p["code_commit"] == "unavailable":
        raise ValueError("freeze and commit before empirical-map computation")
    out.mkdir(parents=True, exist_ok=False)
    manifest = out / "physical_neural_metric_manifest.json"
    p.update(status="running", source_dir=str(source), completed=[], real_event_scoring_performed=False)
    manifest.write_text(json.dumps(p, indent=2) + "\n")
    start = time.monotonic()
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(task, item, source, out) for item in items]
            for future in as_completed(futures):
                row = future.result()
                p["completed"].append(row)
                manifest.write_text(json.dumps(p, indent=2) + "\n")
                print(json.dumps(row), flush=True)
        scores = pd.concat([pd.read_csv(out / f"{x['tag']}_oracle_scores.csv") for x in items], ignore_index=True)
        diagnostics = pd.concat([pd.read_csv(out / f"{x['tag']}_kernel_diagnostics.csv") for x in items], ignore_index=True)
        for name, frame in zip(("sessions", "animals", "summary", "decision"), summarize(scores, items), strict=True):
            frame.to_csv(out / f"physical_neural_metric_{name}.csv", index=False)
        scores.to_csv(out / "physical_neural_metric_scores.csv", index=False)
        diagnostics.to_csv(out / "physical_neural_metric_kernel_diagnostics.csv", index=False)
        p.update(status="complete", rows=len(scores), kernels=len(diagnostics), runtime_s=time.monotonic() - start)
        p["output_sha256"] = {f.name: file_sha256(f) for f in sorted(out.iterdir()) if f != manifest}
    except Exception as exc:
        p.update(status="failed", error=repr(exc), runtime_s=time.monotonic() - start)
        raise
    finally:
        manifest.write_text(json.dumps(p, indent=2) + "\n")
    return p


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    run(args.source_dir, args.output_dir, args.workers)
