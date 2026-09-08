#!/usr/bin/env python3
"""Frozen whole-bin order by map control of proper held-out cell prediction."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
from _provenance import build_script_provenance, file_sha256
from audit_2d_count_conditioned_prediction import IDENTITY, aggregate
from report_2d_count_conditioned_prediction import validate as validate_parent

from hipporeplayimm.conditional_spatial_prediction import SpatialPredictionContext, identity_likelihood
from hipporeplayimm.frozen_posterior_prediction import frozen_smoothed_marginal_log_score, posterior_sha256

PARENT_HASH = "d16542d0bf935929a05a335f84cd2060167c76dd8e255378beb202fb84e97386"
SEED, K = 20260908, 20
MAPS = ("real", "population_code_permuted")
MODELS = ("diffusion", "first_order_imm")
KEYS = IDENTITY + ["event_id", "split"]


def shuffled_indices(identity, event_id, n_bins, shuffle):
    if n_bins < 1 or shuffle < 0:
        raise ValueError("positive bin count and nonnegative shuffle required")
    digest = hashlib.sha256("|".join(map(str, (SEED, *identity, event_id, "whole-bin-order", shuffle))).encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little")).permutation(n_bins)


def permuted_times(edges, permutation):
    widths = np.diff(np.asarray(edges, float))
    if not len(widths) or not np.isfinite(edges).all() or (widths <= 0).any() or sorted(permutation) != list(range(len(widths))):
        raise ValueError("valid bin partition and permutation required")
    widths = widths[permutation]
    right = np.cumsum(widths)
    return right - widths / 2


def shuffled_prediction(tll, hll, times, context, map_permutation):
    orders = (np.arange(len(context.centers)), map_permutation)
    # No target likelihood is inspected until both training posteriors exist.
    inferred = [context.infer(tll[:, order], times)[0] for order in orders]
    hashes = [{m: posterior_sha256(q) for m, q in post.items()} for post in inferred]
    result = []
    for name, order, post, original_hashes in zip(MAPS, orders, inferred, hashes, strict=True):
        scores = {m: frozen_smoothed_marginal_log_score(q, hll[:, order]).total_log_score for m, q in post.items()}
        if any(posterior_sha256(q) != original_hashes[m] for m, q in post.items()):
            raise ValueError("held-out prediction mutated the training posterior")
        result.append(
            {
                "map": name,
                **{f"score_{m}": value for m, value in scores.items()},
                "training_imm_posterior_sha256": original_hashes["first_order_imm"],
                "posterior_unchanged": True,
                "heldout_used_for_inference": False,
            }
        )
    return result


def check_factors(scores, originals, k):
    keys = KEYS + ["map"]
    if scores.empty or originals.empty or k < 1 or originals.duplicated(keys).any() or scores.duplicated(keys + ["shuffle"]).any():
        raise ValueError("empty/duplicate factorial")
    if set(originals["map"]) != set(MAPS) or set(scores["map"]) != set(MAPS):
        raise ValueError("map factor incomplete")
    expected = set(originals[keys].itertuples(index=False, name=None))
    if set(scores[keys].itertuples(index=False, name=None)) != expected or len(scores) != len(originals) * k:
        raise ValueError("event/cell/map coverage mismatch")
    for _, g in scores.groupby(keys, sort=False):
        if set(g.shuffle) != set(range(k)):
            raise ValueError("missing shuffled score")
    columns = [f"score_{m}" for m in (*MODELS, "iid_position", "static_location")]
    if (
        not np.isfinite(scores[columns]).all().all()
        or not scores[columns].le(1e-8).all().all()
        or not scores.posterior_unchanged.eq(True).all()
        or not scores.heldout_used_for_inference.eq(False).all()
    ):
        raise ValueError("invalid or leaking predictive scores")


def contrasts(scores, originals, k=K):
    check_factors(scores, originals, k)
    original = originals.set_index(KEYS + ["map"])
    mean = scores.groupby(KEYS + ["map"], sort=True)[[f"score_{m}" for m in MODELS]].mean()
    median = scores.groupby(KEYS + ["map"], sort=True)[[f"score_{m}" for m in MODELS]].median()
    rows = []
    for model in MODELS:
        column = f"score_{model}"
        real, wrong = original.xs(MAPS[0], level="map"), original.xs(MAPS[1], level="map")
        mr, mw = mean.xs(MAPS[0], level="map"), mean.xs(MAPS[1], level="map")
        real_advantage = real[column] - mr[column]
        wrong_advantage = wrong[column] - mw[column]
        values = {
            "real_order_advantage": real_advantage,
            "wrong_order_advantage": wrong_advantage,
            "order_map_interaction": real_advantage - wrong_advantage,
            "real_original_minus_iid": real[column] - real.score_iid_position,
            "wrong_original_minus_iid": wrong[column] - wrong.score_iid_position,
            "real_shuffle_minus_iid": mr[column] - real.score_iid_position,
            "wrong_shuffle_minus_iid": mw[column] - wrong.score_iid_position,
            "real_order_median_sensitivity": real[column] - median.xs(MAPS[0], level="map")[column],
            "wrong_order_median_sensitivity": wrong[column] - median.xs(MAPS[1], level="map")[column],
        }
        for name, series in values.items():
            if not np.isfinite(series).all():
                raise ValueError("incomplete paired contrast")
            part = series.rename("delta").to_frame()
            part["delta_per_heldout_spike"] = series / real.n_heldout_spikes.replace(0, np.nan)
            rows.append(part.reset_index().assign(contrast=f"{model}__{name}"))
    splits = pd.concat(rows, ignore_index=True)
    events = splits.groupby(IDENTITY + ["event_id", "contrast"], as_index=False)[["delta", "delta_per_heldout_spike"]].median()
    return splits, events


def task(args):
    item, parent, out = args
    started = time.monotonic()
    parent, out = Path(parent), Path(out)
    tag = item["tag"]
    identity = tuple(item[k] for k in IDENTITY)
    fields = dict(zip(IDENTITY, identity, strict=True))
    originals = pd.read_csv(parent / f"{tag}_scores.csv")
    with np.load(parent / f"{tag}_cache.npz") as z:
        cache = {name: z[name] for name in z.files}
    context = SpatialPredictionContext(cache["centers"])
    original_lookup = originals.set_index(["event_id", "split", "map"])
    event_ids = sorted(originals.event_id.unique())
    rows, permutations, support = [], [], []
    for index, eid in enumerate(event_ids):
        counts = cache[f"counts_{eid}"]
        orders = [shuffled_indices(identity, eid, len(counts), k) for k in range(K)]
        times = [permuted_times(cache[f"edges_{eid}"], order) for order in orders]
        for k, order in enumerate(orders):
            permutations.append(fields | {"event_id": eid, "shuffle": k, "permutation": ",".join(map(str, order))})
        support.append(
            fields
            | {
                "event_id": eid,
                "n_bins": len(counts),
                "unique_permutations": len({tuple(p) for p in orders}),
                "identity_permutations": sum(np.array_equal(p, np.arange(len(counts))) for p in orders),
            }
        )
        for split in range(5):
            tr, he = cache[f"train_{split}"], cache[f"held_{split}"]
            tll = identity_likelihood(counts[:, tr], cache["rates"][tr])
            hll = identity_likelihood(counts[:, he], cache["rates"][he])
            for k, (order, clock) in enumerate(zip(orders, times, strict=True)):
                for r in shuffled_prediction(tll[order], hll[order], clock, context, cache["permutation"]):
                    source = original_lookup.loc[(eid, split, r["map"])]
                    for invariant in ("iid_position", "static_location"):
                        if not np.isclose(r[f"score_{invariant}"], source[f"score_{invariant}"], atol=1e-7, rtol=0):
                            raise ValueError("whole-bin order changed an invariant model")
                    rows.append(fields | {"event_id": eid, "split": split, "shuffle": k, "n_heldout_spikes": int(counts[:, he].sum())} | r)
        if (index + 1) % 100 == 0:
            print(f"{tag}: {index + 1}/{len(event_ids)} events", flush=True)
    scores = pd.DataFrame(rows)
    splits, events = contrasts(scores, originals)
    for name, frame in (("scores", scores), ("permutations", pd.DataFrame(permutations)), ("support", pd.DataFrame(support)), ("split_contrasts", splits)):
        frame.to_csv(out / f"{tag}_{name}.csv.gz", index=False)
    events.to_csv(out / f"{tag}_event_contrasts.csv", index=False)
    result = fields | {"tag": tag, "events": len(event_ids), "rows": len(scores), "runtime_s": time.monotonic() - started}
    print(json.dumps(result), flush=True)
    return result


def decision(summary):
    records = []
    for dataset, rats in (("pfeiffer_foster", 4), ("tanni2022", 5)):
        subset = summary[summary.dataset.eq(dataset) & summary.contrast.isin(["first_order_imm__real_order_advantage", "first_order_imm__order_map_interaction"])]
        if len(subset) != 2 or subset.contrast.duplicated().any() or not subset.animals.eq(rats).all():
            raise ValueError("missing primary endpoints")
        passed = subset["mean"].gt(0) & subset.ci_low.gt(0) & subset.positive_animals.eq(rats)
        records.append(
            {
                "dataset": dataset,
                "order_and_adjacency_gate_passed": bool(passed.all()),
                "failed_primary_axes": ",".join(subset.loc[~passed, "contrast"]),
                "parent_replication_gate_changed": False,
                "new_mechanism_established": False,
            }
        )
    return pd.DataFrame(records)


def run(args):
    parent, audit, out = Path(args.parent_dir).resolve(), Path(args.parent_audit).resolve(), Path(args.output_dir).resolve()
    path = parent / "conditional_2d_manifest.json"
    if file_sha256(path) != PARENT_HASH:
        raise ValueError("wrong frozen parent experiment")
    pm, _ = validate_parent(parent, audit)
    if out.exists() and any(out.iterdir()):
        raise ValueError("refusing to overwrite experiment")
    out.mkdir(parents=True, exist_ok=True)
    manifest = build_script_provenance(input_paths={"parent_manifest": path, "parent_audit": audit, "protocol": ROOT / "docs/2d_predictive_order_map_protocol.md"})
    manifest.update(
        status="running",
        k=K,
        seed=SEED,
        n_events=9225,
        n_sessions=33,
        n_animals=9,
        n_splits=5,
        claim_boundary="proper conditional held-out order/map diagnostic; does not rescue failed composition or identify new biology",
    )
    mpath = out / "predictive_order_map_manifest.json"
    mpath.write_text(json.dumps(manifest, indent=2) + "\n")
    started, done = time.monotonic(), []
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(task, (item, parent, out)) for item in sorted(pm["completed"], key=lambda v: -v["runtime_s"])]
            for future in as_completed(futures):
                done.append(future.result())
        events = pd.concat([pd.read_csv(out / f"{v['tag']}_event_contrasts.csv") for v in done], ignore_index=True)
        if len(done) != 33 or sum(v["events"] for v in done) != 9225 or sum(v["rows"] for v in done) != 1845000 or len(events) != 166050:
            raise ValueError("incomplete full-cohort factorial")
        summary, animals, sessions = aggregate(events)
        for name, frame in (("event_contrasts", events), ("summary", summary), ("by_animal", animals), ("by_session", sessions), ("decisions", decision(summary))):
            frame.to_csv(out / f"predictive_order_map_{name}.csv", index=False)
        manifest.update(status="complete", runtime_s=time.monotonic() - started)
    except BaseException as error:
        manifest.update(status="failed", error=repr(error))
        raise
    finally:
        manifest["completed"] = done
        manifest["output_sha256"] = {p.name: file_sha256(p) for p in out.iterdir() if p.is_file() and p != mpath}
        mpath.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-dir", default="/mnt/seagate10tb/florianpfaff/conditional-2d-mua-pf-tanni-all33-20260908")
    parser.add_argument("--parent-audit", default="/mnt/seagate10tb/florianpfaff/conditional-2d-mua-pf-tanni-all33-20260908-audit/conditional_2d_audit.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=12)
    run(parser.parse_args())
