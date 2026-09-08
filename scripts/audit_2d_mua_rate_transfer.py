#!/usr/bin/env python3
"""Matched cross-event rate transfer and full order/map prediction controls."""

from __future__ import annotations

import argparse
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
from audit_2d_count_conditioned_prediction import IDENTITY, aggregate, folds
from audit_2d_predictive_order_map import MAPS, K, permuted_times, shuffled_indices, shuffled_prediction
from audit_hc11_sleep_rate_transfer import calibrate

from hipporeplayimm.conditional_spatial_prediction import SpatialPredictionContext, identity_likelihood, score_event

PARENT_HASH = "d16542d0bf935929a05a335f84cd2060167c76dd8e255378beb202fb84e97386"
ORDER_HASH = "40f4b123142ddd47b22e7db1311602b849a4d996d943076d26c07e0485c524ae"
KEYS = IDENTITY + ["event_id", "split"]
PRIMARY = [
    "alpha100__imm_adaptation_gain",
    "alpha100__imm_minus_iid",
    "alpha100__imm_minus_event_global",
    "alpha100__imm_real_minus_wrong",
    "alpha100__first_order_imm__real_order_advantage",
    "alpha100__first_order_imm__order_map_interaction",
]


def validate_factors(original, shuffled):
    if original.empty or shuffled.empty or original.duplicated(KEYS + ["alpha", "map"]).any() or shuffled.duplicated(KEYS + ["map", "shuffle"]).any():
        raise ValueError("empty or duplicate factors")
    expected = {(a, m) for a in (100, 1000) for m in MAPS}
    if not all(set(g[["alpha", "map"]].itertuples(index=False, name=None)) == expected for _, g in original.groupby(KEYS)):
        raise ValueError("incomplete original maps/calibration")
    a = original[original.alpha.eq(100)]
    key = KEYS + ["map"]
    if set(a[key].itertuples(index=False, name=None)) != set(shuffled[key].itertuples(index=False, name=None)) or not all(
        set(g.shuffle) == set(range(K)) for _, g in shuffled.groupby(key)
    ):
        raise ValueError("missing whole-bin shuffle")
    for frame in (original, shuffled):
        score_columns = [c for c in frame if c.startswith("score_")]
        if (
            not np.isfinite(frame[score_columns]).all().all()
            or not frame[score_columns].le(1e-8).all().all()
            or not frame.posterior_unchanged.eq(True).all()
            or not frame.heldout_used_for_inference.eq(False).all()
        ):
            raise ValueError("invalid or leaking predictions")


def contrasts(original, shuffled, unadapted, old_order):
    validate_factors(original, shuffled)
    old = unadapted[unadapted["map"].eq("real")].set_index(KEYS)
    rows = []

    def add(name, values, support):
        if not np.isfinite(values).all():
            raise ValueError("missing paired scores")
        row = values.rename("delta").to_frame()
        row["delta_per_heldout_spike"] = values / support.replace(0, np.nan)
        rows.append(row.reset_index().assign(contrast=name))

    for alpha in (100, 1000):
        local = original[original.alpha.eq(alpha)].set_index(KEYS + ["map"])
        real, wrong = local.xs("real", level="map"), local.xs(MAPS[1], level="map")
        if not real.index.equals(old.index):
            old = old.reindex(real.index)
        values = {
            "imm_adaptation_gain": real.score_first_order_imm - old.score_first_order_imm,
            "iid_adaptation_gain": real.score_iid_position - old.score_iid_position,
            "global_adaptation_gain": real.score_event_global - old.score_run_global,
            "imm_minus_iid": real.score_first_order_imm - real.score_iid_position,
            "imm_minus_static": real.score_first_order_imm - real.score_static_location,
            "imm_minus_event_global": real.score_first_order_imm - real.score_event_global,
            "iid_minus_event_global": real.score_iid_position - real.score_event_global,
            "imm_real_minus_wrong": real.score_first_order_imm - wrong.score_first_order_imm,
            "diffusion_minus_iid": real.score_diffusion - real.score_iid_position,
            "diffusion_minus_event_global": real.score_diffusion - real.score_event_global,
        }
        for name, value in values.items():
            add(f"alpha{alpha}__{name}", value, real.n_heldout_spikes)
        if alpha == 100:
            means = shuffled.groupby(KEYS + ["map"])[["score_first_order_imm", "score_diffusion"]].mean()
            for model in ("first_order_imm", "diffusion"):
                col = f"score_{model}"
                real_adv = real[col] - means.xs("real", level="map")[col]
                wrong_adv = wrong[col] - means.xs(MAPS[1], level="map")[col]
                for name, value in (("real_order_advantage", real_adv), ("wrong_order_advantage", wrong_adv), ("order_map_interaction", real_adv - wrong_adv)):
                    add(f"alpha100__{model}__{name}", value, real.n_heldout_spikes)
                    if model == "first_order_imm" and name != "wrong_order_advantage":
                        prior = old_order[old_order.contrast.eq(f"{model}__{name}")].set_index(KEYS).delta
                        add(f"alpha100__change_in_{name}", value - prior.reindex(value.index), real.n_heldout_spikes)
    splits = pd.concat(rows, ignore_index=True)
    events = splits.groupby(IDENTITY + ["event_id", "contrast"], as_index=False)[["delta", "delta_per_heldout_spike"]].median()
    return splits, events


def task(job):
    item, parent, order_parent, out = job
    parent, order_parent, out = Path(parent), Path(order_parent), Path(out)
    tick = time.monotonic()
    tag = item["tag"]
    identity = tuple(item[k] for k in IDENTITY)
    fields = dict(zip(IDENTITY, identity, strict=True))
    with np.load(parent / f"{tag}_cache.npz") as z:
        cache = {k: z[k] for k in z.files}
    selection = pd.read_csv(parent / f"{tag}_selection.csv")
    unadapted = pd.read_csv(parent / f"{tag}_scores.csv")
    old_order = pd.read_csv(order_parent / f"{tag}_split_contrasts.csv.gz")
    context = SpatialPredictionContext(cache["centers"])
    original_rows, shuffled_rows, gain_rows, fold_rows = [], [], [], []
    done = 0
    for fold, test, calibration, excluded in folds(selection):
        pooled = np.sum([cache[f"counts_{eid}"].sum(axis=0) for eid in calibration.event_id], axis=0).astype(np.int64)
        fold_rows.append(
            fields
            | {
                "fold": fold,
                "test_ids": ",".join(map(str, test.event_id)),
                "calibration_ids": ",".join(map(str, calibration.event_id)),
                "excluded_ids": ",".join(map(str, excluded.event_id)),
            }
        )
        parameters = {}
        for alpha in (100, 1000):
            probability, gain, reference = calibrate(cache["rates"], pooled, alpha)
            rates = cache["rates"] * gain[:, None]
            if not np.allclose(rates.mean(axis=1) / rates.mean(axis=1).sum(), probability, atol=1e-12):
                raise ValueError("rate transfer failed composition identity")
            if alpha == 100 and not np.allclose(probability, cache[f"global_{fold}"], atol=1e-12):
                raise ValueError("parent baseline changed")
            parameters[alpha] = (rates, probability)
            gain_rows.extend(
                fields | {"fold": fold, "alpha": alpha, "unit_id": uid, "calibration_count": int(c), "reference_probability": p0, "calibrated_probability": p, "gain": g}
                for uid, c, p0, p, g in zip(cache["unit_ids"], pooled, reference, probability, gain, strict=True)
            )
        for event in test.itertuples(index=False):
            eid = int(event.event_id)
            counts, edges, times = cache[f"counts_{eid}"], cache[f"edges_{eid}"], cache[f"times_{eid}"]
            permutations = [shuffled_indices(identity, eid, len(counts), k) for k in range(K)]
            clocks = [permuted_times(edges, p) for p in permutations]
            for split in range(5):
                tr, he = cache[f"train_{split}"], cache[f"held_{split}"]
                base = fields | {"event_id": eid, "split": split, "fold": fold, "n_heldout_spikes": int(counts[:, he].sum()), "n_train_spikes": int(counts[:, tr].sum())}
                for alpha, (rates, probability) in parameters.items():
                    originals = score_event(counts, times, rates, tr, he, context, cache["permutation"], probability)
                    original_rows.extend(base | {"alpha": alpha} | r for r in originals)
                    if alpha == 100:
                        tll, hll = identity_likelihood(counts[:, tr], rates[tr]), identity_likelihood(counts[:, he], rates[he])
                        for k, (p, clock) in enumerate(zip(permutations, clocks, strict=True)):
                            results = shuffled_prediction(tll[p], hll[p], clock, context, cache["permutation"])
                            for r, original in zip(results, originals, strict=True):
                                for invariant in ("iid_position", "static_location"):
                                    if not np.isclose(r[f"score_{invariant}"], original[f"score_{invariant}"], atol=1e-7, rtol=0):
                                        raise ValueError("shuffle changed invariant")
                                shuffled_rows.append(base | {"shuffle": k} | r)
            done += 1
            if done % 100 == 0:
                print(f"{tag} {done}/{len(selection)}", flush=True)
    original, shuffled = pd.DataFrame(original_rows), pd.DataFrame(shuffled_rows)
    splits, events = contrasts(original, shuffled, unadapted, old_order)
    for name, frame in (("original", original), ("shuffled", shuffled), ("split_contrasts", splits)):
        frame.to_csv(out / f"{tag}_{name}.csv.gz", index=False)
    for name, frame in (("event_contrasts", events), ("gains", pd.DataFrame(gain_rows)), ("folds", pd.DataFrame(fold_rows))):
        frame.to_csv(out / f"{tag}_{name}.csv", index=False)
    result = fields | {"tag": tag, "events": len(selection), "original_rows": len(original), "shuffled_rows": len(shuffled), "runtime_s": time.monotonic() - tick}
    print(json.dumps(result), flush=True)
    return result


def decision(summary):
    rows = []
    for dataset, count in (("pfeiffer_foster", 4), ("tanni2022", 5)):
        part = summary[summary.dataset.eq(dataset) & summary.contrast.isin(PRIMARY)]
        if set(part.contrast) != set(PRIMARY) or part.contrast.duplicated().any() or not part.animals.eq(count).all():
            raise ValueError("missing primary endpoint")
        passed = part["mean"].gt(0) & part.ci_low.gt(0) & part.positive_animals.eq(count)
        rows.append(
            {
                "dataset": dataset,
                "full_observation_transfer_gate": bool(passed.all()),
                "failed_primary_axes": ",".join(part.loc[~passed, "contrast"]),
                "independent_confirmation": False,
                "biological_gain_measured": False,
                "new_mechanism_established": False,
            }
        )
    return pd.DataFrame(rows)


def run(args):
    parent, order_parent, out = (Path(p).resolve() for p in (args.parent_dir, args.order_parent_dir, args.output_dir))
    paths = [parent / "conditional_2d_manifest.json", order_parent / "predictive_order_map_manifest.json"]
    if [file_sha256(p) for p in paths] != [PARENT_HASH, ORDER_HASH]:
        raise ValueError("wrong frozen parent")
    pm, om = [json.loads(p.read_text()) for p in paths]
    for directory, manifest in ((parent, pm), (order_parent, om)):
        if manifest["status"] != "complete":
            raise ValueError("incomplete source")
        for name, expected in manifest["output_sha256"].items():
            if file_sha256(directory / name) != expected:
                raise ValueError(f"changed parent output: {name}")
    if out.exists() and any(out.iterdir()):
        raise ValueError("refusing overwrite")
    out.mkdir(parents=True, exist_ok=True)
    manifest = build_script_provenance(input_paths={"parent_manifest": paths[0], "order_parent_manifest": paths[1], "protocol": ROOT / "docs/2d_mua_rate_transfer_protocol.md"})
    manifest.update(status="running", primary_alpha=100, sensitivity_alpha=1000, k=K, n_events=9225, n_splits=5)
    mpath = out / "mua_rate_transfer_manifest.json"
    mpath.write_text(json.dumps(manifest, indent=2) + "\n")
    done, started = [], time.monotonic()
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            tasks = [pool.submit(task, (item, parent, order_parent, out)) for item in sorted(om["completed"], key=lambda x: -x["runtime_s"])]
            for future in as_completed(tasks):
                done.append(future.result())
        if len(done) != 33 or sum(x["events"] for x in done) != 9225 or sum(x["shuffled_rows"] for x in done) != 1845000 or sum(x["original_rows"] for x in done) != 184500:
            raise ValueError("incomplete experiment")
        events = pd.concat([pd.read_csv(out / f"{x['tag']}_event_contrasts.csv") for x in done], ignore_index=True)
        # Each contrast has its own fixed-seed hierarchical bootstrap in the parent.
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            groups = [g for _, g in events.groupby(["dataset", "contrast"])]
            stats = list(pool.map(aggregate, groups))
        summary, animals, sessions = [pd.concat([r[i] for r in stats], ignore_index=True) for i in range(3)]
        for name, frame in (("event_contrasts", events), ("summary", summary), ("by_animal", animals), ("by_session", sessions), ("decisions", decision(summary))):
            frame.to_csv(out / f"mua_rate_transfer_{name}.csv", index=False)
        manifest.update(status="complete", runtime_s=time.monotonic() - started)
        print(decision(summary).to_string(index=False), flush=True)
        print(summary[summary.contrast.isin(PRIMARY)].to_string(index=False), flush=True)
    except BaseException as exc:
        manifest.update(status="failed", error=repr(exc))
        raise
    finally:
        manifest["completed"] = done
        manifest["output_sha256"] = {p.name: file_sha256(p) for p in out.iterdir() if p.is_file() and p != mpath}
        mpath.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-dir", default="/mnt/seagate10tb/florianpfaff/conditional-2d-mua-pf-tanni-all33-20260908")
    parser.add_argument("--order-parent-dir", default="/mnt/seagate10tb/florianpfaff/2d-predictive-order-map-all9225-k20-20260908")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=12)
    run(parser.parse_args())
