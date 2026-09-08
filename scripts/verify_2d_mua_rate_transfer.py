#!/usr/bin/env python3
"""Independent gain, predictive-equation, paired-factorial and interval audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _provenance import build_script_provenance, file_sha256
from verify_2d_count_conditioned_prediction import ll, reference_posterior
from verify_position_free_assembly_prediction import compare_table, same

ID = ["dataset", "animal", "session"]
KEY = ID + ["event_id", "split"]
VALUES = ["delta", "delta_per_heldout_spike"]
MODELS = ["iid_position", "static_location", "diffusion", "first_order_imm"]
PRIMARY = [
    "alpha100__imm_adaptation_gain",
    "alpha100__imm_minus_iid",
    "alpha100__imm_minus_event_global",
    "alpha100__imm_real_minus_wrong",
    "alpha100__first_order_imm__real_order_advantage",
    "alpha100__first_order_imm__order_map_interaction",
]


def independent_contrasts(original, shuffled, parent, parent_order):
    columns = ["score_" + m for m in MODELS] + ["score_event_global", "score_run_global", "n_heldout_spikes"]
    wide = original.pivot(index=KEY, columns=["alpha", "map"], values=columns)
    old = parent[parent["map"].eq("real")].set_index(KEY)
    rows = []

    def append(name, values, counts):
        frame = values.rename("delta").to_frame()
        frame["delta_per_heldout_spike"] = values / counts.replace(0, np.nan)
        rows.append(frame.reset_index().assign(contrast=name))

    for alpha in (100, 1000):

        def score(model, map_name="real", level=alpha):
            return wide[("score_" + model, level, map_name)]

        nspikes = wide[("n_heldout_spikes", alpha, "real")]
        values = {
            "imm_adaptation_gain": score("first_order_imm") - old.score_first_order_imm,
            "iid_adaptation_gain": score("iid_position") - old.score_iid_position,
            "global_adaptation_gain": score("event_global") - old.score_run_global,
            "imm_minus_iid": score("first_order_imm") - score("iid_position"),
            "imm_minus_static": score("first_order_imm") - score("static_location"),
            "imm_minus_event_global": score("first_order_imm") - score("event_global"),
            "iid_minus_event_global": score("iid_position") - score("event_global"),
            "imm_real_minus_wrong": score("first_order_imm") - score("first_order_imm", "population_code_permuted"),
            "diffusion_minus_iid": score("diffusion") - score("iid_position"),
            "diffusion_minus_event_global": score("diffusion") - score("event_global"),
        }
        for name, value in values.items():
            append(f"alpha{alpha}__{name}", value, nspikes)
        if alpha == 100:
            mean = shuffled.groupby(KEY + ["map"])[["score_diffusion", "score_first_order_imm"]].mean()
            for model in ("first_order_imm", "diffusion"):
                a = score(model) - mean.xs("real", level="map")["score_" + model]
                b = score(model, "population_code_permuted") - mean.xs("population_code_permuted", level="map")["score_" + model]
                for name, value in (("real_order_advantage", a), ("wrong_order_advantage", b), ("order_map_interaction", a - b)):
                    append(f"alpha100__{model}__{name}", value, nspikes)
                    if model == "first_order_imm" and name != "wrong_order_advantage":
                        old_value = parent_order[parent_order.contrast.eq(f"{model}__{name}")].set_index(KEY).delta
                        append("alpha100__change_in_" + name, value - old_value, nspikes)
    split = pd.concat(rows, ignore_index=True)
    event = split.groupby(ID + ["event_id", "contrast"], as_index=False)[VALUES].median()
    return split, event


def reference_interval(frame):
    dataset, contrast = frame.iloc[0][["dataset", "contrast"]]
    arrays = [[g[VALUES].to_numpy() for _, g in a.groupby("session")] for _, a in frame.groupby("animal")]
    rng = np.random.default_rng(20260908)
    boot = []
    for _ in range(5000):
        means = []
        for animal in rng.integers(len(arrays), size=len(arrays)):
            recordings = arrays[animal]
            selected = []
            for j in rng.integers(len(recordings), size=len(recordings)):
                values = recordings[j]
                selected.append(np.nanmean(values[rng.integers(len(values), size=len(values))], axis=0))
            means.append(np.nanmean(selected, axis=0))
        boot.append(np.nanmean(means, axis=0))
    interval = np.nanquantile(boot, [0.025, 0.975], axis=0)
    mean_animals = frame.groupby(ID)[VALUES].mean().groupby(["dataset", "animal"]).mean()
    return {
        "dataset": dataset,
        "contrast": contrast,
        "mean": mean_animals.delta.mean(),
        "ci_low": interval[0, 0],
        "ci_high": interval[1, 0],
        "mean_per_heldout_spike": mean_animals.delta_per_heldout_spike.mean(),
        "per_spike_ci_low": interval[0, 1],
        "per_spike_ci_high": interval[1, 1],
        "positive_animals": int(mean_animals.delta.gt(0).sum()),
        "animals": len(arrays),
        "sessions": frame.session.nunique(),
        "events": len(frame),
    }


def verify_session(job):
    item, parent_path, order_path, root = job
    parent_path, order_path, root = Path(parent_path), Path(order_path), Path(root)
    tag = item["tag"]
    fields = {k: item[k] for k in ID}
    with np.load(parent_path / f"{tag}_cache.npz") as z:
        cache = {k: z[k] for k in z.files}
    parent = pd.read_csv(parent_path / f"{tag}_scores.csv")
    original = pd.read_csv(root / f"{tag}_original.csv.gz")
    shuffled = pd.read_csv(root / f"{tag}_shuffled.csv.gz")
    gains = pd.read_csv(root / f"{tag}_gains.csv")
    folds = pd.read_csv(root / f"{tag}_folds.csv")
    selection = pd.read_csv(parent_path / f"{tag}_selection.csv").sort_values(["start_s", "event_id"])
    parent_order = pd.read_csv(order_path / f"{tag}_split_contrasts.csv.gz")
    parent_permutations = pd.read_csv(order_path / f"{tag}_permutations.csv.gz", dtype={"permutation": str}).set_index(["event_id", "shuffle"])
    if (
        len(original) != len(parent) * 2
        or len(shuffled) != len(parent) * 20
        or original.duplicated(KEY + ["alpha", "map"]).any()
        or shuffled.duplicated(KEY + ["map", "shuffle"]).any()
    ):
        raise ValueError("incomplete score dimensions")
    for frame in (original, shuffled):
        cols = [c for c in frame if c.startswith("score_")]
        if not np.isfinite(frame[cols]).all().all() or not frame[cols].le(1e-8).all().all() or frame.heldout_used_for_inference.any() or not frame.posterior_unchanged.all():
            raise ValueError("invalid predictions")
    if (
        set(original[KEY + ["map"]].itertuples(index=False, name=None)) != set(parent[KEY + ["map"]].itertuples(index=False, name=None))
        or not original.groupby(KEY + ["map"]).alpha.apply(lambda x: set(x) == {100, 1000}).all()
    ):
        raise ValueError("source/alpha keys differ")
    if (
        set(shuffled[KEY + ["map"]].itertuples(index=False, name=None)) != set(parent[KEY + ["map"]].itertuples(index=False, name=None))
        or not shuffled.groupby(KEY + ["map"]).shuffle.apply(lambda x: set(x) == set(range(20))).all()
    ):
        raise ValueError("source/shuffle keys differ")
    fixed = original[original.alpha.eq(100)]
    joined = shuffled.merge(fixed, on=KEY + ["map"], suffixes=("", "_original"), validate="many_to_one")
    for model in ("iid_position", "static_location"):
        same(joined["score_" + model], joined["score_" + model + "_original"], "order invariance")
    for frame in (original, shuffled):
        support = frame.merge(parent[KEY + ["map", "n_heldout_spikes"]], on=KEY + ["map"], suffixes=("", "_old"), validate="many_to_one")
        same(support.n_heldout_spikes, support.n_heldout_spikes_old, "counts")
    gain_lookup, event_fold = {}, {}
    for j, ix in enumerate(np.array_split(np.arange(len(selection)), 5)):
        test = selection.iloc[ix]
        cal = selection[~selection.event_id.isin(test.event_id)]
        keep = [all(c.end_s + 1 <= t.start_s or c.start_s >= t.end_s + 1 for t in test.itertuples(index=False)) for c in cal.itertuples(index=False)]
        excluded = cal[np.logical_not(keep)]
        cal = cal[keep]
        row = folds[folds.fold.eq(j)].iloc[0]
        for name, expected in (("test_ids", test), ("calibration_ids", cal), ("excluded_ids", excluded)):
            value = row[name]
            actual = set() if pd.isna(value) or value == "" else {int(float(v)) for v in str(value).split(",")}
            if actual != set(expected.event_id):
                raise ValueError("calibration fold/guard differs")
        c = np.stack([cache[f"counts_{eid}"].sum(axis=0) for eid in cal.event_id]).sum(axis=0)
        p0 = cache["rates"].mean(axis=1)
        p0 /= p0.sum()
        for alpha in (100, 1000):
            p = (c + alpha * p0) / (c.sum() + alpha)
            local = gains[gains.fold.eq(j) & gains.alpha.eq(alpha)].set_index("unit_id").loc[cache["unit_ids"]]
            same(local.calibration_count, c, "calibration counts")
            same(local.reference_probability, p0, "reference composition")
            same(local.calibrated_probability, p, "fitted composition")
            same(local.gain, p / p0, "cell gains")
            if alpha == 100:
                same(p, cache[f"global_{j}"], "matched parent baseline")
            gain_lookup[(j, alpha)] = (cache["rates"] * (p / p0)[:, None], p)
        event_fold.update({eid: j for eid in test.event_id})
    # Global predictions are checked for every original event/split/map/alpha.
    for (eid, split, alpha), group in original.groupby(["event_id", "split", "alpha"]):
        counts = cache[f"counts_{eid}"][:, cache[f"held_{split}"]]
        p = gain_lookup[(event_fold[eid], alpha)][1][cache[f"held_{split}"]]
        predicted = ll(counts, p[:, None]).sum()
        same(group.score_event_global, np.full(len(group), predicted), "independent global score")
        same(group.score_run_global, group.score_event_global, "matched adapted global")
    checks = []
    ids = sorted(selection.event_id)
    selected_ids = {ids[0], ids[len(ids) // 2], ids[-1]}
    orig_lookup = original.set_index(["event_id", "split", "alpha", "map"])
    shuffled_lookup = shuffled.set_index(["event_id", "split", "shuffle", "map"])
    for eid in ids:
        counts, edges = cache[f"counts_{eid}"], cache[f"edges_{eid}"]
        widths = np.diff(edges)
        orders = {}
        for k in range(20):
            key = "|".join(map(str, (20260908, *[fields[x] for x in ID], eid, "whole-bin-order", k)))
            order = np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little")).permutation(len(counts))
            stored = np.array(list(map(int, parent_permutations.loc[(eid, k), "permutation"].split(","))))
            same(order, stored, "parent whole-bin permutation")
            same(counts[order].sum(axis=0), counts.sum(axis=0), "conserved counts")
            orders[k] = order
        if eid not in selected_ids:
            continue
        for split in (0, 4):
            tr, he = cache[f"train_{split}"], cache[f"held_{split}"]
            for alpha in (100, 1000):
                rates = gain_lookup[(event_fold[eid], alpha)][0]
                tll, hll = ll(counts[:, tr], rates[tr]), ll(counts[:, he], rates[he])
                cases = [(-1, np.arange(len(counts)), cache[f"times_{eid}"])]
                if alpha == 100:
                    for k in (0, 19):
                        order = orders[k]
                        new_edges = np.r_[0.0, np.cumsum(widths[order])]
                        cases.append((k, order, (new_edges[:-1] + new_edges[1:]) / 2))
                for k, temporal, times in cases:
                    for name, spatial in (("real", np.arange(rates.shape[1])), ("population_code_permuted", cache["permutation"])):
                        record = orig_lookup.loc[(eid, split, alpha, name)] if k < 0 else shuffled_lookup.loc[(eid, split, k, name)]
                        for model in ("diffusion", "first_order_imm"):
                            post = reference_posterior(tll[temporal][:, spatial], cache["centers"], times, model == "first_order_imm")
                            value = float(logsumexp(post + hll[temporal][:, spatial], axis=1).sum())
                            same(value, record["score_" + model], "independent adapted prediction")
                            checks.append(
                                fields
                                | {
                                    "event_id": eid,
                                    "split": split,
                                    "alpha": alpha,
                                    "shuffle": k,
                                    "map": name,
                                    "model": model,
                                    "absolute_error": abs(value - record["score_" + model]),
                                }
                            )
    splits, events = independent_contrasts(original, shuffled, parent, parent_order)
    compare_table(splits, pd.read_csv(root / f"{tag}_split_contrasts.csv.gz"), KEY + ["contrast"], VALUES, "all split contrasts")
    compare_table(events, pd.read_csv(root / f"{tag}_event_contrasts.csv"), ID + ["event_id", "contrast"], VALUES, "all event contrasts")
    print("verified " + tag, flush=True)
    return events, pd.DataFrame(checks), {"tag": tag, "events": len(ids), "global_scores": len(original), "shuffled_rows": len(shuffled), "split_contrasts": len(splits)}


def run(args):
    root, out = Path(args.run_dir).resolve(), Path(args.output_dir).resolve()
    path = root / "mua_rate_transfer_manifest.json"
    manifest = json.loads(path.read_text())
    if manifest["status"] != "complete" or len(manifest["completed"]) != 33:
        raise ValueError("incomplete run")
    if out.exists() and any(out.iterdir()):
        raise ValueError("refusing overwrite")
    for name, expected in manifest["output_sha256"].items():
        if file_sha256(root / name) != expected:
            raise ValueError("changed evidence output")
    parents = {k: Path(manifest["input_file_paths"][k]) for k in ("parent_manifest", "order_parent_manifest")}
    for name, parent in parents.items():
        if file_sha256(parent) != manifest["input_file_sha256"][name]:
            raise ValueError("changed parent manifest")
        for filename, checksum in json.loads(parent.read_text())["output_sha256"].items():
            if file_sha256(parent.parent / filename) != checksum:
                raise ValueError("changed parent data")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(verify_session, [(item, parents["parent_manifest"].parent, parents["order_parent_manifest"].parent, root) for item in manifest["completed"]]))
    events = pd.concat([r[0] for r in results], ignore_index=True)
    checks = pd.concat([r[1] for r in results], ignore_index=True)
    compare_table(events, pd.read_csv(root / "mua_rate_transfer_event_contrasts.csv"), ID + ["event_id", "contrast"], VALUES, "pooled event contrasts")
    sessions = events.groupby(ID + ["contrast"], as_index=False)[VALUES].mean()
    animals = sessions.groupby(["dataset", "animal", "contrast"], as_index=False)[VALUES].mean()
    compare_table(sessions, pd.read_csv(root / "mua_rate_transfer_by_session.csv"), ID + ["contrast"], VALUES, "session means")
    compare_table(animals, pd.read_csv(root / "mua_rate_transfer_by_animal.csv"), ["dataset", "animal", "contrast"], VALUES, "animal means")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rebuilt = pd.DataFrame(pool.map(reference_interval, [g for _, g in events.groupby(["dataset", "contrast"])]))
    compare_table(
        rebuilt,
        pd.read_csv(root / "mua_rate_transfer_summary.csv"),
        ["dataset", "contrast"],
        [c for c in rebuilt if c not in ("dataset", "contrast")],
        "independent hierarchical intervals",
    )
    decisions = pd.read_csv(root / "mua_rate_transfer_decisions.csv", keep_default_na=False).set_index("dataset")
    for dataset, n in (("pfeiffer_foster", 4), ("tanni2022", 5)):
        primary = rebuilt[rebuilt.dataset.eq(dataset) & rebuilt.contrast.isin(PRIMARY)]
        if len(primary) != 6:
            raise ValueError("missing primary gate")
        passed = primary["mean"].gt(0) & primary.ci_low.gt(0) & primary.positive_animals.eq(n)
        if decisions.loc[dataset, "full_observation_transfer_gate"] != passed.all():
            raise ValueError("gate mismatch")
    if len(events) != 258300 or len(checks) != 3168 or len(rebuilt) != 56:
        raise ValueError("audit coverage incomplete")
    out.mkdir(parents=True, exist_ok=True)
    checks.to_csv(out / "independent_predictions.csv", index=False)
    rebuilt.to_csv(out / "independent_hierarchical_intervals.csv", index=False)
    result = build_script_provenance(input_paths={"run_manifest": path})
    result.update(
        status="pass",
        events=9225,
        sessions=33,
        original_global_scores=184500,
        shuffled_score_rows=1845000,
        independent_dynamic_predictions=len(checks),
        max_prediction_error=float(checks.absolute_error.max()),
        split_contrasts=sum(r[2]["split_contrasts"] for r in results),
        event_contrasts=len(events),
        hierarchical_interval_panels=len(rebuilt),
        scope="All source/output hashes, calibration folds/counts/gains, all analytic global scores, factors and invariants, parent whole-bin permutations, paired event/split summaries and exact hierarchical CIs; separate dense dynamic solver on three events/two splits, both original alphas plus two alpha100 shuffles, both maps/models per session. Native raw counts and RUN maps reuse the frozen audited parent.",
    )
    (out / "mua_rate_transfer_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=8)
    run(parser.parse_args())
