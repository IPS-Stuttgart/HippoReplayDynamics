#!/usr/bin/env python3
"""Independent permutation, predictive-score and factorial summary audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _provenance import build_script_provenance, file_sha256
from report_2d_count_conditioned_prediction import validate as validate_parent
from verify_2d_count_conditioned_prediction import ll, reference_posterior
from verify_position_free_assembly_prediction import compare_table, same

ID = ["dataset", "animal", "session"]
KEY = ID + ["event_id", "split"]
MAPS = ["real", "population_code_permuted"]
MODELS = ["diffusion", "first_order_imm"]
VALUE = ["delta", "delta_per_heldout_spike"]


def reconstruct_contrasts(shuffled, original):
    original = original.set_index(KEY + ["map"])
    all_rows = []
    for model in MODELS:
        column = "score_" + model
        pivot = shuffled.pivot(index=KEY + ["shuffle"], columns="map", values=column)
        mean = pivot.groupby(KEY).mean()
        median = pivot.groupby(KEY).median()
        r, w = original.xs(MAPS[0], level="map"), original.xs(MAPS[1], level="map")
        ur, uw = r[column] - mean[MAPS[0]], w[column] - mean[MAPS[1]]
        values = [
            ur,
            uw,
            ur - uw,
            r[column] - r.score_iid_position,
            w[column] - w.score_iid_position,
            mean[MAPS[0]] - r.score_iid_position,
            mean[MAPS[1]] - w.score_iid_position,
            r[column] - median[MAPS[0]],
            w[column] - median[MAPS[1]],
        ]
        names = [
            "real_order_advantage",
            "wrong_order_advantage",
            "order_map_interaction",
            "real_original_minus_iid",
            "wrong_original_minus_iid",
            "real_shuffle_minus_iid",
            "wrong_shuffle_minus_iid",
            "real_order_median_sensitivity",
            "wrong_order_median_sensitivity",
        ]
        for name, series in zip(names, values, strict=True):
            frame = series.rename("delta").to_frame()
            frame["delta_per_heldout_spike"] = series / r.n_heldout_spikes.replace(0, np.nan)
            all_rows.append(frame.reset_index().assign(contrast=model + "__" + name))
    split = pd.concat(all_rows, ignore_index=True)
    event = split.groupby(ID + ["event_id", "contrast"], as_index=False)[VALUE].median()
    return split, event


def verify_summary(events, root):
    sessions = events.groupby(ID + ["contrast"], as_index=False)[VALUE].mean()
    animals = sessions.groupby(["dataset", "animal", "contrast"], as_index=False)[VALUE].mean()
    compare_table(sessions, pd.read_csv(root / "predictive_order_map_by_session.csv"), ID + ["contrast"], VALUE, "session means")
    compare_table(animals, pd.read_csv(root / "predictive_order_map_by_animal.csv"), ["dataset", "animal", "contrast"], VALUE, "animal means")
    observed = pd.read_csv(root / "predictive_order_map_summary.csv").set_index(["dataset", "contrast"])
    rebuilt = []
    for (dataset, contrast), frame in events.groupby(["dataset", "contrast"]):
        arrays = [[s[VALUE].to_numpy() for _, s in a.groupby("session")] for _, a in frame.groupby("animal")]
        rng = np.random.default_rng(20260908)
        samples = []
        for _ in range(5000):
            rats = []
            for i in rng.integers(len(arrays), size=len(arrays)):
                local = arrays[i]
                sampled = []
                for j in rng.integers(len(local), size=len(local)):
                    values = local[j]
                    sampled.append(np.nanmean(values[rng.integers(len(values), size=len(values))], axis=0))
                rats.append(np.nanmean(sampled, axis=0))
            samples.append(np.nanmean(rats, axis=0))
        interval = np.nanquantile(samples, [0.025, 0.975], axis=0)
        aa = animals[animals.dataset.eq(dataset) & animals.contrast.eq(contrast)]
        vals = [aa.delta.mean(), *interval[:, 0], aa.delta_per_heldout_spike.mean(), *interval[:, 1], aa.delta.gt(0).sum(), len(aa), frame.session.nunique(), len(frame)]
        names = ["mean", "ci_low", "ci_high", "mean_per_heldout_spike", "per_spike_ci_low", "per_spike_ci_high", "positive_animals", "animals", "sessions", "events"]
        same(observed.loc[(dataset, contrast), names].to_numpy(float), vals, "independent hierarchical intervals")
        rebuilt.append(dict(dataset=dataset, contrast=contrast, **dict(zip(names, vals, strict=True))))
    if len(rebuilt) != len(observed):
        raise ValueError("incomplete summary panels")
    decisions = pd.read_csv(root / "predictive_order_map_decisions.csv", keep_default_na=False).set_index("dataset")
    for dataset in ("pfeiffer_foster", "tanni2022"):
        primary = [r for r in rebuilt if r["dataset"] == dataset and r["contrast"] in ("first_order_imm__real_order_advantage", "first_order_imm__order_map_interaction")]
        passing = [r["mean"] > 0 and r["ci_low"] > 0 and r["positive_animals"] == r["animals"] for r in primary]
        if len(primary) != 2 or bool(decisions.loc[dataset, "order_and_adjacency_gate_passed"]) != all(passing):
            raise ValueError("decision mismatch")
        if decisions.loc[dataset, "parent_replication_gate_changed"] or decisions.loc[dataset, "new_mechanism_established"]:
            raise ValueError("unsupported claim upgrade")
    return len(rebuilt)


def run(args):
    root, out = Path(args.run_dir).resolve(), Path(args.output_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError("refusing to overwrite independent audit")
    path = root / "predictive_order_map_manifest.json"
    manifest = json.loads(path.read_text())
    if manifest.get("status") != "complete" or manifest.get("k") != 20 or len(manifest.get("completed", [])) != 33:
        raise ValueError("incomplete factorial run")
    for name, digest in manifest["output_sha256"].items():
        if file_sha256(root / name) != digest:
            raise ValueError("changed output: " + name)
    parent_path = Path(manifest["input_file_paths"]["parent_manifest"])
    parent_audit = Path(manifest["input_file_paths"]["parent_audit"])
    for key, p in (("parent_manifest", parent_path), ("parent_audit", parent_audit)):
        if file_sha256(p) != manifest["input_file_sha256"][key]:
            raise ValueError("changed parent input")
    parent_manifest, _ = validate_parent(parent_path.parent, parent_audit)
    if {i["tag"] for i in manifest["completed"]} != {i["tag"] for i in parent_manifest["completed"]}:
        raise ValueError("source session coverage differs")
    checks, audit_rows, all_events = [], [], []
    invariant_scores, score_rows, perm_count = 0, 0, 0
    for item in sorted(manifest["completed"], key=lambda i: i["tag"]):
        tag = item["tag"]
        identity = {k: item[k] for k in ID}
        original = pd.read_csv(parent_path.parent / f"{tag}_scores.csv")
        shuffled = pd.read_csv(root / f"{tag}_scores.csv.gz")
        permutations = pd.read_csv(root / f"{tag}_permutations.csv.gz", dtype={"permutation": "string"})
        support = pd.read_csv(root / f"{tag}_support.csv.gz")
        with np.load(parent_path.parent / f"{tag}_cache.npz") as z:
            cache = {name: z[name] for name in z.files}
        expected_keys = original[KEY + ["map"]].assign(_join=1).merge(pd.DataFrame({"shuffle": range(20), "_join": 1}), on="_join").drop(columns="_join")
        if (
            shuffled.duplicated(KEY + ["map", "shuffle"]).any()
            or len(shuffled) != len(expected_keys)
            or len(shuffled.merge(expected_keys, on=KEY + ["map", "shuffle"], validate="one_to_one")) != len(expected_keys)
        ):
            raise ValueError("factorial keys missing or duplicate")
        columns = ["score_" + m for m in (*MODELS, "iid_position", "static_location")]
        if (
            not np.isfinite(shuffled[columns]).all().all()
            or not shuffled[columns].le(1e-8).all().all()
            or shuffled.heldout_used_for_inference.any()
            or not shuffled.posterior_unchanged.eq(True).all()
        ):
            raise ValueError("invalid/leaking shuffled scores")
        joined = shuffled.merge(original, on=KEY + ["map"], suffixes=("", "_parent"), validate="many_to_one")
        same(joined.n_heldout_spikes, joined.n_heldout_spikes_parent, "held-out counts")
        for model in ("iid_position", "static_location"):
            same(joined["score_" + model], joined["score_" + model + "_parent"], "invariant scores")
            invariant_scores += len(joined)
        score_rows += len(shuffled)
        event_ids = sorted(original.event_id.unique())
        if permutations.duplicated(["event_id", "shuffle"]).any() or len(permutations) != len(event_ids) * 20:
            raise ValueError("bad permutation coverage")
        if len(support) != len(event_ids) or support.event_id.duplicated().any():
            raise ValueError("bad event support coverage")
        dynamic_ids = {event_ids[0], event_ids[len(event_ids) // 2], event_ids[-1]}
        lookup = shuffled.set_index(["event_id", "split", "shuffle", "map"])
        for eid in event_ids:
            counts, widths = cache[f"counts_{eid}"], np.diff(cache[f"edges_{eid}"])
            local = permutations[permutations.event_id.eq(eid)].sort_values("shuffle")
            if list(local.shuffle) != list(range(20)):
                raise ValueError("missing permutation")
            used = []
            for record in local.itertuples(index=False):
                p = np.array(list(map(int, record.permutation.split(","))))
                key = "|".join(map(str, (20260908, *[identity[k] for k in ID], eid, "whole-bin-order", record.shuffle)))
                random = np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little"))
                same(p, random.permutation(len(counts)), "deterministic permutation")
                if sorted(p) != list(range(len(counts))):
                    raise ValueError("not a bin permutation")
                same(counts[p].sum(axis=0), counts.sum(axis=0), "count conservation")
                same(widths[p].sum(), widths.sum(), "duration conservation")
                used.append(tuple(p))
                perm_count += 1
                if eid not in dynamic_ids or record.shuffle not in (0, 19):
                    continue
                new_edges = np.r_[0.0, np.cumsum(widths[p])]
                times = (new_edges[:-1] + new_edges[1:]) / 2
                for split in (0, 4):
                    tr, he = cache[f"train_{split}"], cache[f"held_{split}"]
                    tll, hll = ll(counts[p][:, tr], cache["rates"][tr]), ll(counts[p][:, he], cache["rates"][he])
                    for map_name, order in zip(MAPS, (np.arange(len(cache["centers"])), cache["permutation"]), strict=True):
                        row = lookup.loc[(eid, split, record.shuffle, map_name)]
                        for model in MODELS:
                            post = reference_posterior(tll[:, order], cache["centers"], times, model == "first_order_imm")
                            value = float(logsumexp(post + hll[:, order], axis=1).sum())
                            observed = row["score_" + model]
                            same(value, observed, "independent shuffled prediction")
                            checks.append(
                                identity | {"event_id": eid, "split": split, "shuffle": record.shuffle, "map": map_name, "model": model, "absolute_error": abs(value - observed)}
                            )
            sr = support[support.event_id.eq(eid)]
            if len(sr) != 1:
                raise ValueError("missing permutation support row")
            same(
                sr[["n_bins", "unique_permutations", "identity_permutations"]].iloc[0].to_numpy(),
                [len(counts), len(set(used)), sum(p == tuple(range(len(counts))) for p in used)],
                "short-event permutation support",
            )
        split, events = reconstruct_contrasts(shuffled, original)
        compare_table(split, pd.read_csv(root / f"{tag}_split_contrasts.csv.gz"), KEY + ["contrast"], VALUE, "split factorials")
        compare_table(events, pd.read_csv(root / f"{tag}_event_contrasts.csv"), ID + ["event_id", "contrast"], VALUE, "event factorials")
        all_events.append(events)
        audit_rows.append(identity | {"events": len(event_ids), "score_rows": len(shuffled), "permutations": len(permutations)})
        print("independently audited " + tag, flush=True)
    events = pd.concat(all_events, ignore_index=True)
    compare_table(events, pd.read_csv(root / "predictive_order_map_event_contrasts.csv"), ID + ["event_id", "contrast"], VALUE, "pooled event factorials")
    n_summary = verify_summary(events, root)
    if score_rows != 1845000 or perm_count != 184500 or len(checks) != 1584 or len(events) != 166050 or n_summary != 36:
        raise ValueError("incomplete independent audit coverage")
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(checks).to_csv(out / "independent_shuffled_predictions.csv", index=False)
    pd.DataFrame(audit_rows).to_csv(out / "independent_session_checks.csv", index=False)
    result = build_script_provenance(input_paths={"run_manifest": path, "parent_audit": parent_audit})
    result.update(
        status="pass",
        sessions=33,
        events=9225,
        score_rows=score_rows,
        invariant_scores=invariant_scores,
        permutations=perm_count,
        independent_predictions=len(checks),
        max_prediction_error=max(r["absolute_error"] for r in checks),
        split_contrasts=830250,
        event_contrasts=len(events),
        bootstrap_panels=n_summary,
        scope="All permutations/count conservation, invariant scores, factors, paired summaries and intervals; separate dynamic solver on two shuffles/two splits/three events per session. Parent raw-count/map audit is hash-pinned and reused, not refitted.",
    )
    result["output_sha256"] = {p.name: file_sha256(p) for p in out.iterdir()}
    (out / "predictive_order_map_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    run(parser.parse_args())
