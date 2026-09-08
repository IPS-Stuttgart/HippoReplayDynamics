#!/usr/bin/env python3
"""Independent raw-timestamp, geometry, predictive-join and aggregation audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax, xlogy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _provenance import build_script_provenance, file_sha256
from audit_2d_geometry_predictive_dissociation import CRITERIA, EXPECTED, FACTORS, ID, KEY, METRICS, STRATA, VALUES
from report_2d_predictive_order_map import validate as validate_factorial
from verify_position_free_assembly_prediction import compare_table, same


def independent_geometry(xy, counts, filtered, minimum):
    support = np.flatnonzero(counts.sum(1) >= 2)
    valid = np.zeros(len(counts), bool)
    if len(support):
        valid[support.min() : support.max() + 1] = True
    if filtered:
        valid &= (counts.sum(1) >= 3) & (np.count_nonzero(counts, axis=1) >= 2)
    indices = np.flatnonzero(valid)
    if len(indices):
        disconnected = (np.diff(indices) != 1) | (np.linalg.norm(np.diff(xy[indices], axis=0), axis=1) >= 20 - 1e-9)
        segments = np.split(indices, np.flatnonzero(disconnected) + 1)
        longest = max(segments, key=len)
        start, end, length = int(longest[0]), int(longest[-1] + 1), len(longest)
        displacement = float(np.linalg.norm(xy[end - 1] - xy[start]))
    else:
        start, end, length, displacement = 0, 0, 0, 0.0
    passed = length >= minimum and displacement >= 40 - 1e-9
    reason = "passed"
    if not passed:
        if valid.sum() < minimum:
            reason = "insufficient_duration_or_support"
        elif length < minimum:
            reason = "jumps_or_unsupported_gaps"
        else:
            reason = "insufficient_displacement"
    adjacent = np.flatnonzero(valid[1:] & valid[:-1])
    distances = np.linalg.norm(xy[adjacent + 1] - xy[adjacent], axis=1)
    return {
        "geometric_pass": passed,
        "failure_reason": reason,
        "n_frames": len(xy),
        "n_valid_frames": int(valid.sum()),
        "continuous_start": start,
        "continuous_end_exclusive": end,
        "continuous_frames": length,
        "continuous_displacement_cm": displacement,
        "large_jump_fraction": np.mean(distances >= 20 - 1e-9) if len(distances) else np.nan,
    }


def raw_session(record, root, parent, factorial):
    tag = record["tag"]
    path = Path(record["source_cache"])
    if file_sha256(path) != record["source_sha256"]:
        raise ValueError("changed source cache")
    with np.load(path) as z:
        source = {
            k: z[k]
            for k in (
                "spikes",
                "cell_ids",
                "unit_qc_mask",
                "candidate_event_indices",
                "candidate_offsets",
                "candidate_base_starts_s",
                "candidate_base_durations_s",
                "candidate_base_counts",
            )
        }
    with np.load(parent / f"{tag}_cache.npz") as z:
        cache = {k: z[k] for k in z.files}
    selected = pd.read_csv(parent / f"{tag}_selection.csv")
    labels = pd.read_csv(root / f"{tag}_labels.csv")
    mask = source["unit_qc_mask"].astype(bool)
    same(source["cell_ids"][mask], cache["unit_ids"], "frozen unit IDs")
    event_lookup = {int(e): i for i, e in enumerate(source["candidate_event_indices"])}
    spans = [source["candidate_offsets"][event_lookup[int(e)] : event_lookup[int(e)] + 2] for e in selected.event_id]
    indices = np.concatenate([np.arange(a, b) for a, b in spans])
    times, widths = source["candidate_base_starts_s"][indices], source["candidate_base_durations_s"][indices]
    raw = np.empty((len(indices), len(cache["unit_ids"])), int)
    for j, unit in enumerate(cache["unit_ids"]):
        spikes = np.sort(source["spikes"][source["spikes"][:, 1] == unit, 0])
        raw[:, j] = np.searchsorted(spikes, times + widths, side="left") - np.searchsorted(spikes, times, side="left")
    same(raw, source["candidate_base_counts"][indices][:, mask], "raw 5 ms counts")
    rebuilt, offset, total_frames = [], 0, 0
    for event, (a, b) in zip(selected.itertuples(index=False), spans, strict=True):
        event_counts, durations = raw[offset : offset + b - a], widths[offset : offset + b - a]
        offset += b - a
        n_complete = len(durations) - int(not np.isclose(durations[-1], 0.005, atol=1e-9, rtol=0))
        frames = np.asarray([event_counts[t : t + 4].sum(axis=0) for t in range(max(0, n_complete - 3))], dtype=int).reshape(-1, len(cache["unit_ids"]))
        total_frames += len(frames)
        identity = {k: getattr(event, k) for k in ID} | {"event_id": event.event_id}
        for split in range(5):
            tr, he = cache[f"train_{split}"], cache[f"held_{split}"]
            if set(tr) & set(he) or sorted(np.r_[tr, he]) != list(range(len(cache["unit_ids"]))):
                raise ValueError("invalid training/held-out partition")
            rates = cache["rates"][tr]
            ll = frames[:, tr] @ np.log(rates * 0.02) - 0.02 * rates.sum(axis=0)
            xy = cache["centers"][ll.argmax(axis=1)]
            posterior = softmax(ll, axis=1)
            entropy = -np.sum(xlogy(posterior, posterior), axis=1)
            for criterion, (filtered, minimum) in CRITERIA.items():
                rebuilt.append(
                    identity
                    | {
                        "split": split,
                        "criterion": criterion,
                        **independent_geometry(xy, frames[:, tr], filtered, minimum),
                        "n_train_cells": len(tr),
                        "mean_training_entropy_nats": entropy.mean() if len(entropy) else np.nan,
                        "heldout_used_for_label": False,
                    }
                )
    rebuilt = pd.DataFrame(rebuilt)
    keys = KEY + ["criterion"]
    numeric = [c for c in rebuilt if c not in keys + ["failure_reason"]]
    compare_table(rebuilt, labels, keys, numeric, "training geometry labels")
    if not rebuilt.sort_values(keys).failure_reason.reset_index(drop=True).equals(labels.sort_values(keys).failure_reason.reset_index(drop=True)):
        raise ValueError("rejection reason mismatch")
    original = pd.read_csv(parent / f"{tag}_scores.csv")
    real = original[original["map"].eq("real")].set_index(KEY)
    shuffled = pd.read_csv(factorial / f"{tag}_split_contrasts.csv.gz").set_index(KEY + ["contrast"])
    predictions = real[["n_train_cells", "n_heldout_cells", "n_train_spikes", "n_heldout_spikes", "duration_s"]].copy()
    for metric, contrast in FACTORS.items():
        predictions[metric] = shuffled.xs(contrast, level="contrast").delta
    predictions["imm_minus_event_global"] = real.score_first_order_imm - real.score_event_global
    for m in METRICS:
        predictions[m + "_per_spike"] = predictions[m] / real.n_heldout_spikes.replace(0, np.nan)
    predictions = predictions.reset_index()
    actual_predictions = pd.read_csv(root / f"{tag}_predictions.csv")
    compare_table(predictions, actual_predictions, KEY, [c for c in predictions if c not in KEY], "unaltered predictive scores")
    return (
        labels,
        actual_predictions,
        {k: record[k] for k in ID}
        | {"events": len(selected), "raw_base_bins": len(raw), "overlapping_frames": total_frames, "label_rows": len(rebuilt), "predictive_rows": len(predictions)},
    )


def independent_tables(labels, scores):
    joined = labels.merge(scores, on=KEY, validate="many_to_one", suffixes=("_label", ""))
    sessions, animals, summary = [], [], []
    keys_sessions = list(scores[ID].drop_duplicates().itertuples(index=False, name=None))
    for split in range(5):
        for criterion in sorted(CRITERIA):
            group = joined[joined.split.eq(split) & joined.criterion.eq(criterion)]
            for stratum in STRATA:
                chosen = group if stratum == "all" else group[group.geometric_pass.eq(stratum == "geometric_accepted")]
                common = {"split": split, "criterion": criterion, "stratum": stratum}
                ss = []
                for dataset, animal, session in keys_sessions:
                    p = chosen[chosen.dataset.eq(dataset) & chosen.animal.eq(animal) & chosen.session.eq(session)]
                    ss.append(common | {"dataset": dataset, "animal": animal, "session": session, "events": len(p)} | p[VALUES].mean().to_dict())
                ss = pd.DataFrame(ss)
                sessions.append(ss)
                for dataset in EXPECTED:
                    aa = []
                    for animal, g in ss[ss.dataset.eq(dataset)].groupby("animal"):
                        aa.append(
                            common
                            | {"dataset": dataset, "animal": animal, "events": int(g.events.sum()), "contributing_sessions": int(g.events.gt(0).sum()), "total_sessions": len(g)}
                            | g[VALUES].mean().to_dict()
                        )
                    aa = pd.DataFrame(aa)
                    animals.append(aa)
                    e = chosen[chosen.dataset.eq(dataset)]
                    ci = np.full((2, len(VALUES)), np.nan)
                    if split == 0 and criterion == "edge10" and len(e):
                        blocks = [[s[VALUES].to_numpy() for _, s in a.groupby("session")] for _, a in e.groupby("animal")]
                        rng = np.random.default_rng(20260908)
                        draws = []
                        for _ in range(5000):
                            rat_means = []
                            for i in rng.integers(len(blocks), size=len(blocks)):
                                session_means = []
                                for j in rng.integers(len(blocks[i]), size=len(blocks[i])):
                                    x = blocks[i][j]
                                    session_means.append(np.nanmean(x[rng.integers(len(x), size=len(x))], axis=0))
                                rat_means.append(np.nanmean(session_means, axis=0))
                            draws.append(np.nanmean(rat_means, axis=0))
                        ci = np.nanquantile(draws, [0.025, 0.975], axis=0)
                    for i, metric in enumerate(METRICS):
                        summary.append(
                            common
                            | {
                                "dataset": dataset,
                                "metric": metric,
                                "events": len(e),
                                "total_sessions": EXPECTED[dataset][1],
                                "contributing_sessions": e.session.nunique(),
                                "total_animals": EXPECTED[dataset][2],
                                "contributing_animals": int(aa[metric].notna().sum()),
                                "positive_animals": int(aa[metric].gt(0).sum()),
                                "mean": aa[metric].mean(),
                                "ci_low": ci[0, i],
                                "ci_high": ci[1, i],
                                "mean_per_spike": aa[metric + "_per_spike"].mean(),
                                "per_spike_ci_low": ci[0, i + 7],
                                "per_spike_ci_high": ci[1, i + 7],
                                "primary_setting": split == 0 and criterion == "edge10",
                                "interval_draws": 5000 if split == 0 and criterion == "edge10" else 0,
                            }
                        )
    return pd.DataFrame(summary), pd.concat(sessions, ignore_index=True), pd.concat(animals, ignore_index=True)


def run(args):
    root, out = Path(args.run_dir).resolve(), Path(args.output_dir).resolve()
    path = root / "geometry_predictive_manifest.json"
    manifest = json.loads(path.read_text())
    if manifest.get("status") != "complete" or len(manifest.get("completed", [])) != 33:
        raise ValueError("complete full-cohort run required")
    if out.exists() and any(out.iterdir()):
        raise ValueError("refusing to overwrite audit")
    for k, p in manifest["input_file_paths"].items():
        if file_sha256(p) != manifest["input_file_sha256"][k]:
            raise ValueError("changed input: " + k)
    for name, digest in manifest["output_sha256"].items():
        if file_sha256(root / name) != digest:
            raise ValueError("changed output: " + name)
    factorial = Path(manifest["input_file_paths"]["factorial_manifest"]).parent
    validate_factorial(factorial, Path(manifest["input_file_paths"]["factorial_audit"]))
    parent = Path(manifest["input_file_paths"]["parent_manifest"]).parent
    all_labels, all_scores, checks = [], [], []
    for r in manifest["completed"]:
        labels, scores, check = raw_session(r, root, parent, factorial)
        all_labels.append(labels)
        all_scores.append(scores)
        checks.append(check)
        print("independently checked " + r["tag"], flush=True)
    labels, scores = pd.concat(all_labels, ignore_index=True), pd.concat(all_scores, ignore_index=True)
    actual = pd.read_csv(root / "geometry_predictive_labels.csv")
    compare_table(labels, actual, KEY + ["criterion"], [c for c in labels if c not in KEY + ["criterion", "failure_reason"]], "pooled labels")
    actual = pd.read_csv(root / "geometry_predictive_predictions.csv")
    compare_table(scores, actual, KEY, [c for c in scores if c not in KEY], "pooled predictions")
    summary, sessions, animals = independent_tables(labels, scores)
    for name, frame, keys in (
        ("summary", summary, ["dataset", "split", "criterion", "stratum", "metric"]),
        ("by_session", sessions, ID + ["split", "criterion", "stratum"]),
        ("by_animal", animals, ["dataset", "animal", "split", "criterion", "stratum"]),
    ):
        compare_table(frame, pd.read_csv(root / f"geometry_predictive_{name}.csv"), keys, [c for c in frame if c not in keys], "independent " + name)
    observed = pd.read_csv(root / "geometry_predictive_decisions.csv").set_index("dataset")
    for dataset, (_, _, n_animals) in EXPECTED.items():
        g = summary[summary.dataset.eq(dataset) & summary.split.eq(0) & summary.criterion.eq("edge10") & summary.stratum.eq("geometric_rejected")].set_index("metric")
        passed = {m: g.loc[m, "mean"] > 0 and g.loc[m, "ci_low"] > 0 and g.loc[m, "positive_animals"] == n_animals and g.loc[m, "events"] > 0 for m in METRICS}
        order = all(passed[m] for m in ("imm_minus_iid", "imm_order_advantage", "imm_order_map_interaction"))
        if bool(observed.loc[dataset, "rejected_order_adjacency_supported"]) != order or bool(observed.loc[dataset, "rejected_beyond_other_event_composition_supported"]) != (
            order and passed["imm_minus_event_global"]
        ):
            raise ValueError("gate decision mismatch")
        if observed.loc[dataset, "novel_mechanism_established"] or observed.loc[dataset, "biological_false_negative_rate_identified"]:
            raise ValueError("unsupported claim upgrade")
    if (len(labels), len(scores), len(summary), len(sessions), len(animals)) != (184500, 46125, 840, 1980, 540):
        raise ValueError("incomplete audit coverage")
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(checks).to_csv(out / "independent_geometry_session_checks.csv", index=False)
    provenance = build_script_provenance(input_paths={"run_manifest": path})
    provenance.update(
        status="pass",
        sessions=33,
        events=9225,
        labels=184500,
        predictions=46125,
        summary_rows=840,
        primary_interval_rows=42,
        raw_base_bins=sum(r["raw_base_bins"] for r in checks),
        overlapping_frames=sum(r["overlapping_frames"] for r in checks),
        scope="All overlapping counts from cached native spike timestamps; all training geometry/entropy, fixed predictive joins, strata, coverage, primary intervals and decisions. Parent independent predictive/native audits reused; original MAT/NWB files and RUN map fitting not repeated.",
    )
    provenance["output_sha256"] = {p.name: file_sha256(p) for p in out.iterdir()}
    (out / "geometry_predictive_audit.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    run(parser.parse_args())
