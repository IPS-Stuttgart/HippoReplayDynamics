#!/usr/bin/env python3
"""Training-only geometric screen joined to existing proper predictive scores."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp, xlogy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _provenance import build_script_provenance, file_sha256
from report_2d_predictive_order_map import validate as validate_factorial

ID = ["dataset", "animal", "session"]
KEY = ID + ["event_id", "split"]
CRITERIA = {"edge10": (False, 10), "edge11": (False, 11), "supported10": (True, 10), "supported11": (True, 11)}
FACTORS = {
    "imm_minus_iid": "first_order_imm__real_original_minus_iid",
    "imm_order_advantage": "first_order_imm__real_order_advantage",
    "imm_order_map_interaction": "first_order_imm__order_map_interaction",
    "diffusion_minus_iid": "diffusion__real_original_minus_iid",
    "diffusion_order_advantage": "diffusion__real_order_advantage",
    "diffusion_order_map_interaction": "diffusion__order_map_interaction",
}
METRICS = list(FACTORS) + ["imm_minus_event_global"]
VALUES = METRICS + [m + "_per_spike" for m in METRICS]
STRATA = ["all", "geometric_accepted", "geometric_rejected"]
EXPECTED = {"pfeiffer_foster": (4001, 8, 4), "tanni2022": (5224, 25, 5)}
FACTORIAL_HASH = "40f4b123142ddd47b22e7db1311602b849a4d996d943076d26c07e0485c524ae"


def frame_counts(base, durations):
    base, durations = np.asarray(base), np.asarray(durations)
    if base.ndim != 2 or len(base) != len(durations) or not len(base) or not np.isfinite(base).all() or (base < 0).any() or (base != np.floor(base)).any():
        raise ValueError("invalid source counts")
    if not np.isfinite(durations).all() or (durations <= 0).any() or (durations > 0.005 + 1e-9).any() or not np.allclose(durations[:-1], 0.005, atol=1e-9, rtol=0):
        raise ValueError("expected complete 5 ms bins and optional final partial bin")
    n = len(base) - int(not np.isclose(durations[-1], 0.005, atol=1e-9, rtol=0))
    cs = np.vstack([np.zeros((1, base.shape[1]), dtype=np.int64), np.cumsum(base[:n], axis=0, dtype=np.int64)])
    return cs[4:] - cs[:-4] if n >= 4 else np.empty((0, base.shape[1]), dtype=np.int64)


def screen(path, counts, filtered=False, minimum=10):
    if path.shape != (len(counts), 2) or not np.isfinite(path).all() or minimum < 1:
        raise ValueError("invalid MAP path")
    edge = np.flatnonzero(counts.sum(axis=1) >= 2)
    valid = np.zeros(len(path), bool)
    if len(edge):
        valid[edge[0] : edge[-1] + 1] = True
    if filtered:
        valid &= (counts.sum(axis=1) >= 3) & ((counts > 0).sum(axis=1) >= 2)
    best_start, best_end, begin = 0, 0, None
    for t in range(len(path)):
        if not valid[t]:
            begin = None
            continue
        if begin is None or np.linalg.norm(path[t] - path[t - 1]) >= 20 - 1e-9:
            begin = t
        if t + 1 - begin > best_end - best_start:
            best_start, best_end = begin, t + 1
    length = best_end - best_start
    displacement = float(np.linalg.norm(path[best_end - 1] - path[best_start])) if length > 1 else 0.0
    passed = length >= minimum and displacement >= 40 - 1e-9
    reason = (
        "passed" if passed else ("insufficient_duration_or_support" if valid.sum() < minimum else "jumps_or_unsupported_gaps" if length < minimum else "insufficient_displacement")
    )
    adjacent = valid[:-1] & valid[1:]
    jumps = np.linalg.norm(np.diff(path, axis=0), axis=1)[adjacent]
    return {
        "geometric_pass": bool(passed),
        "failure_reason": reason,
        "n_frames": len(path),
        "n_valid_frames": int(valid.sum()),
        "continuous_start": best_start,
        "continuous_end_exclusive": best_end,
        "continuous_frames": length,
        "continuous_displacement_cm": displacement,
        "large_jump_fraction": float(np.mean(jumps >= 20 - 1e-9)) if len(jumps) else np.nan,
    }


def classify_training(counts, rates, centers, training):
    c, r = counts[:, training], rates[training]
    if not len(training) or r.shape[1] != len(centers) or (r <= 0).any() or not np.isfinite(r).all():
        raise ValueError("invalid encoding/training cells")
    ll = c @ np.log(r * 0.02) - 0.02 * r.sum(axis=0)
    post = np.exp(ll - logsumexp(ll, axis=1, keepdims=True))
    path = centers[ll.argmax(axis=1)]
    entropy = -xlogy(post, post).sum(axis=1)
    records = []
    for name, (filtered, minimum) in CRITERIA.items():
        records.append(
            {
                "criterion": name,
                **screen(path, c, filtered, minimum),
                "n_train_cells": len(training),
                "mean_training_entropy_nats": float(entropy.mean()) if len(entropy) else np.nan,
                "heldout_used_for_label": False,
            }
        )
    return records


def prediction_table(parent, factorial, tag):
    original = pd.read_csv(parent / f"{tag}_scores.csv")
    original = original[original["map"].eq("real")].copy()
    source = pd.read_csv(factorial / f"{tag}_split_contrasts.csv.gz")
    source = source[source.contrast.isin(FACTORS.values())]
    if source.duplicated(KEY + ["contrast"]).any() or original.duplicated(KEY).any():
        raise ValueError("duplicate score keys")
    pivot = source.pivot(index=KEY, columns="contrast", values="delta").rename(columns={v: k for k, v in FACTORS.items()}).reset_index()
    metadata = original[KEY + ["n_train_cells", "n_heldout_cells", "n_train_spikes", "n_heldout_spikes", "duration_s"]].copy()
    metadata["imm_minus_event_global"] = original.score_first_order_imm - original.score_event_global
    out = pivot.merge(metadata, on=KEY, validate="one_to_one")
    if len(out) != len(original) or len(source) != len(original) * 6 or not np.isfinite(out[METRICS]).all().all():
        raise ValueError("missing predictive scores")
    for m in METRICS:
        out[m + "_per_spike"] = out[m] / out.n_heldout_spikes.replace(0, np.nan)
    return out


def task(args):
    item, parent, factorial, out = args
    parent, factorial, out = Path(parent), Path(factorial), Path(out)
    tag = item["tag"]
    selected = pd.read_csv(parent / f"{tag}_selection.csv")
    with np.load(parent / f"{tag}_cache.npz") as z:
        cache = {k: z[k] for k in z.files}
    source = Path(selected.iloc[0].source_cache_path)
    with np.load(source) as z:
        a = {
            k: z[k]
            for k in (
                "cell_ids",
                "unit_qc_mask",
                "rates_hz",
                "valid_spatial_bins",
                "bin_centers_cm",
                "candidate_event_indices",
                "candidate_offsets",
                "candidate_base_counts",
                "candidate_base_durations_s",
            )
        }
    mask, support = a["unit_qc_mask"].astype(bool), a["valid_spatial_bins"].astype(bool)
    if (
        not np.array_equal(a["cell_ids"][mask], cache["unit_ids"])
        or not np.array_equal(a["rates_hz"][mask][:, support], cache["rates"])
        or not np.array_equal(a["bin_centers_cm"][support], cache["centers"])
    ):
        raise ValueError("source encoding mismatch")
    lookup = {int(e): j for j, e in enumerate(a["candidate_event_indices"])}
    labels = []
    for e in selected.itertuples(index=False):
        j = lookup[int(e.event_id)]
        left, right = a["candidate_offsets"][j : j + 2]
        frames = frame_counts(a["candidate_base_counts"][left:right][:, mask], a["candidate_base_durations_s"][left:right])
        fields = {k: getattr(e, k) for k in ID} | {"event_id": e.event_id}
        for split in range(5):
            for row in classify_training(frames, cache["rates"], cache["centers"], cache[f"train_{split}"]):
                labels.append(fields | {"split": split} | row)
    labels = pd.DataFrame(labels)
    scores = prediction_table(parent, factorial, tag)
    if labels.duplicated(KEY + ["criterion"]).any() or len(labels) != len(selected) * 20 or labels.heldout_used_for_label.any():
        raise ValueError("incomplete or leaking labels")
    if set(labels[KEY].itertuples(index=False, name=None)) != set(scores[KEY].itertuples(index=False, name=None)):
        raise ValueError("label-prediction key mismatch")
    labels.to_csv(out / f"{tag}_labels.csv", index=False)
    scores.to_csv(out / f"{tag}_predictions.csv", index=False)
    result = {k: item[k] for k in ID} | {"tag": tag, "events": len(selected), "label_rows": len(labels), "source_cache": str(source), "source_sha256": file_sha256(source)}
    print(json.dumps(result), flush=True)
    return result


def interval(frame, seed=20260908, draws=5000):
    if frame.empty:
        return np.full((2, len(VALUES)), np.nan)
    groups = [[s[VALUES].to_numpy() for _, s in a.groupby("session")] for _, a in frame.groupby("animal")]
    rng, boot = np.random.default_rng(seed), []
    for _ in range(draws):
        animals = []
        for i in rng.integers(len(groups), size=len(groups)):
            sessions = []
            for j in rng.integers(len(groups[i]), size=len(groups[i])):
                values = groups[i][j]
                sessions.append(np.nanmean(values[rng.integers(len(values), size=len(values))], axis=0))
            animals.append(np.nanmean(sessions, axis=0))
        boot.append(np.nanmean(animals, axis=0))
    return np.nanquantile(boot, [0.025, 0.975], axis=0)


def summarize(labels, scores, draws=5000):
    labels = labels.drop(columns="n_train_cells")
    joined = labels.merge(scores, on=KEY, validate="many_to_one")
    if len(joined) != len(labels) or joined.heldout_used_for_label.any():
        raise ValueError("incomplete/leaking label-score join")
    all_sessions = scores[ID].drop_duplicates()
    all_animals = scores[["dataset", "animal"]].drop_duplicates()
    summaries, session_rows, animal_rows = [], [], []
    for (split, criterion), group in joined.groupby(["split", "criterion"]):
        for stratum in STRATA:
            selected = group if stratum == "all" else group[group.geometric_pass.eq(stratum == "geometric_accepted")]
            ss = selected.groupby(ID)[VALUES].mean().reset_index()
            ss = all_sessions.merge(ss, on=ID, how="left", validate="one_to_one")
            counts = selected.groupby(ID).size().rename("events").reset_index()
            ss = ss.merge(counts, on=ID, how="left", validate="one_to_one")
            ss["events"] = ss.events.fillna(0).astype(int)
            aa = ss.groupby(["dataset", "animal"])[VALUES].mean().reset_index()
            aa = all_animals.merge(aa, on=["dataset", "animal"], how="left", validate="one_to_one")
            coverage = (
                ss.assign(nonempty=ss.events.gt(0).astype(int))
                .groupby(["dataset", "animal"])
                .agg(events=("events", "sum"), contributing_sessions=("nonempty", "sum"), total_sessions=("session", "size"))
                .reset_index()
            )
            aa = aa.merge(coverage, on=["dataset", "animal"], validate="one_to_one")
            common = {"split": split, "criterion": criterion, "stratum": stratum}
            session_rows.append(ss.assign(**common))
            animal_rows.append(aa.assign(**common))
            for dataset, (_, n_sessions, n_animals) in EXPECTED.items():
                a = aa[aa.dataset.eq(dataset)]
                e = selected[selected.dataset.eq(dataset)]
                ci = interval(e, draws=draws) if split == 0 and criterion == "edge10" else np.full((2, len(VALUES)), np.nan)
                for index, metric in enumerate(METRICS):
                    summaries.append(
                        common
                        | {
                            "dataset": dataset,
                            "metric": metric,
                            "events": len(e),
                            "total_sessions": n_sessions,
                            "contributing_sessions": e.session.nunique(),
                            "total_animals": n_animals,
                            "contributing_animals": int(a[metric].notna().sum()),
                            "positive_animals": int(a[metric].gt(0).sum()),
                            "mean": a[metric].mean(),
                            "ci_low": ci[0, index],
                            "ci_high": ci[1, index],
                            "mean_per_spike": a[metric + "_per_spike"].mean(),
                            "per_spike_ci_low": ci[0, index + len(METRICS)],
                            "per_spike_ci_high": ci[1, index + len(METRICS)],
                            "primary_setting": split == 0 and criterion == "edge10",
                            "interval_draws": draws if split == 0 and criterion == "edge10" else 0,
                        }
                    )
    return pd.DataFrame(summaries), pd.concat(session_rows, ignore_index=True), pd.concat(animal_rows, ignore_index=True)


def decision(summary):
    rows = []
    for dataset, (events, sessions, animals) in EXPECTED.items():
        all_rows = summary[summary.dataset.eq(dataset) & summary.split.eq(0) & summary.criterion.eq("edge10") & summary.stratum.eq("all")]
        p = summary[summary.dataset.eq(dataset) & summary.split.eq(0) & summary.criterion.eq("edge10") & summary.stratum.eq("geometric_rejected")]
        if (
            len(p) != 7
            or p.metric.duplicated().any()
            or set(p.metric) != set(METRICS)
            or len(all_rows) != 7
            or not all_rows.events.eq(events).all()
            or not all_rows.contributing_sessions.eq(sessions).all()
        ):
            raise ValueError("incomplete primary summaries")
        p = p.set_index("metric")
        passed = p["mean"].gt(0) & p.ci_low.gt(0) & p.positive_animals.eq(animals) & p.contributing_animals.eq(animals) & p.events.gt(0)
        first = ["imm_minus_iid", "imm_order_advantage", "imm_order_map_interaction"]
        order = bool(passed.loc[first].all())
        full = order and bool(passed.loc["imm_minus_event_global"])
        rows.append(
            {
                "dataset": dataset,
                "rejected_events": int(p.events.iloc[0]),
                "rejected_order_adjacency_supported": order,
                "rejected_beyond_other_event_composition_supported": full,
                "failed_primary_metrics": ",".join(m for m in first + ["imm_minus_event_global"] if not passed.loc[m]),
                "biological_false_negative_rate_identified": False,
                "novel_mechanism_established": False,
            }
        )
    return pd.DataFrame(rows)


def run(args):
    factorial, audit, out = Path(args.factorial_dir).resolve(), Path(args.factorial_audit).resolve(), Path(args.output_dir).resolve()
    path = factorial / "predictive_order_map_manifest.json"
    if file_sha256(path) != FACTORIAL_HASH:
        raise ValueError("wrong frozen predictive experiment")
    fm, _ = validate_factorial(factorial, audit)
    parent_manifest = Path(fm["input_file_paths"]["parent_manifest"])
    if file_sha256(parent_manifest) != fm["input_file_sha256"]["parent_manifest"]:
        raise ValueError("changed parent manifest")
    parent = parent_manifest.parent
    pm = json.loads(parent_manifest.read_text())
    event_manifest = Path(pm["input_file_paths"]["event_manifest"])
    if file_sha256(event_manifest) != pm["input_file_sha256"]["event_manifest"]:
        raise ValueError("changed source event manifest")
    source_records = json.loads(event_manifest.read_text())["results"]
    for record in source_records:
        if file_sha256(record["source_cache_path"]) != record["source_cache_sha256"]:
            raise ValueError("changed native source cache")
    for name, digest in pm["output_sha256"].items():
        if file_sha256(parent / name) != digest:
            raise ValueError("changed parent artifact: " + name)
    if out.exists() and any(out.iterdir()):
        raise ValueError("refusing to overwrite experiment")
    out.mkdir(parents=True, exist_ok=True)
    manifest = build_script_provenance(
        input_paths={"factorial_manifest": path, "factorial_audit": audit, "parent_manifest": parent_manifest, "protocol": ROOT / "docs/2d_geometry_predictive_protocol.md"}
    )
    manifest.update(status="running", events=9225, sessions=33, animals=9, primary_split=0, primary_criterion="edge10", non_rescoring_predictive=True)
    target = out / "geometry_predictive_manifest.json"
    target.write_text(json.dumps(manifest, indent=2) + "\n")
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            done = list(pool.map(task, [(r, parent, factorial, out) for r in pm["completed"]]))
        source_hashes = {str(Path(r["source_cache_path"]).resolve()): r["source_cache_sha256"] for r in source_records}
        if any(source_hashes[str(Path(r["source_cache"]).resolve())] != r["source_sha256"] for r in done):
            raise ValueError("source cache changed during labeling")
        labels = pd.concat([pd.read_csv(out / f"{r['tag']}_labels.csv") for r in done], ignore_index=True)
        scores = pd.concat([pd.read_csv(out / f"{r['tag']}_predictions.csv") for r in done], ignore_index=True)
        if len(done) != 33 or len(labels) != 184500 or len(scores) != 46125:
            raise ValueError("incomplete full cohort")
        summary, sessions, animals = summarize(labels, scores)
        for name, frame in (("labels", labels), ("predictions", scores), ("summary", summary), ("by_session", sessions), ("by_animal", animals), ("decisions", decision(summary))):
            frame.to_csv(out / f"geometry_predictive_{name}.csv", index=False)
        manifest.update(status="complete", completed=done)
    except BaseException as error:
        manifest.update(status="failed", error=repr(error))
        raise
    finally:
        manifest["output_sha256"] = {p.name: file_sha256(p) for p in out.iterdir() if p.is_file() and p != target}
        target.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factorial-dir", default="/mnt/seagate10tb/florianpfaff/2d-predictive-order-map-all9225-k20-20260908")
    parser.add_argument("--factorial-audit", default="/mnt/seagate10tb/florianpfaff/2d-predictive-order-map-all9225-k20-20260908-audit/predictive_order_map_audit.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=8)
    run(parser.parse_args())
