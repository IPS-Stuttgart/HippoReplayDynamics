#!/usr/bin/env python3
"""First-half RUN context validation and proper cross-context MUA prediction."""

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
from audit_2d_count_conditioned_prediction import folds, pool_counts

from hipporeplayimm.encoding import _speed_cm_s, _times_in_intervals
from hipporeplayimm.multicontext_prediction import predict_contexts

SOURCE_HASH = "2ed1c4d910a3ddc4027eefb398d48a55bae6829c3bf16e8f57955beb9ee55613"
PARENT_HASH = "d16542d0bf935929a05a335f84cd2060167c76dd8e255378beb202fb84e97386"
PRIMARY = ("mix_iid_minus_current_iid", "mix_iid_minus_mix_global", "mix_iid_minus_event_global")
CONTRASTS = {
    PRIMARY[0]: ("mix_iid", "current_iid"),
    PRIMARY[1]: ("mix_iid", "mix_global"),
    PRIMARY[2]: ("mix_iid", "event_global"),
    "mix_iid_minus_blind_iid": ("mix_iid", "blind_iid"),
    "current_iid_minus_current_global": ("current_iid", "current_global"),
    "mix_global_minus_current_global": ("mix_global", "current_global"),
}


def seed(*parts):
    return int.from_bytes(hashlib.sha256("|".join(map(str, (20260908, *parts))).encode()).digest()[:8], "little")


def build_bank(sources):
    common = sorted(set.intersection(*(set(a["cell_ids"].astype(int)) for a in sources)))
    if len(common) < 10:
        raise ValueError("insufficient aligned unit IDs")
    maps, occupancy, indices = [], [], []
    for a in sources:
        lookup = {int(x): i for i, x in enumerate(a["cell_ids"])}
        ind = np.array([lookup[x] for x in common])
        valid = a["occupancy_first_half_s"] >= 0.05
        if valid.sum() < 2:
            raise ValueError("insufficient training-half spatial coverage")
        maps.append(a["rates_first_half_hz"][ind][:, valid])
        occupancy.append(a["occupancy_first_half_s"][valid])
        indices.append(ind)
    expected = sum((rates @ occ for rates, occ in zip(maps, occupancy, strict=True)), start=np.zeros(len(common)))
    mean = expected / sum(x.sum() for x in occupancy)
    peak = np.stack([x.max(axis=1) for x in maps]).max(axis=0)
    keep = (expected >= 30) & (mean <= 4) & (peak >= 2)
    if keep.sum() < 10:
        raise ValueError("insufficient training-only QC cells")
    unit_table = pd.DataFrame({"unit_id": common, "first_half_expected_spikes": expected, "pooled_first_half_rate_hz": mean, "maximum_first_half_rate_hz": peak, "included": keep})
    maps = [m[keep] for m in maps]
    indices = [i[keep] for i in indices]
    compositions = [m @ o / o.sum() for m, o in zip(maps, occupancy, strict=True)]
    return np.array(common)[keep], maps, compositions, indices, unit_table


def running_windows(source, midpoint, maximum=100):
    p = source["position"]
    speed = _speed_cm_s(p[:, 0], p[:, 1:3])
    good = (p[:, 0] >= midpoint + 1) & (speed >= 10) & _times_in_intervals(p[:, 0], source["supported_run_intervals"])
    link = good[:-1] & good[1:] & (np.diff(p[:, 0]) <= 0.1)
    starts = np.flatnonzero(good & ~np.r_[False, link])
    stops = np.flatnonzero(good & ~np.r_[link, False])
    windows = []
    for lo, hi in zip(starts, stops, strict=True):
        duration = p[hi, 0] - p[lo, 0]
        for j in range(int(np.floor(duration / 0.2 + 1e-10))):
            start = float(p[lo, 0] + j * 0.2)
            windows.append((start, start + 0.2))
    if len(windows) > maximum:
        chosen = np.linspace(0, len(windows) - 1, maximum).astype(int)
        windows = [windows[i] for i in chosen]
    return np.asarray(windows).reshape(-1, 2)


def count_running_windows(spikes, units, windows):
    result = np.zeros((len(windows), 10, len(units)), dtype=np.int32)
    for k, unit in enumerate(units):
        times = np.sort(spikes[spikes[:, 1] == unit, 0])
        for j, (start, end) in enumerate(windows):
            edges = start + np.arange(11) * 0.02
            edges[-1] = end
            subset = times[np.searchsorted(times, start) : np.searchsorted(times, end)]
            result[j, :, k] = np.histogram(subset, edges)[0]
    return result


def neural_splits(animal, n_units):
    result = []
    for split in range(5):
        order = np.random.default_rng(seed(animal, "common-context-cells", split)).permutation(n_units)
        nheld = max(1, round(0.3 * n_units))
        result.append((np.sort(order[nheld:]), np.sort(order[:nheld])))
    return result


def worker(args):
    animal, records, parent, out = args
    out, parent = Path(out), Path(parent)
    started = time.monotonic()
    sources, meta = [], []
    for record in records:
        path = Path(record["artifact_path"])
        if file_sha256(path) != record["artifact_sha256"]:
            raise ValueError("changed source NPZ")
        with np.load(path) as z:
            sources.append({k: z[k] for k in z.files})
        meta.append(json.loads(path.with_suffix(".json").read_text()))
    units, maps, compositions, indices, unit_qc = build_bank(sources)
    contexts = ["ABCD"[round(np.log2(np.prod(np.diff(a["arena_bounds_cm"], axis=0)) / 10000 / 1.09375))] for a in sources]
    if sorted(contexts) != list("AABCD"):
        raise ValueError("four contexts with repeated A required")
    splits = neural_splits(animal, len(units))
    bank = {"unit_ids": units, "context_ids": np.array(contexts)}
    for i, (m, c) in enumerate(zip(maps, compositions, strict=True)):
        bank[f"rates_{i}"] = m
        bank[f"composition_{i}"] = c
    for i, (tr, he) in enumerate(splits):
        bank[f"train_{i}"], bank[f"held_{i}"] = tr, he
    np.savez_compressed(out / f"{animal}_bank.npz", **bank)
    unit_qc.assign(animal=animal).to_csv(out / f"{animal}_unit_qc.csv", index=False)
    rows, selected, fold_rows, template_rows = [], [], [], []
    for i, (a, info, record) in enumerate(zip(sources, meta, records, strict=True)):
        session, current = record["session"], contexts[i]
        tag = f"tanni2022__{animal}__{session}"
        selection_path = parent / f"{tag}_selection.csv"
        events = pd.read_csv(selection_path).sort_values(["start_s", "event_id"])
        midpoint = info["rate_map_half_split_time_s"]
        windows = running_windows(a, midpoint)
        if not len(windows):
            raise ValueError("no held-out running windows")
        run_counts = count_running_windows(a["spikes"], units, windows)
        saved = {"run_windows": windows, "run_counts": run_counts}
        lookup = {int(e): k for k, e in enumerate(a["candidate_event_indices"])}
        for e in events.itertuples(index=False):
            ix = lookup[e.event_id]
            lo, hi = a["candidate_offsets"][ix : ix + 2]
            counts, _, edges = pool_counts(a["candidate_base_counts"][lo:hi][:, indices[i]], a["candidate_base_starts_s"][lo:hi], a["candidate_base_durations_s"][lo:hi])
            if not np.allclose(edges[[0, -1]], [e.start_s, e.end_s], atol=1e-8, rtol=0):
                raise ValueError("source event interval mismatch")
            saved[f"counts_{e.event_id}"], saved[f"edges_{e.event_id}"] = counts, edges
        global_prob, event_folds = {}, {}
        reference = compositions[i] / compositions[i].sum()
        for fold, test, calibration, excluded in folds(events):
            cal_counts = sum((saved[f"counts_{eid}"].sum(axis=0) for eid in calibration.event_id), start=np.zeros(len(units), dtype=int))
            probability = (cal_counts + 100 * reference) / (cal_counts.sum() + 100)
            saved[f"event_global_{fold}"] = probability
            for eid in test.event_id:
                global_prob[eid], event_folds[eid] = probability, fold
            fold_rows.append(
                {
                    "animal": animal,
                    "session": session,
                    "fold": fold,
                    "test_ids": ",".join(map(str, test.event_id)),
                    "calibration_ids": ",".join(map(str, calibration.event_id)),
                    "guard_excluded_ids": ",".join(map(str, excluded.event_id)),
                }
            )
        np.savez_compressed(out / f"{animal}__{session}_observations.npz", **saved)
        template_rows.append(
            {
                "animal": animal,
                "session": session,
                "context": current,
                "template_index": i,
                "n_common_available_units": len(unit_qc),
                "n_encoding_units": len(units),
                "first_half_spatial_bins": maps[i].shape[1],
                "rate_map_cutoff_s": midpoint,
                "run_windows": len(windows),
                "MUA_events": len(events),
                "source_path": record["artifact_path"],
                "source_sha256": record["artifact_sha256"],
                "source_metadata_path": str(Path(record["artifact_path"]).with_suffix(".json")),
                "source_metadata_sha256": file_sha256(Path(record["artifact_path"]).with_suffix(".json")),
                "selection_path": str(selection_path),
                "selection_sha256": file_sha256(selection_path),
            }
        )
        jobs = [("RUN", j, c, start, end, reference, -1) for j, (c, (start, end)) in enumerate(zip(run_counts, windows, strict=True))]
        jobs += [
            ("MUA", int(e.event_id), saved[f"counts_{e.event_id}"], e.start_s, e.end_s, global_prob[e.event_id], event_folds[e.event_id]) for e in events.itertuples(index=False)
        ]
        for phase, eid, counts, start, end, baseline, fold in jobs:
            selected.append({"animal": animal, "session": session, "context": current, "phase": phase, "event_id": eid, "start_s": start, "end_s": end, "fold": fold})
            for split, (tr, he) in enumerate(splits):
                r = predict_contexts(counts, maps, compositions, tr, he, contexts, current, baseline)
                p = np.array([r[f"iid_p_{c}"] for c in "ABCD"])
                global_p = np.array([r[f"global_p_{c}"] for c in "ABCD"])
                rows.append(
                    {
                        "animal": animal,
                        "session": session,
                        "context": current,
                        "phase": phase,
                        "event_id": eid,
                        "split": split,
                        "n_train_cells": len(tr),
                        "n_heldout_cells": len(he),
                        "n_train_spikes": int(counts[:, tr].sum()),
                        "n_heldout_spikes": int(counts[:, he].sum()),
                        "duration_s": end - start,
                        "iid_context_correct": "ABCD"[int(p.argmax())] == current,
                        "global_context_correct": "ABCD"[int(global_p.argmax())] == current,
                        "iid_remote_mass": 1 - r[f"iid_p_{current}"],
                        "iid_correct_context_probability": r[f"iid_p_{current}"],
                        **r,
                    }
                )
        print(f"{animal} {session}: RUN={len(windows)}, MUA={len(events)}, common QC cells={len(units)}", flush=True)
    scores = pd.DataFrame(rows)
    scores.to_csv(out / f"{animal}_scores.csv", index=False)
    pd.DataFrame(selected).to_csv(out / f"{animal}_selection.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(out / f"{animal}_calibration_folds.csv", index=False)
    pd.DataFrame(template_rows).to_csv(out / f"{animal}_templates.csv", index=False)
    return {"animal": animal, "rows": len(scores), "units": len(units), "runtime_s": time.monotonic() - started}


def event_contrasts(scores):
    keys = ["animal", "session", "context", "phase", "event_id"]
    if scores.duplicated(keys + ["split"]).any() or not scores.heldout_used_for_inference.eq(False).all():
        raise ValueError("duplicate or leaking prediction")
    if not scores.groupby(keys).split.apply(lambda s: set(s) == set(range(5))).all():
        raise ValueError("missing cell split")
    s = scores.copy()
    for name, (left, right) in CONTRASTS.items():
        s[name] = s[left] - s[right]
        s[name + "_per_spike"] = s[name] / s.n_heldout_spikes.replace(0, np.nan)
    metrics = [v for k in CONTRASTS for v in (k, k + "_per_spike")]
    e = s.groupby(keys, as_index=False)[metrics + ["n_heldout_spikes", "n_train_spikes", "duration_s"]].median()
    return s, e


def balanced_means(events, metrics):
    session = events.groupby(["animal", "session", "context", "phase"], as_index=False)[metrics].mean()
    counts = events.groupby(["animal", "session", "context", "phase"]).size().rename("events").reset_index()
    session = session.merge(counts, validate="one_to_one")
    context = session.groupby(["animal", "context", "phase"], as_index=False)[metrics].mean()
    animal = context.groupby(["animal", "phase"], as_index=False)[metrics].mean()
    return session, context, animal


def bootstrap_mua(events, metrics, draws=5000):
    rng = np.random.default_rng(20260908)
    animals = sorted(events.animal.unique())
    estimates = {}
    for animal in animals:
        context_samples = []
        for _, g in events[events.animal.eq(animal)].groupby("context", sort=True):
            session_samples = []
            for _, f in g.groupby("session", sort=True):
                values = f[metrics].to_numpy(float)
                n_resamples = draws * len(animals)
                samples = np.empty((n_resamples, len(metrics)))
                for lo in range(0, n_resamples, 100):
                    ix = rng.integers(0, len(values), size=(min(100, n_resamples - lo), len(values)))
                    samples[lo : lo + len(ix)] = np.nanmean(values[ix], axis=1)
                session_samples.append(samples)
            context_samples.append(np.mean(session_samples, axis=0))
        if len(context_samples) != 4:
            raise ValueError("all physical contexts required")
        estimates[animal] = np.mean(context_samples, axis=0).reshape(draws, len(animals), len(metrics))
    chosen = rng.integers(0, len(animals), size=(draws, len(animals)))
    tensor = np.stack([estimates[a] for a in animals])
    distribution = tensor[chosen, np.arange(draws)[:, None], np.arange(len(animals))[None, :]].mean(axis=1)
    _, _, observed = balanced_means(events, metrics)
    rows = []
    for j, metric in enumerate(metrics):
        values = observed[metric].to_numpy()
        lo, hi = np.quantile(distribution[:, j], [0.025, 0.975])
        rows.append(
            {
                "metric": metric,
                "mean": float(values.mean()),
                "ci_low": float(lo),
                "ci_high": float(hi),
                "positive_animals": int((values > 0).sum()),
                "animals": len(animals),
                "primary": metric in PRIMARY,
            }
        )
    return pd.DataFrame(rows)


def run(args):
    source, parent, out = (Path(x).resolve() for x in (args.source_dir, args.parent_dir, args.output_dir))
    if file_sha256(source / "coverage_input_manifest.json") != SOURCE_HASH or file_sha256(parent / "conditional_2d_manifest.json") != PARENT_HASH:
        raise ValueError("wrong frozen source/selection parent")
    if out.exists() and any(out.iterdir()):
        raise ValueError("refusing overwrite")
    out.mkdir(parents=True, exist_ok=True)
    manifest = build_script_provenance(
        input_paths={
            "source_manifest": source / "coverage_input_manifest.json",
            "source_sessions": source / "coverage_input_sessions.csv",
            "parent_manifest": parent / "conditional_2d_manifest.json",
            "protocol": ROOT / "docs/tanni_multicontext_protocol.md",
        }
    )
    manifest.update(
        status="running",
        encoding="first_half_RUN_only",
        target_score="proper_conditional_multinomial_frozen_training_posterior",
        independent_confirmation=False,
        temporal_order_test=False,
    )
    path = out / "tanni_multicontext_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    table = pd.read_csv(source / "coverage_input_sessions.csv")
    table = table[table.dataset.eq("tanni2022")].sort_values(["animal", "session"])
    jobs = [(animal, g.to_dict("records"), str(parent), str(out)) for animal, g in table.groupby("animal", sort=True)]
    if len(jobs) != 5 or any(len(j[1]) != 5 for j in jobs):
        raise ValueError("expected five animals with five recordings each")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        tasks = [pool.submit(worker, job) for job in jobs]
        manifest["tasks"] = [f.result() for f in as_completed(tasks)]
    scores = pd.concat([pd.read_csv(out / f"{j[0]}_scores.csv") for j in jobs], ignore_index=True)
    splits, events = event_contrasts(scores)
    if len(events[events.phase.eq("MUA")]) != 5224:
        raise ValueError("frozen MUA coverage changed")
    metrics = [v for k in CONTRASTS for v in (k, k + "_per_spike")]
    sessions, contexts, animals = balanced_means(events, metrics)
    summary = bootstrap_mua(events[events.phase.eq("MUA")], metrics)
    raw_run = scores[scores.phase.eq("RUN")]
    accuracy = raw_run.groupby(["animal", "session", "context"], as_index=False)[["iid_context_correct", "global_context_correct", "iid_correct_context_probability"]].mean()
    accuracy = accuracy.groupby(["animal", "context"], as_index=False)[["iid_context_correct", "global_context_correct", "iid_correct_context_probability"]].mean()
    accuracy = accuracy.groupby("animal", as_index=False)[["iid_context_correct", "global_context_correct", "iid_correct_context_probability"]].mean()
    run_animal = animals[animals.phase.eq("RUN")].merge(accuracy, on="animal", validate="one_to_one")
    enough_run = sessions[sessions.phase.eq("RUN")].events.ge(50).all()
    run_pass = bool(enough_run and run_animal.iid_context_correct.gt(0.5).all() and run_animal.current_iid_minus_current_global.gt(0).all())
    primary = summary[summary.primary]
    predictive_pass = bool(len(primary) == 3 and primary["mean"].gt(0).all() and primary.ci_low.gt(0).all() and primary.positive_animals.eq(5).all())
    decisions = {
        "RUN_context_validation_passed": run_pass,
        "all_candidate_predictive_contrasts_passed": predictive_pass,
        "cross_context_spatial_reactivation_lead": run_pass and predictive_pass,
        "biological_replay_claim": False,
        "temporal_order_validated": False,
        "new_high_importance_mechanism_established": False,
    }
    for name, frame in (
        ("split_contrasts", splits),
        ("event_contrasts", events),
        ("sessions", sessions),
        ("contexts", contexts),
        ("animals", animals),
        ("summary", summary),
        ("RUN_validation", run_animal),
    ):
        frame.to_csv(out / f"tanni_multicontext_{name}.csv", index=False)
    pd.DataFrame([decisions]).to_csv(out / "tanni_multicontext_decisions.csv", index=False)
    remote = scores[scores.phase.eq("MUA") & scores.split.eq(0)].copy()
    remote["training_remote_ge_0_8"] = remote.iid_remote_mass.ge(0.8)
    remote.groupby(["animal", "session", "context"], as_index=False)[["iid_remote_mass", "training_remote_ge_0_8"]].mean().to_csv(
        out / "tanni_multicontext_remote_descriptive.csv", index=False
    )
    manifest.update(status="complete", events=5224, RUN_windows=int(events.phase.eq("RUN").sum()), decisions=decisions)
    manifest["output_sha256"] = {f.name: file_sha256(f) for f in out.iterdir() if f != path}
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(decisions), flush=True)
    print(run_animal[["animal", "iid_context_correct", "current_iid_minus_current_global"]].to_string(index=False), flush=True)
    print(primary.to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--parent-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=5)
    run(parser.parse_args())
