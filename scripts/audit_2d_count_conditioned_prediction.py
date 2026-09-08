#!/usr/bin/env python3
"""Frozen common-pipeline PF/Tanni conditional cross-cell predictive experiment."""

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

from hipporeplayimm.conditional_spatial_prediction import MODELS, SpatialPredictionContext, score_event

SEED = 20260908
IDENTITY = ["dataset", "animal", "session"]
KEYS = IDENTITY + ["event_id", "split"]
INPUT_HASH = "2ed1c4d910a3ddc4027eefb398d48a55bae6829c3bf16e8f57955beb9ee55613"
WINDOW_HASH = "116a278789c0d3b9ce1f67e04954924231987db8e8e8daa16f03282e29f4f5b8"


def seed(*parts):
    return int.from_bytes(hashlib.sha256("|".join(map(str, (SEED, *parts))).encode()).digest()[:4], "little")


def load_inputs(input_dir, event_dir):
    for root, filename, digest in ((input_dir, "coverage_input_manifest.json", INPUT_HASH), (event_dir, "coverage_event_definition_manifest.json", WINDOW_HASH)):
        if file_sha256(root / filename) != digest:
            raise ValueError("source experiment changed")
    manifest = json.loads((event_dir / "coverage_event_definition_manifest.json").read_text())
    frames = []
    for item in manifest["results"]:
        for path_key, hash_key in (("windows_path", "windows_sha256"), ("source_cache_path", "source_cache_sha256")):
            if file_sha256(item[path_key]) != item[hash_key]:
                raise ValueError("source windows/cache changed")
        table = pd.read_csv(item["windows_path"])
        selected = table[table.detector.eq("source_high_mua") & table.window_variant.eq("detected_core") & table.eligible.eq(True)].copy()
        selected = selected.rename(columns={"source_event_id": "event_id"})
        if selected.empty or selected[IDENTITY].drop_duplicates().shape[0] != 1 or selected.event_id.duplicated().any():
            raise ValueError("incomplete/ambiguous session cohort")
        frames.append(selected)
    selected = pd.concat(frames, ignore_index=True)
    if (
        selected.groupby("dataset").size().to_dict() != {"pfeiffer_foster": 4001, "tanni2022": 5224}
        or len(frames) != 33
        or selected.groupby("dataset").animal.nunique().to_dict() != {"pfeiffer_foster": 4, "tanni2022": 5}
    ):
        raise ValueError("frozen all-session cohort differs")
    return selected, manifest


def folds(events):
    events = events.sort_values(["start_s", "event_id"])
    if len(events) < 5 or events.event_id.duplicated().any():
        raise ValueError("five nonempty event folds required")
    result = []
    for fold, indices in enumerate(np.array_split(np.arange(len(events)), 5)):
        test = events.iloc[indices]
        candidates = events[~events.event_id.isin(test.event_id)]
        good = np.ones(len(candidates), dtype=bool)
        for t in test.itertuples(index=False):
            good &= (candidates.end_s.to_numpy() + 1 <= t.start_s) | (candidates.start_s.to_numpy() >= t.end_s + 1)
        if not good.any():
            raise ValueError("no guarded calibration events")
        result.append((fold, test, candidates[good], candidates[~good]))
    return result


def pool_counts(base, starts, durations):
    if len(base) != len(starts) or len(base) != len(durations) or not len(base) or (durations <= 0).any():
        raise ValueError("invalid source count clock")
    if not np.allclose(durations[:-1], 0.005, atol=1e-8) or durations[-1] > 0.005 + 1e-8:
        raise ValueError("expected native cached 5 ms bins")
    indices = np.arange(0, len(base), 4)
    counts = np.add.reduceat(base, indices, axis=0)
    dt = np.add.reduceat(durations, indices)
    left = starts[indices]
    if not np.array_equal(counts.sum(axis=0), base.sum(axis=0)) or not np.allclose(left[1:], left[:-1] + dt[:-1], atol=1e-8):
        raise ValueError("pooling changed counts or continuity")
    return counts, left + dt / 2, np.r_[left, left[-1] + dt[-1]]


def prepare_cache(selected):
    source = Path(selected.iloc[0].source_cache_path)
    with np.load(source) as loaded:
        a = {k: loaded[k] for k in loaded.files if k not in ("position", "spikes")}
    mask, bins = a["unit_qc_mask"].astype(bool), a["valid_spatial_bins"].astype(bool)
    units, rates, centers = a["cell_ids"][mask], a["rates_hz"][mask][:, bins], a["bin_centers_cm"][bins]
    if len(units) < 5 or len(centers) < 2 or not np.isfinite(rates).all() or (rates <= 0).any():
        raise ValueError("invalid frozen encoding")
    cache = {"unit_ids": units, "rates": rates, "centers": centers, "occupancy": a["occupancy_s"][bins], "arena_bounds_cm": a["arena_bounds_cm"]}
    lookup = {int(e): j for j, e in enumerate(a["candidate_event_indices"])}
    for e in selected.itertuples(index=False):
        j = lookup[int(e.event_id)]
        start, end = a["candidate_offsets"][j : j + 2]
        if not np.allclose([a["candidate_start_s"][j], a["candidate_end_s"][j]], [e.start_s, e.end_s], atol=1e-8, rtol=0):
            raise ValueError("source event interval mismatch")
        c, times, edges = pool_counts(a["candidate_base_counts"][start:end][:, mask], a["candidate_base_starts_s"][start:end], a["candidate_base_durations_s"][start:end])
        if c.sum() != e.n_spikes_qc_units or np.count_nonzero(c.sum(axis=0)) != e.n_active_qc_units:
            raise ValueError("candidate support mismatch")
        cache[f"counts_{e.event_id}"], cache[f"times_{e.event_id}"], cache[f"edges_{e.event_id}"] = c, times, edges
    identity = tuple(selected.iloc[0][IDENTITY])
    for split in range(5):
        indices = np.random.default_rng(seed(*identity, "cells", split)).permutation(len(units))
        nheld = max(1, round(0.3 * len(units)))
        cache[f"held_{split}"], cache[f"train_{split}"] = np.sort(indices[:nheld]), np.sort(indices[nheld:])
    cache["permutation"] = np.random.default_rng(seed(*identity, "map")).permutation(len(centers))
    return cache


def task(args):
    selected, out = args
    out = Path(out)
    identity = dict(zip(IDENTITY, selected.iloc[0][IDENTITY], strict=True))
    tag = "__".join(str(identity[k]).replace("/", "_") for k in IDENTITY)
    cache = prepare_cache(selected)
    context = SpatialPredictionContext(cache["centers"])
    reference = cache["rates"].mean(axis=1)
    reference /= reference.sum()
    rows, metadata = [], []
    started = time.monotonic()
    for fold, test, calibration, excluded in folds(selected):
        counts_cal = sum((cache[f"counts_{i}"].sum(axis=0) for i in calibration.event_id), start=np.zeros(len(reference), dtype=np.int64))
        probability = (counts_cal + 100 * reference) / (counts_cal.sum() + 100)
        cache[f"global_{fold}"] = probability
        metadata.append(
            identity
            | {
                "fold": fold,
                "test_ids": ",".join(map(str, test.event_id)),
                "calibration_ids": ",".join(map(str, calibration.event_id)),
                "excluded_ids": ",".join(map(str, excluded.event_id)),
                "n_calibration_events": len(calibration),
                "n_calibration_spikes": int(counts_cal.sum()),
            }
        )
        for e in test.itertuples(index=False):
            counts, times = cache[f"counts_{e.event_id}"], cache[f"times_{e.event_id}"]
            for split in range(5):
                tr, he = cache[f"train_{split}"], cache[f"held_{split}"]
                tick = time.monotonic()
                values = score_event(counts, times, cache["rates"], tr, he, context, cache["permutation"], probability)
                for row in values:
                    rows.append(
                        identity
                        | {
                            "event_id": e.event_id,
                            "fold": fold,
                            "split": split,
                            "n_time_bins": len(counts),
                            "n_train_cells": len(tr),
                            "n_heldout_cells": len(he),
                            "n_train_spikes": int(counts[:, tr].sum()),
                            "n_heldout_spikes": int(counts[:, he].sum()),
                            "duration_s": e.end_s - e.start_s,
                            "runtime_s": time.monotonic() - tick,
                            "status": "success",
                        }
                        | row
                    )
        print(f"{tag} fold {fold + 1}/5: {len(test)} events", flush=True)
    np.savez_compressed(out / f"{tag}_cache.npz", **cache)
    pd.DataFrame(rows).to_csv(out / f"{tag}_scores.csv", index=False)
    pd.DataFrame(metadata).to_csv(out / f"{tag}_folds.csv", index=False)
    selected.to_csv(out / f"{tag}_selection.csv", index=False)
    return identity | {"tag": tag, "events": len(selected), "rows": len(rows), "runtime_s": time.monotonic() - started}


def contrasts(scores):
    if scores.duplicated(KEYS + ["map"]).any():
        raise ValueError("duplicate predictive score rows")
    real = scores[scores["map"].eq("real")].set_index(KEYS)
    wrong = scores[scores["map"].eq("population_code_permuted")].set_index(KEYS).reindex(real.index)
    values = {f"imm_minus_{m}": real.score_first_order_imm - real[f"score_{m}"] for m in ("iid_position", "static_location", "diffusion", "event_global", "run_global")}
    values.update(
        real_minus_wrong_imm=real.score_first_order_imm - wrong.score_first_order_imm,
        iid_minus_event_global=real.score_iid_position - real.score_event_global,
        diffusion_minus_iid=real.score_diffusion - real.score_iid_position,
        event_global_minus_run_global=real.score_event_global - real.score_run_global,
    )
    records = []
    for name, value in values.items():
        if not np.isfinite(value).all():
            raise ValueError("missing paired prediction")
        part = value.rename("delta").to_frame()
        part["delta_per_heldout_spike"] = value / real.n_heldout_spikes.replace(0, np.nan)
        records.append(part.reset_index().assign(contrast=name))
    split = pd.concat(records, ignore_index=True)
    events = split.groupby(IDENTITY + ["event_id", "contrast"], as_index=False)[["delta", "delta_per_heldout_spike"]].median()
    return split, events


def aggregate(events):
    columns = ["delta", "delta_per_heldout_spike"]
    sessions = events.groupby(IDENTITY + ["contrast"], as_index=False)[columns].mean()
    animals = sessions.groupby(["dataset", "animal", "contrast"], as_index=False)[columns].mean()
    rows = []
    for (dataset, contrast), group in events.groupby(["dataset", "contrast"]):
        local = animals[animals.dataset.eq(dataset) & animals.contrast.eq(contrast)]
        rats = sorted(local.animal.unique())
        # Hierarchical resampling retains equal session weights within each animal.
        cached = {rat: [g[columns].to_numpy() for _, g in group[group.animal.eq(rat)].groupby("session")] for rat in rats}
        rng, boot = np.random.default_rng(SEED), []
        for _ in range(5000):
            estimates = []
            for i in rng.integers(0, len(rats), len(rats)):
                source = cached[rats[i]]
                selected = [source[j] for j in rng.integers(0, len(source), len(source))]
                estimates.append(np.nanmean([np.nanmean(a[rng.integers(0, len(a), len(a))], axis=0) for a in selected], axis=0))
            boot.append(np.nanmean(estimates, axis=0))
        ci = np.nanquantile(boot, [0.025, 0.975], axis=0)
        rows.append(
            {
                "dataset": dataset,
                "contrast": contrast,
                "mean": local.delta.mean(),
                "ci_low": ci[0, 0],
                "ci_high": ci[1, 0],
                "mean_per_heldout_spike": local.delta_per_heldout_spike.mean(),
                "per_spike_ci_low": ci[0, 1],
                "per_spike_ci_high": ci[1, 1],
                "positive_animals": int(local.delta.gt(0).sum()),
                "animals": len(rats),
                "sessions": group.session.nunique(),
                "events": group.event_id.size,
            }
        )
    return pd.DataFrame(rows), animals, sessions


def technical_gates(scores, selected):
    if scores.empty or selected.empty:
        return False
    keys = IDENTITY + ["event_id"]
    columns = [f"score_{m}" for m in (*MODELS, "event_global", "run_global")]
    return bool(
        len(scores) == len(selected) * 10
        and not scores.duplicated(KEYS + ["map"]).any()
        and set(scores[keys].itertuples(index=False, name=None)) == set(selected[keys].itertuples(index=False, name=None))
        and all(
            set(g[["split", "map"]].itertuples(index=False, name=None)) == {(s, m) for s in range(5) for m in ("real", "population_code_permuted")} for _, g in scores.groupby(keys)
        )
        and np.isfinite(scores[columns]).all().all()
        and scores[columns].le(1e-8).all().all()
        and scores.status.eq("success").all()
        and scores.posterior_unchanged.eq(True).all()
        and scores.heldout_used_for_inference.eq(False).all()
    )


def run(args):
    input_dir, event_dir, out = Path(args.input_dir).resolve(), Path(args.event_dir).resolve(), Path(args.output_dir).resolve()
    selected, _parent = load_inputs(input_dir, event_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError("refusing to overwrite frozen experiment")
    out.mkdir(parents=True, exist_ok=True)
    selected.to_csv(out / "frozen_selection.csv", index=False)
    manifest = build_script_provenance(
        input_paths={
            "input_manifest": input_dir / "coverage_input_manifest.json",
            "event_manifest": event_dir / "coverage_event_definition_manifest.json",
            "protocol": ROOT / "docs/2d_count_conditioned_prediction_protocol.md",
        }
    )
    manifest.update(
        status="running",
        source_input_dir=str(input_dir),
        source_event_dir=str(event_dir),
        n_events=len(selected),
        n_sessions=33,
        n_animals=9,
        n_splits=5,
        scoring_bin_s=0.02,
        encoding_bin_cm=8,
        diffusion_sigma_cm_sqrt_s=60,
        stationary_sigma_cm=2,
        max_step_sigma=3,
        imm_switch_tau_s=0.06,
        time_duration_round_decimals=9,
        global_pseudocount=100,
        seed=SEED,
        claim_boundary="retrospective common-pipeline cross-cell prediction; not replay prevalence or a unique mechanism",
    )
    path = out / "conditional_2d_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    started, completed = time.monotonic(), []
    try:
        tasks = [(g.copy(), out) for _, g in selected.groupby(IDENTITY)]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for future in as_completed([pool.submit(task, item) for item in tasks]):
                item = future.result()
                completed.append(item)
                print(json.dumps(item), flush=True)
        scores = pd.concat([pd.read_csv(out / f"{v['tag']}_scores.csv") for v in completed], ignore_index=True)
        passed = technical_gates(scores, selected)
        pd.DataFrame([{"gate": "complete_nonleaking_proper_prediction", "passed": passed}, {"gate": "overall", "passed": passed}]).to_csv(
            out / "conditional_2d_gates.csv", index=False
        )
        if not passed:
            raise ValueError("technical gates failed")
        split, events = contrasts(scores)
        summary, animals, sessions = aggregate(events)
        for name, frame in (("split_contrasts", split), ("event_contrasts", events), ("summary", summary), ("by_animal", animals), ("by_session", sessions)):
            frame.to_csv(out / f"conditional_2d_{name}.csv", index=False)
        manifest.update(status="complete", rows=len(scores), runtime_s=time.monotonic() - started)
    except BaseException as exc:
        manifest.update(status="failed", error=repr(exc))
        raise
    finally:
        manifest["completed"] = completed
        manifest["output_sha256"] = {p.name: file_sha256(p) for p in out.iterdir() if p.is_file() and p != path}
        path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="/mnt/seagate10tb/florianpfaff/replay-coverage-real-inputs-all33-20260905")
    parser.add_argument("--event-dir", default="/mnt/seagate10tb/florianpfaff/replay-coverage-event-definitions-all33-20260905-v2")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=8)
    run(parser.parse_args())
