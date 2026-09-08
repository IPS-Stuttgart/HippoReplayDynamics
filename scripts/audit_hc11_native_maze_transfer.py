#!/usr/bin/env python3
"""Frozen, temporally held-out native MAZE encoding/prediction control."""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import audit_hc11_count_conditioned_prediction as frozen
from _provenance import build_script_provenance, file_sha256

from hipporeplayimm.benchmarks import _split_cells
from hipporeplayimm.frozen_posterior_prediction import posterior_sha256

native = frozen.native
UNIT_REGIMES = ("train_only_qc", "frozen_parent_units")
COUNT_REGIMES = ("native", "sleep_total_cap")
PARENT_DIGEST = "bc846d25e27b9400d2870b2cad534c781d1fdee12f27496e5a8d96db153d3d07"
SEED = 20260908
GUARD = 5.0
WINDOW = 0.2
BIN = 0.02
MAX_WINDOWS = 100
CONTRASTS = {
    "behavior_minus_global": ("behavior", "global"),
    "behavior_real_minus_wrong": ("behavior", "behavior_wrong"),
    "iid_minus_global": ("iid_position", "global"),
    "imm_minus_iid": ("first_order_imm", "iid_position"),
    "imm_minus_static": ("first_order_imm", "static_location"),
    "diffusion_minus_iid": ("diffusion", "iid_position"),
    "static_minus_iid": ("static_location", "iid_position"),
}


def fold_intervals(track, fold):
    if fold not in (0, 1) or np.shape(track.maze_epoch) != (1, 2):
        raise ValueError("requires two folds of one native MAZE epoch")
    start, stop = track.maze_epoch[0]
    middle = (start + stop) / 2
    intervals = ((start + GUARD, middle - GUARD), (middle + GUARD, stop - GUARD))
    if any(b - a <= WINDOW for a, b in intervals):
        raise ValueError("MAZE too short for guarded folds")
    return intervals[fold], intervals[1 - fold]


def training_inputs(track, spikes, interval):
    start, stop = interval
    mask = track.maze_mask & (track.times_s >= start) & (track.times_s < stop)
    restricted = native.SpikeData(spikes.unit_ids, {uid: values[(values >= start) & (values < stop)] for uid, values in spikes.times_by_unit.items()})
    return replace(track, maze_mask=mask), restricted, mask


def select_windows(track, interval, session, fold):
    start, stop = interval
    n = max(0, int(np.floor((stop - start) / WINDOW)))
    starts = start + np.arange(n) * WINDOW
    centers = starts[:, None] + (np.arange(10) + 0.5) * BIN
    indices = native.nearest_frame_indices(track.times_s, centers.ravel()).reshape(n, 10)
    good = track.maze_mask[indices] & np.isfinite(track.position_cm[indices]) & np.isfinite(track.speed_cm_s[indices])
    good &= np.abs(track.times_s[indices] - centers) <= 0.05
    good &= track.speed_cm_s[indices] >= frozen.ENCODING_PARAMETERS["min_run_speed_cm_s"]
    directions = track.direction[indices]
    supported = good.all(axis=1)
    constant = (directions == directions[:, :1]).all(axis=1) & (directions[:, 0] != 0)
    eligible = np.flatnonzero(supported & constant)
    if len(eligible) > MAX_WINDOWS:
        rng = np.random.default_rng(frozen.stable_seed(SEED, "hc11_maze_transfer_windows", session, fold))
        eligible = np.sort(rng.choice(eligible, MAX_WINDOWS, replace=False))
    windows = pd.DataFrame({"window_id": eligible, "start_time_s": starts[eligible], "end_time_s": starts[eligible] + WINDOW})
    qc = {"candidate_windows": n, "behavior_supported_windows": int(supported.sum()), "eligible_windows": int((supported & constant).sum()), "selected_windows": len(windows)}
    return windows, qc


def fit_maps(track, spikes, interval, unit_regime, parent_units):
    restricted, train_spikes, mask = training_inputs(track, spikes, interval)
    if unit_regime == "train_only_qc":
        maps, qc = native.build_session_encodings(restricted, train_spikes, **frozen.ENCODING_PARAMETERS)
    elif unit_regime == "frozen_parent_units":
        edges = native.make_bin_edges(track.track_length_cm, frozen.ENCODING_PARAMETERS["position_bin_size_cm"])
        moving = restricted.maze_mask & (restricted.speed_cm_s >= frozen.ENCODING_PARAMETERS["min_run_speed_cm_s"])
        encodings = [
            native.fit_encoding_map(
                restricted,
                train_spikes,
                tuple(parent_units),
                frame_mask=moving & direction_mask,
                bin_edges_cm=edges,
                smoothing_sigma_bins=frozen.ENCODING_PARAMETERS["smoothing_sigma_bins"],
                name=name,
            )
            for name, direction_mask in (("pooled", np.ones(len(mask), bool)), ("negative", track.direction < 0), ("positive", track.direction > 0))
        ]
        maps = {"pooled": [encodings[0]], "direction_mixture": encodings[1:]}
        qc = pd.DataFrame({"unit_id": parent_units, "unit_qc_passed": True, "selection_scope": "previous_full_RUN_selection_sensitivity_only"})
    else:
        raise ValueError("unknown unit regime")
    return maps, qc, mask


def capped_counts(counts, target, seed):
    counts = np.asarray(counts)
    if not np.issubdtype(counts.dtype, np.integer) or np.any(counts < 0) or target < 0 or int(target) != target:
        raise ValueError("nonnegative integer counts and target required")
    n = int(counts.sum())
    if n <= target:
        return counts.copy(), n == target
    rng = np.random.default_rng(seed)
    result = rng.multivariate_hypergeometric(counts.ravel(), int(target)).reshape(counts.shape)
    if np.any(result > counts) or result.sum() != target:
        raise ValueError("thinning changed or duplicated a spike")
    return result, True


def infer_and_predict(counts, edges, rates, train, held, kernels):
    if len(np.intersect1d(train, held)) or not len(train) or not len(held):
        raise ValueError("invalid neural split")
    parts = [frozen.pf.likelihood_parts(counts[:, train], r[train], np.diff(edges)) for r in rates]
    post = {model: frozen.infer_training(parts, "count_conditioned", 1.0, model, kernels) for model in frozen.MODELS}
    hashes = {model: posterior_sha256(q) for model, q in post.items()}
    likelihood = [frozen.pf.likelihood_parts(counts[:, held], r[held], np.diff(edges)) for r in rates]
    scores = {f"score_{model}": frozen.heldout_score(q, likelihood, "count_conditioned") for model, q in post.items()}
    if any(posterior_sha256(post[m]) != digest for m, digest in hashes.items()):
        raise ValueError("held-out observations altered training inference")
    return scores, post, hashes


def behavior_scores(counts, edges, rates, pooled_rates, occupancy, held, positions, directions, bin_edges, permutation):
    n = len(bin_edges) - 1
    bins = np.clip(np.searchsorted(bin_edges, positions, side="right") - 1, 0, n - 1)
    dd = np.zeros(len(bins), dtype=int) if len(rates) == 1 else (directions > 0).astype(int)
    real = np.stack([frozen.pf.likelihood_parts(counts[:, held], r[held], np.diff(edges))["count_conditioned"] for r in rates])
    score = real[dd, np.arange(len(bins)), bins].sum()
    wrong = real[dd, np.arange(len(bins)), permutation[bins]].sum()
    global_rate = (pooled_rates @ (occupancy / occupancy.sum()))[:, None]
    baseline = frozen.pf.likelihood_parts(counts[:, held], global_rate[held], np.diff(edges))["count_conditioned"].sum()
    return {"score_behavior": float(score), "score_behavior_wrong": float(wrong), "score_global": float(baseline)}


def position_metrics(logq, rates, centers, bin_edges, truth, counts_train, topology, length):
    q = np.exp(logsumexp(logq.reshape(len(logq), len(rates), len(centers)), axis=1))
    map_position = centers[np.argmax(q, axis=1)]
    if topology == "circular":
        resultant = q @ np.exp(2j * np.pi * centers / length)
        mean_position = np.mod(np.angle(resultant), 2 * np.pi) * length / (2 * np.pi)
        mean_position[np.abs(resultant) < 1e-8] = np.nan
    else:
        mean_position = q @ centers
    map_error = native.topology_distance(map_position, truth, topology, length)
    mean_error = native.topology_distance(mean_position, truth, topology, length)
    ranked = np.argsort(-q, axis=1)
    mass_before = np.cumsum(np.take_along_axis(q, ranked, axis=1), axis=1) - np.take_along_axis(q, ranked, axis=1)
    in_set = np.zeros_like(q, dtype=bool)
    np.put_along_axis(in_set, ranked, mass_before < 0.95, axis=1)
    bins = np.clip(np.searchsorted(bin_edges, truth, side="right") - 1, 0, len(centers) - 1)
    covered = in_set[np.arange(len(q)), bins]
    supported = (counts_train.sum(axis=1) >= 2) & ((counts_train > 0).sum(axis=1) >= 2)
    result = {"fraction_supported_bins": supported.mean(), "mean_posterior_entropy": float(-(q * np.log(np.maximum(q, 1e-300))).sum(axis=1).mean())}
    for name, mask in (("all", np.ones(len(q), bool)), ("supported", supported)):
        result[f"median_map_error_cm_{name}"] = float(np.median(map_error[mask])) if mask.any() else np.nan
        finite = mask & np.isfinite(mean_error)
        result[f"median_mean_error_cm_{name}"] = float(np.median(mean_error[finite])) if finite.any() else np.nan
        result[f"coverage_95_{name}"] = float(covered[mask].mean()) if mask.any() else np.nan
    return result


def score_fold(task):
    session, animal, fold, source_dir, dataset_root, output = task
    source, output = Path(source_dir), Path(output)
    folders = list(Path(dataset_root).glob(f"*/{session}"))
    if len(folders) != 1:
        raise ValueError("ambiguous native source session")
    folder = folders[0]
    track, spikes = native.load_track_samples(folder), native.load_spikes(folder)
    with np.load(source / f"{session}_cache.npz") as previous:
        parent_units = previous["unit_ids"].copy()
    selection = pd.read_csv(source / "frozen_selection.csv")
    totals = selection[(selection.session == session) & (selection.phase == "POST")].sort_values("event_id").n_spikes.to_numpy(int)
    train_interval, test_interval = fold_intervals(track, fold)
    windows, window_qc = select_windows(track, test_interval, session, fold)
    target_rng = np.random.default_rng(frozen.stable_seed(SEED, "hc11_maze_transfer_targets", session, fold))
    targets = np.resize(target_rng.permutation(totals), len(windows))
    windows = windows.assign(session=session, rat=animal, fold=fold, target_sleep_spikes=targets)
    tag = f"{session}_fold{fold}"
    windows.to_csv(output / f"{tag}_selection.csv", index=False)
    cache = {
        "topology": np.array(track.topology),
        "track_length": np.array(track.track_length_cm),
        "train_interval": np.array(train_interval),
        "test_interval": np.array(test_interval),
        "parent_unit_ids": parent_units,
    }
    rows, qc_rows = [], []
    for regime in UNIT_REGIMES:
        started = time.monotonic()
        try:
            maps, unit_qc, train_mask = fit_maps(track, spikes, train_interval, regime, parent_units)
        except ValueError as exc:
            if regime != "train_only_qc" or "place-like units pass QC" not in str(exc):
                raise
            qc_rows.append(
                {"session": session, "rat": animal, "fold": fold, "unit_regime": regime, "status": "insufficient_training_units", "failure_reason": str(exc), **window_qc}
            )
            continue
        pooled = maps["pooled"][0]
        units = np.asarray(pooled.unit_ids)
        rates = {variant: [enc.rates_hz for enc in maps[variant]] for variant in frozen.VARIANTS}
        n = len(pooled.bin_centers_cm)
        permutation = np.random.default_rng(frozen.stable_seed(SEED, "hc11_maze_transfer_map", session, fold)).permutation(n)
        cache.update(centers=pooled.bin_centers_cm, bin_edges=pooled.bin_edges_cm, permutation=permutation, train_mask=train_mask)
        cache[f"unit_ids_{regime}"] = units
        cache[f"occupancy_{regime}"] = pooled.occupancy_s
        for variant, encodings in rates.items():
            for d, rate in enumerate(encodings):
                cache[f"rates_{regime}_{variant}_{d}"] = rate
        splits = []
        for split in range(frozen.N_SPLITS):
            tr_uid, he_uid = _split_cells(units, 0.3, frozen.SPLIT_SEED + split)
            tr, he = np.flatnonzero(np.isin(units, tr_uid)), np.flatnonzero(np.isin(units, he_uid))
            cache[f"train_{regime}_{split}"], cache[f"held_{regime}_{split}"] = tr, he
            splits.append((tr, he))
        unit_qc.to_csv(output / f"{tag}_{regime}_unit_qc.csv", index=False)
        for window in windows.itertuples(index=False):
            edges = float(window.start_time_s) + np.arange(11) * BIN
            counts = frozen.count_spikes(spikes, tuple(units), edges)
            indices = native.nearest_frame_indices(track.times_s, 0.5 * (edges[:-1] + edges[1:]))
            positions, directions = track.position_cm[indices], track.direction[indices]
            kernels = frozen.transitions(pooled.bin_centers_cm, edges, track.topology, track.track_length_cm)
            cap_seed = frozen.stable_seed(SEED, "hc11_maze_transfer_cap", session, fold, regime, window.window_id)
            capped, exact = capped_counts(counts, int(window.target_sleep_spikes), cap_seed)
            key = str(window.window_id)
            cache[f"edges_{key}"], cache[f"position_{key}"], cache[f"direction_{key}"] = edges, positions, directions
            for count_regime, observed in zip(COUNT_REGIMES, (counts, capped), strict=True):
                cache[f"counts_{regime}_{count_regime}_{key}"] = observed
                for split, (tr, he) in enumerate(splits):
                    for variant in frozen.VARIANTS:
                        prediction, post, hashes = infer_and_predict(observed, edges, rates[variant], tr, he, kernels)
                        known = behavior_scores(observed, edges, rates[variant], pooled.rates_hz, pooled.occupancy_s, he, positions, directions, pooled.bin_edges_cm, permutation)
                        metrics = position_metrics(
                            post["iid_position"], rates[variant], pooled.bin_centers_cm, pooled.bin_edges_cm, positions, observed[:, tr], track.topology, track.track_length_cm
                        )
                        rows.append(
                            {
                                "session": session,
                                "rat": animal,
                                "fold": fold,
                                "window_id": window.window_id,
                                "unit_regime": regime,
                                "count_regime": count_regime,
                                "encoding_variant": variant,
                                "split": split,
                                **prediction,
                                **known,
                                **metrics,
                                "n_spikes": int(observed.sum()),
                                "n_train_spikes": int(observed[:, tr].sum()),
                                "n_heldout_spikes": int(observed[:, he].sum()),
                                "n_active_units": int((observed.sum(axis=0) > 0).sum()),
                                "n_units": len(units),
                                "target_sleep_spikes": int(window.target_sleep_spikes),
                                "cap_attained": bool(exact) if count_regime != "native" else False,
                                "train_cell_ids": ",".join(map(str, units[tr])),
                                "heldout_cell_ids": ",".join(map(str, units[he])),
                                "counts_sha256": frozen.pf._hash_array(observed),
                                "iid_posterior_sha256": hashes["iid_position"],
                                "status": "success",
                                "posterior_unchanged": True,
                                "heldout_used_for_inference": False,
                            }
                        )
        qc_rows.append(
            {
                "session": session,
                "rat": animal,
                "fold": fold,
                "unit_regime": regime,
                "status": "success",
                "failure_reason": "",
                "n_units": len(units),
                "parent_units": len(parent_units),
                "parent_overlap_units": len(np.intersect1d(parent_units, units)),
                "n_bins": n,
                "train_occupancy_s": pooled.occupancy_s.sum(),
                "occupancy_nonzero_fraction": float((pooled.occupancy_s > 0).mean()),
                "train_start": train_interval[0],
                "train_stop": train_interval[1],
                "test_start": test_interval[0],
                "test_stop": test_interval[1],
                "runtime_s": time.monotonic() - started,
                **window_qc,
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(output / f"{tag}_scores.csv", index=False)
    pd.DataFrame(qc_rows).to_csv(output / f"{tag}_qc.csv", index=False)
    np.savez_compressed(output / f"{tag}_cache.npz", **cache)
    return {"session": session, "rat": animal, "fold": fold, "tag": tag, "selected_windows": len(windows), "rows": len(table), "qc": qc_rows}


def event_contrasts(scores):
    keys = ["session", "rat", "fold", "window_id", "unit_regime", "count_regime", "encoding_variant"]
    if scores.empty or scores.duplicated(keys + ["split"]).any():
        raise ValueError("empty/duplicate score rows")
    parts = []
    for contrast, (a, b) in CONTRASTS.items():
        part = scores[keys + ["split"]].copy()
        part["contrast"] = contrast
        part["delta"] = scores[f"score_{a}"] - scores[f"score_{b}"]
        part["delta_per_heldout_spike"] = part.delta / scores.n_heldout_spikes.replace(0, np.nan)
        parts.append(part)
    paired = pd.concat(parts, ignore_index=True)
    return paired, paired.groupby(keys + ["contrast"], as_index=False)[["delta", "delta_per_heldout_spike"]].median()


def summaries(events):
    conditions = ["unit_regime", "count_regime", "encoding_variant", "contrast"]
    half = events.groupby(conditions + ["rat", "session", "fold"], as_index=False)[["delta", "delta_per_heldout_spike"]].mean()
    session = half.groupby(conditions + ["rat", "session"], as_index=False)[["delta", "delta_per_heldout_spike"]].mean()
    animals = session.groupby(conditions + ["rat"], as_index=False)[["delta", "delta_per_heldout_spike"]].mean()
    records = []
    for key, group in animals.groupby(conditions):
        group = group.sort_values("rat")
        values = group.delta.to_numpy()
        rng = np.random.default_rng(SEED)
        means = values[rng.integers(0, len(values), size=(5000, len(values)))].mean(axis=1)
        ci = np.quantile(means, [0.025, 0.975])
        records.append(
            dict(zip(conditions, key, strict=True))
            | {
                "equal_animal_mean": values.mean(),
                "ci_low": ci[0],
                "ci_high": ci[1],
                "positive_animals": int((values > 0).sum()),
                "animals": len(values),
                "mean_per_heldout_spike": group.delta_per_heldout_spike.mean(),
                "interval_scope": "animal_clusters_only_fixed_maps_and_within_animal_data",
            }
        )
    return pd.DataFrame(records), animals, session, half


def technical_gates(scores, windows, qc):
    if scores.empty or windows.empty or qc.empty:
        return pd.DataFrame([{"gate": "nonempty_inputs", "passed": False}, {"gate": "overall", "passed": False}])
    keys = ["session", "fold", "window_id", "unit_regime", "count_regime", "encoding_variant", "split"]
    expected_per_window = len(UNIT_REGIMES) * len(COUNT_REGIMES) * len(frozen.VARIANTS) * frozen.N_SPLITS
    expected_conditions = {(a, b, c, d) for a in UNIT_REGIMES for b in COUNT_REGIMES for c in frozen.VARIANTS for d in range(5)}
    coverage = scores.groupby(["session", "fold", "window_id"])
    complete = all(set(g[keys[3:]].itertuples(index=False, name=None)) == expected_conditions for _, g in coverage)
    score_columns = [f"score_{name}" for name in (*frozen.MODELS, "global", "behavior", "behavior_wrong")]
    cell_disjoint = all(
        not (set(tr.split(",")) & set(he.split(","))) for tr, he in scores[["train_cell_ids", "heldout_cell_ids"]].drop_duplicates().itertuples(index=False, name=None)
    )
    successful_qc = qc[qc.status == "success"]
    disjoint_time = ((successful_qc.train_stop <= successful_qc.test_start) | (successful_qc.test_stop <= successful_qc.train_start)).all()
    checks = {
        "nonempty_and_exact_rows": len(scores) > 0 and len(scores) == len(windows) * expected_per_window and not scores.duplicated(keys).any(),
        "all_window_factors": bool(len(scores)) and complete,
        "all_sessions_halves_populations": len(qc) == 32
        and qc.status.eq("success").all()
        and windows.session.nunique() == 8
        and windows.rat.nunique() == 4
        and windows.groupby("session").fold.nunique().eq(2).all(),
        "all_selected_windows": bool(len(windows)) and set(scores[keys[:3]].itertuples(index=False, name=None)) == set(windows[keys[:3]].itertuples(index=False, name=None)),
        "proper_finite_scores": bool(len(scores)) and np.isfinite(scores[score_columns]).all().all() and scores[score_columns].le(1e-8).all().all(),
        "disjoint_times_and_cells": bool(len(successful_qc)) and disjoint_time and cell_disjoint,
        "count_partitions": scores.n_spikes.eq(scores.n_train_spikes + scores.n_heldout_spikes).all(),
        "shared_population_draws": scores.groupby(keys[:5]).counts_sha256.nunique().eq(1).all(),
        "no_failures_or_updates": scores.status.eq("success").all() and scores.posterior_unchanged.eq(True).all() and scores.heldout_used_for_inference.eq(False).all(),
    }
    checks["overall"] = all(checks.values())
    return pd.DataFrame([{"gate": k, "passed": bool(v)} for k, v in checks.items()])


def run(args):
    source, output = Path(args.source_dir).resolve(), Path(args.output_dir).resolve()
    parent_path = source / "hc11_conditional_manifest.json"
    if file_sha256(parent_path) != PARENT_DIGEST:
        raise ValueError("unexpected frozen sleep parent")
    parent = json.loads(parent_path.read_text())
    for name in ["frozen_selection.csv"] + [f"{s['session']}_cache.npz" for s in parent["sessions"]]:
        if file_sha256(source / name) != parent["output_sha256"][name]:
            raise ValueError("changed parent selection/cache")
    raw_hashes = {}
    for session in parent["sessions"]:
        folders = list(Path(args.dataset_root).glob(f"*/{session['session']}"))
        if len(folders) != 1:
            raise ValueError("ambiguous raw dataset session")
        raw_hashes.update({str(folders[0] / Path(path).name): digest for path, digest in session["source_hashes"].items()})
    if any(file_sha256(path) != digest for path, digest in raw_hashes.items()):
        raise ValueError("changed native spikes/position")
    if output.exists() and any(output.iterdir()):
        raise ValueError("refusing to overwrite an experiment")
    output.mkdir(parents=True, exist_ok=True)
    manifest = build_script_provenance(input_paths={"parent_manifest": parent_path})
    manifest.update(
        experiment="hc11_native_maze_transfer",
        source_dir=str(source),
        dataset_root=str(Path(args.dataset_root).resolve()),
        status="running",
        seed=SEED,
        encoding_parameters=frozen.ENCODING_PARAMETERS,
        guard_s=GUARD,
        window_s=WINDOW,
        bin_s=BIN,
        max_windows_per_fold=MAX_WINDOWS,
        raw_file_sha256=raw_hashes,
        primary="train_only_qc; native_counts; direction_mixture; behavior-conditioned encoding controls",
        n_neural_splits=5,
        heldout_fraction=0.3,
        claim_boundary="empirical encoding/prediction transfer control, not replay or a retuned sleep analysis",
    )
    path = output / "maze_transfer_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    completed = []
    started = time.monotonic()
    try:
        tasks = [(s["session"], s["animal"], fold, source, args.dataset_root, output) for s in parent["sessions"] for fold in range(2)]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(score_fold, task) for task in tasks]
            for future in as_completed(futures):
                result = future.result()
                completed.append(result)
                print(json.dumps({key: value for key, value in result.items() if key != "qc"}), flush=True)
        scores = pd.concat([pd.read_csv(output / f"{s['tag']}_scores.csv") for s in completed], ignore_index=True)
        windows = pd.concat([pd.read_csv(output / f"{s['tag']}_selection.csv") for s in completed], ignore_index=True)
        qc = pd.concat([pd.read_csv(output / f"{s['tag']}_qc.csv") for s in completed], ignore_index=True)
        gates = technical_gates(scores, windows, qc)
        gates.to_csv(output / "maze_transfer_gate_summary.csv", index=False)
        windows.to_csv(output / "maze_transfer_windows.csv", index=False)
        qc.to_csv(output / "maze_transfer_encoding_qc.csv", index=False)
        if not gates.passed.all():
            raise ValueError("incomplete technical coverage; no biological interpretation")
        paired, events = event_contrasts(scores)
        paired.to_csv(output / "maze_transfer_split_contrasts.csv", index=False)
        events.to_csv(output / "maze_transfer_window_contrasts.csv", index=False)
        summary, animal, session, half = summaries(events)
        for name, table in (("summary", summary), ("by_animal", animal), ("by_session", session), ("by_half", half)):
            table.to_csv(output / f"maze_transfer_{name}.csv", index=False)
        scores.groupby(["unit_regime", "count_regime", "encoding_variant", "session", "rat", "fold"], as_index=False)[
            [
                c
                for c in scores
                if "error_cm" in c
                or "coverage_95" in c
                or c in ("n_spikes", "n_heldout_spikes", "n_active_units", "fraction_supported_bins", "mean_posterior_entropy", "cap_attained")
            ]
        ].median().to_csv(output / "maze_transfer_decoder_qc.csv", index=False)
        manifest.update(status="complete", elapsed_s=time.monotonic() - started)
    except BaseException as exc:
        manifest.update(status="failed", error=repr(exc))
        raise
    finally:
        manifest["completed_folds"] = sorted(completed, key=lambda item: (item["session"], item["fold"]))
        manifest["output_sha256"] = {p.name: file_sha256(p) for p in output.iterdir() if p.is_file() and p != path}
        path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", default="/mnt/seagate10tb/florianpfaff/hc11-conditional-cross-cell-prediction-320x5-20260908")
    parser.add_argument("--dataset-root", default=str(frozen.DEFAULT_DATASET))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=8)
    run(parser.parse_args())
