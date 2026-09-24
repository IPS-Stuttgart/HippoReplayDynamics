#!/usr/bin/env python3
"""Paired within/cross reward-epoch RUN decoding; never replay scoring."""

from __future__ import annotations

import argparse
import hashlib
import json
import socket
import sys
import time
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _provenance import build_script_provenance, file_sha256
from validate_kleinman_run_decoder import PARAMETERS, align_behavior, fit_maps, interval_counts, interval_data, make_traversals, split_units, window_behavior

SEED = 20260924
SPLITS = 5
PAIR_KEYS = ["animal", "session", "source_epoch", "target_epoch", "arm"]


def epoch_groups(runs):
    result = []
    for epoch in sorted({r["epoch"] for r in runs}):
        rows = [dict(r) for r in runs if r["epoch"] == epoch]
        for i, row in enumerate(rows[: 2 * (len(rows) // 2)]):
            row["epoch_group"] = i // 2
            result.append(row)
    return result


def group_split(runs, animal, session, source, target, split):
    groups = {e: np.array(sorted({r["epoch_group"] for r in runs if r["epoch"] == e})) for e in [source, target]}
    n = min(len(groups[source]) // 2, len(groups[target]) // 2)
    if n < 2:
        raise ValueError("insufficient_epoch_lap_groups")
    digest = hashlib.sha256((animal + "/" + session).encode()).digest()
    identity = int.from_bytes(digest[:8], "little")
    selections = {}
    for epoch in [source, target]:
        rng = np.random.default_rng(np.random.SeedSequence([SEED, identity, split, epoch]))
        selections[epoch] = rng.permutation(groups[epoch])[:n]
    test = np.setdiff1d(groups[target], selections[target])
    return selections[source], selections[target], test


def run_mask(t, runs, epoch, groups):
    mask = np.zeros(len(t) - 1, bool)
    for row in runs:
        if row["epoch"] == epoch and row["epoch_group"] in groups:
            mask |= (t[:-1] >= row["start_s"]) & (t[1:] <= row["end_s"])
    return mask


def common_maps(a, b):
    ar, sa, ua, _ = a
    br, sb, ub, _ = b
    common = ua & ub
    support = sa & sb
    return ar[:, common[ua]], br[:, common[ub]], support, common


def decode_metrics(counts, rates, supported, centers, truth, direction, edges, arm):
    if arm == "poisson":
        values = counts @ np.log(rates).T - PARAMETERS["test_window_s"] * rates.sum(axis=1)
    elif arm == "composition":
        values = counts @ (np.log(rates) - np.log(rates.sum(axis=1, keepdims=True))).T
    else:
        raise ValueError("unknown likelihood")
    values[:, ~supported] = -np.inf
    logp = values - logsumexp(values, axis=1, keepdims=True)
    p = np.exp(logp)
    n_bins = len(centers)
    mean = p @ np.tile(centers, 2)
    right = p[:, n_bins:].sum(axis=1)
    state = np.clip(np.searchsorted(edges, truth, side="right") - 1, 0, n_bins - 1) + direction * n_bins
    true_support = supported[state]
    score = logp[np.arange(len(p)), state] + np.log(supported.sum())
    score[~true_support] = np.nan
    return {
        "mean_error_cm": np.abs(mean - truth),
        "map_error_cm": np.abs(np.tile(centers, 2)[p.argmax(axis=1)] - truth),
        "mean_position_cm": mean,
        "direction_correct": np.where(np.isclose(right, 0.5, atol=1e-12, rtol=0), 0.5, (right > 0.5) == direction),
        "entropy": -np.sum(p[:, supported] * logp[:, supported], axis=1) / np.log(supported.sum()),
        "truth_supported": true_support,
        "true_log_score_above_uniform": score,
    }


def compare_session(folder, animal, session):
    info = loadmat(folder / "session_info.mat", simplify_cells=True)["session_info"]
    t, x, speed, ends, visits, epochs = align_behavior(info)
    if len(epochs) != 3:
        raise ValueError("expected_three_reward_epochs")
    runs = epoch_groups(make_traversals(visits, epochs))
    _, trains, _ = split_units(loadmat(folder / "spike_data.mat", simplify_cells=True)["spike_data"])
    step = PARAMETERS["bin_cm"]
    edges = np.arange(np.floor(x.min() / step) * step, np.ceil(x.max() / step) * step + step, step)
    centers = (edges[:-1] + edges[1:]) / 2
    counts = interval_counts(trains, t)
    dt, valid, trainable, _, direction, spatial = interval_data(t, x, speed, runs, edges)
    splits, windows = [], []
    for source, target in permutations([1, 2, 3], 2):
        for split in range(SPLITS):
            base = {"animal": animal, "session": session, "source_epoch": source, "target_epoch": target, "split": split}
            try:
                src, tgt, test = group_split(runs, animal, session, source, target, split)
            except ValueError as exc:
                splits.append(dict(**base, status=str(exc), n_windows=0))
                continue
            assert not set(tgt) & set(test)
            assert len(src) == len(tgt)
            mask_a = trainable & run_mask(t, runs, target, tgt)
            mask_b = trainable & run_mask(t, runs, source, src)
            heldout = run_mask(t, runs, target, test)
            assert not np.any((mask_a | mask_b) & heldout)
            a = fit_maps(counts, dt, direction, spatial, mask_a, len(centers))
            b = fit_maps(counts, dt, direction, spatial, mask_b, len(centers))
            ra, rb, support, units = common_maps(a, b)
            base.update(
                n_training_groups=len(tgt),
                n_test_groups=len(test),
                n_common_units=int(units.sum()),
                within_units=int(a[2].sum()),
                cross_units=int(b[2].sum()),
                within_support_states=int(a[1].sum()),
                cross_support_states=int(b[1].sum()),
                common_support_states=int(support.sum()),
                within_training_s=float(a[3].sum()),
                cross_training_s=float(b[3].sum()),
                source_train_groups=json.dumps(src.tolist()),
                target_train_groups=json.dumps(tgt.tolist()),
                target_test_groups=json.dumps(test.tolist()),
            )
            if units.sum() < PARAMETERS["minimum_units"] or support.sum() < 2:
                splits.append(dict(**base, status="insufficient_common_cells_or_support", n_windows=0))
                continue
            qa = ra[support] / ra[support].sum(axis=1, keepdims=True)
            qb = rb[support] / rb[support].sum(axis=1, keepdims=True)
            base["mean_map_hellinger_squared"] = float(np.mean(0.5 * np.sum((np.sqrt(qa) - np.sqrt(qb)) ** 2, axis=1)))
            selected = [s for s, keep in zip(trains, units, strict=True) if keep]
            records, obs = [], []
            rejected = {"tracking_rejected": 0, "behavior_rejected": 0, "spike_rejected": 0}
            width = PARAMETERS["test_window_s"]
            for row in runs:
                if row["epoch"] != target or row["epoch_group"] not in test:
                    continue
                for start in np.arange(row["start_s"], row["end_s"] - width + 1e-9, width):
                    end = start + width
                    behavior = window_behavior(t, x, speed, start, end, valid)
                    if behavior is None:
                        rejected["tracking_rejected"] += 1
                        continue
                    truth, velocity = behavior
                    if velocity <= PARAMETERS["test_speed_cm_s"] or truth <= ends[0] + PARAMETERS["interior_margin_cm"] or truth >= ends[1] - PARAMETERS["interior_margin_cm"]:
                        rejected["behavior_rejected"] += 1
                        continue
                    c = np.array([np.searchsorted(s, end, side="left") - np.searchsorted(s, start, side="left") for s in selected])
                    if c.sum() < PARAMETERS["minimum_test_spikes"]:
                        rejected["spike_rejected"] += 1
                        continue
                    obs.append(c)
                    records.append(
                        dict(
                            **{k: row[k] for k in ["traversal", "epoch_group", "direction"]},
                            start_s=start,
                            end_s=end,
                            true_position_cm=truth,
                            speed_cm_s=velocity,
                            n_spikes=int(c.sum()),
                            n_active_units=int((c > 0).sum()),
                        )
                    )
            base.update(**rejected, n_windows=len(records))
            splits.append(dict(**base, status="scored" if records else "no_eligible_windows"))
            if not records:
                continue
            table = pd.DataFrame(records)
            for arm in ["poisson", "composition"]:
                results = {}
                for label, rates in [("within", ra), ("cross", rb)]:
                    metrics = decode_metrics(np.asarray(obs), rates, support, centers, table.true_position_cm.to_numpy(), table.direction.to_numpy(), edges, arm)
                    results.update({label + "_" + name: values for name, values in metrics.items()})
                combined = table.assign(**base, arm=arm, **results)
                combined["error_increase_cm"] = combined.cross_mean_error_cm - combined.within_mean_error_cm
                windows.append(combined)
    return pd.DataFrame(splits), pd.concat(windows, ignore_index=True) if windows else pd.DataFrame()


def summarize_windows(frame):
    keys = [*PAIR_KEYS, "split"]
    metrics = [
        "within_mean_error_cm",
        "cross_mean_error_cm",
        "within_map_error_cm",
        "cross_map_error_cm",
        "error_increase_cm",
        "within_direction_correct",
        "cross_direction_correct",
        "within_entropy",
        "cross_entropy",
        "within_truth_supported",
        "within_true_log_score_above_uniform",
        "cross_true_log_score_above_uniform",
    ]
    grouped = frame.groupby(keys)
    result = grouped[metrics].mean()
    result["n_windows"] = grouped.size()
    result["n_common_units"] = grouped.n_common_units.first()
    result["mean_map_hellinger_squared"] = grouped.mean_map_hellinger_squared.first()
    result["within_qc"] = (result.within_mean_error_cm <= 35) & (result.within_direction_correct >= 0.6) & (result.n_windows >= 20) & (result.n_common_units >= 5)
    result["cross_qc"] = (result.cross_mean_error_cm <= 35) & (result.cross_direction_correct >= 0.6) & (result.n_windows >= 20) & (result.n_common_units >= 5)
    return result.reset_index()


def run(args):
    start = time.monotonic()
    source = pd.read_csv(args.cohort_dir / "kleinman_extent_cohort_sessions.csv")
    manifest_path = args.cohort_dir / "manifest.json"
    prior = json.loads(manifest_path.read_text())
    name = "kleinman_extent_cohort_sessions.csv"
    if file_sha256(args.cohort_dir / name) != prior["outputs"][name]:
        raise ValueError("changed prior eligibility")
    if len(source) != 135 or source.run_pass.sum() != 127 or source.extent_eligible.sum() != 14:
        raise ValueError("frozen cohort mismatch")
    inputs = {
        "cohort_manifest": manifest_path,
        "cohort_sessions": args.cohort_dir / name,
        "producer": Path(__file__),
        "encoder": ROOT / "scripts/validate_kleinman_run_decoder.py",
        "protocol": ROOT / "docs/kleinman_epoch_map_protocol.md",
    }
    folders = {}
    for row in source.loc[source.run_pass].itertuples():
        folder = args.dataset_root / "Experiment_1" / row.animal / row.session
        folders[(row.animal, row.session)] = folder
        for name in ["session_info.mat", "spike_data.mat"]:
            inputs[row.animal + "/" + row.session + "/" + name] = folder / name
    provenance = build_script_provenance(cwd=ROOT, input_paths=inputs)
    if provenance["git_dirty"] or provenance["code_commit"] == "unavailable":
        raise ValueError("clean committed producer required")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    sessions, all_splits, all_summaries = [], [], []
    for row in source.itertuples():
        key = {k: getattr(row, k) for k in ["animal", "session", "drug", "novel", "run_pass", "extent_eligible"]}
        if not row.run_pass:
            sessions.append(dict(**key, status="run_qc_ineligible"))
            continue
        try:
            splits, windows = compare_session(folders[(row.animal, row.session)], row.animal, row.session)
            out = args.output_dir / "sessions" / row.animal / row.session
            out.mkdir(parents=True)
            splits.to_csv(out / "splits.csv", index=False)
            if len(windows):
                windows.to_csv(out / "windows.csv.gz", index=False)
                summary = summarize_windows(windows)
                all_summaries.append(summary)
            all_splits.append(splits)
            sessions.append(dict(**key, status="completed", n_split_records=len(splits), n_scored_splits=int(splits.status.eq("scored").sum()), n_window_rows=len(windows)))
        except (ValueError, AssertionError, KeyError, IndexError) as exc:
            sessions.append(dict(**key, status="technical_failure", failure_reason=str(exc)))
        pd.DataFrame(sessions).to_csv(args.output_dir / "progress.csv", index=False)
        print(json.dumps({**sessions[-1], "elapsed_s": time.monotonic() - start}), flush=True)
    frame = pd.DataFrame(sessions)
    split_results = pd.concat(all_splits, ignore_index=True)
    summary = pd.concat(all_summaries, ignore_index=True)
    pair = summary.groupby(PAIR_KEYS).median(numeric_only=True).reset_index()
    pair["n_scored_splits"] = summary.groupby(PAIR_KEYS).size().to_numpy()
    stats = pair.groupby(["animal", "session", "arm"]).median(numeric_only=True).drop(columns=["source_epoch", "target_epoch"]).reset_index()
    stats["n_epoch_pairs"] = pair.groupby(["animal", "session", "arm"]).size().to_numpy()
    stats = stats.merge(frame, on=["animal", "session"], validate="many_to_one", suffixes=("", "_session"))
    gates = pd.DataFrame(
        [
            {"gate": "source_sessions_accounted", "passed": len(frame) == 135 and not frame.duplicated(["animal", "session"]).any()},
            {"gate": "all_run_pass_attempted", "passed": frame.run_pass.sum() == 127 and frame.loc[frame.run_pass, "status"].ne("run_qc_ineligible").all()},
            {"gate": "all_split_records_present", "passed": len(split_results) == 127 * 30},
            {"gate": "no_technical_failures", "passed": not frame.status.eq("technical_failure").any()},
        ]
    )
    outputs = {"sessions": frame, "splits": split_results, "split_metrics": summary, "epoch_pairs": pair, "session_metrics": stats, "gates": gates}
    for subset in ["all_run_pass", "extent_eligible"]:
        subset_frame = stats if subset == "all_run_pass" else stats.loc[stats.extent_eligible]
        by_animal = subset_frame.groupby(["animal", "arm"]).median(numeric_only=True).reset_index()
        by_animal["n_sessions"] = subset_frame.groupby(["animal", "arm"]).size().to_numpy()
        outputs["by_animal_" + subset] = by_animal
    for label, data in outputs.items():
        data.to_csv(args.output_dir / ("kleinman_epoch_map_" + label + ".csv"), index=False)
    for key, path in inputs.items():
        if file_sha256(path) != provenance["input_file_sha256"][key]:
            raise ValueError("changed input " + key)
    manifest = {
        **provenance,
        "host": socket.gethostname(),
        "runtime_s": time.monotonic() - start,
        "run_parameters": PARAMETERS,
        "seed": SEED,
        "splits": SPLITS,
        "replay_scored": False,
        "reward_replay_contrast_scored": False,
        "outputs": {str(p.relative_to(args.output_dir)): file_sha256(p) for p in args.output_dir.rglob("*") if p.is_file()},
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--cohort-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    run(parser.parse_args())
