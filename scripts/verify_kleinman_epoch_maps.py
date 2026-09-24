#!/usr/bin/env python3
"""Verify epoch-map accounting and independently refit sampled RUN windows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter1d
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _provenance import file_sha256
from audit_kleinman_epoch_maps import PAIR_KEYS, epoch_groups, group_split, run_mask, summarize_windows
from validate_kleinman_run_decoder import PARAMETERS, align_behavior, interval_data, make_traversals, split_units


def independent_map(trains, t, dt, direction, spatial, mask, n_bins):
    occupancy = np.zeros((2, n_bins))
    counts = np.zeros((2, n_bins, len(trains)))
    left, right = t[:-1][mask], t[1:][mask]
    d, s, weights = direction[mask], spatial[mask], dt[mask]
    for k in [0, 1]:
        occupancy[k] = np.bincount(s[d == k], weights=weights[d == k], minlength=n_bins)
        for j, spikes in enumerate(trains):
            c = np.searchsorted(spikes, right, side="left") - np.searchsorted(spikes, left, side="left")
            counts[k, :, j] = np.bincount(s[d == k], weights=c[d == k], minlength=n_bins)
    sigma = PARAMETERS["smooth_cm"] / PARAMETERS["bin_cm"]
    rates = gaussian_filter1d(counts, sigma, axis=1, mode="constant", truncate=4) / np.maximum(
        gaussian_filter1d(occupancy, sigma, axis=1, mode="constant", truncate=4)[:, :, None], 1e-12
    )
    units = (counts.sum(axis=(0, 1)) >= 10) & (rates.max(axis=(0, 1)) >= 1)
    return np.maximum(rates, 1e-5).reshape(2 * n_bins, len(trains)), (occupancy >= 0.1).reshape(-1), units


def scalar_metrics(counts, rates, support, centers, truth, direction, edges, arm):
    logs = []
    for rate, keep in zip(rates, support, strict=True):
        if not keep:
            logs.append(-np.inf)
        elif arm == "poisson":
            logs.append(sum(float(n) * np.log(lam) - 0.25 * lam for n, lam in zip(counts, rate, strict=True)))
        else:
            logs.append(sum(float(n) * np.log(lam / rate.sum()) for n, lam in zip(counts, rate, strict=True)))
    lp = np.array(logs) - logsumexp(logs)
    p = np.exp(lp)
    x = np.tile(centers, 2)
    mean = sum(p * x)
    right = sum(p[len(centers) :])
    state = int(np.clip(np.searchsorted(edges, truth, side="right") - 1, 0, len(centers) - 1)) + direction * len(centers)
    return {
        "mean_error_cm": abs(mean - truth),
        "map_error_cm": abs(x[p.argmax()] - truth),
        "mean_position_cm": mean,
        "direction_correct": 0.5 if np.isclose(right, 0.5, atol=1e-12, rtol=0) else float((right > 0.5) == direction),
        "entropy": -sum(p[support] * lp[support]) / np.log(support.sum()),
        "truth_supported": bool(support[state]),
        "true_log_score_above_uniform": lp[state] + np.log(support.sum()) if support[state] else np.nan,
    }


def verify_session(folder, splits, windows):
    candidates = splits.loc[splits.status.eq("scored")].sort_values(["source_epoch", "target_epoch", "split"])
    if candidates.empty:
        return 0
    row = candidates.iloc[0]
    info = loadmat(folder / "session_info.mat", simplify_cells=True)["session_info"]
    t, x, speed, _, visits, epochs = align_behavior(info)
    runs = epoch_groups(make_traversals(visits, epochs))
    _, trains, _ = split_units(loadmat(folder / "spike_data.mat", simplify_cells=True)["spike_data"])
    edges = np.arange(np.floor(x.min() / 2) * 2, np.ceil(x.max() / 2) * 2 + 2, 2)
    centers = (edges[:-1] + edges[1:]) / 2
    dt, _, trainable, _, direction, spatial = interval_data(t, x, speed, runs, edges)
    src, tgt, test = group_split(runs, row.animal, row.session, row.source_epoch, row.target_epoch, row.split)
    assert src.tolist() == json.loads(row.source_train_groups)
    assert tgt.tolist() == json.loads(row.target_train_groups)
    assert test.tolist() == json.loads(row.target_test_groups)
    train_a = run_mask(t, runs, row.target_epoch, tgt) & trainable
    train_b = run_mask(t, runs, row.source_epoch, src) & trainable
    test_mask = run_mask(t, runs, row.target_epoch, test)
    assert not ((train_a | train_b) & test_mask).any()
    a, sa, ua = independent_map(trains, t, dt, direction, spatial, train_a, len(centers))
    b, sb, ub = independent_map(trains, t, dt, direction, spatial, train_b, len(centers))
    cells, support = ua & ub, sa & sb
    assert cells.sum() == row.n_common_units and support.sum() == row.common_support_states
    selected = [s for s, keep in zip(trains, cells, strict=True) if keep]
    sample = windows.loc[(windows.source_epoch == row.source_epoch) & (windows.target_epoch == row.target_epoch) & (windows.split == row.split)]
    times = np.unique(sample.start_s)
    chosen = times[np.unique(np.linspace(0, len(times) - 1, min(4, len(times)), dtype=int))]
    checked = 0
    for obs in sample.loc[sample.start_s.isin(chosen)].itertuples():
        c = np.array([np.count_nonzero((s >= obs.start_s - 1e-12) & (s < obs.end_s - 1e-12)) for s in selected])
        assert c.sum() == obs.n_spikes and (c > 0).sum() == obs.n_active_units
        for label, rates in [("within", a[:, cells]), ("cross", b[:, cells])]:
            scores = scalar_metrics(c, rates, support, centers, obs.true_position_cm, obs.direction, edges, obs.arm)
            for metric, value in scores.items():
                np.testing.assert_allclose(value, getattr(obs, label + "_" + metric), atol=1e-8, rtol=1e-8, equal_nan=True)
        checked += 1
    return checked


def run(args):
    out = args.output_dir
    manifest_path = out / "manifest.json"
    m = json.loads(manifest_path.read_text())
    for name, digest in m["outputs"].items():
        assert file_sha256(out / name) == digest, name
    for name, path in m["input_file_paths"].items():
        assert file_sha256(Path(path)) == m["input_file_sha256"][name], name
    sessions = pd.read_csv(out / "kleinman_epoch_map_sessions.csv")
    all_splits = pd.read_csv(out / "kleinman_epoch_map_splits.csv")
    assert len(sessions) == 135 and sessions.run_pass.sum() == 127
    assert not sessions.status.eq("technical_failure").any()
    assert len(all_splits) == 127 * 30
    rebuilt, checked, rows = [], 0, 0
    for session in sessions.loc[sessions.run_pass].itertuples():
        folder = out / "sessions" / session.animal / session.session
        splits = pd.read_csv(folder / "splits.csv")
        assert len(splits) == 30 and not splits.duplicated(["source_epoch", "target_epoch", "split"]).any()
        for row in splits.loc[splits.n_windows > 0].itertuples():
            assert len(json.loads(row.source_train_groups)) == len(json.loads(row.target_train_groups)) == row.n_training_groups
            assert not set(json.loads(row.target_train_groups)) & set(json.loads(row.target_test_groups))
        if not (folder / "windows.csv.gz").exists():
            assert session.n_window_rows == 0 and not splits.status.eq("scored").any()
            continue
        w = pd.read_csv(folder / "windows.csv.gz")
        assert len(w) == session.n_window_rows
        assert w.groupby(["source_epoch", "target_epoch", "split", "start_s"]).arm.nunique().eq(2).all()
        for key, sub in w.groupby(["source_epoch", "target_epoch", "split"]):
            matched = splits.loc[(splits.source_epoch == key[0]) & (splits.target_epoch == key[1]) & (splits.split == key[2])].iloc[0]
            assert len(sub) == 2 * matched.n_windows
        np.testing.assert_allclose(w.error_increase_cm, w.cross_mean_error_cm - w.within_mean_error_cm, atol=1e-9)
        assert (w.within_truth_supported == w.cross_truth_supported).all()
        rebuilt.append(summarize_windows(w))
        raw = args.dataset_root / "Experiment_1" / session.animal / session.session
        checked += verify_session(raw, splits, w)
        rows += len(w)
        print(json.dumps({"animal": session.animal, "session": session.session, "window_rows": rows, "independent_fits": checked}), flush=True)
    r = pd.concat(rebuilt).sort_values([*PAIR_KEYS, "split"]).reset_index(drop=True)
    expected = pd.read_csv(out / "kleinman_epoch_map_split_metrics.csv").sort_values([*PAIR_KEYS, "split"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(r, expected, check_dtype=False, atol=1e-8, rtol=1e-8)
    pair = r.groupby(PAIR_KEYS).median(numeric_only=True).reset_index()
    pair["n_scored_splits"] = r.groupby(PAIR_KEYS).size().to_numpy()
    pd.testing.assert_frame_equal(pair, pd.read_csv(out / "kleinman_epoch_map_epoch_pairs.csv"), check_dtype=False, atol=1e-8, rtol=1e-8)

    stats = pair.groupby(["animal", "session", "arm"]).median(numeric_only=True).drop(columns=["source_epoch", "target_epoch"]).reset_index()
    stats["n_epoch_pairs"] = pair.groupby(["animal", "session", "arm"]).size().to_numpy()
    stats = stats.merge(sessions, on=["animal", "session"], validate="many_to_one", suffixes=("", "_session"))
    pd.testing.assert_frame_equal(stats, pd.read_csv(out / "kleinman_epoch_map_session_metrics.csv"), check_dtype=False, atol=1e-8, rtol=1e-8)
    for subset in ["all_run_pass", "extent_eligible"]:
        selected = stats if subset == "all_run_pass" else stats.loc[stats.extent_eligible]
        animals = selected.groupby(["animal", "arm"]).median(numeric_only=True).reset_index()
        animals["n_sessions"] = selected.groupby(["animal", "arm"]).size().to_numpy()
        pd.testing.assert_frame_equal(animals, pd.read_csv(out / ("kleinman_epoch_map_by_animal_" + subset + ".csv")), check_dtype=False, atol=1e-8, rtol=1e-8)
    assert pd.read_csv(out / "kleinman_epoch_map_gates.csv").passed.all()
    old = json.loads((args.initial_dir / "manifest.json").read_text())
    for name, digest in old["outputs"].items():
        assert file_sha256(args.initial_dir / name) == digest, name
    original = pd.read_csv(args.initial_dir / "kleinman_epoch_map_split_metrics.csv").sort_values([*PAIR_KEYS, "split"]).reset_index(drop=True)
    unchanged = r.loc[~((r.animal == "Con_3") & (r.session == "20220527_run1"))].reset_index(drop=True)
    pd.testing.assert_frame_equal(original, unchanged, check_dtype=False, atol=1e-8, rtol=1e-8)

    result = {
        "status": "passed",
        "unchanged_prior_sessions": 126,
        "producer_commit": m["code_commit"],
        "manifest_sha256": file_sha256(manifest_path),
        "window_rows_verified": rows,
        "independent_raw_window_arm_fits": checked,
        "scope": "All hashes/accounting/split and pair summaries; one independently refitted map pair and up to four windows per scored session, both likelihoods. Shared source-alignment/traversal helpers, not an independent behavioral annotation audit.",
        "verifier_sha256": file_sha256(Path(__file__)),
    }
    (out / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--initial-dir", required=True, type=Path)
    run(p.parse_args())
