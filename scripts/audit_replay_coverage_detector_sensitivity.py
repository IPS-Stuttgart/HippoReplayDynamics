#!/usr/bin/env python3
"""Recount every spike window and independently reconstruct paired path metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from hipporeplayimm.replay_coverage_detector_sensitivity import (
    IDENTITY,
    session_summary,
    summarize_animals,
)
from scipy.special import gammaln, logsumexp, xlogy

from scripts._provenance import build_script_provenance, file_sha256
from scripts.analyze_replay_coverage_detector_sensitivity import METRIC_KEY


def check_hash(path, digest):
    if file_sha256(path) != digest:
        raise ValueError(f"hash mismatch: {path}")


def independent_metrics(path, counts, rms, entropy, full_path, full_counts, filtered):
    """Separate run segmentation and explicit gap checks, without scorer helpers."""
    def support(c):
        return ((c > 0).sum(axis=1) >= 2) & (c.sum(axis=1) >= 3) if filtered else np.ones(len(c), bool)

    def longest(p, valid):
        jumps = np.linalg.norm(np.diff(p, axis=0), axis=1)
        connected = valid[:-1] & valid[1:] & (jumps < 20)
        starts = np.flatnonzero(valid & ~np.r_[False, connected]) if len(p) else []
        ends = np.flatnonzero(valid & ~np.r_[connected, False]) + 1 if len(p) else []
        if len(starts):
            best = int(np.argmax(ends - starts))
            a, b = int(starts[best]), int(ends[best])
            distance = float(np.linalg.norm(p[b - 1] - p[a]))
        else:
            a, b, distance = 0, 0, 0.
        return a, b, distance, bool(b - a >= 10 and distance >= 40)

    valid, full_valid = support(counts), support(full_counts)
    a, b, distance, passed = longest(path, valid)
    full_passed = longest(full_path, full_valid)[-1]
    starts = np.arange(0, max(0, len(path) - 4), 4)
    speed = np.linalg.norm(path[starts + 4] - path[starts], axis=1) / .02
    full_speed = np.linalg.norm(full_path[starts + 4] - full_path[starts], axis=1) / .02
    measured = np.array([valid[i:i + 5].all() for i in starts], bool)
    common = measured & np.array([full_valid[i:i + 5].all() for i in starts], bool)
    selected = measured & passed & (starts >= a) & (starts + 4 < b)
    adjacent = valid[:-1] & valid[1:]
    jumps = np.linalg.norm(np.diff(path, axis=0), axis=1)

    def median(values):
        return float(np.median(values)) if len(values) else np.nan

    return {"continuous_start": a, "continuous_end_exclusive": b, "continuous_frames": b - a,
        "continuous_displacement_cm": distance, "continuity_pass": passed, "full_continuity_pass": full_passed,
        "paired_continuity_delta": int(passed) - int(full_passed), "overlapping_frames": len(path),
        "supported_frames": int(valid.sum()), "large_jump_fraction": float((jumps[adjacent] >= 20).mean()) if adjacent.any() else np.nan,
        "nonoverlap_measurable_steps": int(measured.sum()), "median_event_speed_cm_s": median(speed[measured]),
        "selected_measurable_steps": int(selected.sum()), "median_selected_speed_cm_s": median(speed[selected]),
        "common_measurable_steps": int(common.sum()), "median_common_step_speed_delta_cm_s": median(speed[common] - full_speed[common]),
        "median_posterior_rms_cm": median(np.asarray(rms)[valid]), "median_posterior_entropy_nats": median(np.asarray(entropy)[valid])}


def analytic_checks(arrays, subset, likelihood, decoded):
    n = len(arrays["frame_counts"])
    if not n:
        return 0
    total = arrays["frame_counts"][:, subset].sum(axis=1)
    frames = np.unique(np.r_[np.linspace(0, n - 1, min(12, n)).astype(int), total.argmin(), total.argmax()])
    rates = arrays["rates_hz"][subset]
    grid = arrays["grid_cm"]
    for frame in frames:
        counts = arrays["frame_counts"][frame, subset]
        if likelihood == "poisson":
            mu = .02 * rates
            ll = np.sum(counts[:, None] * np.log(mu) - mu - gammaln(counts[:, None] + 1), axis=0)
        else:
            probs = rates / rates.sum(axis=0)
            ll = np.sum(counts[:, None] * np.log(probs), axis=0) + gammaln(counts.sum() + 1) - gammaln(counts + 1).sum()
        p = np.exp(ll - logsumexp(ll))
        mean = np.sum(p[:, None] * grid, axis=0)
        rms = np.sqrt(np.sum(p * np.sum((grid - mean)**2, axis=1)))
        expected = {"posterior_mean": mean, "posterior_rms_cm": rms, "posterior_entropy_nats": -xlogy(p, p).sum()}
        for name, value in expected.items():
            np.testing.assert_allclose(decoded[name][frame], value, rtol=1e-8, atol=1e-6)
        # Permit only genuine floating-point MAP ties, not a different location.
        locations = np.flatnonzero(np.all(grid == decoded["map"][frame], axis=1))
        assert len(locations) == 1 and ll.max() - ll[locations[0]] < 1e-8
    return len(frames)


def assert_table(actual, expected, keys):
    a = actual.sort_values(keys).reset_index(drop=True)
    b = expected.sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(a[b.columns], b, check_dtype=False, rtol=1e-8, atol=1e-7)


def audit_session(record, seed, replicates):
    for prefix in ["source_windows", "source_cache", "input_arrays", "metrics", "population"]:
        check_hash(record[f"{prefix}_path"], record[f"{prefix}_sha256"])
    source = pd.read_csv(record["source_windows_path"], float_precision="round_trip")
    windows = source[source.eligible].sort_values("window_uid").reset_index(drop=True)
    with np.load(record["input_arrays_path"], allow_pickle=False) as f:
        arrays = dict(f)
    with np.load(record["source_cache_path"], allow_pickle=False) as f:
        mask = f["unit_qc_mask"].astype(bool)
        ids, grid, bounds = f["cell_ids"][mask], f["bin_centers_cm"], f["arena_bounds_cm"]
        support = f["valid_spatial_bins"].astype(bool)
        if np.isfinite(bounds).all():
            support &= np.all((grid >= bounds[0]) & (grid <= bounds[1]), axis=1)
        np.testing.assert_array_equal(arrays["cell_ids"], ids)
        np.testing.assert_array_equal(arrays["grid_cm"], grid[support])
        np.testing.assert_array_equal(arrays["rates_hz"], f["rates_hz"][mask][:, support])
        spikes = f["spikes"]
    np.testing.assert_array_equal(arrays["window_uids"], windows.window_uid.to_numpy(str))
    assert len(arrays["frame_offsets"]) == len(windows) + 1 and arrays["frame_offsets"][0] == 0
    assert arrays["frame_offsets"][-1] == len(arrays["frame_counts"])
    for i, w in enumerate(windows.itertuples(index=False)):
        a, b = arrays["frame_offsets"][i:i + 2]
        expected_n = max(0, int(np.floor((w.end_s - w.start_s) / .005 + 1e-8)) - 3)
        assert b - a == expected_n
        np.testing.assert_allclose(arrays["frame_start_s"][a:b], w.start_s + .005 * np.arange(expected_n), rtol=0, atol=1e-9)
        np.testing.assert_allclose(arrays["frame_end_s"][a:b] - arrays["frame_start_s"][a:b], .020, rtol=0, atol=1e-9)
        assert not expected_n or arrays["frame_end_s"][b - 1] <= w.end_s + 1e-9
    for column, cell in enumerate(ids):
        times = np.sort(spikes[spikes[:, 1] == cell, 0])
        counted = np.searchsorted(times, arrays["frame_end_s"], side="left") - np.searchsorted(times, arrays["frame_start_s"], side="left")
        np.testing.assert_array_equal(counted, arrays["frame_counts"][:, column])
    token = int.from_bytes(hashlib.sha256(":".join(record[k] for k in IDENTITY).encode()).digest()[:8], "little")
    expected_specs = [(1., 0, np.arange(len(ids)))]
    for rep in range(replicates):
        permutation = np.random.default_rng(np.random.SeedSequence([seed, token, rep])).permutation(len(ids))
        expected_specs.append((.5, rep, np.sort(permutation[:max(1, len(ids) // 2)])))
    assert len(record["population_specs"]) == len(expected_specs)
    for spec, (fraction, rep, indices) in zip(record["population_specs"], expected_specs, strict=True):
        assert spec["cell_fraction"] == fraction and spec["population_replicate"] == rep
        np.testing.assert_array_equal(spec["indices"], indices)
        np.testing.assert_array_equal(spec["cell_ids"], ids[indices])
    metrics = pd.read_csv(record["metrics_path"], float_precision="round_trip")
    assert not metrics.duplicated(METRIC_KEY).any()
    assert len(metrics) == len(windows) * (replicates + 1) * 8 == record["metric_rows"]
    assert len(record["decoded_paths"]) == (replicates + 1) * 2
    full, checked, seen = {}, 0, set()
    for item in record["decoded_paths"]:
        condition = (item["cell_fraction"], item["population_replicate"], item["likelihood"])
        assert condition not in seen
        seen.add(condition)
        matched = [s for s in record["population_specs"] if (s["cell_fraction"], s["population_replicate"]) == condition[:2]]
        assert len(matched) == 1 and item["indices"] == matched[0]["indices"]
        check_hash(item["path"], item["sha256"])
        with np.load(item["path"], allow_pickle=False) as f:
            decoded = dict(f)
        for value in decoded.values():
            assert len(value) == len(arrays["frame_counts"]) and np.isfinite(value).all()
        indices = np.asarray(item["indices"], int)
        checked += analytic_checks(arrays, indices, item["likelihood"], decoded)
        if item["cell_fraction"] == 1.:
            full[item["likelihood"]] = decoded
        ref = full[item["likelihood"]]
        selected = metrics[(metrics.cell_fraction == condition[0]) & (metrics.population_replicate == condition[1]) & metrics.likelihood.eq(condition[2])]
        lookup = selected.set_index(["window_uid", "estimator", "bin_filter"])
        for i, w in enumerate(windows.itertuples(index=False)):
            a, b = arrays["frame_offsets"][i:i + 2]
            counts = arrays["frame_counts"][a:b]
            for estimator in ["map", "posterior_mean"]:
                for filtered, filter_name in [(False, "unfiltered"), (True, "at_least_2cells_3spikes")]:
                    expected = independent_metrics(decoded[estimator][a:b], counts[:, indices], decoded["posterior_rms_cm"][a:b], decoded["posterior_entropy_nats"][a:b], ref[estimator][a:b], counts, filtered)
                    row = lookup.loc[(w.window_uid, estimator, filter_name)]
                    np.testing.assert_allclose(row[list(expected)].to_numpy(float), list(expected.values()), rtol=1e-8, atol=1e-7, equal_nan=True)
    population = pd.read_csv(record["population_path"])
    assert_table(population, session_summary(metrics, source, record["population_specs"]), list(population.columns[:7]) + ["cell_fraction", "population_replicate", "likelihood", "estimator", "bin_filter"])
    result = {**{k: record[k] for k in IDENTITY}, "status": "pass", "windows": len(windows),
        "directly_counted_frames": len(arrays["frame_counts"]), "independent_metric_rows": len(metrics), "analytic_posterior_frames": checked}
    print(json.dumps(result), flush=True)
    return result


def run(root, workers):
    path = root / "coverage_detector_sensitivity_manifest.json"
    meta = json.loads(path.read_text())
    if meta["status"] != "complete" or meta["eligible_windows"] < 1:
        raise ValueError("completed nonempty decoding required")
    for name, source in meta["input_file_paths"].items():
        check_hash(source, meta["input_file_sha256"][name])
    for name, digest in meta["output_sha256"].items():
        check_hash(root / name, digest)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(audit_session, r, meta["seed"], meta["replicates"]) for r in meta["results"]]
        results = [f.result() for f in as_completed(futures)]
    population = pd.concat([pd.read_csv(r["population_path"]) for r in meta["results"]], ignore_index=True)
    session, animal, summary = summarize_animals(population, meta["bootstraps"], meta["seed"] + 1)
    for name, frame in [("population_summary", population), ("session_summary", session), ("animal_summary", animal), ("summary", summary)]:
        actual = pd.read_csv(root / f"coverage_detector_sensitivity_{name}.csv")
        pd.testing.assert_frame_equal(actual, frame, check_dtype=False, rtol=1e-8, atol=1e-7)
    result = build_script_provenance(input_paths={"decoding_manifest": path, "auditor": Path(__file__)}, cwd=ROOT)
    result.update(status="pass", sessions=sorted(results, key=lambda r: tuple(r[k] for k in IDENTITY)),
        verification_scope="all source and output hashes, all direct raw-spike counts, all independently computed path metrics, sampled analytic posteriors, exact summary reconstruction")
    (root / "coverage_detector_sensitivity_reconstruction_audit.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    run(args.input_dir.resolve(), args.workers)
