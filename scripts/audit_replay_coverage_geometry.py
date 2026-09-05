#!/usr/bin/env python3
"""Reconstruct geometry inputs and directly recount support for every metric row."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from hipporeplayimm.replay_coverage_data import array_sha256
from hipporeplayimm.replay_coverage_geometry import CORE, KEY, centered_grid, decoder_settings
from scripts._provenance import build_script_provenance, file_sha256
from scripts.simulate_replay_coverage_geometry import (
    FIELDS,
    build_batch,
    observation_manifest,
    summarize_batch,
)


def direct_support(fine, window, stride, filtered):
    counts = np.stack([fine[start:start + window].sum(axis=0) for start in range(0, len(fine) - window + 1, stride)])
    good = (counts.sum(axis=1) >= 3) & ((counts > 0).sum(axis=1) >= 2) if filtered else np.ones(len(counts), bool)
    idx = np.arange(0, len(counts), window // stride)
    steps = sum(good[a:b + 1].all() for a, b in itertools.pairwise(idx))
    return {"valid_bins": int(good.sum()), "valid_adjacent_steps": int((good[:-1] & good[1:]).sum()),
            "all_steps": int(steps), "n_decoded_bins": len(counts),
            "spikes": int(fine.sum()), "window_counts_sha256": array_sha256(counts)}, counts


def compare_tables(actual, expected, key):
    if actual.duplicated(key).any() or expected.duplicated(key).any():
        raise AssertionError("duplicate rows")
    pd.testing.assert_frame_equal(actual.sort_values(key).reset_index(drop=True)[expected.columns],
                                  expected.sort_values(key).reset_index(drop=True), check_dtype=False, rtol=1e-9, atol=1e-9)


def direct_truth_eligibility(path, window, stride, rule):
    xy = path["midpoints_cm"]
    means = np.stack([xy[k:k + window].mean(axis=0) for k in range(0, len(xy) - window + 1, stride)])
    threshold = 20 if rule == "literal_20cm_10frames" else 4 * stride
    frames = 10 if rule == "literal_20cm_10frames" else int(np.ceil(45 / stride)) + 1
    splits = np.r_[0, np.flatnonzero(np.linalg.norm(np.diff(means, axis=0), axis=1) >= threshold) + 1, len(means)]
    lengths = np.diff(splits)
    best = int(np.argmax(lengths))
    start, end = splits[best:best + 2]
    return bool(lengths[best] >= frames and np.linalg.norm(means[end - 1] - means[start]) >= 40)


def sampled_likelihood_check(frame, trial, obs, centers, bounds, config):
    """Independent analytic Gaussian likelihood and position error, first path."""
    sampled = frame[frame.path_id.eq(0) & frame.truth_kind.eq("continuous") & frame.gradient.eq(-.5)]
    n_checked = 0
    grouping = ["n_cells", "support_domain", "grid_cm", "window_ms", "stride_ms", "observation", "likelihood"]
    for key, group in sampled.groupby(grouping, sort=True):
        cells, domain, spacing, window, stride, family, likelihood = key
        grid = centered_grid(CORE if domain == "common_core" else bounds, spacing)
        sigma = float(group.sigma_cm.iloc[0])
        rates = config.floor_hz + (config.peak_hz - config.floor_hz) * np.exp(-np.sum((centers[:cells, None] - grid[None])**2, axis=2) / (2*sigma**2))
        fine = obs[family, cells]
        _, counts = direct_support(fine, window, stride, False)
        if likelihood == "poisson":
            log_likelihood = counts @ np.log(rates) - window / 1000 * rates.sum(axis=0)
        else:
            log_likelihood = counts @ (np.log(rates) - np.log(rates.sum(axis=0)))
        p = np.exp(log_likelihood - logsumexp(log_likelihood, axis=1)[:, None])
        paths = {"map": grid[p.argmax(axis=1)], "posterior_mean": p @ grid}
        starts = np.arange(0, len(fine) - window + 1, stride)
        true_center = trial["path"]["edges_cm"][starts + window // 2]
        for row in group.itertuples(index=False):
            valid = (counts.sum(axis=1) >= 3) & ((counts > 0).sum(axis=1) >= 2) if row.bin_filter != "unfiltered" else np.ones(len(counts), bool)
            errors = np.linalg.norm(paths[row.estimator] - true_center, axis=1)
            median = float(np.median(errors[valid])) if valid.any() else np.nan
            np.testing.assert_allclose(row.median_position_error_cm, median, rtol=1e-8, atol=1e-8, equal_nan=True)
            n_checked += 1
    return n_checked


def audit(root):
    manifest = json.loads((root / "geometry_manifest.json").read_text())
    if manifest["status"] != "complete":
        raise AssertionError("run is not terminal complete")
    for name, expected in manifest["output_sha256"].items():
        if file_sha256(root / name) != expected:
            raise AssertionError(f"output hash mismatch: {name}")
    for name, path in manifest["input_file_paths"].items():
        if file_sha256(path) != manifest["input_file_sha256"][name]:
            raise AssertionError(f"input code changed: {name}")
        if file_sha256(root / "inputs" / f"{name}{Path(path).suffix}") != manifest["input_file_sha256"][name]:
            raise AssertionError(f"input snapshot changed: {name}")
    args = SimpleNamespace(**manifest["parameters"])
    batches = pd.read_csv(root / "geometry_batches.csv")
    expected_batches = {(p, f) for p in range(args.populations) for f in args.field_ids}
    if set(zip(batches.population_seed, batches.field_id, strict=True)) != expected_batches or len(batches) != len(expected_batches):
        raise AssertionError("missing or duplicate batch")
    observed = pd.read_csv(root / "geometry_observations.csv")
    summaries = pd.read_csv(root / "geometry_population_summary.csv")
    gradients = pd.read_csv(root / "geometry_population_gradients.csv")
    grid_meta = pd.read_csv(root / "geometry_decoder_grids.csv")
    audit_rows = []
    obs_key = ["population_seed", "field_id", "path_id", "truth_kind", "gradient", "observation", "n_cells"]
    for batch in batches.itertuples(index=False):
        paths, obs, evaluate, centers, bounds = build_batch(args, batch.population_seed, batch.field_id)
        if array_sha256(centers) != batch.centers_sha256:
            raise AssertionError("population centers changed")
        expected = observation_manifest(paths, obs, batch.population_seed, batch.field_id)
        selected = observed[observed.population_seed.eq(batch.population_seed) & observed.field_id.eq(batch.field_id)]
        compare_tables(selected, expected, obs_key)
        for trial, values in zip(paths, obs, strict=True):
            np.testing.assert_array_equal(values["native_poisson", 64], values["native_poisson", 128][:, :64])
            np.testing.assert_array_equal(values["common_count_schedule", 64].sum(axis=1), values["common_count_schedule", 128].sum(axis=1))
        frame = pd.read_csv(root / batch.file)
        if frame.duplicated(KEY).any() or len(frame) != batch.rows:
            raise AssertionError("metric keys incomplete or duplicated")
        lookup = {(p["path_id"], p["truth_kind"], p["gradient"]): o for p, o in zip(paths, obs, strict=True)}
        path_lookup = {(p["path_id"], p["truth_kind"], p["gradient"]): p["path"] for p in paths}
        for key, rows in frame.groupby(["path_id", "truth_kind", "gradient", "window_ms", "stride_ms", "continuity_rule"], sort=False):
            path_id, kind, grad, window, stride, rule = key
            expected_eligible = direct_truth_eligibility(path_lookup[path_id, kind, grad], window, stride, rule)
            if not rows.truth_geometric_eligible.eq(expected_eligible).all():
                raise AssertionError("truth eligibility differs from direct geometric calculation")
        comparisons = 0
        for key, rows in frame.groupby(["path_id", "truth_kind", "gradient", "n_cells", "observation", "window_ms", "stride_ms", "bin_filter"], sort=False):
            path_id, kind, gradient, cells, family, window, stride, support = key
            fine = lookup[path_id, kind, gradient][family, cells]
            direct, _ = direct_support(fine, window, stride, support != "unfiltered")
            if not rows.fine_counts_sha256.eq(array_sha256(fine)).all():
                raise AssertionError("decoder setting changed input spikes")
            for name, value in direct.items():
                if not rows[name].eq(value).all():
                    raise AssertionError(f"direct support/count mismatch: {name}")
            comparisons += len(rows)
        expected_count = 0
        _, aspect, sigma = FIELDS[batch.field_id]
        for cells in [64, 128]:
            settings = [(8, 20, 5)] if args.baseline_only else decoder_settings(cells, sigma, aspect)
            for spacing, window, stride in settings:
                expected_count += len(paths) * 2 * 2 * 2 * 2 * (1 if stride == 5 else 2)
        if comparisons != len(frame) or len(frame) != expected_count:
            raise AssertionError("incomplete setting coverage")
        grids = grid_meta[grid_meta.population_seed.eq(batch.population_seed) & grid_meta.field_id.eq(batch.field_id)]
        for row in grids.itertuples(index=False):
            grid = centered_grid(CORE if row.support_domain == "common_core" else bounds, row.grid_cm)
            rates = evaluate(grid)[:, :row.n_cells].T
            if array_sha256(grid) != row.grid_sha256 or array_sha256(rates) != row.rates_sha256:
                raise AssertionError("grid or field-rate hash mismatch")
        summary, gradient = summarize_batch(frame)
        match = summaries[summaries.population_seed.eq(batch.population_seed) & summaries.sigma_cm.eq(sigma)
                          & summaries.aspect.eq(aspect) & summaries.area_m2.eq(FIELDS[batch.field_id][0])]
        compare_tables(match, summary, [c for c in KEY if c != "path_id"])
        match = gradients[gradients.population_seed.eq(batch.population_seed) & gradients.sigma_cm.eq(sigma)
                          & gradients.aspect.eq(aspect) & gradients.area_m2.eq(FIELDS[batch.field_id][0])]
        compare_tables(match, gradient, [c for c in KEY if c != "path_id"] + ["selection", "coordinate", "readout"])
        check = sampled_likelihood_check(frame, paths[0], obs[0], centers, bounds, args)
        audit_rows.append({"population_seed": batch.population_seed, "field_id": batch.field_id, "metric_rows_recounted": comparisons,
                           "observation_arrays_reconstructed": len(expected), "analytic_likelihood_rows_checked": check, "pass": True})
        print(json.dumps(audit_rows[-1]), flush=True)
    # Across geometry, path hashes and imposed population totals must be identical.
    identity = ["population_seed", "path_id", "truth_kind", "gradient"]
    if observed.groupby(identity).path_sha256.nunique().max() != 1:
        raise AssertionError("geometry changed prescribed paths")
    if observed[observed.observation.eq("common_count_schedule")].groupby(identity).total_schedule_sha256.nunique().max() != 1:
        raise AssertionError("common count schedule changed across geometry")
    return pd.DataFrame(audit_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = audit(args.input_dir)
    rows.to_csv(args.input_dir / "geometry_reconstruction_audit.csv", index=False)
    record = {**build_script_provenance(input_paths={"scoring_manifest": args.input_dir / "geometry_manifest.json", "audit_code": __file__}),
              "status": "pass", "batches": len(rows), "metric_rows_recounted": int(rows.metric_rows_recounted.sum()),
              "observations_reconstructed": int(rows.observation_arrays_reconstructed.sum()),
              "analytic_likelihood_rows_checked": int(rows.analytic_likelihood_rows_checked.sum())}
    (args.input_dir / "geometry_reconstruction_audit.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
