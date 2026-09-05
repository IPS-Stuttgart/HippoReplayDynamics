#!/usr/bin/env python3
"""Recount raw observations and independently reconstruct shuffle decisions."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from scripts._provenance import build_script_provenance, file_sha256
from scripts.analyze_replay_coverage_shuffle_baseline import summarize


def independent_screen(path, grid, counts, filtered, frames):
    support = np.flatnonzero(counts.sum(axis=1) >= 2)
    if not len(support):
        return False
    best_start, best_end, start, previous = 0, -1, None, None
    for t in range(int(support[0]), int(support[-1]) + 1):
        if filtered and (counts[t].sum() < 3 or np.count_nonzero(counts[t]) < 2):
            start, previous = None, None
            continue
        if previous is None or np.linalg.norm(grid[path[t]] - grid[path[previous]]) >= 20:
            start = t
        if t - start > best_end - best_start:
            best_start, best_end = start, t
        previous = t
    return bool(best_end - best_start + 1 >= frames and np.linalg.norm(grid[path[best_end]] - grid[path[best_start]]) >= 40)


def explicit_rates(rates, support, shape, bank, family):
    if family == "cell_identity":
        assert np.array_equal(np.sort(bank), np.arange(len(rates)))
        return rates[bank][:, support]
    assert np.all(bank.sum(axis=1) > 0)
    return np.array([np.roll(rate.reshape(tuple(shape)), tuple(shift), axis=(0, 1)).ravel()[support] for rate, shift in zip(rates, bank, strict=True)])


def audit_dense_map(counts, rates, actual):
    ll = counts @ np.log(rates * .02) - .02 * rates.sum(axis=0)
    best = ll.argmax(axis=1)
    gap = ll[np.arange(len(ll)), best] - ll[np.arange(len(ll)), actual]
    assert np.isfinite(ll).all() and np.all(gap <= 1e-10), "decoded bin is not a floating-point likelihood maximum"
    return int(np.count_nonzero(best != actual))


def audit_session(record, shuffles, seed=20260905):
    for name in ["input_arrays", "metrics", "population"]:
        assert file_sha256(record[f"{name}_path"]) == record[f"{name}_sha256"]
    source = record["source"]
    assert file_sha256(source["source_cache_path"]) == source["source_cache_sha256"]
    assert file_sha256(source["windows_path"]) == source["windows_sha256"]
    source_windows = pd.read_csv(source["windows_path"], float_precision="round_trip")
    windows = source_windows[source_windows.eligible & source_windows.window_variant.eq("detected_core")].sort_values("window_uid").reset_index(drop=True)
    a = dict(np.load(record["input_arrays_path"], allow_pickle=False))
    raw = dict(np.load(source["source_cache_path"], allow_pickle=False))
    np.testing.assert_array_equal(a["cell_ids"], raw["cell_ids"][raw["unit_qc_mask"]])
    np.testing.assert_array_equal(a["rates_hz"], raw["rates_hz"][raw["unit_qc_mask"]])
    np.testing.assert_array_equal(a["window_uids"], windows.window_uid.to_numpy(str))
    for j, cell in enumerate(a["cell_ids"]):
        times = np.sort(raw["spikes"][raw["spikes"][:, 1] == cell, 0])
        counts = np.searchsorted(times, a["base_starts_s"] + a["base_durations_s"], side="left") - np.searchsorted(times, a["base_starts_s"], side="left")
        np.testing.assert_array_equal(counts, a["base_counts"][:, j])
    for i, w in enumerate(windows.itertuples(index=False)):
        begin, end = a["base_offsets"][i:i + 2]
        np.testing.assert_allclose(a["base_starts_s"][begin], w.start_s, rtol=0, atol=1e-10)
        np.testing.assert_allclose(a["base_starts_s"][end - 1] + a["base_durations_s"][end - 1], w.end_s, rtol=0, atol=1e-10)
        perm = a["permutation"][begin:end]
        np.testing.assert_array_equal(np.sort(perm), np.arange(begin, end))
        complete = np.isclose(a["base_durations_s"][begin:end], .005, atol=1e-9, rtol=0)
        np.testing.assert_array_equal(perm[~complete], np.arange(begin, end)[~complete])
        np.testing.assert_array_equal(a["base_counts"][perm].sum(axis=0), a["base_counts"][begin:end].sum(axis=0))
        for order in range(2):
            x, y = a["frame_offsets"][2 * i + order:2 * i + order + 2]
            assert y - x == max(0, int(np.count_nonzero(complete)) - 3)
            fine = a["base_counts"][begin:end] if order == 0 else a["base_counts"][perm]
            expected = np.array([fine[t:t + 4].sum(axis=0) for t in range(y - x)], dtype=np.int64).reshape(y - x, len(a["cell_ids"]))
            np.testing.assert_array_equal(expected, a["frame_counts"][x:y])
    metrics = pd.read_csv(record["metrics_path"])
    columns = [(False, 10), (True, 10), (False, 11), (True, 11)]
    independent_tests, map_frames, tied_map_disagreements, null_bits = 0, 0, 0, 0
    rng = np.random.default_rng(seed)
    for spec in record["shuffle_files"]:
        assert file_sha256(spec["path"]) == spec["sha256"]
        np.testing.assert_array_equal(a["cell_ids"][spec["indices"]], spec["cell_ids"])
        z = dict(np.load(spec["path"], allow_pickle=False))
        counts = a["frame_counts"][:, spec["indices"]]
        rates = a["rates_hz"][spec["indices"]]
        total = len(a["frame_offsets"]) - 1
        original = np.zeros((total, 4), bool)
        for i, (x, y) in enumerate(zip(a["frame_offsets"][:-1], a["frame_offsets"][1:], strict=True)):
            for c, (filtered, minimum) in enumerate(columns):
                original[i, c] = independent_screen(z["original_path"][x:y], a["grid_cm"], counts[x:y], filtered, minimum)
                independent_tests += 1
        np.testing.assert_array_equal(original, z["geometric_pass"])
        np.testing.assert_array_equal(z["tested_observations"], np.flatnonzero(original.any(axis=1)))
        chosen = np.concatenate([np.arange(a["frame_offsets"][i], a["frame_offsets"][i + 1]) for i in z["tested_observations"]]) if len(z["tested_observations"]) else np.empty(0, int)
        np.testing.assert_array_equal(chosen, z["selected_frame_indices"])
        np.testing.assert_array_equal(z["selected_offsets"], np.r_[0, np.cumsum(np.diff(a["frame_offsets"])[z["tested_observations"]])])
        assert z["null_pass"].dtype == bool and z["null_pass"].shape == (2, shuffles, len(z["tested_observations"]), 4)
        null_bits += z["null_pass"].size
        picked = np.sort(rng.choice(len(counts), min(512, len(counts)), replace=False))
        tied_map_disagreements += audit_dense_map(counts[picked], rates[:, a["support"]], z["original_path"][picked])
        map_frames += len(picked)
        for f, family in enumerate(["cell_identity", "independent_xy_roll"]):
            banks = z[f"bank__{family}"]
            assert len(banks) == shuffles
            for k, path in enumerate(z[f"sample_paths__{family}"]):
                shifted = explicit_rates(rates, a["support"], a["grid_shape"], banks[k], family)
                local = np.sort(rng.choice(len(chosen), min(512, len(chosen)), replace=False))
                tied_map_disagreements += audit_dense_map(counts[chosen[local]], shifted, path[local])
                map_frames += len(local)
                for j, (x, y) in enumerate(zip(z["selected_offsets"][:-1], z["selected_offsets"][1:], strict=True)):
                    for c, (filtered, minimum) in enumerate(columns):
                        expected = independent_screen(path[x:y], a["grid_cm"], counts[chosen[x:y]], filtered, minimum)
                        assert expected == z["null_pass"][f, k, j, c]
                        independent_tests += 1
        p = np.full((total, 4, 2), np.nan)
        p[z["tested_observations"]] = (z["null_pass"].sum(axis=1).transpose(1, 2, 0) + 1) / (shuffles + 1)
        p[~original] = np.nan
        np.testing.assert_array_equal(p, z["p_values"])
        np.testing.assert_array_equal(z["accepted"], original & (p < .02).all(axis=-1))
        rows = metrics[(metrics.cell_fraction == spec["cell_fraction"]) & (metrics.population_replicate == spec["population_replicate"])]
        assert len(rows) == total * 4
        rows = rows.set_index(["window_uid", "observation", "bin_filter", "min_frames"])
        assert rows.index.is_unique
        for i, uid in enumerate(a["window_uids"]):
            for order, obs in enumerate(["original_order", "order_randomized"]):
                for c, (filtered, minimum) in enumerate(columns):
                    row = rows.loc[(uid, obs, "at_least_2cells_3spikes" if filtered else "edge_only", minimum)]
                    ix = 2 * i + order
                    assert bool(row.geometric_pass) == original[ix, c]
                    assert row.n_shuffles == (shuffles if original[ix, c] else 0)
                    assert row.test_status == ("tested" if original[ix, c] else "geometric_fail_not_tested")
                    np.testing.assert_allclose([row.cell_identity_p, row.xy_roll_p], p[ix, c], rtol=1e-12, atol=1e-15, equal_nan=True)
                    np.testing.assert_allclose([row.cell_identity_successes, row.xy_roll_successes],
                        p[ix, c] * (shuffles + 1) - 1, rtol=1e-12, atol=1e-9, equal_nan=True)
                    for alpha in [.01, .02, .05]:
                        assert bool(row[f"accepted_alpha_{alpha:g}"]) == (original[ix, c] and (p[ix, c] < alpha).all())
    population = pd.read_csv(record["population_path"])
    for row in population.itertuples(index=False):
        subset = metrics[(metrics.cell_fraction == row.cell_fraction) & (metrics.population_replicate == row.population_replicate)
            & (metrics.observation == row.observation) & (metrics.detector == row.detector) & (metrics.bin_filter == row.bin_filter) & (metrics.min_frames == row.min_frames)]
        assert row.eligible_events == len(subset)
        assert row.geometric_events == subset.geometric_pass.sum()
        assert row.accepted_events == subset[f"accepted_alpha_{row.alpha:g}"].sum()
        np.testing.assert_allclose([row.geometric_fraction, row.accepted_fraction],
            [subset.geometric_pass.mean(), subset[f"accepted_alpha_{row.alpha:g}"].mean()], equal_nan=True)
    return {"dataset": record["dataset"], "animal": record["animal"], "session": record["session"], "status": "pass",
        "base_bins_directly_counted": len(a["base_counts"]), "original_and_sample_null_geometric_tests": independent_tests,
        "dense_analytic_map_frames": map_frames, "tied_map_index_disagreements": tied_map_disagreements,
        "decision_rows": len(metrics), "null_bits_recounted": null_bits}


def main(args):
    source = args.input_dir / "coverage_shuffle_baseline_manifest.json"
    meta = json.loads(source.read_text())
    assert meta["status"] == "complete" and meta["results"]
    for key, path in meta["input_file_paths"].items():
        assert file_sha256(path) == meta["input_file_sha256"][key]
    if "frozen_subsets" in meta["input_file_paths"]:
        frozen = json.loads(Path(meta["input_file_paths"]["frozen_subsets"]).read_text())["population_subsets"]
        lookup = {(p["session_key"], p["cell_fraction"], p["population_replicate"]): p["cell_ids"] for p in frozen}
        for record in meta["results"]:
            for spec in record["population_specs"]:
                key = (":".join(record[k] for k in ["dataset", "animal", "session"]), spec["cell_fraction"], spec["population_replicate"])
                assert spec["cell_ids"] == lookup[key]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(audit_session, meta["results"], [meta["shuffles_per_family"]] * len(meta["results"])))
    population = pd.concat([pd.read_csv(r["population_path"]) for r in meta["results"]], ignore_index=True)
    session, animal, summary = summarize(population, args.bootstraps, meta["seed"] + 1)
    for name, frame in [("population_summary", population), ("session_summary", session), ("animal_summary", animal), ("summary", summary)]:
        actual = pd.read_csv(args.input_dir / f"coverage_shuffle_baseline_{name}.csv")
        pd.testing.assert_frame_equal(actual, frame, check_dtype=False, rtol=1e-10, atol=1e-12)
    for name, digest in meta["output_sha256"].items():
        assert file_sha256(args.input_dir / name) == digest
    provenance = build_script_provenance(input_paths={"scoring_manifest": source, "auditor": Path(__file__)}, cwd=ROOT)
    provenance.update(status="pass", results=results, verification_scope="all raw base counts, order preservation, original geometry, p/acceptance and tables; first three null paths and sampled independent dense MAP scores")
    (args.input_dir / "coverage_shuffle_baseline_reconstruction_audit.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"status": "pass", "sessions": len(results), "decision_rows": sum(r["decision_rows"] for r in results),
        "dense_analytic_map_frames": sum(r["dense_analytic_map_frames"] for r in results)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--bootstraps", type=int, default=5000)
    main(parser.parse_args())
