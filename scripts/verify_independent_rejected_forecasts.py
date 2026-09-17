#!/usr/bin/env python3
"""Reconstruct detection, geometry and primary forecasting without producer kernels."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.special import gammaln, logsumexp
from scipy.stats import binomtest

from scripts._provenance import build_script_provenance, file_sha256

ID = ["dataset", "animal", "session"]
BASELINES = ["matched_own", "matched_shared", "frozen", "no_history", "global"]


def ref_runs(mask):
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) != 1) + 1
    return [(int(g[0]), int(g[-1]) + 1) for g in np.split(idx, breaks)]


def ref_detection(raw, detector_ids):
    pos = raw["position"]
    start = pos[0, 0]
    times = start + (np.arange(int(np.floor((pos[-1, 0] - start) / 0.001))) + 0.5) * 0.001
    smoothed_speed = np.full(len(times), np.nan)
    raw_speed = np.hypot(np.gradient(pos[:, 1], pos[:, 0]), np.gradient(pos[:, 2], pos[:, 0]))
    for lo, hi in raw["supported_run_intervals"]:
        inds = np.flatnonzero((pos[:, 0] >= lo) & (pos[:, 0] <= hi))
        if len(inds) < 3:
            continue
        blocks = np.split(inds, np.flatnonzero(np.diff(pos[inds, 0]) > 0.1) + 1)
        for block in blocks:
            if len(block) < 3:
                continue
            p = pos[block]
            v = np.hypot(np.gradient(p[:, 1], p[:, 0]), np.gradient(p[:, 2], p[:, 0]))
            v = gaussian_filter1d(v, 0.1 / np.median(np.diff(p[:, 0])), mode="nearest")
            a, b = np.searchsorted(times, p[1, 0]), np.searchsorted(times, p[-2, 0], side="right")
            smoothed_speed[a:b] = np.interp(times[a:b], p[:, 0], v)
    for a, b in ref_runs(np.isfinite(smoothed_speed)):
        t = times[a:b]
        speed = np.maximum(np.interp(t - 0.0005, pos[:, 0], raw_speed), np.interp(t + 0.0005, pos[:, 0], raw_speed))
        left, right = np.searchsorted(pos[:, 0], t[0] - 0.0005), np.searchsorted(pos[:, 0], t[-1] + 0.0005, side="right")
        inds = np.floor((pos[left:right, 0] - (t[0] - 0.0005)) / 0.001).astype(int)
        valid = (inds >= 0) & (inds < len(speed))
        np.maximum.at(speed, inds[valid], raw_speed[left:right][valid])
        smoothed_speed[a:b] = np.maximum(smoothed_speed[a:b], speed)
    spikes = raw["spikes"][np.isin(raw["spikes"][:, 1], detector_ids)]
    edges = times[0] - 0.0005 + np.arange(len(times) + 1) * 0.001
    activity = gaussian_filter1d(np.histogram(spikes[:, 0], edges)[0].astype(float), 10, mode="constant", truncate=4)
    valid = np.isfinite(smoothed_speed) & (smoothed_speed < 5)
    if valid.sum() < 10 or np.std(activity[valid]) == 0:
        return []
    mean, sd = activity[valid].mean(), activity[valid].std()
    z = (activity - mean) / sd
    # Expand threshold peaks independently, then merge their mean-bounded runs.
    intervals = []
    for a, b in ref_runs(valid & (z > 3)):
        while a and valid[a - 1] and z[a - 1] > 0:
            a -= 1
        while b < len(z) and valid[b] and z[b] > 0:
            b += 1
        if intervals and a <= intervals[-1][1]:
            intervals[-1] = (intervals[-1][0], max(b, intervals[-1][1]))
        else:
            intervals.append((a, b))
    events = []
    for a, b in intervals:
        if not 0.05 - 1e-12 <= (b - a) * 0.001 <= 2 + 1e-12:
            continue
        local = spikes[(spikes[:, 0] >= edges[a]) & (spikes[:, 0] < edges[b])]
        active = len(np.unique(local[:, 1]))
        if active >= max(2, int(np.ceil(0.1 * len(detector_ids)))):
            events.append((edges[a], edges[b], len(local), active))
    return events


def ref_geometry(base, durations, rates, grid, cells):
    n = int(np.count_nonzero(np.isclose(durations, 0.005, rtol=0, atol=1e-9)))
    windows = np.array([base[i : i + 4, cells].sum(axis=0) for i in range(max(0, n - 3))])
    if len(windows) == 0:
        windows = np.zeros((0, len(cells)), int)
    ll = windows @ np.log(rates[cells]) - 0.020 * rates[cells].sum(axis=0)
    path = ll.argmax(axis=1)
    edges = np.flatnonzero(windows.sum(axis=1) >= 2)
    valid = np.zeros(len(windows), bool)
    if len(edges):
        valid[edges[0] : edges[-1] + 1] = True
    best_start, best_length = 0, 0
    for a, b in ref_runs(valid):
        jumps = np.linalg.norm(np.diff(grid[path[a:b]], axis=0), axis=1)
        boundaries = np.r_[a, a + np.flatnonzero(jumps >= 20 - 1e-9) + 1, b]
        for x, y in zip(boundaries[:-1], boundaries[1:], strict=True):
            if y - x > best_length:
                best_start, best_length = int(x), int(y - x)
    disp = float(np.linalg.norm(grid[path[best_start + best_length - 1]] - grid[path[best_start]])) if best_length else 0
    return path, dict(valid_frames=int(valid.sum()), longest_run_frames=best_length, run_displacement_cm=disp, geometric_pass=bool(best_length >= 10 and disp >= 40 - 1e-9))


def ref_ll(counts, rates):
    p = rates / rates.sum(axis=0, keepdims=True)
    return counts @ np.log(p) + (gammaln(counts.sum(axis=1) + 1) - gammaln(counts + 1).sum(axis=1))[:, None]


def ref_filter(ll, initial, transition):
    q, values = initial.copy(), []
    for t, x in enumerate(ll):
        if t:
            q = q @ transition
        q = q * np.exp(x - np.max(x))
        q = q / q.sum()
        values.append(q.copy())
    return np.asarray(values)


def ref_null(a):
    n = len(a)
    lhs, rhs = a.T - np.eye(n), np.zeros(n)
    lhs[-1] = 1
    rhs[-1] = 1
    pi = np.linalg.solve(lhs, rhs)
    target = pi * (1 - np.diag(a))
    flow = np.ones_like(a) - np.eye(n)
    for _ in range(100000):
        flow *= (target / flow.sum(axis=1))[:, None]
        flow *= (target / flow.sum(axis=0))[None, :]
        if np.max(np.abs(flow.sum(axis=1) - target)) < 1e-14 and np.max(np.abs(flow.sum(axis=1) - target) / pi) < 1e-11:
            break
    else:
        raise AssertionError("reference null did not converge")
    null = flow / pi[:, None] + np.diag(np.diag(a))
    np.testing.assert_allclose(null.sum(axis=1), 1, atol=1e-10)
    np.testing.assert_allclose(pi @ null, pi, atol=1e-10)
    return null


def ref_scores(counts, train, held, fit):
    ll = ref_ll(counts[:, train], fit["probabilities"][train])
    a, initial = fit["transition"], fit["initial"]
    null = ref_null(a)
    filtered, null_filtered = ref_filter(ll, initial, a), ref_filter(ll, initial, null)
    out = {}
    for h in (1, 2, 4):
        if h >= len(counts):
            continue
        candidates = {
            "dynamic": filtered[:-h] @ np.linalg.matrix_power(a, h),
            "matched_shared": filtered[:-h] @ np.linalg.matrix_power(null, h),
            "matched_own": null_filtered[:-h] @ np.linalg.matrix_power(null, h),
            "frozen": filtered[:-h],
            "no_history": np.array([initial @ np.linalg.matrix_power(a, t) for t in range(h, len(counts))]),
        }
        hl = ref_ll(counts[h:, held], fit["probabilities"][held])
        totals = counts[h:, held].sum(axis=1)
        row = {}
        for name, q in candidates.items():
            v = logsumexp(np.log(np.maximum(q, 1e-300)) + hl, axis=1)
            v[totals == 0] = 0
            row["score_" + name] = float(v.sum())
        row["score_global"] = float(ref_ll(counts[h:, held], fit["global_probability"][held, None]).sum())
        out[h] = row
    return out


def primary_summary(rows):
    chosen = rows[rows.level.eq("full") & rows.model.eq("learned_hmm") & rows.horizon.eq(2) & rows.rejected_with_opportunity & rows.status.eq("scored")]
    animals, summary = [], []
    for baseline in BASELINES:
        x = chosen[ID + ["event_id", "split"]].copy()
        x["value"] = (chosen.score_dynamic - chosen["score_" + baseline]) / chosen.n_heldout_target_spikes.replace(0, np.nan)
        event = x.groupby(ID + ["event_id"]).value.median()
        session = event.groupby(ID).mean()
        rat = session.groupby(["dataset", "animal"]).mean()
        for (dataset, animal), val in rat.items():
            animals.append(dict(dataset=dataset, animal=animal, contrast="dynamic_minus_" + baseline, value=val))
        for dataset in rat.index.get_level_values(0).unique():
            v = rat.loc[dataset].dropna().sort_index().to_numpy()
            draws = np.asarray(list(itertools.product(v, repeat=len(v))))
            lo, hi = np.quantile(draws.mean(axis=1), [0.025, 0.975])
            nonzero = v[v != 0]
            summary.append(
                dict(
                    dataset=dataset,
                    contrast="dynamic_minus_" + baseline,
                    mean=v.mean(),
                    ci_low=lo,
                    ci_high=hi,
                    animals=len(v),
                    positive_animals=int((v > 0).sum()),
                    sign_p_value=binomtest(int((nonzero > 0).sum()), len(nonzero)).pvalue if len(nonzero) else 1,
                )
            )
    return pd.DataFrame(animals), pd.DataFrame(summary)


def audit(root, output):
    mp = root / "independent_rejected_forecast_manifest.json"
    m = json.loads(mp.read_text())
    if m["status"] != "complete" or m["subset_debug"] or len(m["completed"]) != 33:
        raise AssertionError("complete full-cohort run required")
    for key, path in m["input_file_paths"].items():
        assert file_sha256(path) == m["input_file_sha256"][key], key
    for name, digest in m["output_sha256"].items():
        assert file_sha256(root / name) == digest, name
    sources = pd.read_csv(m["input_file_paths"]["source_sessions"])
    results, frames = [], []
    for rec in m["completed"]:
        local = root / rec["tag"]
        source = sources
        for k in ID:
            source = source[source[k].eq(rec[k])]
        assert len(source) == 1
        raw = dict(np.load(source.iloc[0].artifact_path, allow_pickle=False))
        cache = dict(np.load(local / "cache.npz", allow_pickle=False))
        sel = pd.read_csv(local / "selection.csv", float_precision="round_trip")
        labels = pd.read_csv(local / "labels.csv.gz").set_index(["event_id", "split"])
        paths = dict(np.load(local / "paths.npz", allow_pickle=False))
        scores = pd.read_csv(local / "scores.csv.gz")
        frames.append(scores)
        units = raw["cell_ids"][raw["unit_qc_mask"].astype(bool)]
        det = cache["detector_unit_ids"]
        assert set(det).isdisjoint(cache["unit_ids"])
        assert set(det) | set(cache["unit_ids"]) == set(units)
        detected = ref_detection(raw, det)
        assert len(detected) == len(sel)
        np.testing.assert_allclose(np.asarray(detected)[:, :2], sel[["start_s", "end_s"]], atol=1e-9, rtol=0)
        np.testing.assert_array_equal(np.asarray(detected)[:, 2:], sel[["detector_spikes", "detector_active_cells"]])
        qc = raw["unit_qc_mask"].astype(bool)
        lookup = {u: i for i, u in enumerate(units)}
        order = [lookup[u] for u in cache["unit_ids"]]
        states = raw["valid_spatial_bins"].astype(bool)
        np.testing.assert_array_equal(cache["rates"], raw["rates_hz"][qc][order][:, states])
        np.testing.assert_array_equal(cache["centers"], raw["bin_centers_cm"][states])
        geometry_checks, count_checks = 0, 0
        # All event counts are reconstructed in one vectorized search per cell.
        lefts, rights, offsets = [], [], [0]
        for e in sel.itertuples(index=False):
            dt = cache[f"durations_{e.event_id}"]
            n = len(dt)
            edges = e.start_s + np.arange(n + 1) * 0.005
            edges[-1] = e.end_s
            np.testing.assert_allclose(np.diff(edges), dt, rtol=0, atol=1e-10)
            lefts.append(edges[:-1])
            rights.append(edges[1:])
            offsets.append(offsets[-1] + n)
        lefts, rights = np.concatenate(lefts), np.concatenate(rights)
        all_counts = np.empty((len(lefts), len(cache["unit_ids"])), int)
        for j, unit in enumerate(cache["unit_ids"]):
            ts = np.sort(raw["spikes"][raw["spikes"][:, 1] == unit, 0])
            all_counts[:, j] = np.searchsorted(ts, rights, side="left") - np.searchsorted(ts, lefts, side="left")
        for i, e in enumerate(sel.itertuples(index=False)):
            base = all_counts[offsets[i] : offsets[i + 1]]
            np.testing.assert_array_equal(base, cache[f"base_{e.event_id}"])
            count_checks += base.size
            n = len(base) - int(not np.isclose(cache[f"durations_{e.event_id}"][-1], 0.005, atol=1e-9, rtol=0))
            n = n // 4 * 4
            counted = np.array([base[j : j + 4].sum(axis=0) for j in range(0, n, 4)])
            np.testing.assert_array_equal(counted, cache[f"counts_{e.event_id}"])
            assert base[n:].sum() == cache[f"discarded_{e.event_id}"]
            for split in range(5):
                tr, held, half = cache[f"train_{split}"], cache[f"held_{split}"], cache[f"half_{split}"]
                assert set(tr).isdisjoint(held) and sorted([*tr, *held]) == list(range(len(cache["unit_ids"])))
                assert set(half).issubset(tr)
                for level, cells in [("full", tr), ("half", half)]:
                    path, metrics = ref_geometry(base, cache[f"durations_{e.event_id}"], cache["rates"], cache["centers"], cells)
                    np.testing.assert_array_equal(path, paths[f"{level}_{e.event_id}_{split}"])
                    lab = labels.loc[(e.event_id, split)]
                    for key, value in metrics.items():
                        assert abs(float(lab[level + "_" + key]) - float(value)) < 1e-8, (rec["tag"], e.event_id, key)
                    geometry_checks += 1
                lab = labels.loc[(e.event_id, split)]
                assert lab.rejected_with_opportunity == (not lab.full_geometric_pass and lab.full_valid_frames >= 10)
                assert lab.lost_with_thinning == (lab.full_geometric_pass and not lab.half_geometric_pass)
        meta = json.loads((local / "folds.json").read_text())
        tested, score_checks, error = [], 0, 0.0
        for f in meta:
            assert f["converged"] and not set(f["test_ids"]) & set(f["calibration_ids"])
            tested.extend(f["test_ids"])
            test, cal = sel[sel.event_id.isin(f["test_ids"])], sel[sel.event_id.isin(f["calibration_ids"])]
            for e in test.itertuples(index=False):
                assert ((cal.end_s + 1 <= e.start_s) | (cal.start_s >= e.end_s + 1)).all()
            fit = dict(np.load(local / f"fit_{f['fold']}.npz", allow_pickle=False))
            assert (np.diff(fit["objective_trace"]) >= -1e-7 * (1 + abs(fit["objective_trace"][:-1]))).all()
            eid = int(test.sort_values("start_s").iloc[0].event_id)
            for split in range(5):
                for level in ["full", "half"]:
                    train = cache[f"train_{split}" if level == "full" else f"half_{split}"]
                    expected = ref_scores(cache[f"counts_{eid}"], train, cache[f"held_{split}"], fit)
                    for h, values in expected.items():
                        scored = scores[scores.event_id.eq(eid) & scores.split.eq(split) & scores.level.eq(level) & scores.model.eq("learned_hmm") & scores.horizon.eq(h)]
                        assert len(scored) == 1
                        for key, value in values.items():
                            discrepancy = abs(scored.iloc[0][key] - value)
                            assert discrepancy < 1e-8, (rec["tag"], eid, key, discrepancy)
                            error = max(error, discrepancy)
                            score_checks += 1
        assert sorted(tested) == sorted(sel.event_id)
        result = {k: rec[k] for k in ID} | dict(
            events=len(sel), counts_checked=count_checks, geometry_paths_checked=geometry_checks, scores_reconstructed=score_checks, maximum_score_error=error
        )
        results.append(result)
        print(json.dumps(result), flush=True)
    rows = pd.concat(frames, ignore_index=True)
    animals, summary = primary_summary(rows)
    produced = pd.read_csv(root / "rejected_forecast_summary.csv.gz")
    produced = produced[
        produced.level.eq("full")
        & produced.model.eq("learned_hmm")
        & produced.horizon.eq(2)
        & produced.group.eq("rejected_with_opportunity")
        & produced.metric.eq("delta_per_spike")
    ]
    merged = summary.merge(produced, on=["dataset", "contrast"], suffixes=("_audit", "_produced"), validate="one_to_one")
    assert len(merged) == 10
    for k in ["mean", "ci_low", "ci_high", "animals", "positive_animals", "sign_p_value"]:
        np.testing.assert_allclose(merged[k + "_audit"], merged[k + "_produced"], rtol=0, atol=1e-10)
    output.mkdir(parents=True, exist_ok=False)
    animals.to_csv(output / "independent_primary_animals.csv", index=False)
    summary.to_csv(output / "independent_primary_summary.csv", index=False)
    provenance = build_script_provenance(input_paths={"run_manifest": mp, "verifier": Path(__file__)}, cwd=ROOT)
    provenance.update(
        status="pass",
        source_status=m["status"],
        sessions=results,
        all_detections_reconstructed=True,
        all_count_arrays_reconstructed=True,
        all_geometric_paths_reconstructed=True,
        all_primary_aggregates_reconstructed=True,
        forecast_scope="first chronological event per fold; both inference levels, five partitions, all eligible horizons; neural model only",
        model_fits_refitted=False,
        spatial_diffusion_scores_independently_reconstructed=False,
    )
    (output / "independent_rejected_forecast_audit.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    audit(a.run_dir, a.output_dir)
