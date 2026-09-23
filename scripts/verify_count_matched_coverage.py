#!/usr/bin/env python3
"""Independently check exact-count interventions and a stratified score sample."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import gammaln, logsumexp


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ll(counts, rates):
    p = rates / rates.sum(axis=0)
    return counts @ np.log(p) + (gammaln(counts.sum(axis=1) + 1) - gammaln(counts + 1).sum(axis=1))[:, None]


def predicted_score(counts, rates, held, held_rates, initial, transition):
    emission = ll(counts, rates)
    target = ll(held, held_rates)
    q = initial.copy()
    total = 0.0
    for t in range(len(counts) - 2):
        if t:
            q = q @ transition
        logq = np.log(q) + emission[t]
        q = np.exp(logq - logsumexp(logq))
        prediction = q @ transition @ transition
        total += float(logsumexp(np.log(prediction) + target[t + 2]))
    return total


def trajectory(path, centers, counts):
    edges = np.flatnonzero(counts.sum(axis=1) >= 2)
    if not len(edges):
        return False, 0, 0.0
    # First longest run is retained, matching the declared tie convention.
    runs = []
    start = int(edges[0])
    for t in range(start + 1, int(edges[-1]) + 1):
        if np.linalg.norm(centers[path[t]] - centers[path[t - 1]]) >= 20 - 1e-9:
            runs.append((start, t))
            start = t
    runs.append((start, int(edges[-1]) + 1))
    a, b = max(runs, key=lambda r: r[1] - r[0])
    displacement = float(np.linalg.norm(centers[path[b - 1]] - centers[path[a]]))
    return b - a >= 10 and displacement >= 40 - 1e-9, b - a, displacement


def null_transition(transition):
    n = len(transition)
    equation = transition.T - np.eye(n)
    equation[-1] = 1
    rhs = np.zeros(n)
    rhs[-1] = 1
    pi = np.linalg.solve(equation, rhs)
    stay = np.diag(transition)
    margin = pi * (1 - stay)
    v = margin.copy()
    for _ in range(20000):
        u = margin / (v.sum() - v)
        v = margin / (u.sum() - u)
        v /= v.sum()
        u = margin / (v.sum() - v)
        achieved = v * (u.sum() - u)
        if np.max(np.abs(achieved - margin)) < 1e-13 and np.max(np.abs(achieved - margin) / pi) < 1e-9:
            break
    else:
        raise AssertionError("independent null fitting did not converge")
    result = (u / pi)[:, None] * v[None, :]
    np.fill_diagonal(result, stay)
    np.testing.assert_allclose(result.sum(axis=1), 1, atol=1e-10)
    np.testing.assert_allclose(pi @ result, pi, atol=1e-10)
    return result


def read(path):
    return pd.read_csv(path, float_precision="round_trip")


def compare_table(actual, expected, keys, metrics):
    if actual.duplicated(keys).any() or expected.duplicated(keys).any():
        raise AssertionError("duplicate aggregate key")
    a, e = actual.set_index(keys).sort_index(), expected.set_index(keys).sort_index()
    if not a.index.equals(e.index):
        raise AssertionError("aggregate key mismatch")
    np.testing.assert_allclose(a[metrics].to_numpy(float), e[metrics].to_numpy(float), rtol=1e-9, atol=1e-9, equal_nan=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    root, source = args.output_dir, args.source_dir
    manifest = json.loads((root / "count_matched_manifest.json").read_text())
    original = json.loads((source / "independent_rejected_forecast_manifest.json").read_text())
    assert manifest["status"] == "complete_pending_independent_verification"
    assert len(manifest["completed"]) == 33
    assert sha(source / "independent_rejected_forecast_manifest.json") == manifest["input_file_sha256"]["source_manifest"]
    source_records = {r["tag"]: r for r in original["completed"]}
    event_tables, counters = [], {"rows": 0, "matched_samples": 0, "geometry_samples": 0, "predictive_samples": 0, "hashes": 1}
    for record in manifest["completed"]:
        assert record["status"] == "complete"
        folder, inp = root / record["tag"], source / record["tag"]
        for path, digest in record["output_sha256"].items():
            assert sha(folder / path) == digest
            counters["hashes"] += 1
        for path, digest in source_records[record["tag"]]["output_sha256"].items():
            assert sha(inp / path) == digest
            counters["hashes"] += 1
        table = read(folder / "scores.csv.gz")
        counters["rows"] += len(table)
        keys = ["dataset", "animal", "session", "event_id", "split"]
        assert not table.duplicated(keys + ["arm", "draw"]).any()
        assert len(table) == record["events"] * 35
        assert set(table.status) <= {"scored", "insufficient_bins"}
        assert table.groupby(keys).size().eq(7).all()
        for field in ["n_target_bins", "n_heldout_target_spikes"]:
            assert table.groupby(keys)[field].nunique().eq(1).all()
        z = np.load(inp / "cache.npz", allow_pickle=False)
        folds = json.loads((inp / "folds.json").read_text())
        sampled = {int(e) for f in folds for e in [f["test_ids"][0], f["test_ids"][-1]]}
        fits = {f["fold"]: dict(np.load(inp / f"fit_{f['fold']}.npz")) for f in folds}
        nulls = {}
        for f in folds:
            nulls[f["fold"]] = null_transition(fits[f["fold"]]["transition"])
        geometry_verified = 0
        for eid, group in table.groupby("event_id", sort=True):
            base, dt = z[f"base_{eid}"], z[f"durations_{eid}"]
            n = int(np.isclose(dt, 0.005, rtol=0, atol=1e-9).sum())
            for split, local in group.groupby("split"):
                train, half, held = (z[f"{x}_{split}"] for x in ["train", "half", "held"])
                assert not set(train) & set(held) and set(half) <= set(train)
                target = base[:, half].sum(axis=1)
                for row in local.itertuples(index=False):
                    cells = half if row.arm == "half_neurons" else train
                    observed = base[:, cells]
                    if row.arm == "matched_spikes":
                        parts = (20260917, "count_matched_v1", record["dataset"], record["animal"], record["session"], eid, split, row.draw)
                        seed = int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:4], "little")
                        rng = np.random.default_rng(seed)
                        sampled_counts = np.zeros_like(observed)
                        for t, total in enumerate(target):
                            if total:
                                sampled_counts[t] = rng.multivariate_hypergeometric(observed[t].astype(np.int64), int(total), method="marginals")
                        assert (sampled_counts <= observed).all()
                        assert np.array_equal(sampled_counts.sum(axis=1), target)
                        observed = sampled_counts
                        counters["matched_samples"] += 1
                    assert row.n_inference_spikes == observed.sum()
                    assert row.n_active_inference_cells == (observed.sum(axis=0) > 0).sum()
                    if eid not in sampled:
                        continue
                    windows = np.array([observed[t : t + 4].sum(axis=0) for t in range(n - 3)])
                    path = ll(windows, z["rates"][cells]).argmax(axis=1)
                    passed, length, displacement = trajectory(path, z["centers"], windows)
                    assert passed == row.geometric_pass and length == row.longest_run_frames
                    np.testing.assert_allclose(displacement, row.run_displacement_cm, atol=1e-9)
                    counters["geometry_samples"] += 1
                    geometry_verified += 1
                    count = observed[: n // 4 * 4].reshape(-1, 4, len(cells)).sum(axis=1)
                    held_count = base[: n // 4 * 4, held].reshape(-1, 4, len(held)).sum(axis=1)
                    assert held_count[2:].sum() == row.n_heldout_target_spikes
                    if row.status != "scored":
                        assert len(count) <= 2
                        continue
                    fit = fits[row.fold]
                    for name, transition in [("dynamic", fit["transition"]), ("matched_own", nulls[row.fold])]:
                        value = predicted_score(count, fit["probabilities"][cells], held_count, fit["probabilities"][held], fit["initial"], transition)
                        np.testing.assert_allclose(value, getattr(row, "score_" + name), atol=1e-8, rtol=1e-10)
                        counters["predictive_samples"] += 1
        z.close()
        assert geometry_verified > 0
        table["predictive_per_spike"] = table.score_dynamic / table.n_heldout_target_spikes.replace(0, np.nan)
        table["advantage_per_spike"] = (table.score_dynamic - table.score_matched_own) / table.n_heldout_target_spikes.replace(0, np.nan)
        metrics = ["geometric_pass", "predictive_per_spike", "advantage_per_spike", "n_active_inference_cells"]
        table["geometric_pass"] = table.geometric_pass.astype(float)
        wide = table.groupby(keys + ["arm"])[metrics].mean().unstack("arm")
        paired = wide.index.to_frame(index=False)
        for metric in metrics:
            for arm in ["full", "half_neurons", "matched_spikes"]:
                paired[metric + "__" + arm] = wide[(metric, arm)].to_numpy()
            paired[metric + "__matched_minus_half"] = (wide[(metric, "matched_spikes")] - wide[(metric, "half_neurons")]).to_numpy()
        paired["group"] = "all"
        lost = (wide[("geometric_pass", "full")] == 1) & (wide[("geometric_pass", "half_neurons")] == 0)
        paired = pd.concat([paired, paired.loc[lost.to_numpy()].assign(group="conditional_full_pass_half_fail")], ignore_index=True)
        metric_cols = [c for c in paired if "__" in c]
        compare_table(read(folder / "paired.csv.gz"), paired, keys + ["group"], metric_cols)
        event_tables.append(paired.groupby(keys[:-1] + ["group"], as_index=False)[metric_cols].mean())
        print(record["tag"], "verified", flush=True)
    events = pd.concat(event_tables, ignore_index=True)
    sessions = events.groupby(["dataset", "animal", "session", "group"], as_index=False)[metric_cols].mean()
    animals = sessions.groupby(["dataset", "animal", "group"], as_index=False)[metric_cols].mean()
    for name, table, keys in [
        ("events", events, ["dataset", "animal", "session", "event_id", "group"]),
        ("sessions", sessions, ["dataset", "animal", "session", "group"]),
        ("animals", animals, ["dataset", "animal", "group"]),
    ]:
        compare_table(read(root / f"count_matched_{name}.csv"), table, keys, metric_cols)
    summary = []
    for (dataset, group), table in animals.groupby(["dataset", "group"]):
        for metric in metric_cols:
            values = table[metric].dropna().to_numpy()
            if len(values):
                boot = [np.mean(values[list(i)]) for i in itertools.product(range(len(values)), repeat=len(values))]
                low, high = np.quantile(boot, [0.025, 0.975])
                summary.append(
                    {
                        "dataset": dataset,
                        "group": group,
                        "metric": metric,
                        "mean": float(values.mean()),
                        "ci_low": low,
                        "ci_high": high,
                        "animals": len(values),
                        "positive_animals": int((values > 0).sum()),
                    }
                )
    compare_table(read(root / "count_matched_summary.csv"), pd.DataFrame(summary), ["dataset", "group", "metric"], ["mean", "ci_low", "ci_high", "animals", "positive_animals"])
    for name, digest in manifest["output_sha256"].items():
        assert sha(root / name) == digest
        counters["hashes"] += 1
    result = dict(
        status="passed",
        producer_commit=manifest["code_commit"],
        verification_code_sha256=sha(__file__),
        **counters,
        sessions=33,
        events=len(events[events.group.eq("all")]),
        scope="all manipulation and aggregation rows; first/last event per fold, all splits/arms for independent geometry and dense predictive scoring; upstream map/model fitting not repeated",
    )
    (root / "independent_verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
