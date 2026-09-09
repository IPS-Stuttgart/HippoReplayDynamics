#!/usr/bin/env python3
"""Independent marked-count and data-processing audit of paired timing recovery."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np
import pandas as pd
from scipy.integrate import fixed_quad

from scripts._provenance import build_script_provenance, file_sha256


def close(x, y, atol=1e-8):
    np.testing.assert_allclose(x, y, rtol=1e-9, atol=atol)


def random(*parts):
    text = "clock_timing_v1|20260909|" + "|".join(map(str, parts))
    return np.random.default_rng(int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little"))


def integrate(clock, values, n, order):
    def integrand(q):
        times = (np.arange(n * 20)[:, None] + q) / (n * 20)
        return np.array([np.interp(times.ravel(), clock, v) for v in values.T]).reshape(values.shape[1], n, 20, len(q)).transpose(1, 2, 0, 3)

    return fixed_quad(integrand, 0, 1, n=order)[0]


def scoring(x, q):
    time = q.sum(axis=2)
    cell = q.sum(axis=1)
    return {
        "fine_joint": float(np.sum(x * np.log(q))),
        "fine_identity": float(np.sum(x * np.log(q / time[:, :, None]))),
        "coarse_identity": float(np.sum(x.sum(axis=1) * np.log(cell))),
        "fine_timing": float(np.sum(x.sum(axis=2) * np.log(time))),
    }


def record_audit(item, parent, run, n_paths=32, repeats=10):
    tag = item["tag"]
    table = pd.read_csv(run / f"{tag}_scores.csv.gz")
    assert len(table) == n_paths * 4 * repeats * 4
    assert not table.duplicated(["path_id", "condition", "generator", "repeat", "observation"]).any()
    max_error = 0.0
    parent_error = 0.0
    checked = 0
    min_excess = float("inf")
    for i in range(n_paths):
        with np.load(parent / f"{tag}_p{i:03d}.npz") as z, np.load(run / f"{tag}_p{i:03d}.npz") as saved:
            totals = z["totals"]
            n = len(totals)
            candidates = {}
            close(saved["totals"], totals)
            for model in ["physical", "neural"]:
                rates = integrate(z[f"clock_{model}"], z["path_rates"], n, 16)
                hi = integrate(z[f"clock_{model}"], z["path_rates"], n, 32)
                close(saved[f"rates_{model}"], rates)
                q = rates / rates.sum(axis=(1, 2), keepdims=True)
                h = hi / hi.sum(axis=(1, 2), keepdims=True)
                candidates[model] = q
                max_error = max(max_error, float(np.max(abs(q - h))))
                old = z[f"bin_rates_{model}"]
                old = old / old.sum(axis=1, keepdims=True)
                parent_error = max(parent_error, float(np.max(abs(q.sum(axis=1) - old))))
            for condition in ["exact", "gain_drift"]:
                for truth in ["physical", "neural"]:
                    q = candidates[truth] * (1 if condition == "exact" else z["gains"])
                    q = q / q.sum(axis=(1, 2), keepdims=True)
                    expected = {model: scoring(q * totals[:, None, None], v) for model, v in candidates.items()}
                    if condition == "exact":
                        wrong = "physical" if truth == "neural" else "neural"
                        kl = {mode: expected[truth][mode] - expected[wrong][mode] for mode in expected[truth]}
                        close(kl["fine_joint"], kl["fine_identity"] + kl["fine_timing"])
                        assert min(kl.values()) >= -1e-8
                        min_excess = min(min_excess, kl["fine_joint"] - kl["coarse_identity"])
                    for repeat in range(repeats):
                        gen = random(tag, i, condition, truth, repeat)
                        x = np.array([gen.multinomial(int(t), v.ravel()).reshape(v.shape) for t, v in zip(totals, q, strict=True)])
                        close(saved[f"counts_{condition}_{truth}"][repeat], x)
                        close(x.sum(axis=(1, 2)), totals)
                        score = {k: scoring(x, p) for k, p in candidates.items()}
                        for k in candidates:
                            close(score[k]["fine_joint"], score[k]["fine_identity"] + score[k]["fine_timing"])
                        rows = table[(table.path_id == i) & (table.condition == condition) & (table.generator == truth) & (table.repeat == repeat)]
                        assert set(rows.observation) == set(score["physical"])
                        for row in rows.itertuples():
                            delta = score["neural"][row.observation] - score["physical"][row.observation]
                            ex = expected["neural"][row.observation] - expected["physical"][row.observation]
                            directed = delta if truth == "neural" else -delta
                            close(row.log_score_neural, score["neural"][row.observation])
                            close(row.log_score_physical, score["physical"][row.observation])
                            close(row.delta_neural_minus_physical, delta)
                            close(row.oracle_correct, 0.5 if abs(directed) < 1e-10 else float(directed > 0))
                            close(row.expected_signed_margin, ex if truth == "neural" else -ex)
                            assert row.n_bins == n and row.n_spikes == totals.sum()
                            checked += 1
    assert min_excess >= -1e-9
    return {
        "tag": tag,
        "scores_checked": checked,
        "max_probability_error": max_error,
        "max_parent_marginal_error": parent_error,
        "minimum_exact_data_processing_excess": min_excess,
    }


def audit(args):
    run = args.run_dir
    mp = run / "clock_timing_manifest.json"
    m = json.loads(mp.read_text())
    assert m["status"] == "complete" and len(m["completed"]) == 33 and not m["real_events_rescored"]
    parent = Path(m["parent_dir"])
    pm = parent / "literal_clock_manifest.json"
    assert file_sha256(pm) == m["input_file_sha256"]["parent_manifest"]
    for folder, hashes in [(run, m["output_sha256"]), (parent, json.loads(pm.read_text())["output_sha256"])]:
        for f, h in hashes.items():
            assert file_sha256(folder / f) == h
    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(record_audit, x, parent, run) for x in m["completed"]]
        for f in as_completed(jobs):
            records.append(f.result())
            print(records[-1], flush=True)
    table = pd.concat([pd.read_csv(run / f"{x['tag']}_scores.csv.gz") for x in m["completed"]], ignore_index=True)
    fields = ["oracle_correct", "expected_signed_margin"]
    expected = table.groupby(["dataset", "animal", "session", "path_id", "condition", "observation"], as_index=False)[fields].mean()
    for name, keys in [
        ("paths", None),
        ("recordings", ["dataset", "animal", "session", "condition", "observation"]),
        ("animals", ["dataset", "animal", "condition", "observation"]),
        ("summary", ["dataset", "condition", "observation"]),
    ]:
        if keys:
            expected = expected.groupby(keys, as_index=False)[fields].mean()
        actual = pd.read_csv(run / f"clock_timing_{name}.csv")
        sort = [c for c in expected.columns if c not in fields]
        close(expected.sort_values(sort)[fields], actual.sort_values(sort)[fields])
    animal = pd.read_csv(run / "clock_timing_animals.csv")
    summary = pd.read_csv(run / "clock_timing_summary.csv")
    for row in summary.itertuples():
        sub = animal[(animal.dataset == row.dataset) & (animal.condition == row.condition) & (animal.observation == row.observation)]
        close(row.min_animal_accuracy, sub.oracle_correct.min())
        assert row.practical_pass == bool(row.oracle_correct >= 0.80 and sub.oracle_correct.min() > 0.50)
    paired = pd.read_csv(run / "clock_timing_paired_animals.csv")
    for row in paired.itertuples():
        sub = animal[(animal.dataset == row.dataset) & (animal.animal == row.animal) & (animal.condition == row.condition)].set_index("observation").oracle_correct
        for mode in sub.index:
            close(getattr(row, mode), sub[mode])
        close(row.identity_timing_gain, sub.fine_identity - sub.coarse_identity)
        close(row.joint_timing_gain, sub.fine_joint - sub.coarse_identity)
    p = build_script_provenance(input_paths={"run_manifest": mp, "verifier": Path(__file__)}, cwd=ROOT)
    p.update(
        status="pass",
        records=records,
        scores_checked=sum(x["scores_checked"] for x in records),
        max_probability_error=max(x["max_probability_error"] for x in records),
        max_parent_marginal_error=max(x["max_parent_marginal_error"] for x in records),
        minimum_exact_data_processing_excess=min(x["minimum_exact_data_processing_excess"] for x in records),
    )
    p["numerical_readiness_pass"] = max(p["max_probability_error"], p["max_parent_marginal_error"]) <= 1e-4
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "clock_timing_audit.json").write_text(json.dumps(p, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    audit(parser.parse_args())
