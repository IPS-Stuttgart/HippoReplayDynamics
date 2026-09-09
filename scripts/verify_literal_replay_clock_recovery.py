#!/usr/bin/env python3
"""Independent algebra, resampling and quadrature audit of literal clock recovery."""

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
from scipy.spatial import Delaunay
from scipy.special import softmax

from scripts._provenance import build_script_provenance, file_sha256


def random(*parts):
    digest = hashlib.sha256(("literal_clock_v1|20260909|" + "|".join(map(str, parts))).encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def close(a, b, atol=1e-8):
    np.testing.assert_allclose(a, b, atol=atol, rtol=1e-9)


def average(t, values, n, order=256):
    def evaluate(q):
        locations = (np.arange(n)[:, None] + q) / n
        result = np.array([np.interp(locations.ravel(), t, v) for v in values.T])
        return result.reshape(values.shape[1], n, len(q)).transpose(1, 0, 2)

    return fixed_quad(evaluate, 0, 1, n=order)[0]


def record_audit(item, source, run, n_paths=32, repeats=5):
    tag = item["tag"]
    with np.load(source / f"{tag}_cache.npz") as z:
        rates, centers = z["rates"], z["centers"]
        profiles = {}
        for k in sorted(k for k in z.files if k.startswith("counts_")):
            event = int(k.split("_")[1])
            dt = np.diff(z[f"edges_{event}"])
            counts = z[k]
            keep = np.isclose(dt, 0.02, atol=1e-8, rtol=0)
            if not keep.all() and not (keep[:-1].all() and dt[-1] < 0.02 + 1e-8):
                raise ValueError("unexpected partial bin")
            counts = counts[keep]
            if len(counts) >= 3:
                profiles[event] = counts.sum(axis=1).astype(int)
    path_table = pd.read_csv(run / f"{tag}_paths.csv")
    scores = pd.read_csv(run / f"{tag}_scores.csv.gz")
    assert len(path_table) == n_paths and len(scores) == n_paths * 4 * repeats * 2
    assert not scores.duplicated(["path_id", "condition", "generator", "repeat", "decoder_grid_cm"]).any()
    selected = random(tag, "profiles").choice(sorted(profiles), n_paths, replace=True)
    gain = np.exp(0.35 * random(tag, "gains").normal(size=len(rates)) - 0.35**2 / 2)
    tri = Delaunay(centers)
    verts = centers[tri.simplices]
    allowed = np.max(np.sqrt(np.sum((verts[:, :, None] - verts[:, None]) ** 2, axis=-1)), axis=(1, 2)) <= np.sqrt(2) * 8 * 1.001
    groups = np.floor((centers - centers.min(axis=0)) / 16 + 1e-9).astype(int)
    unique = np.unique(groups, axis=0)
    coarse = [np.all(groups == group, axis=1) for group in unique]
    grids = {8: (rates.T, centers), 16: (np.vstack([rates[:, g].mean(axis=1) for g in coarse]), np.vstack([centers[g].mean(axis=0) for g in coarse]))}
    maximum_error, flips, checked = 0.0, 0, 0
    for metadata in path_table.itertuples():
        i = metadata.path_id
        totals = profiles[int(selected[i])]
        assert metadata.template_event == selected[i] and metadata.n_bins == len(totals) and metadata.n_spikes == totals.sum()
        with np.load(run / f"{tag}_p{i:03d}.npz") as saved:
            close(saved["totals"], totals)
            close(saved["gains"], gain)
            gen = random(tag, i, "geometry")
            for attempt in range(1, metadata.sampling_attempts + 1):
                a, b = centers[gen.choice(len(centers), 2, replace=False)]
                d = b - a
                length = np.linalg.norm(d)
                if not 40 <= length <= 120:
                    continue
                amp = 0.2 * length * gen.choice([-1, 1]) if i % 2 else 0
                u = np.linspace(0, 1, 801)
                pts = a + u[:, None] * d + amp * np.sin(np.pi * u[:, None]) * np.array([-d[1], d[0]]) / length
                ids = tri.find_simplex(pts)
                valid = (ids >= 0).all() and allowed[ids].all()
                if valid:
                    assert attempt == metadata.sampling_attempts
                    break
            else:
                raise ValueError("no matching sampled path")
            close(saved["points"], pts)
            bary = np.einsum("nij,nj->ni", tri.transform[ids, :2], pts - tri.transform[ids, 2])
            bary = np.c_[bary, 1 - bary.sum(axis=1)]
            pr = np.einsum("nk,cnk->nc", bary, rates[:, tri.simplices[ids]])
            close(pr, saved["path_rates"])
            p = pr / pr.sum(axis=1, keepdims=True)
            s = np.r_[0, np.sqrt(np.sum(np.diff(pts, axis=0) ** 2, axis=1)).cumsum()]
            c = np.r_[0, np.sqrt(0.5 * np.sum(np.diff(np.sqrt(p), axis=0) ** 2, axis=1)).cumsum()]
            close(saved["s"], s)
            close(saved["c"], c)
            close(metadata.path_length_cm, s[-1])
            close(metadata.code_arc_length, c[-1])
            clocks = {"physical": s / s[-1], "neural": c / c[-1]}
            averages, reference, mean_positions = {}, {}, {}
            for model, t in clocks.items():
                close(saved[f"clock_{model}"], t)
                av = average(t, pr, len(totals), 128)
                close(saved[f"bin_rates_{model}"], av)
                hi = average(t, pr, len(totals))
                averages[model] = av / av.sum(axis=1, keepdims=True)
                reference[model] = hi / hi.sum(axis=1, keepdims=True)
                maximum_error = max(maximum_error, float(np.max(abs(averages[model] - reference[model]))))
                mean_positions[model] = average(t, pts, len(totals), 128)
                close(saved[f"position_mean_{model}"], mean_positions[model])
            for condition in ["exact", "gain_drift"]:
                for truth in ["physical", "neural"]:
                    emitted = averages[truth] * (1 if condition == "exact" else gain)
                    emitted /= emitted.sum(axis=1, keepdims=True)
                    for repeat in range(repeats):
                        gen = random(tag, i, condition, truth, repeat, "counts")
                        x = np.array([gen.multinomial(int(n), p) for n, p in zip(totals, emitted, strict=True)])
                        close(saved[f"counts_{condition}_{truth}_{repeat}"], x)
                        exact_delta = float(np.sum(x * (np.log(averages["neural"]) - np.log(averages["physical"]))))
                        hi_delta = float(np.sum(x * (np.log(reference["neural"]) - np.log(reference["physical"]))))
                        flips += int(np.sign(exact_delta) != np.sign(hi_delta))
                        sub = scores[(scores.path_id == i) & (scores.condition == condition) & (scores.generator == truth) & (scores.repeat == repeat)]
                        assert len(sub) == 2
                        for row in sub.itertuples():
                            close(row.delta_neural_minus_physical, exact_delta)
                            for model in clocks:
                                close(getattr(row, f"log_score_{model}"), np.sum(x * np.log(averages[model])))
                            directed = exact_delta if truth == "neural" else -exact_delta
                            close(row.oracle_correct, 0.5 if abs(directed) < 1e-10 else float(directed > 0))
                            rr, xy = grids[row.decoder_grid_cm]
                            posterior = softmax(x @ np.log(rr / rr.sum(axis=1, keepdims=True)).T, axis=1)
                            mean = posterior @ xy
                            maximum = xy[posterior.argmax(axis=1)]
                            width = np.sqrt(np.maximum(posterior @ np.sum(xy**2, axis=1) - np.sum(mean**2, axis=1), 0))
                            steps = np.linalg.norm(np.diff(mean, axis=0), axis=1)
                            close(row.posterior_mean_error_cm, np.mean(np.linalg.norm(mean - mean_positions[truth], axis=1)))
                            close(row.map_error_cm, np.mean(np.linalg.norm(maximum - mean_positions[truth], axis=1)))
                            close(row.posterior_rms_cm, width.mean())
                            close(row.decoded_mean_step_speed_cm_s, steps.mean() / 0.02)
                            close(row.decoded_map_step_speed_cm_s, np.linalg.norm(np.diff(maximum, axis=0), axis=1).mean() / 0.02)
                            close(row.true_bin_mean_step_speed_cm_s, np.linalg.norm(np.diff(mean_positions[truth], axis=0), axis=1).mean() / 0.02)
                            close(row.hard_continuous_step_fraction, np.mean(steps < 20))
                            checked += 1
    return {
        "tag": tag,
        "paths": n_paths,
        "observations": n_paths * 4 * repeats,
        "score_rows_checked": checked,
        "quadrature_max_probability_error": maximum_error,
        "quadrature_label_flips": flips,
    }


def audit(args):
    run = args.run_dir
    mp = run / "literal_clock_manifest.json"
    m = json.loads(mp.read_text())
    assert m["status"] == "complete" and len(m["completed"]) == 33
    assert not m["real_events_rescored"] and m["oracle_knows_path"]
    source = Path(m["source_dir"])
    sm = source / "conditional_2d_manifest.json"
    assert file_sha256(sm) == m["input_file_sha256"]["source_manifest"]
    source_hashes = json.loads(sm.read_text())["output_sha256"]
    for name, digest in m["output_sha256"].items():
        assert file_sha256(run / name) == digest
    for item in m["completed"]:
        name = item["tag"] + "_cache.npz"
        assert file_sha256(source / name) == source_hashes[name]
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(record_audit, item, source, run) for item in m["completed"]]
        for f in as_completed(futures):
            results.append(f.result())
            print(results[-1], flush=True)
    table = pd.concat([pd.read_csv(run / f"{item['tag']}_scores.csv.gz") for item in m["completed"]], ignore_index=True)
    fields = ["oracle_correct", "delta_neural_minus_physical", "posterior_mean_error_cm", "decoded_mean_step_speed_cm_s", "true_arc_speed_cm_s", "clock_separation_rms_cm"]
    expected = table[table.decoder_grid_cm == 8].groupby(["dataset", "animal", "session", "condition", "path_id"], as_index=False)[fields].mean()
    for name, keys in [
        ("paths", None),
        ("recordings", ["dataset", "animal", "session", "condition"]),
        ("animals", ["dataset", "animal", "condition"]),
        ("summary", ["dataset", "condition"]),
    ]:
        if keys:
            expected = expected.groupby(keys, as_index=False)[fields].mean()
        actual = pd.read_csv(run / f"literal_clock_{name}.csv")
        sort = [c for c in expected.columns if c not in fields]
        close(expected.sort_values(sort)[fields], actual.sort_values(sort)[fields])
    summary = pd.read_csv(run / "literal_clock_summary.csv")
    animals = pd.read_csv(run / "literal_clock_animals.csv")
    for row in summary.itertuples():
        subset = animals[(animals.dataset == row.dataset) & (animals.condition == row.condition)]
        close(row.minimum_animal_accuracy, subset.oracle_correct.min())
        assert bool(row.oracle_practical_pass) == bool(row.oracle_correct >= 0.80 and subset.oracle_correct.min() > 0.50)
    p = build_script_provenance(input_paths={"run_manifest": mp, "verifier": Path(__file__)}, cwd=ROOT)
    p.update(
        status="pass",
        completed=results,
        score_rows_checked=sum(x["score_rows_checked"] for x in results),
        quadrature_max_probability_error=max(x["quadrature_max_probability_error"] for x in results),
        quadrature_label_flips=sum(x["quadrature_label_flips"] for x in results),
    )
    p["numerical_readiness_pass"] = p["quadrature_max_probability_error"] <= 1e-4
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "literal_clock_audit.json").write_text(json.dumps(p, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    audit(parser.parse_args())
