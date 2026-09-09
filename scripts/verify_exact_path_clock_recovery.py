#!/usr/bin/env python3
"""Stratified independent score audit, exhaustive merge and mixture certification."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np
import pandas as pd
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist
from scipy.special import logsumexp

from scripts._provenance import build_script_provenance, file_sha256
from scripts.verify_dense_path_clock_recovery import integral_bins
from scripts.verify_unknown_path_clock_populations import certify, close

MODELS = ("physical", "neural", "stationary", "physical_reset", "neural_reset")


def independent_geometries(centers, rates, start, stop):
    tri = Delaunay(centers)
    vertices = centers[tri.simplices]
    allowed = np.max(np.linalg.norm(vertices[:, :, None] - vertices[:, None], axis=-1), axis=(1, 2)) <= np.sqrt(2) * 8 * 1.001
    distance = cdist(centers, centers)
    for a in range(start, stop):
        for b in np.flatnonzero((distance[a] >= 40) & (distance[a] <= 120)):
            for sign in (0, -1, 1):
                delta = centers[b] - centers[a]
                amplitude = 0.2 * distance[a, b] * sign
                u = np.linspace(0, 1, 801)
                points = centers[a] + u[:, None] * delta + amplitude * np.sin(np.pi * u[:, None]) * (np.array([-delta[1], delta[0]]) / distance[a, b])
                ids = tri.find_simplex(points)
                descriptor = (a, int(b), sign)
                if not (ids >= 0).all() or not allowed[ids].all():
                    yield descriptor, None
                    continue
                bary = np.einsum("nij,nj->ni", tri.transform[ids, :2], points - tri.transform[ids, 2])
                bary = np.c_[bary, 1 - bary.sum(axis=1)]
                values = np.einsum("nk,cnk->nc", bary, rates[:, tri.simplices[ids]])
                p = values / values.sum(axis=1, keepdims=True)
                s = np.r_[0, np.linalg.norm(np.diff(points, axis=0), axis=1).cumsum()]
                c = np.r_[0, np.sqrt(0.5 * np.sum(np.diff(np.sqrt(p), axis=0) ** 2, axis=1)).cumsum()]
                yield descriptor, (values, (s / s[-1], c / c[-1]))


def independent_block(centers, rates, groups, start, stop, chunk=128):
    sums, queues, descriptors = {}, [[], []], []
    proposed, rejected = np.zeros(2, int), np.zeros(2, int)
    for n, (_index, x) in groups.items():
        for f, c in product(range(2), repeat=2):
            sums[f"coherent_{f}_{c}_{n}"] = np.full(len(x), -np.inf)
            sums[f"reset_{f}_{c}_{n}"] = np.full((len(x), n), -np.inf)

    def flush(family):
        if not queues[family]:
            return
        for n, (_index, x) in groups.items():
            for clock in range(2):
                p = np.array([integral_bins(clocks[clock], values, n) for values, clocks in queues[family]])
                p /= p.sum(axis=2, keepdims=True)
                ll = np.einsum("etc,htc->eht", x, np.log(p), optimize=True)
                key = f"coherent_{family}_{clock}_{n}"
                sums[key] = np.logaddexp(sums[key], logsumexp(ll.sum(axis=2), axis=1))
                key = f"reset_{family}_{clock}_{n}"
                sums[key] = np.logaddexp(sums[key], logsumexp(ll, axis=1))
        queues[family].clear()

    for descriptor, path in independent_geometries(centers, rates, start, stop):
        family = int(descriptor[2] != 0)
        proposed[family] += 1
        if path is None:
            rejected[family] += 1
            continue
        descriptors.append(descriptor)
        queues[family].append(path)
        if len(queues[family]) == chunk:
            flush(family)
    for family in range(2):
        flush(family)
    return sums | {"descriptors": np.array(descriptors, dtype=np.int32).reshape(-1, 3), "proposed": proposed, "rejected": rejected, "accepted": proposed - rejected}


def audit_block(block, source, run):
    tag = block["tag"]
    metadata = pd.read_csv(run / f"{tag}_observations.csv.gz")
    chosen = metadata[metadata.event_in_population.lt(2)].row_index.to_numpy()
    groups, local = {}, {}
    with np.load(run / f"{tag}_counts.npz") as z:
        for k in z.files:
            if k.startswith("counts_"):
                n = int(k.split("_")[1])
                ids = z[f"index_{n}"]
                keep = np.flatnonzero(np.isin(ids, chosen))
                if len(keep):
                    local[n] = keep
                    groups[n] = ids[keep], z[k][keep]
    with np.load(source / f"{tag}_cache.npz") as z:
        reference = independent_block(z["centers"], z["rates"], groups, block["start"], block["stop"])
    checked, error = 0, 0.0
    with np.load(run / block["file"]) as actual:
        for k in ("descriptors", "proposed", "rejected", "accepted"):
            np.testing.assert_array_equal(actual[k], reference[k])
        for k in reference:
            if k.startswith(("coherent_", "reset_")):
                n = int(k.split("_")[-1])
                a, b = actual[k][local[n]], reference[k]
                close(a, b, atol=1e-7)
                finite = np.isfinite(a) & np.isfinite(b)
                error = max(error, float(np.max(abs(a[finite] - b[finite]), initial=0)))
                checked += a.size
    return {
        "tag": tag,
        "start": block["start"],
        "stop": block["stop"],
        "proposals_regenerated": int(reference["proposed"].sum()),
        "paths_regenerated": len(reference["descriptors"]),
        "partial_log_sums_recomputed": checked,
        "max_error": error,
    }


def audit_merge(tag, manifest, run):
    source = Path(manifest["source_dir"])
    blocks = sorted((b for b in manifest["blocks"] if b["tag"] == tag), key=lambda b: b["start"])
    with np.load(source / f"{tag}_cache.npz") as z:
        rates, centers = z["rates"], z["centers"]
    distance = cdist(centers, centers)
    np.testing.assert_array_equal(np.concatenate([np.arange(b["start"], b["stop"]) for b in blocks]), np.arange(len(centers)))
    descriptors = []
    actual = np.load(run / f"{tag}_exact_scores.npy", mmap_mode="r")
    with ExitStack() as stack:
        parts = [stack.enter_context(np.load(run / b["file"])) for b in blocks]
        for b, part in zip(blocks, parts, strict=True):
            pair_count = ((distance[b["start"] : b["stop"]] >= 40) & (distance[b["start"] : b["stop"]] <= 120)).sum()
            np.testing.assert_array_equal(part["proposed"], [pair_count, 2 * pair_count])
            np.testing.assert_array_equal(part["proposed"] - part["rejected"], part["accepted"])
            desc = part["descriptors"]
            assert np.all((desc[:, 0] >= b["start"]) & (desc[:, 0] < b["stop"]))
            assert np.isin(desc[:, 2], [-1, 0, 1]).all()
            d = distance[desc[:, 0], desc[:, 1]]
            assert ((d >= 40) & (d <= 120)).all()
            np.testing.assert_array_equal(part["accepted"], [(desc[:, 2] == 0).sum(), (desc[:, 2] != 0).sum()])
            descriptors.append(desc)
        desc = np.concatenate(descriptors)
        assert len(np.unique(desc, axis=0)) == len(desc)
        totals = np.sum([p["accepted"] for p in parts], axis=0)
        with np.load(run / f"{tag}_counts.npz") as counts:
            for k in counts.files:
                if not k.startswith("counts_"):
                    continue
                n, x = int(k.split("_")[1]), counts[k]
                index = counts[f"index_{n}"]
                expected = np.empty((len(index), 5))
                for c in range(2):
                    coherent, reset = [], []
                    for f in range(2):
                        coherent.append(logsumexp(np.stack([p[f"coherent_{f}_{c}_{n}"] for p in parts]), axis=0) - np.log(totals[f]) - np.log(2))
                        reset.append(logsumexp(np.stack([p[f"reset_{f}_{c}_{n}"] for p in parts]), axis=0) - np.log(totals[f]) - np.log(2))
                    expected[:, c] = logsumexp(coherent, axis=0)
                    expected[:, c + 3] = logsumexp(reset, axis=0).sum(axis=1)
                static = rates.T / rates.sum(axis=0)[:, None]
                expected[:, 2] = logsumexp(x.sum(axis=1) @ np.log(static).T, axis=1) - np.log(len(static))
                close(actual[index], expected)
    return {"tag": tag, "paths": len(desc), "full_merged_likelihoods_checked": int(actual.size)}


def audit_fit_repeat(repeat, fits, run, dense, tags):
    checked = 0
    for tag in tags:
        metadata = pd.read_csv(run / f"{tag}_observations.csv.gz")
        metadata = metadata[metadata.repeat.eq(repeat)]
        exact = np.load(run / f"{tag}_exact_scores.npy", mmap_mode="r")
        mc = np.load(dense / f"{tag}_scores.npy", mmap_mode="r")
        for row in fits[fits.repeat.eq(repeat) & fits.session.eq(metadata.session.iloc[0])].itertuples():
            index = metadata[metadata.scenario.eq(row.scenario) & metadata.source_teacher.eq(row.teacher)].row_index.to_numpy()
            assert len(index) == row.n_events == 128
            if row.support == "exhaustive":
                ll = exact[index]
            else:
                _, h, b = row.support.split("_")
                ll = mc[int(b[1:]), (1024, 4096, 8192).index(int(h)), index]
            ll = ll - ll.max(axis=1, keepdims=True)
            w = np.array([getattr(row, "weight_" + m) for m in MODELS])
            optimum, _ = certify(ll, w)
            close(row.relative_log_likelihood, optimum)
            close(row.phi_hat, w[1] / w[:2].sum())
            close(row.coherent_weight, w[:2].sum())
            for label, phi in (("low", row.phi_low), ("high", row.phi_high), ("null", 0.5)):
                with np.errstate(divide="ignore"):
                    moving = np.logaddexp(ll[:, 0] + np.log1p(-phi), ll[:, 1] + np.log(phi))
                weights = [getattr(row, label + "_weight_" + m) for m in ("coherent", *MODELS[2:])]
                value, _ = certify(np.column_stack([moving, ll[:, 2:]]), weights)
                close(getattr(row, label + "_profile_log_likelihood"), value)
                lr = 2 * (optimum - value)
                if label == "null":
                    close(row.null_lr, max(0, lr))
                elif phi in (0, 1):
                    assert lr <= 3.841458820694124 + 1e-4
                else:
                    close(lr, 3.841458820694124, atol=2e-4)
            checked += 1
    assert checked == 84
    return checked


def audit(args):
    run = args.run_dir
    mp = run / "exact_path_clock_manifest.json"
    m = json.loads(mp.read_text())
    assert m["status"] == "complete" and m["n_observations"] == 76800 and m["n_fits"] == 4200
    assert not m["all_33_encoders"] and not m["real_events_rescored"] and not m["oracle_knows_path"]
    for key, p in m["input_file_paths"].items():
        assert file_sha256(Path(p)) == m["input_file_sha256"][key]
    parent, dense, source = Path(m["parent_dir"]), Path(m["dense_dir"]), Path(m["source_dir"])
    pm = json.loads((parent / "unknown_path_clock_manifest.json").read_text())
    dm = json.loads((dense / "dense_path_clock_manifest.json").read_text())
    sm = source / "conditional_2d_manifest.json"
    assert file_sha256(sm) == pm["input_file_sha256"]["source_manifest"]
    source_hashes = json.loads(sm.read_text())["output_sha256"]
    for tag in m["tags"]:
        name = tag + "_cache.npz"
        assert file_sha256(source / name) == source_hashes[name]
        for tail in ("scores.npy", "observations.csv.gz"):
            name = f"{tag}_{tail}"
            assert file_sha256(dense / name) == dm["output_sha256"][name]
        metadata = pd.read_csv(run / f"{tag}_observations.csv.gz")
        pd.testing.assert_frame_equal(metadata, pd.read_csv(dense / f"{tag}_observations.csv.gz"))
        with np.load(run / f"{tag}_counts.npz") as z:
            groups = {int(k.split("_")[1]): (z[k.replace("counts_", "index_")], z[k]) for k in z.files if k.startswith("counts_")}
        locations = {int(row): (n, i) for n, (index, _x) in groups.items() for i, row in enumerate(index)}
        assert len(locations) == len(metadata) == 38400
        for repeat in range(50):
            name = f"{tag}_r{repeat:03d}_observations.npz"
            assert file_sha256(parent / name) == pm["output_sha256"][name]
            with np.load(parent / name) as z:
                counts, offsets = z["counts"], z["offsets"]
            for row in metadata[metadata.repeat.eq(repeat)].itertuples():
                a, b = offsets[row.observation_index : row.observation_index + 2]
                n, i = locations[row.row_index]
                np.testing.assert_array_equal(groups[n][1][i], counts[a:b])
    for name, digest in m["output_sha256"].items():
        assert file_sha256(run / name) == digest
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    result = build_script_provenance(input_paths={"run_manifest": mp}, cwd=ROOT) | {"status": "running", "blocks": [], "all_geometry_likelihoods_independently_recomputed": False}
    path = out / "exact_path_clock_audit.json"
    try:
        selected = []
        for tag in m["tags"]:
            blocks = sorted((b for b in m["blocks"] if b["tag"] == tag), key=lambda b: b["start"])
            selected.extend(blocks[i] for i in (0, len(blocks) // 2, len(blocks) - 1))
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for f in as_completed([pool.submit(audit_block, b, Path(m["source_dir"]), run) for b in selected]):
                record = f.result()
                result["blocks"].append(record)
                path.write_text(json.dumps(result, indent=2) + "\n")
                print(record, flush=True)
        result["merges"] = [audit_merge(tag, m, run) for tag in m["tags"]]
        fits = pd.read_csv(run / "exact_path_clock_fits.csv")
        supports = tuple(f"mc_{h}_b{b}" for h in (1024, 4096, 8192) for b in (0, 1)) + ("exhaustive",)
        keys = ["dataset", "teacher", "scenario", "repeat", "support"]
        expected = set(product(("pfeiffer_foster", "tanni2022"), ("original_bank_0", "original_bank_1"), (0.25, 0.5, 0.75), range(50), supports))
        assert len(fits) == 4200 and set(fits[keys].itertuples(index=False, name=None)) == expected
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            result["fits_certified"] = sum(pool.map(audit_fit_repeat, range(50), [fits] * 50, [run] * 50, [Path(m["dense_dir"])] * 50, [m["tags"]] * 50))
        summary = pd.read_csv(run / "exact_path_clock_summary.csv")
        gates = pd.read_csv(run / "exact_path_clock_gates.csv")
        assert len(summary) == 84 and len(gates) == 28
        for row in summary.itertuples():
            g = fits[fits.dataset.eq(row.dataset) & fits.teacher.eq(row.teacher) & fits.scenario.eq(row.scenario) & fits.support.eq(row.support)]
            close(
                [row.mean_estimate, row.bias, row.rmse, row.median_interval_width, row.mean_coherent_weight],
                [g.phi_hat.mean(), (g.phi_hat - g.scenario).mean(), np.sqrt(np.mean((g.phi_hat - g.scenario) ** 2)), (g.phi_high - g.phi_low).median(), g.coherent_weight.mean()],
            )
            for name, values in (
                ("covered", (g.phi_low <= g.scenario + 1e-8) & (g.phi_high >= g.scenario - 1e-8)),
                ("directional_claim", (g.phi_low > 0.5) | (g.phi_high < 0.5)),
                ("correct_direction", ((g.scenario > 0.5) & (g.phi_low > 0.5)) | ((g.scenario < 0.5) & (g.phi_high < 0.5))),
            ):
                fraction, n, z = values.mean(), len(values), 1.959963984540054
                center = (fraction + z * z / (2 * n)) / (1 + z * z / n)
                radius = z * np.sqrt(fraction * (1 - fraction) / n + z * z / (4 * n * n)) / (1 + z * z / n)
                close([getattr(row, name + s) for s in ("_fraction", "_mc_low", "_mc_high")], [fraction, center - radius, center + radius])
        for row in gates.itertuples():
            g = summary[summary.dataset.eq(row.dataset) & summary.teacher.eq(row.teacher) & summary.support.eq(row.support)]
            bias, coverage = g.bias.abs().max(), g.covered_fraction.min()
            power = g[g.scenario.ne(0.5)].correct_direction_fraction.min()
            false = g[g.scenario.eq(0.5)].directional_claim_fraction.iloc[0]
            close([row.max_absolute_bias, row.min_coverage, row.min_direction_power, row.null_false_direction_fraction], [bias, coverage, power, false])
            assert row.practical_pass == bool(bias <= 0.1 and coverage >= 0.9 and power >= 0.8 and false <= 0.05)
        result.update(status="pass", max_score_error=max(b["max_error"] for b in result["blocks"]))
    except BaseException as error:
        result.update(status="failed", error=repr(error))
        raise
    finally:
        path.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    audit(parser.parse_args())
