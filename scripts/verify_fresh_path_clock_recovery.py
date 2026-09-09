#!/usr/bin/env python3
"""Independently regenerate fresh observations, certify integration and fits."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np
import pandas as pd
from scipy.spatial import Delaunay

from scripts._provenance import build_script_provenance, file_sha256
from scripts.diagnose_exact_path_clock_pooling import certify_fit
from scripts.run_exact_path_clock_recovery import TAGS
from scripts.run_fresh_path_clock_recovery import TEACHERS
from scripts.verify_dense_path_clock_recovery import integral_bins
from scripts.verify_exact_path_clock_recovery import audit_block, audit_merge
from scripts.verify_unknown_path_clock_populations import close


def reference_event(totals, model, centers, rates, tri, descriptors, families, generator):
    n = len(totals)
    if model == "stationary":
        latent = np.full(n, generator.integers(len(centers)), dtype=np.int32)
        probabilities = rates[:, latent].T.copy()
        probabilities /= probabilities.sum(axis=1, keepdims=True)
    else:
        ids = []
        for _ in range(n if model.endswith("reset") else 1):
            family = families[generator.integers(2)]
            ids.append(family[generator.integers(len(family))])
        latent = np.array(ids if len(ids) > 1 else ids * n, dtype=np.int32)
        probabilities = np.empty((n, rates.shape[0]))
        for index in np.unique(latent):
            a, b, sign = descriptors[index]
            delta = centers[b] - centers[a]
            length = np.linalg.norm(delta)
            u = np.linspace(0, 1, 801)
            points = centers[a] + u[:, None] * delta + (0.2 * length * sign) * np.sin(np.pi * u[:, None]) * (np.array([-delta[1], delta[0]]) / length)
            cells = tri.find_simplex(points)
            if (cells < 0).any():
                raise ValueError("sampled prior path leaves support")
            bary = np.einsum("nij,nj->ni", tri.transform[cells, :2], points - tri.transform[cells, 2])
            bary = np.c_[bary, 1 - bary.sum(axis=1)]
            values = np.einsum("nk,cnk->nc", bary, rates[:, tri.simplices[cells]])
            if model.startswith("neural"):
                p = values / values.sum(axis=1, keepdims=True)
                increments = np.sqrt(np.sum(np.diff(np.sqrt(p), axis=0) ** 2, axis=1) / 2)
            else:
                increments = np.linalg.norm(np.diff(points, axis=0), axis=1)
            clock = np.r_[0, increments.cumsum()]
            average = integral_bins(clock / clock[-1], values, n)
            p = average / average.sum(axis=1, keepdims=True)
            probabilities[latent == index] = p[latent == index]
    counts = np.array([generator.multinomial(int(k), p) for k, p in zip(totals, probabilities, strict=True)], dtype=np.int32)
    return counts, latent


def audit_generation(record, manifest, run):
    tag, repeat = record["tag"], record["repeat"]
    meta = pd.read_csv(run / f"{tag}_observations.csv.gz")
    meta = meta[meta.repeat.eq(repeat)].sort_values("row_index")
    source = Path(manifest["source_dir"])
    with np.load(source / f"{tag}_cache.npz") as z:
        centers, rates = z["centers"], z["rates"]
    tri = Delaunay(centers)
    descriptors = np.load(run / f"{tag}_prior_descriptors.npy", mmap_mode="r")
    families = [np.flatnonzero(descriptors[:, 2] == 0), np.flatnonzero(descriptors[:, 2] != 0)]
    with np.load(Path(manifest["parent_dir"]) / f"{tag}_r{repeat:03d}_observations.npz") as z:
        original, original_offsets = z["counts"], z["offsets"]
    with np.load(run / record["file"]) as z:
        counts, latent, offsets, rows = z["counts"], z["latent"], z["offsets"], z["row_index"]
    np.testing.assert_array_equal(rows, meta.row_index)
    generated_bins = 0
    for i, row in enumerate(meta.itertuples()):
        a, b = original_offsets[row.observation_index : row.observation_index + 2]
        totals = original[a:b].sum(axis=1)
        parts = (tag, repeat, row.source_teacher, row.event_in_population, row.scenario)
        digest = hashlib.sha256(("fresh_path_clocks_v1|20260910|" + "|".join(map(str, parts))).encode()).digest()
        generator = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        expected, ids = reference_event(totals, row.generator, centers, rates, tri, descriptors, families, generator)
        a, b = offsets[i : i + 2]
        np.testing.assert_array_equal(counts[a:b].sum(axis=1), totals)
        np.testing.assert_array_equal(latent[a:b], ids)
        np.testing.assert_array_equal(counts[a:b], expected)
        generated_bins += len(totals)
    return {"tag": tag, "repeat": repeat, "count_vectors_regenerated": generated_bins, "observations_regenerated": len(meta)}


def audit_fit_batch(repeat, fits, run):
    checked, error = 0, 0.0
    for tag in TAGS:
        meta = pd.read_csv(run / f"{tag}_observations.csv.gz")
        if repeat >= 0:
            meta = meta[meta.repeat.eq(repeat)]
        scores = np.load(run / f"{tag}_exact_scores.npy", mmap_mode="r")
        subset = fits[fits.repeat.eq(repeat) & fits.session.eq(meta.session.iloc[0])]
        if len(subset) != 6:
            raise ValueError("six unique source conditions required")
        for _, row in subset.iterrows():
            group = meta[meta.source_teacher.eq(row.teacher) & meta.scenario.eq(row.scenario)]
            if len(group) != row.n_events or row.n_events != (128 if repeat >= 0 else 6400):
                raise ValueError("incomplete fit denominator")
            error = max(error, certify_fit(scores[group.row_index.to_numpy()], row.to_dict()))
            checked += 1
    return {"fits_certified": checked, "max_profile_error": error}


def check_summary(fits, summary, gates):
    keys = ["dataset", "teacher", "scenario", "repeat", "support"]
    expected = set(product(("pfeiffer_foster", "tanni2022"), TEACHERS, (0.25, 0.5, 0.75), range(50), ("exhaustive",)))
    if len(fits) != 600 or set(fits[keys].itertuples(index=False, name=None)) != expected or len(summary) != 12 or len(gates) != 4:
        raise ValueError("complete primary fit and summary grid required")
    expected_summary = set(product(("pfeiffer_foster", "tanni2022"), TEACHERS, (0.25, 0.5, 0.75)))
    if set(summary[["dataset", "teacher", "scenario"]].itertuples(index=False, name=None)) != expected_summary:
        raise ValueError("duplicate or missing summary condition")
    for row in summary.itertuples():
        group = fits[fits.dataset.eq(row.dataset) & fits.teacher.eq(row.teacher) & fits.scenario.eq(row.scenario)]
        close(
            [row.mean_estimate, row.bias, row.rmse, row.median_interval_width, row.mean_coherent_weight],
            [
                group.phi_hat.mean(),
                (group.phi_hat - group.scenario).mean(),
                np.sqrt(np.mean((group.phi_hat - group.scenario) ** 2)),
                (group.phi_high - group.phi_low).median(),
                group.coherent_weight.mean(),
            ],
        )
        for name, success in (
            ("covered", (group.phi_low <= group.scenario + 1e-8) & (group.phi_high >= group.scenario - 1e-8)),
            ("directional_claim", (group.phi_low > 0.5) | (group.phi_high < 0.5)),
            ("correct_direction", ((group.scenario > 0.5) & (group.phi_low > 0.5)) | ((group.scenario < 0.5) & (group.phi_high < 0.5))),
        ):
            p, n, z = success.mean(), len(success), 1.959963984540054
            center = (p + z * z / (2 * n)) / (1 + z * z / n)
            radius = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
            close([getattr(row, name + s) for s in ("_fraction", "_mc_low", "_mc_high")], [p, center - radius, center + radius])
    if set(gates[["dataset", "teacher"]].itertuples(index=False, name=None)) != set(product(("pfeiffer_foster", "tanni2022"), TEACHERS)):
        raise ValueError("duplicate or missing gate condition")
    for row in gates.itertuples():
        group = summary[summary.dataset.eq(row.dataset) & summary.teacher.eq(row.teacher)]
        bias, coverage, power, false = (
            group.bias.abs().max(),
            group.covered_fraction.min(),
            group[group.scenario.ne(0.5)].correct_direction_fraction.min(),
            group[group.scenario.eq(0.5)].directional_claim_fraction.iloc[0],
        )
        close([row.max_absolute_bias, row.min_coverage, row.min_direction_power, row.null_false_direction_fraction], [bias, coverage, power, false])
        if row.practical_pass != bool(bias <= 0.1 and coverage >= 0.9 and power >= 0.8 and false <= 0.05):
            raise ValueError("incorrect recovery gate")


def audit(args):
    run, out = args.run_dir, args.output_dir
    mp = run / "fresh_path_clock_manifest.json"
    m = json.loads(mp.read_text())
    if (
        m["status"] != "complete"
        or m["n_observations"] != 76800
        or m["n_fits"] != 600
        or m["n_pooled_fits"] != 12
        or not m["fresh_path_per_observation"]
        or m["finite_teacher_bank"]
    ):
        raise ValueError("completed fresh-path calibration required")
    for key, p in m["input_file_paths"].items():
        if file_sha256(Path(p)) != m["input_file_sha256"][key]:
            raise ValueError("input provenance changed")
    for name, digest in m["output_sha256"].items():
        if file_sha256(run / name) != digest:
            raise ValueError("output hash mismatch")
    previous = Path(m["previous_dir"])
    old = json.loads((previous / "exact_path_clock_manifest.json").read_text())
    for name, digest in old["output_sha256"].items():
        if file_sha256(previous / name) != digest:
            raise ValueError("prior artifacts changed")
    source = Path(m["source_dir"])
    source_hashes = json.loads((source / "conditional_2d_manifest.json").read_text())["output_sha256"]
    parent = Path(m["parent_dir"])
    parent_manifest = json.loads((parent / "unknown_path_clock_manifest.json").read_text())
    if file_sha256(source / "conditional_2d_manifest.json") != parent_manifest["input_file_sha256"]["source_manifest"]:
        raise ValueError("source encoder manifest changed")
    parent_hashes = parent_manifest["output_sha256"]
    for tag in TAGS:
        p = source / f"{tag}_cache.npz"
        if file_sha256(p) != source_hashes[p.name]:
            raise ValueError("map hash mismatch")
        meta = pd.read_csv(run / f"{tag}_observations.csv.gz")
        expected = pd.read_csv(previous / f"{tag}_observations.csv.gz")
        expected["source_teacher"] = expected.source_teacher.map({"original_bank_0": TEACHERS[0], "original_bank_1": TEACHERS[1]})
        pd.testing.assert_frame_equal(meta, expected)
        old_descriptors = []
        for b in sorted((b for b in old["blocks"] if b["tag"] == tag), key=lambda b: b["start"]):
            with np.load(previous / b["file"]) as z:
                old_descriptors.append(z["descriptors"])
        np.testing.assert_array_equal(np.load(run / f"{tag}_prior_descriptors.npy"), np.concatenate(old_descriptors))
        with np.load(run / f"{tag}_counts.npz") as z:
            groups = {int(k.split("_")[1]): (z[k.replace("counts_", "index_")], z[k]) for k in z.files if k.startswith("counts_")}
        locations = {int(row): (n, i) for n, (index, _x) in groups.items() for i, row in enumerate(index)}
        if set(locations) != set(range(38400)):
            raise ValueError("incomplete scoring count array")
        for repeat in range(50):
            p = parent / f"{tag}_r{repeat:03d}_observations.npz"
            if file_sha256(p) != parent_hashes[p.name]:
                raise ValueError("source profile hash mismatch")
            with np.load(run / f"{tag}_fresh_r{repeat:03d}.npz") as z:
                for row, a, b in zip(z["row_index"], z["offsets"][:-1], z["offsets"][1:], strict=True):
                    n, i = locations[int(row)]
                    np.testing.assert_array_equal(groups[n][1][i], z["counts"][a:b])
    out.mkdir(parents=True, exist_ok=False)
    result = build_script_provenance(input_paths={"run_manifest": mp}, cwd=ROOT) | {
        "status": "running",
        "generations": [],
        "blocks": [],
        "all_geometry_likelihoods_independently_recomputed": False,
    }
    path = out / "fresh_path_clock_audit.json"
    try:
        if len(m["generations"]) != 100 or {(r["tag"], r["repeat"]) for r in m["generations"]} != set(product(TAGS, range(50))):
            raise ValueError("all 100 fresh source batches required")
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for f in as_completed([pool.submit(audit_generation, r, m, run) for r in m["generations"]]):
                record = f.result()
                result["generations"].append(record)
                print(record, flush=True)
                path.write_text(json.dumps(result, indent=2) + "\n")
        selected = []
        for tag in TAGS:
            blocks = sorted((b for b in m["blocks"] if b["tag"] == tag), key=lambda b: b["start"])
            selected.extend(blocks[i] for i in (0, len(blocks) // 2, len(blocks) - 1))
            for b in blocks:
                with np.load(run / b["file"]) as z, np.load(previous / b["file"]) as ref:
                    for key in ("descriptors", "proposed", "rejected", "accepted", "start_stop"):
                        np.testing.assert_array_equal(z[key], ref[key])
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for f in as_completed([pool.submit(audit_block, b, source, run) for b in selected]):
                result["blocks"].append(f.result())
                path.write_text(json.dumps(result, indent=2) + "\n")
        result["merges"] = [audit_merge(tag, m, run) for tag in TAGS]
        fits = pd.read_csv(run / "fresh_path_clock_fits.csv")
        pooled = pd.read_csv(run / "fresh_path_clock_pooled_fits.csv")
        check_summary(fits, pd.read_csv(run / "fresh_path_clock_summary.csv"), pd.read_csv(run / "fresh_path_clock_gates.csv"))
        if len(pooled) != 12 or pooled.duplicated(["dataset", "teacher", "scenario"]).any() or not pooled.repeat.eq(-1).all():
            raise ValueError("all distinct pooled fits required")
        combined = pd.concat([fits, pooled], ignore_index=True)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            certificates = list(pool.map(audit_fit_batch, range(-1, 50), [combined] * 51, [run] * 51))
        result.update(
            status="pass",
            observations_regenerated=sum(r["observations_regenerated"] for r in result["generations"]),
            fits_certified=sum(r["fits_certified"] for r in certificates),
            max_score_error=max(b["max_error"] for b in result["blocks"]),
        )
    except BaseException as error:
        result.update(status="failed", error=repr(error))
        raise
    finally:
        path.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    audit(parser.parse_args())
