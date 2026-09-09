#!/usr/bin/env python3
"""Independent geometry/subset-score audit and complete mixture certification."""

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
from scipy.spatial import Delaunay
from scipy.special import logsumexp

from scripts._provenance import build_script_provenance, file_sha256
from scripts.verify_unknown_path_clock_populations import certify, close, integral_bins

MODELS = ("physical", "neural", "stationary", "physical_reset", "neural_reset")


def random(*parts):
    digest = hashlib.sha256(("dense_path_clocks_v1|20260909|" + "|".join(map(str, parts))).encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def record_audit(item, source, parent, run, supports=(1024, 4096, 8192), banks=(0, 1), repeats=50, chunk=128):
    tag = item["tag"]
    metadata = pd.read_csv(run / f"{tag}_observations.csv.gz")
    paths = pd.read_csv(run / f"{tag}_paths.csv.gz")
    scores = np.load(run / f"{tag}_scores.npy", mmap_mode="r")
    assert scores.shape == (len(banks), len(supports), len(metadata), 5) and np.isfinite(scores).all()
    assert len(paths) == len(banks) * max(supports) and not paths.duplicated(["candidate_bank", "path_index"]).any()
    selected = metadata[metadata.event_in_population.lt(2)].copy()
    observations, parent_rows = {}, []
    for repeat in range(repeats):
        original = pd.read_csv(parent / f"{tag}_r{repeat:03d}_scores.csv.gz")
        original = original[original.support.eq(256)].sort_values("observation_index").drop(columns=[c for c in original if c.startswith("score_") or c == "support"])
        parent_rows.append(original)
        with np.load(parent / f"{tag}_r{repeat:03d}_observations.npz") as z:
            for row in selected[selected.repeat.eq(repeat)].itertuples():
                a, b = z["offsets"][row.observation_index : row.observation_index + 2]
                observations[row.row_index] = z["counts"][a:b]
    original = pd.concat(parent_rows, ignore_index=True).rename(columns={"teacher": "source_teacher"})
    original["source_teacher"] = original.source_teacher.map({"matched": "original_bank_0", "independent": "original_bank_1"})
    original["row_index"] = np.arange(len(original))
    pd.testing.assert_frame_equal(metadata[original.columns], original, check_dtype=False)
    assert len(selected) == repeats * 12
    groups = {n: (g.row_index.to_numpy(), np.stack([observations[i] for i in g.row_index])) for n, g in selected.groupby("n_bins")}
    with np.load(source / f"{tag}_cache.npz") as z:
        rates, centers = z["rates"], z["centers"]
    static = rates.T / rates.sum(axis=0)[:, None]
    tri = Delaunay(centers)
    vertices = centers[tri.simplices]
    allowed = np.max(np.linalg.norm(vertices[:, :, None] - vertices[:, None], axis=-1), axis=(1, 2)) <= np.sqrt(2) * 8 * 1.001
    checked, maximum_error = 0, 0.0
    for bi, bank in enumerate(banks):
        descriptors = paths[paths.candidate_bank.eq(bank)].sort_values("path_index")
        np.testing.assert_array_equal(descriptors.path_index, np.arange(max(supports)))
        coherent = {n: np.full((2, len(x)), -np.inf) for n, (_index, x) in groups.items()}
        reset = {n: np.full((2, len(x), n), -np.inf) for n, (_index, x) in groups.items()}
        for start in range(0, max(supports), chunk):
            library = []
            for row in descriptors.iloc[start : start + chunk].itertuples():
                generator = random(tag, bank, row.path_index, "geometry")
                for attempt in range(1, row.attempts + 1):
                    a, b = centers[generator.choice(len(centers), 2, replace=False)]
                    delta = b - a
                    distance = np.linalg.norm(delta)
                    if not 40 <= distance <= 120:
                        continue
                    amplitude = 0.2 * distance * generator.choice([-1, 1]) if row.path_index % 2 else 0
                    u = np.linspace(0, 1, 801)
                    pts = a + u[:, None] * delta + amplitude * np.sin(np.pi * u[:, None]) * np.array([-delta[1], delta[0]]) / distance
                    ids = tri.find_simplex(pts)
                    if (ids >= 0).all() and allowed[ids].all():
                        assert attempt == row.attempts
                        break
                else:
                    raise ValueError("geometry regeneration failed")
                close([row.start_x, row.start_y, row.end_x, row.end_y, row.amplitude], [*a, *b, amplitude])
                bary = np.einsum("nij,nj->ni", tri.transform[ids, :2], pts - tri.transform[ids, 2])
                bary = np.c_[bary, 1 - bary.sum(axis=1)]
                values = np.einsum("nk,cnk->nc", bary, rates[:, tri.simplices[ids]])
                p = values / values.sum(axis=1, keepdims=True)
                s = np.r_[0, np.linalg.norm(np.diff(pts, axis=0), axis=1).cumsum()]
                c = np.r_[0, np.sqrt(0.5 * np.sum(np.diff(np.sqrt(p), axis=0) ** 2, axis=1)).cumsum()]
                close([row.arc_cm, row.code_arc], [s[-1], c[-1]])
                library.append((values, (s / s[-1], c / c[-1])))
            for n, (index, counts) in groups.items():
                for mi in range(2):
                    p = np.array([integral_bins(clock[mi], values, n) for values, clock in library])
                    p /= p.sum(axis=2, keepdims=True)
                    per_bin = np.einsum("etc,htc->eht", counts, np.log(p), optimize=True)
                    coherent[n][mi] = np.logaddexp(coherent[n][mi], logsumexp(per_bin.sum(axis=2), axis=1))
                    reset[n][mi] = np.logaddexp(reset[n][mi], logsumexp(per_bin, axis=1))
                if start + chunk in supports:
                    h = start + chunk
                    si = supports.index(h)
                    expected = np.column_stack(
                        [
                            coherent[n][0] - np.log(h),
                            coherent[n][1] - np.log(h),
                            logsumexp(counts.sum(axis=1) @ np.log(static).T, axis=1) - np.log(len(static)),
                            (reset[n][0] - np.log(h)).sum(axis=1),
                            (reset[n][1] - np.log(h)).sum(axis=1),
                        ]
                    )
                    actual = scores[bi, si, index]
                    close(expected, actual, atol=1e-7)
                    maximum_error = max(maximum_error, float(np.max(abs(expected - actual))))
                    checked += actual.size
    return {
        "tag": tag,
        "observations_in_run": len(metadata),
        "observations_audited": len(selected),
        "paths_regenerated": len(paths),
        "likelihoods_recomputed": checked,
        "max_score_error": maximum_error,
    }


def certify_repeat(repeat, items, run, fits, supports=(1024, 4096, 8192), banks=(0, 1)):
    tables = []
    for item in items:
        metadata = pd.read_csv(run / f"{item['tag']}_observations.csv.gz")
        subset = metadata[metadata.repeat.eq(repeat)]
        scores = np.load(run / f"{item['tag']}_scores.npy", mmap_mode="r")
        for bi, bank in enumerate(banks):
            for si, h in enumerate(supports):
                frame = subset.assign(candidate_bank=bank, support=h)
                frame[["score_" + m for m in MODELS]] = scores[bi, si, subset.row_index.to_numpy()]
                tables.append(frame)
    full = pd.concat(tables, ignore_index=True)
    checked = 0
    for row in fits[fits.repeat.eq(repeat)].itertuples():
        group = full[
            full.dataset.eq(row.dataset)
            & full.source_teacher.eq(row.source_teacher)
            & full.scenario.eq(row.scenario)
            & full.candidate_bank.eq(row.candidate_bank)
            & full.support.eq(row.support)
        ]
        assert len(group) == row.n_events and group.animal.nunique() == row.n_animals and group.session.nunique() == row.n_recordings
        ll = group[["score_" + m for m in MODELS]].to_numpy()
        ll -= ll.max(axis=1, keepdims=True)
        w = np.array([getattr(row, "weight_" + m) for m in MODELS])
        optimum, _ = certify(ll, w)
        close(row.relative_log_likelihood, optimum)
        close(row.coherent_weight, w[:2].sum())
        close(row.phi_hat, w[1] / w[:2].sum())
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
        assert row.direction == ("neural" if row.phi_low > 0.5 else "physical" if row.phi_high < 0.5 else "undetermined")
        checked += 1
    assert checked == 72
    return checked


def verify_summaries(run, fits):
    summary = pd.read_csv(run / "dense_path_clock_summary.csv")
    gates = pd.read_csv(run / "dense_path_clock_gates.csv")
    convergence = pd.read_csv(run / "dense_path_clock_convergence.csv")
    assert len(summary) == 72 and len(gates) == 24 and len(convergence) == 8
    for row in summary.itertuples():
        group = fits[
            fits.dataset.eq(row.dataset)
            & fits.source_teacher.eq(row.source_teacher)
            & fits.scenario.eq(row.scenario)
            & fits.support.eq(row.support)
            & fits.candidate_bank.eq(row.candidate_bank)
        ]
        assert len(group) == 50
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
        for name, values in (
            ("covered", (group.phi_low <= group.scenario + 1e-8) & (group.phi_high >= group.scenario - 1e-8)),
            ("directional_claim", (group.phi_low > 0.5) | (group.phi_high < 0.5)),
            ("correct_direction", ((group.scenario > 0.5) & (group.phi_low > 0.5)) | ((group.scenario < 0.5) & (group.phi_high < 0.5))),
        ):
            fraction, n, z = values.mean(), len(values), 1.959963984540054
            center = (fraction + z * z / (2 * n)) / (1 + z * z / n)
            radius = z * np.sqrt(fraction * (1 - fraction) / n + z * z / (4 * n * n)) / (1 + z * z / n)
            close(getattr(row, name + "_fraction"), fraction)
            close(getattr(row, name + "_mc_low"), center - radius)
            close(getattr(row, name + "_mc_high"), center + radius)
    for row in gates.itertuples():
        s = summary[
            summary.dataset.eq(row.dataset) & summary.source_teacher.eq(row.source_teacher) & summary.support.eq(row.support) & summary.candidate_bank.eq(row.candidate_bank)
        ]
        bias, coverage = s.bias.abs().max(), s.covered_fraction.min()
        power = s[s.scenario.ne(0.5)].correct_direction_fraction.min()
        false = s[s.scenario.eq(0.5)].directional_claim_fraction.iloc[0]
        close([row.max_absolute_bias, row.min_coverage, row.min_direction_power, row.null_false_direction_fraction], [bias, coverage, power, false])
        assert row.practical_pass == bool(bias <= 0.1 and coverage >= 0.9 and power >= 0.8 and false <= 0.05)
    for row in convergence.itertuples():
        g = fits[fits.dataset.eq(row.dataset) & fits.source_teacher.eq(row.source_teacher)]
        high = g[g.support.eq(8192)]
        a, b = [high[high.candidate_bank.eq(bank)].set_index(["scenario", "repeat"]).sort_index() for bank in (0, 1)]
        d = abs(a.phi_hat - b.phi_hat)
        hi = high[high.candidate_bank.eq(row.candidate_bank)].set_index(["scenario", "repeat"]).sort_index()
        lo = g[g.support.eq(4096) & g.candidate_bank.eq(row.candidate_bank)].set_index(["scenario", "repeat"]).sort_index()
        shift = abs(hi.phi_hat - lo.phi_hat)
        interval = np.maximum(abs(hi.phi_low - lo.phi_low), abs(hi.phi_high - lo.phi_high))
        close(
            [
                row.median_cross_bank_phi_difference,
                row.p95_cross_bank_phi_difference,
                row.median_support_phi_shift,
                row.p95_support_phi_shift,
                row.median_support_interval_endpoint_shift,
            ],
            [d.median(), d.quantile(0.95), shift.median(), shift.quantile(0.95), interval.median()],
        )
        assert row.integration_stable == bool(d.median() <= 0.025 and shift.median() <= 0.025 and interval.median() <= 0.05)


def audit(args):
    run = args.run_dir
    mp = run / "dense_path_clock_manifest.json"
    m = json.loads(mp.read_text())
    assert m["status"] == "complete" and len(m["completed"]) == 33 and m["observations"] == 1267200 and m["score_rows"] == 7603200 and m["n_fits"] == 3600
    assert not m["oracle_knows_path"] and not m["real_events_rescored"] and m["all_scorer_libraries_independent"]
    parent, source = Path(m["parent_dir"]), Path(m["source_dir"])
    parent_manifest = parent / "unknown_path_clock_manifest.json"
    assert file_sha256(parent_manifest) == m["input_file_sha256"]["parent_manifest"]
    pm = json.loads(parent_manifest.read_text())
    sm = source / "conditional_2d_manifest.json"
    assert file_sha256(sm) == pm["input_file_sha256"]["source_manifest"]
    source_hashes = json.loads(sm.read_text())["output_sha256"]
    for item in m["completed"]:
        name = item["tag"] + "_cache.npz"
        assert file_sha256(source / name) == source_hashes[name]
        for repeat in range(50):
            for suffix in ("scores.csv.gz", "observations.npz"):
                name = f"{item['tag']}_r{repeat:03d}_{suffix}"
                assert file_sha256(parent / name) == pm["output_sha256"][name]
    for name, digest in m["output_sha256"].items():
        assert file_sha256(run / name) == digest
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    result = build_script_provenance(input_paths={"run_manifest": mp}, cwd=ROOT) | {
        "status": "running",
        "records": [],
        "all_likelihoods_recomputed": False,
        "audit_selection": "first 2 frozen event indices per source teacher/scenario/replicate",
    }
    path = out / "dense_path_clock_audit.json"
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(record_audit, item, source, parent, run) for item in m["completed"]]
            for f in as_completed(futures):
                r = f.result()
                result["records"].append(r)
                path.write_text(json.dumps(result, indent=2) + "\n")
                print(r, flush=True)
        fits = pd.read_csv(run / "dense_path_clock_fits.csv")
        assert len(fits) == 3600 and not fits.duplicated(["dataset", "scenario", "source_teacher", "repeat", "candidate_bank", "support"]).any()
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            n = list(pool.map(certify_repeat, range(50), [m["completed"]] * 50, [run] * 50, [fits] * 50))
        verify_summaries(run, fits)
        result.update(
            status="pass",
            paths_regenerated=sum(r["paths_regenerated"] for r in result["records"]),
            observations_audited=sum(r["observations_audited"] for r in result["records"]),
            likelihoods_recomputed=sum(r["likelihoods_recomputed"] for r in result["records"]),
            population_fits_certified=sum(n),
            max_score_error=max(r["max_score_error"] for r in result["records"]),
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
    parser.add_argument("--workers", type=int, default=8)
    audit(parser.parse_args())
