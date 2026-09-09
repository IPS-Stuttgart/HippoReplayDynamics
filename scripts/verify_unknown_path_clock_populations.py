#!/usr/bin/env python3
"""Regenerate observations and verify marginalized scores and mixture certificates."""

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

MODELS = ("physical", "neural", "stationary", "physical_reset", "neural_reset")
IDS = ["dataset", "animal", "session"]


def random(*parts):
    digest = hashlib.sha256(("unknown_path_clocks_v1|20260909|" + "|".join(map(str, parts))).encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def close(a, b, atol=1e-8):
    np.testing.assert_allclose(a, b, atol=atol, rtol=1e-9)


def integral_bins(clock, values, n):
    # Insert bin edges into the knots, integrate trapezoids, then sum by bin.
    edges = np.linspace(0, 1, n + 1)
    knots = np.unique(np.r_[clock, edges])
    y = np.array([np.interp(knots, clock, column) for column in values.T]).T
    area = np.diff(knots)[:, None] * (y[1:] + y[:-1]) / 2
    bins = np.minimum(np.floor((knots[:-1] + knots[1:]) / 2 * n).astype(int), n - 1)
    result = np.zeros((n, values.shape[1]))
    np.add.at(result, bins, area)
    return result * n


def record_audit(item, source, run, n_paths=256, repeats=50, events=128, supports=(128, 256)):
    tag = item["tag"]
    with np.load(source / f"{tag}_cache.npz") as z:
        rates, centers = z["rates"], z["centers"]
        profiles = {}
        for name in sorted(k for k in z.files if k.startswith("counts_")):
            event = int(name.split("_")[1])
            widths = np.diff(z[f"edges_{event}"])
            keep = np.isclose(widths, 0.02, atol=1e-8, rtol=0)
            assert keep.all() or (keep[:-1].all() and 0 < widths[-1] < 0.02 + 1e-8)
            counts = z[name][keep]
            if len(counts) >= 3:
                profiles[event] = counts.sum(axis=1)
    tri = Delaunay(centers)
    vertices = centers[tri.simplices]
    allowed = np.max(np.linalg.norm(vertices[:, :, None] - vertices[:, None], axis=-1), axis=(1, 2)) <= np.sqrt(2) * 8 * 1.001
    metadata = pd.read_csv(run / f"{tag}_libraries.csv")
    assert len(metadata) == n_paths * 2 and not metadata.duplicated(["bank", "path_index"]).any()
    libraries = {0: [], 1: []}
    with np.load(run / f"{tag}_libraries.npz") as saved:
        for row in metadata.itertuples():
            generator = random(tag, row.bank, row.path_index, "geometry")
            for attempt in range(1, row.attempts + 1):
                a, b = centers[generator.choice(len(centers), 2, replace=False)]
                delta = b - a
                distance = np.linalg.norm(delta)
                if not 40 <= distance <= 120:
                    continue
                amplitude = 0.2 * distance * generator.choice([-1, 1]) if row.path_index % 2 else 0
                u = np.linspace(0, 1, 801)
                points = a + u[:, None] * delta + amplitude * np.sin(np.pi * u[:, None]) * np.array([-delta[1], delta[0]]) / distance
                ids = tri.find_simplex(points)
                if (ids >= 0).all() and allowed[ids].all():
                    assert attempt == row.attempts
                    break
            else:
                raise ValueError("geometry sampling does not match")
            close(points, saved[f"points_{row.bank}_{row.path_index}"])
            bary = np.einsum("nij,nj->ni", tri.transform[ids, :2], points - tri.transform[ids, 2])
            bary = np.c_[bary, 1 - bary.sum(axis=1)]
            values = np.einsum("nk,cnk->nc", bary, rates[:, tri.simplices[ids]])
            p = values / values.sum(axis=1, keepdims=True)
            s = np.r_[0, np.linalg.norm(np.diff(points, axis=0), axis=1).cumsum()]
            c = np.r_[0, np.sqrt(0.5 * np.sum(np.diff(np.sqrt(p), axis=0) ** 2, axis=1)).cumsum()]
            close(row.length_cm, s[-1])
            close(row.code_arc_length, c[-1])
            clocks = (s / s[-1], c / c[-1])
            for model, clock in zip(MODELS[:2], clocks, strict=True):
                close(clock, saved[f"clock_{model}_{row.bank}_{row.path_index}"])
            libraries[row.bank].append((values, clocks))
    static = rates.T / rates.sum(axis=0)[:, None]
    cached = {}

    def bank_probabilities(bank, n):
        if (bank, n) not in cached:
            arrays = []
            for i in range(2):
                array = np.array([integral_bins(clocks[i], values, n) for values, clocks in libraries[bank]])
                arrays.append(array / array.sum(axis=2, keepdims=True))
            cached[bank, n] = arrays
        return cached[bank, n]

    checked, maximum_error = 0, 0.0
    for repeat in range(repeats):
        table = pd.read_csv(run / f"{tag}_r{repeat:03d}_scores.csv.gz")
        assert len(table) == 3 * 2 * events * len(supports)
        assert not table.duplicated(["observation_index", "support"]).any()
        base = table[table.support.eq(max(supports))].sort_values("observation_index")
        np.testing.assert_array_equal(base.observation_index, np.arange(len(base)))
        assert base.repeat.eq(repeat).all()
        for key in IDS:
            assert base[key].eq(item[key]).all()
        with np.load(run / f"{tag}_r{repeat:03d}_observations.npz") as saved:
            counts, latent, offsets = saved["counts"], saved["latent"], saved["offsets"]
        assert len(offsets) == len(base) + 1 and offsets[0] == 0 and offsets[-1] == len(counts) == len(latent)
        k = 0
        for scenario in (0.25, 0.5, 0.75):
            selected = random(tag, repeat, scenario, "profiles").choice(sorted(profiles), events, replace=True)
            labels = random(tag, repeat, scenario, "labels").choice(5, events, p=[0.6 * (1 - scenario), 0.6 * scenario, 0.2, 0.1, 0.1])
            for bank, teacher in enumerate(("matched", "independent")):
                for j, (event, label) in enumerate(zip(selected, labels, strict=True)):
                    row = base.iloc[k]
                    totals = profiles[int(event)]
                    n = len(totals)
                    assert row.template_event == event and row.generator == MODELS[label] and row.teacher == teacher and row.scenario == scenario and row.event_in_population == j
                    assert row.n_spikes == totals.sum() and row.n_bins == n
                    generator = random(tag, repeat, scenario, teacher, j, "observation")
                    if label == 2:
                        indices = np.repeat(generator.integers(len(static)), n)
                        p = static[indices]
                    else:
                        indices = generator.integers(n_paths, size=n) if label >= 3 else np.repeat(generator.integers(n_paths), n)
                        p = bank_probabilities(bank, n)[int(label in (1, 4))][indices, np.arange(n)]
                    x = np.array([generator.multinomial(int(total), probability) for total, probability in zip(totals, p, strict=True)])
                    a, b = offsets[k : k + 2]
                    np.testing.assert_array_equal(counts[a:b], x)
                    np.testing.assert_array_equal(latent[a:b], indices)
                    k += 1
        for n, group in base.groupby("n_bins"):
            indices = group.observation_index.to_numpy()
            x = np.stack([counts[offsets[i] : offsets[i + 1]] for i in indices])
            probabilities = bank_probabilities(0, n)
            expected = {h: np.zeros((len(x), 5)) for h in supports}
            st = logsumexp(x.sum(axis=1) @ np.log(static).T, axis=1) - np.log(len(static))
            for h in supports:
                expected[h][:, 2] = st
            for i, p in enumerate(probabilities):
                per_bin = np.einsum("etc,htc->eht", x, np.log(p), optimize=True)
                for h in supports:
                    expected[h][:, i] = logsumexp(per_bin[:, :h].sum(axis=2), axis=1) - np.log(h)
                    expected[h][:, i + 3] = (logsumexp(per_bin[:, :h], axis=1) - np.log(h)).sum(axis=1)
            for h in supports:
                actual = table[table.support.eq(h)].set_index("observation_index").loc[indices]
                ll = actual[["score_" + m for m in MODELS]].to_numpy()
                close(ll, expected[h])
                maximum_error = max(maximum_error, float(np.max(abs(ll - expected[h]))))
                for field in [*IDS, "repeat", "scenario", "teacher", "event_in_population", "generator", "template_event", "n_bins", "n_spikes"]:
                    np.testing.assert_array_equal(actual[field], group[field])
                checked += ll.size
    return {"tag": tag, "library_paths": n_paths * 2, "observations": repeats * 6 * events, "likelihoods_checked": checked, "max_score_error": maximum_error}


def certify(log_scores, weights):
    weights = np.asarray(weights)
    assert np.isfinite(weights).all() and np.all(weights >= 0.5e-9)
    close(weights.sum(), 1, atol=1e-7)
    shift = log_scores.max(axis=1)
    density = np.exp(log_scores - shift[:, None])
    mass = density @ weights
    gradient = (density / mass[:, None]).mean(axis=0)
    free = weights > 1e-8
    level = gradient[free].mean()
    error = max(float(np.max(abs(gradient[free] - level), initial=0)), float(np.max(gradient[~free] - level, initial=0)))
    assert error <= 2.01e-5
    return np.log(mass).sum() + shift.sum(), error


def audit_fits(repeat, items, run, fits):
    table = pd.concat([pd.read_csv(run / f"{item['tag']}_r{repeat:03d}_scores.csv.gz") for item in items], ignore_index=True)
    count = 0
    for row in fits[fits.repeat.eq(repeat)].itertuples():
        group = table[table.dataset.eq(row.dataset) & table.scenario.eq(row.scenario) & table.teacher.eq(row.teacher) & table.support.eq(row.support)]
        assert len(group) == row.n_events and group.animal.nunique() == row.n_animals and group.session.nunique() == row.n_recordings
        ll = group[["score_" + m for m in MODELS]].to_numpy()
        ll -= ll.max(axis=1, keepdims=True)
        weights = np.array([getattr(row, "weight_" + m) for m in MODELS])
        maximum, _ = certify(ll, weights)
        close(row.relative_log_likelihood, maximum)
        close(row.coherent_weight, weights[:2].sum())
        close(row.phi_hat, weights[1] / weights[:2].sum())
        for name, phi in (("low", row.phi_low), ("high", row.phi_high), ("null", 0.5)):
            with np.errstate(divide="ignore"):
                dynamic = np.logaddexp(ll[:, 0] + np.log1p(-phi), ll[:, 1] + np.log(phi))
            w = [getattr(row, name + "_weight_" + m) for m in ("coherent", *MODELS[2:])]
            value, _ = certify(np.column_stack([dynamic, ll[:, 2:]]), w)
            close(getattr(row, name + "_profile_log_likelihood"), value)
            lr = 2 * (maximum - value)
            if name == "null":
                close(row.null_lr, max(0, lr))
            elif phi in (0, 1):
                assert lr <= 3.841458820694124 + 1e-4
            else:
                close(lr, 3.841458820694124, atol=2e-4)
        assert row.direction == ("neural" if row.phi_low > 0.5 else "physical" if row.phi_high < 0.5 else "undetermined")
        assert row.covered == (row.phi_low <= row.scenario + 1e-8 and row.phi_high >= row.scenario - 1e-8)
        count += 1
    assert count == 24
    return count


def audit(args):
    run, out = args.run_dir, args.output_dir
    mp = run / "unknown_path_clock_manifest.json"
    m = json.loads(mp.read_text())
    assert m["status"] == "complete" and not m["oracle_knows_path"] and not m["real_events_rescored"]
    assert len(m["completed"]) == 33 and m["n_fits"] == 1200 and m["n_observations"] == 1267200 and m["n_rows"] == 2534400
    source = Path(m["source_dir"])
    sm = source / "conditional_2d_manifest.json"
    assert file_sha256(sm) == m["input_file_sha256"]["source_manifest"]
    hashes = json.loads(sm.read_text())["output_sha256"]
    for name, digest in m["output_sha256"].items():
        assert file_sha256(run / name) == digest
    for item in m["completed"]:
        name = item["tag"] + "_cache.npz"
        assert file_sha256(source / name) == hashes[name]
    out.mkdir(parents=True, exist_ok=False)
    result = build_script_provenance(input_paths={"run_manifest": mp}, cwd=ROOT) | {"status": "running", "records": []}
    path = out / "unknown_path_clock_audit.json"
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(record_audit, item, source, run) for item in m["completed"]]
            for future in as_completed(futures):
                checked = future.result()
                result["records"].append(checked)
                path.write_text(json.dumps(result, indent=2) + "\n")
                print(checked, flush=True)
        fits = pd.read_csv(run / "unknown_path_clock_fits.csv")
        assert len(fits) == 1200 and not fits.duplicated(["dataset", "scenario", "repeat", "teacher", "support"]).any()
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            checked = list(pool.map(audit_fits, range(50), [m["completed"]] * 50, [run] * 50, [fits] * 50))
        summary = pd.read_csv(run / "unknown_path_clock_summary.csv")
        gates = pd.read_csv(run / "unknown_path_clock_gates.csv")
        assert len(summary) == 24 and len(gates) == 8
        for row in summary.itertuples():
            group = fits[fits.dataset.eq(row.dataset) & fits.scenario.eq(row.scenario) & fits.teacher.eq(row.teacher) & fits.support.eq(row.support)]
            assert len(group) == 50
            close(row.bias, (group.phi_hat - group.scenario).mean())
            close(row.mean_estimate, group.phi_hat.mean())
            close(row.rmse, np.sqrt(np.mean((group.phi_hat - group.scenario) ** 2)))
            close(row.median_interval_width, (group.phi_high - group.phi_low).median())
            close(row.mean_coherent_weight, group.coherent_weight.mean())
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
            s = summary[summary.dataset.eq(row.dataset) & summary.teacher.eq(row.teacher) & summary.support.eq(row.support)]
            bias, coverage = s.bias.abs().max(), s.covered_fraction.min()
            power = s[s.scenario.ne(0.5)].correct_direction_fraction.min()
            false = s[s.scenario.eq(0.5)].directional_claim_fraction.iloc[0]
            close([row.max_absolute_bias, row.min_coverage, row.min_direction_power, row.null_false_direction_fraction], [bias, coverage, power, false])
            assert row.practical_pass == bool(bias <= 0.1 and coverage >= 0.9 and power >= 0.8 and false <= 0.05)
        result.update(
            status="pass",
            likelihoods_checked=sum(r["likelihoods_checked"] for r in result["records"]),
            observations_regenerated=sum(r["observations"] for r in result["records"]),
            population_fits_certified=sum(checked),
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
