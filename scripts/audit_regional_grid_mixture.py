"""Independent raw-count, objective, first-iterate and gate audit for grid mixtures."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.special import logsumexp

from scripts._provenance import build_script_provenance, file_sha256
from scripts.audit_edge_support_content import load_npz
from scripts.build_regional_blind_bank import STRATA, read_csv, write_json


def checked_manifest(folder):
    m = json.loads((folder / "manifest.json").read_text())
    for name, expected in m["outputs"].items():
        if file_sha256(folder / name) != expected:
            raise AssertionError("changed artifact: " + str(folder / name))
    return m


def close(a, b, tol=1e-9):
    if not np.allclose(a, b, atol=tol, rtol=tol, equal_nan=True):
        raise AssertionError((np.max(np.abs(np.asarray(a) - b)), tol))


def audit_likelihood(counts, rates, observation, total, full):
    if observation == "conditional_multinomial":
        probability = rates / rates.sum(axis=0)
        return counts @ np.log(probability)
    if observation == "fixed_total_censored":
        other = full - rates.sum(axis=0)
        if (other <= 0).any():
            raise ValueError("censored category unavailable")
        all_counts = np.column_stack([counts, total - counts.sum(axis=1)])
        probability = np.vstack([rates, other]) / full
        return all_counts @ np.log(probability)
    return counts @ np.log(rates) - 0.02 * rates.sum(axis=0)


def objective_audit(ll, pi, penalty, edges):
    i, j = edges
    ll = ll - ll.max(axis=1, keepdims=True)
    with np.errstate(divide="ignore"):
        z = logsumexp(ll + np.log(pi), axis=1)
    factor = len(pi) ** 2 / len(i)
    diff = pi[i] - pi[j]
    regularizer = 0.5 * factor * np.sum(diff**2)
    gradient = -np.exp(ll - z[:, None]).mean(axis=0)
    gradient += penalty * factor * (np.bincount(i, weights=diff, minlength=len(pi)) - np.bincount(j, weights=diff, minlength=len(pi)))
    gap = max(0.0, float(gradient @ pi - gradient.min()))
    return float(-z.mean() + penalty * regularizer), gap


def audit_population(task):
    parent, j = task
    folder = Path(parent["folder"])
    out = folder / f"pop{j}"
    checked_manifest(out)
    a = load_npz(folder / "inputs.npz")
    data = load_npz(parent["source"])
    coords = load_npz(folder / "encoding_coordinates.npz")
    grid = coords[f"grid{j}"]
    full = coords[f"full_rate_sum{j}"]
    distances = cdist(grid, grid)
    spacing = distances[distances > 0].min()
    edges = np.where(np.triu((distances > 0) & (distances <= spacing * 1.01), 1))
    lookup = {int(c): i for i, c in enumerate(data["cell_ids"])}
    ix = [lookup[int(c)] for c in data[f"pop{j}_ids"]]
    counts = a["counts"][..., ix]
    rates = data[f"pop{j}_rates"]
    region = data[f"pop{j}_region"]
    meta = read_csv(folder / "panels.csv")
    cv = read_csv(out / "cv.csv")
    cw = load_npz(out / "cv_weights.npz")["pi"]
    cal = int(np.flatnonzero(meta.phase.eq("calibration"))[0])
    clf = audit_likelihood(counts[:, cal].reshape(-1, len(rates)), rates, "conditional_multinomial", None, None)
    folds = np.broadcast_to(a["folds"][None, :, None], counts[:, cal].shape[:-1]).ravel()
    for row, pi in zip(cv.itertuples(), cw, strict=True):
        _, gap = objective_audit(clf[folds != row.fold], pi, row.penalty, edges)
        close(gap, row.dual_gap)
        assert gap <= 1e-6
        with np.errstate(divide="ignore"):
            score = logsumexp(clf[folds == row.fold] + np.log(pi), axis=1).mean()
        close(score, row.score)
    estimates = read_csv(out / "estimates.csv")
    cache = load_npz(out / "fit_weights.npz")
    likelihoods = {}
    for observation in estimates.observation.unique():
        likelihoods[observation] = audit_likelihood(counts.reshape(-1, len(rates)), rates, observation, a["counts"].sum(axis=-1).ravel(), full).reshape(*counts.shape[:-1], -1)
    max_gap = 0.0
    for r in estimates.itertuples():
        pi, first = cache["pi"][r.fit_id], cache["first"][r.fit_id]
        close(pi.sum(), 1)
        assert (pi >= 0).all()
        panel = int(np.flatnonzero(meta.phase.eq("validation") & meta.prevalence.eq(r.binary_label_quota))[0])
        g = None if r.generator == "equal_seven" else STRATA.index((r.generator, r.scale))
        ll = likelihoods[r.observation][:, panel] if g is None else likelihoods[r.observation][g, panel]
        ll = ll.reshape(-1, ll.shape[-1])
        expected_first = np.exp(ll - logsumexp(ll, axis=1, keepdims=True)).mean(axis=0)
        close(first, expected_first)
        close(r.first_em_mass, first[region].sum())
        close(r.estimate, pi[region].sum())
        truth = a["truth"][:, panel].mean() if g is None else a["truth"][g, panel].mean()
        close(r.true_occupancy, truth)
        close(r.absolute_error, abs(pi[region].sum() - truth))
        value, gap = objective_audit(ll, pi, r.penalty, edges)
        close(value, r.objective)
        close(gap, r.dual_gap)
        assert bool(r.converged) == bool(gap <= 1e-6)
        max_gap = max(max_gap, gap)
    return {"session": parent["session"], "population_index": j, "status": "pass", "cv_fits": len(cv), "validation_fits": len(estimates), "maximum_dual_gap": max_gap}


def audit_raw_inputs(parent, bank):
    folder = Path(parent["folder"])
    checked_manifest(folder)
    a = load_npz(folder / "inputs.npz")
    data = load_npz(parent["source"])
    native = Path(bank) / parent["session"].replace("/", "_")
    templates = load_npz(native / "templates.npz")
    meta = read_csv(folder / "panels.csv")
    close(a["event_ids"], templates["event_ids"])
    assert len(np.unique(a["event_ids"])) == len(a["event_ids"])
    assert set(a["folds"]) == {0, 1, 2}
    checked = 0
    for g, (kind, scale) in enumerate(STRATA):
        directory = native / f"{kind}_scale{scale:g}_delta40"
        m = json.loads((directory / "manifest.json").read_text())
        for pos, row in enumerate(meta.itertuples()):
            path = directory / f"panel_{row.panel:03}.npz"
            assert file_sha256(path) == m["sha256"][path.name]
            p = load_npz(path)
            close(a["truth"][g, pos], p["occupancy_fraction"][:, 1])
            for e in range(min(3, len(a["event_ids"]))):
                left, right = templates["offsets"][e : e + 2]
                t = templates["times"][left:right]
                ids = p["identities"][left:right]
                end, start = templates["endpoints"][e], templates["starts"][e]
                steps = round((end - start) / 0.005)
                edges = start + 0.005 * (steps - np.array([8, 4, 0]))
                edges[-1] = end
                edges[-2] = data["windows"][templates["source_indices"][e], 0]
                for b in range(2):
                    direct = np.bincount(ids[(t >= edges[b]) & (t < edges[b + 1])], minlength=len(data["cell_ids"]))
                    close(direct, a["counts"][g, pos, e, b], 0)
                    checked += 1
    return {"session": parent["session"], "raw_windows_checked": checked, "status": "pass"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    root, out = Path(args.experiment), Path(args.output_dir)
    m = checked_manifest(root)
    assert m["status"] == "complete_pending_audit"
    sessions = read_csv(root / "sessions.csv").to_dict("records")
    assert len(sessions) == 8 and sum(s["populations"] for s in sessions) == 22
    raw = [audit_raw_inputs(s, m["parameters"]["bank"]) for s in sessions]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(audit_population, (s, j)) for s in sessions for j in range(s["populations"])]):
            row = future.result()
            rows.append(row)
            print("AUDIT", row["session"], row["population_index"], "pass", flush=True)
    d = read_csv(root / "estimates.csv")
    cv = read_csv(root / "cross_validation.csv")
    assert len(d) == 4224 and len(cv) == 264
    scores = cv.groupby(["penalty", "animal", "session"]).score.mean().groupby(["penalty", "animal"]).mean().groupby("penalty").mean()
    chosen = float(scores[scores >= scores.max() - 1e-6].index.max())
    assert chosen == json.loads((root / "frozen_penalty.json").read_text())["penalty"]
    decision = json.loads((root / "decision.json").read_text())
    primary = d[d.observation.eq("conditional_multinomial") & d.fit_kind.eq("selected_penalty")]
    assert decision["primary_truth_budget_pass"] == bool(primary.absolute_error.max() <= 0.05)
    pairs = read_csv(root / "paired_population_gaps.csv")
    for p in pairs.itertuples():
        f = d[
            d.session.eq(p.session)
            & d.family.eq(p.family)
            & d.observation.eq(p.observation)
            & d.fit_kind.eq(p.fit_kind)
            & d.generator.eq(p.generator)
            & d.binary_label_quota.eq(p.binary_label_quota)
        ]
        f = f[f.scale.isna()] if pd.isna(p.scale) else f[f.scale.eq(p.scale)]
        assert len(f) == 2
        high = f[f.population.str.endswith("_high")].iloc[0]
        low = f[f.population.str.endswith("_low")].iloc[0]
        close(p.fitted_gap, high.estimate - low.estimate)
        close(p.first_gap, high.first_em_mass - low.first_em_mass)
    pp = pairs[pairs.observation.eq("conditional_multinomial") & pairs.fit_kind.eq("selected_penalty")]
    assert len(pp) == 224 and decision["primary_gap_budget_pass"] == bool(pp.absolute_fitted_gap.max() <= 0.05)
    out.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(rows).to_csv(out / "fit_audit.csv", index=False)
    pd.DataFrame(raw).to_csv(out / "raw_count_audit.csv", index=False)
    manifest = build_script_provenance(input_paths={"experiment": root / "manifest.json", "audit_code": __file__})
    manifest.update(status="technical_pass", outputs={p.name: file_sha256(p) for p in out.iterdir()})
    write_json(out / "manifest.json", manifest)


if __name__ == "__main__":
    main()
