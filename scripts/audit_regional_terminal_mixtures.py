"""Independent arithmetic, optimization and denominator audit of segment tests."""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

from hipporeplayimm.regional_content_frontier import calls_from_bf
from hipporeplayimm.regional_readout_endpoint import class_likelihoods
from scripts._provenance import build_script_provenance, file_sha256
from scripts.audit_edge_support_content import load_npz
from scripts.build_regional_blind_bank import STRATA, read_csv, write_json
from scripts.diagnose_regional_zero_mass import mixed_measure_likelihoods


def separate_optimizer(f, weights=None):
    f = np.asarray(f, float)
    if weights is None:
        weights = np.ones(f.shape[:-1])
    f = np.maximum(f/f.max(axis=-1, keepdims=True), 1e-250)
    if np.max(abs(f[..., 0]-f[..., 1])) < 1e-10:
        return np.nan
    def loss(pi):
        return -float(np.sum(weights*np.log((1-pi)*f[..., 0]+pi*f[..., 1])))
    fit = minimize_scalar(loss, bounds=(0., 1.), method="bounded", options={"xatol": 1e-11})
    if not fit.success:
        raise ValueError("independent optimizer failed")
    return float(min((0., fit.x, 1.), key=loss))


def exhaustive_any_bf(bin_scores, bin_totals, area):
    s = np.where(np.asarray(bin_totals) == 0, 0., bin_scores)
    if not np.sum(bin_totals):
        return 0.
    odds = s + np.log(area/(1-area))
    lh, ln = -np.logaddexp(0., -odds), -np.logaddexp(0., odds)
    values = [float(np.where(state, lh, ln).sum()) for state in itertools.product((0, 1), repeat=len(s)) if any(state)]
    prior_none = len(s)*np.log1p(-area)
    return float(logsumexp(values)-ln.sum()-(np.log(-np.expm1(prior_none))-prior_none))


def audit_task(task):
    root, bank = Path(task["folder"]), Path(task["bank"])
    manifest = json.loads((root/"manifest.json").read_text())
    for name, sha in manifest["outputs"].items():
        assert file_sha256(root/name) == sha, name
    arrays = load_npz(root/"readouts_and_likelihoods.npz")
    meta = read_csv(root/"panels_used.csv")
    assert set(meta.phase) == {"calibration", "validation"}
    cal, val = arrays["calibration_indices"], arrays["validation_indices"]
    assert len(cal) == 4 and len(val) == 16 and not set(cal) & set(val)
    labels = arrays["truth"]
    np.testing.assert_array_equal(labels.sum(axis=2), np.broadcast_to(labels[0].sum(axis=1), labels.shape[:2]))
    pure = read_csv(root/"pure_validation.csv")
    envelope = read_csv(root/"mixture_envelopes.csv")
    grid = read_csv(root/"mixture_grid_checks.csv")
    assert len(pure) == manifest["populations"]*7*16*6
    assert len(envelope) == manifest["populations"]*16*4
    assert len(grid) == 212*4
    expected = pure.loc[pure.calibration_mode.eq("pooled")].set_index(
        ["population", "readout", "representation", "generator", "scale", "prevalence", "replica"])
    assert expected.index.is_unique
    definitions = json.loads((bank/task["session"].replace("/", "_")/"population_definitions.json").read_text())
    curves, refits = 0, 0
    for j, definition in enumerate(definitions):
        for readout in ("independent_bin_any_home", "pooled_counts"):
            score = arrays[f"pop{j}_{readout}_score"]
            totals = arrays[f"pop{j}_totals"]
            for representation in ("continuous_zero_mass", "ternary"):
                x = calls_from_bf(score, totals) if representation == "ternary" else score
                query = x[:, val, 0].ravel()
                if representation == "ternary":
                    check = class_likelihoods(x[:, cal].ravel(), labels[:, cal].ravel(), query, "ternary")
                else:
                    check = mixed_measure_likelihoods(x[:, cal].ravel(), labels[:, cal].ravel(), query)
                f = arrays[f"pop{j}_{readout}_{representation}_likelihoods"]
                assert np.isfinite(f).all() and (f > 0).all()
                np.testing.assert_allclose(check, f[:, :, 0].reshape(-1, 2), rtol=1e-10, atol=1e-12)
                curves += len(query)
                if j == 0:
                    for g, (kind, scale) in enumerate(STRATA):
                        for v in (0, 4, 8, 12):
                            row = meta.iloc[val[v]]
                            estimate = separate_optimizer(f[g, v])
                            actual = expected.loc[("full", readout, representation, kind, scale, row.prevalence, row.replica), "estimate"]
                            np.testing.assert_allclose(estimate, actual, atol=2e-5, rtol=0, equal_nan=True)
                            refits += 1
    for row in envelope.itertuples():
        e = [expected.loc[(row.population, row.readout, row.representation, kind, scale, row.prevalence, row.replica), "estimate"] for kind, scale in STRATA]
        valid = np.isfinite(e).all()
        lo, hi = (min(e), max(e)) if valid else (0., 1.)
        np.testing.assert_allclose([lo, hi], [row.lower_estimate, row.upper_estimate], atol=1e-9)
        error = max(abs(lo-row.truth), abs(hi-row.truth))
        assert abs(error-row.worst_absolute_error) < 1e-9
        assert row.within_5pp == bool(valid and error <= .05)
    f = arrays["pop0_independent_bin_any_home_continuous_zero_mass_likelihoods"]
    mixed_refits = 0
    for prevalence, rows in grid.groupby("prevalence"):
        v = int(np.flatnonzero(meta.iloc[val].prevalence.eq(prevalence).to_numpy())[0])
        for row in rows.iloc[[0, 50, 100, 210, 211]].itertuples():
            w = np.array(json.loads(row.weights))
            check = separate_optimizer(f[:, v], w[:, None])
            np.testing.assert_allclose(check, row.estimate, atol=2e-5, rtol=0, equal_nan=True)
            mixed_refits += 1
    parent = bank/task["session"].replace("/", "_")
    source, templates = load_npz(parent/"source_inputs.npz"), load_npz(parent/"templates.npz")
    ids = {int(c): i for i, c in enumerate(source["cell_ids"])}
    use = [ids[int(c)] for c in source["pop0_ids"]]
    rates, region = source["pop0_rates"], source["pop0_region"]
    area = float(region.mean())
    raw_checks, max_error = 0, 0.
    for g, (kind, scale) in enumerate(STRATA):
        folder = parent/f"{kind}_scale{scale:g}_delta{task['delta_ms']}"
        actual_panel = int(meta.iloc[val[0]].panel)
        p = load_npz(folder/f"panel_{actual_panel:03}.npz")
        for i in (0, len(templates["event_ids"])//2, len(templates["event_ids"])-1):
            a, b = templates["offsets"][i:i+2]
            times, cell = templates["times"][a:b], p["identities"][a:b]
            steps = round((templates["endpoints"][i]-templates["starts"][i])/.005)
            edges = templates["starts"][i]+.005*np.arange(steps-task["delta_ms"]//5, steps+1, 4)
            edges[-1] = templates["endpoints"][i]
            edges[-2] = source["windows"][templates["source_indices"][i], 0]
            counts = np.array([np.bincount(cell[(times >= l) & (times < r)], minlength=len(ids))[use] for l, r in itertools.pairwise(edges)])
            ll = counts@np.log(rates)-.02*rates.sum(axis=0)
            bf = logsumexp(ll[:, region], axis=1)-np.log(region.sum())-logsumexp(ll[:, ~region], axis=1)+np.log((~region).sum())
            any_bf = exhaustive_any_bf(bf, counts.sum(axis=1), area)
            pooled_ll = counts.sum(axis=0)@np.log(rates)-(task["delta_ms"]/1000.)*rates.sum(axis=0)
            pooled = float(logsumexp(pooled_ll[region])-np.log(region.sum())-logsumexp(pooled_ll[~region])+np.log((~region).sum())) if counts.sum() else 0.
            expected_bfs = [arrays[f"pop0_{name}_score"][g, val[0], i] for name in ("independent_bin_any_home", "pooled_counts")]
            error = float(np.max(np.abs(np.array([any_bf, pooled])-expected_bfs)))
            assert error < 1e-8
            max_error = max(max_error, error)
            raw_checks += 2
    return {"session": task["session"], "delta_ms": task["delta_ms"], "status": "pass",
            "calibration_likelihood_pairs_checked": curves, "independent_pure_refits": refits,
            "independent_mixture_refits": mixed_refits, "envelopes_checked": len(envelope),
            "raw_readouts_checked": raw_checks, "maximum_raw_readout_error": max_error}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    root, out = Path(args.experiment), Path(args.output_dir)
    manifest = json.loads((root/"manifest.json").read_text())
    assert manifest["status"] == "complete_pending_report_audit"
    for key, path in manifest["input_file_paths"].items():
        assert file_sha256(path) == manifest["input_file_sha256"][key], key
    for name, sha in manifest["outputs"].items():
        assert file_sha256(root/name) == sha, name
    tasks = read_csv(root/"tasks.csv")
    assert len(tasks) == 32 and tasks.session.nunique() == 8
    assert not tasks.duplicated(["session", "delta_ms"]).any()
    assert (tasks.groupby("session").delta_ms.nunique() == 4).all()
    out.mkdir(parents=True, exist_ok=False)
    records = [r | {"bank": manifest["parameters"]["bank"]} for r in tasks.to_dict("records")]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for row in executor.map(audit_task, records):
            rows.append(row)
            print("AUDIT", row["session"], row["delta_ms"], "pass", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(out/"technical_audit.csv", index=False)
    provenance = build_script_provenance(input_paths={"experiment": root/"manifest.json", "auditor": __file__})
    provenance.update(status="technical_pass", tasks=len(rows), outputs={"technical_audit.csv": file_sha256(out/"technical_audit.csv")})
    write_json(out/"manifest.json", provenance)


if __name__ == "__main__":
    main()
