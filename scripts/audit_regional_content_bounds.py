#!/usr/bin/env python3
"""Independent dense readout, count, calibration and LP reconstruction."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.special import logsumexp
from scipy.stats import beta

from scripts._provenance import build_script_provenance, file_sha256


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def interval(x, n, alpha):
    if not n:
        return (0., 1.)
    return (float(beta.ppf(alpha/2, x, n-x+1)) if x else 0.,
            float(beta.isf(alpha/2, x+1, n-x)) if x < n else 1.)


def calibration(calls, truth):
    bounds = np.empty((2, 4, 3, 2))
    for z in range(2):
        for k in range(4):
            for category in range(3):
                bounds[z, k, category] = interval(int(((truth == z) & (calls[:, k] == category)).sum()),
                                                    int((truth == z).sum()), .025/24)
    return bounds


def dense(counts, archive):
    rates, region, assignment = archive["rates"], archive["region"], archive["group_assignment"]
    masses, factors, calls, support = [], [], [], []
    prior = region.mean()
    for k in range(4):
        ids = np.flatnonzero(assignment == k)
        ll = np.zeros((len(counts), rates.shape[1]))
        for cell in ids:
            ll += counts[:, cell, None]*np.log(rates[cell])[None, :] - .02*rates[cell][None, :]
        q = np.exp(ll-logsumexp(ll, axis=1)[:, None])[:, region].sum(axis=1)
        bf = logsumexp(ll[:, region], axis=1)-logsumexp(ll[:, ~region], axis=1)-np.log(prior/(1-prior))
        n = counts[:, ids].sum(axis=1)
        y = np.ones(len(counts), int)
        y[(bf > np.log(3)) & (n > 0)] = 2
        y[(bf < -np.log(3)) & (n > 0)] = 0
        masses.append(q)
        factors.append(bf)
        calls.append(y)
        support.append(n)
    return {"mass": np.array(masses).T, "log_bf": np.array(factors).T,
                "calls": np.array(calls).T, "spikes": np.array(support).T}


def independent_lp(calls, cal, slack, conditional=None):
    patterns = np.indices((3, 3, 3, 3)).reshape(4, -1).T
    idx = np.array([sum(int(v)*3**(3-k) for k, v in enumerate(row)) for row in calls])
    hist = np.bincount(idx, minlength=81)
    constraints, rhs = [], []
    for j, count in enumerate(hist):
        lower, upper = interval(int(count), len(calls), .025/81)
        row = np.zeros((2, 81))
        row[:, j] = 1
        constraints.extend([row.ravel(), -row.ravel()])
        rhs.extend([upper, -lower])
    for z in range(2):
        for k in range(4):
            for category in range(3):
                lower = max(0, cal[z, k, category, 0]-slack)
                upper = min(1, cal[z, k, category, 1]+slack)
                hit = (patterns[:, k] == category).astype(float)
                row = np.zeros((2, 81))
                row[z] = hit-upper
                constraints.append(row.ravel().copy())
                row[z] = lower-hit
                constraints.append(row.ravel().copy())
                rhs.extend([0., 0.])
    a, b = np.array(constraints), np.array(rhs)
    c = np.r_[np.zeros(81), np.ones(81)]
    eq, eq_rhs = np.ones((1, 162)), [1.]
    if conditional is not None:
        # Independent fractional program, same estimand but separate construction.
        a, b = np.c_[a, -b], np.zeros(len(b))
        eq = np.zeros((2, 163))
        eq[0, :162], eq[0, -1] = 1, -1
        eq[1, [conditional, 81+conditional]] = 1
        eq_rhs = [0., 1.]
        c = np.zeros(163)
        c[81+conditional] = 1
    solutions = [linprog(c*sign, A_ub=a, b_ub=b, A_eq=eq, b_eq=eq_rhs,
                         bounds=(0, None), method="highs-ipm") for sign in (1, -1)]
    if any(sol.status == 2 for sol in solutions):
        return np.array([np.nan, np.nan])
    if not all(sol.success for sol in solutions):
        raise ValueError("independent optimizer failed")
    return np.array([c @ sol.x for sol in solutions])


def regenerated(archive, rng, n, balanced, multiplier, drift=False, assembly_fraction=0):
    rates = archive["drift_rates"] if drift else archive["rates"]
    region = archive["region"]
    z = np.repeat([0, 1], n//2) if balanced else rng.binomial(1, .3, n)
    loc = np.array([rng.choice(np.where(region == bool(v))[0]) for v in z])
    total = rng.choice(archive["real_counts"].sum(axis=1), n)*multiplier
    u = rng.binomial(1, .7, n)
    state_u = np.array([rng.choice(np.where(region == bool(v))[0]) for v in u])
    p = rates[:, loc].T.copy()
    p /= p.sum(axis=1, keepdims=True)
    nuisance = rates[:, state_u].T.copy()
    nuisance /= nuisance.sum(axis=1, keepdims=True)
    p = (1-assembly_fraction)*p + assembly_fraction*nuisance
    counts = np.array([rng.multinomial(int(t), row) for t, row in zip(total, p, strict=True)])
    return {"truth": z, "states": loc, "totals": total, "assembly_truth": u, "assembly_states": state_u, "counts": counts}


def check_one(row, args, manifest):
    folder = Path(row.folder)
    for name, sha in json.loads((folder/"outputs.json").read_text()).items():
        if file_sha256(folder/name) != sha:
            raise ValueError("output changed: "+str(folder/name))
    source = json.loads((folder/"source.json").read_text())
    frozen = json.loads(Path(source["frozen_input"]).read_text())
    if file_sha256(source["frozen_input"]) != source["source_sha256"]:
        raise ValueError("frozen input changed")
    enc = load(frozen["freeze"]["encoding_path"])
    data = load(folder/"real_and_calibration.npz")
    mask, valid = enc["unit_qc_mask"].astype(bool), enc["valid_spatial_bins"].astype(bool)
    np.testing.assert_array_equal(data["rates"], np.maximum(enc["rates_hz"][mask][:, valid], 1e-4))
    np.testing.assert_array_equal(data["drift_rates"], np.maximum(enc["rates_second_half_hz"][mask][:, valid], 1e-4))
    np.testing.assert_array_equal(data["grid"], enc["bin_centers_cm"][valid])
    np.testing.assert_array_equal(data["cell_ids"], enc["cell_ids"][mask])
    home = np.array(source["home_xy"])
    np.testing.assert_array_equal(data["region"], np.linalg.norm(data["grid"]-home, axis=1) <= 20)
    score = data["rates"][:, data["region"]].mean(axis=1)/data["rates"].mean(axis=1)
    order = np.lexsort((data["cell_ids"], score))
    for k, ix in enumerate(np.array_split(order, 4)):
        assert np.all(data["group_assignment"][ix] == k)
    count_checks, posterior_checks = 0, 0
    for name, source_name in (("real", "real"), ("run", "run_q4")):
        old = load(Path(frozen["edge_source"])/f"{source_name}_audit.npz")
        starts, ends, truth = [], [], []
        for j, start in enumerate(old["starts_s"]):
            length = old["offsets"][j+1]-old["offsets"][j]
            edges = start+.005*np.arange(length+1)
            starts.append(edges[-5])
            ends.append(edges[-1])
            if name == "run":
                samples = start+(np.arange(200)+.5)*.001
                xy = np.column_stack([np.interp(samples, enc["position"][:, 0], enc["position"][:, d]) for d in (1, 2)])
                truth.append(xy[-20:].mean(axis=0))
        for j, cell in enumerate(data["cell_ids"]):
            spikes = np.sort(enc["spikes"][enc["spikes"][:, 1] == cell, 0])
            expected = np.searchsorted(spikes, ends, side="left")-np.searchsorted(spikes, starts, side="left")
            np.testing.assert_array_equal(expected, data[f"{name}_counts"][:, j])
            count_checks += len(expected)
        np.testing.assert_allclose(data[f"{name}_starts"], starts, atol=1e-10, rtol=0)
        if name == "run":
            np.testing.assert_allclose(data["run_truth"], truth, atol=1e-8, rtol=0)
        check = dense(data[f"{name}_counts"], data)
        for k, expected in check.items():
            np.testing.assert_allclose(data[f"{name}_{k}"], expected, atol=2e-8, rtol=1e-10)
        posterior_checks += len(check["calls"])*4
    np.testing.assert_array_equal(data["run_labels"], np.linalg.norm(data["run_truth"]-home, axis=1) <= 20)
    order_run = np.argsort(data["run_starts"], kind="stable")
    np.testing.assert_array_equal(data["run_cal_indices"], order_run[:len(order_run)//2])
    np.testing.assert_array_equal(data["run_val_indices"], order_run[len(order_run)//2:])
    run_ci = calibration(data["run_calls"][data["run_cal_indices"]], data["run_labels"][data["run_cal_indices"]])
    np.testing.assert_allclose(run_ci, data["run_ci"], atol=1e-10, rtol=0)
    rng = np.random.default_rng(source["seed"])
    parameters = manifest["parameters"]
    cis = {}
    for multiplier in (1, 8):
        generated = regenerated(data, rng, 2*parameters["calibration_per_class"], True, multiplier)
        for k, value in generated.items():
            np.testing.assert_array_equal(data[f"cal{multiplier}_{k}"], value)
        check = dense(generated["counts"], data)
        for k, value in check.items():
            np.testing.assert_allclose(data[f"cal{multiplier}_{k}"], value, atol=2e-8, rtol=1e-10)
        cis[multiplier] = calibration(check["calls"], generated["truth"])
        np.testing.assert_allclose(cis[multiplier], data[f"cal{multiplier}_ci"], atol=1e-10, rtol=0)
        count_checks += generated["counts"].size
        posterior_checks += len(check["calls"])*4
    bounds = pd.read_csv(folder/"bounds.csv")
    scores = pd.read_csv(folder/"scores.csv")
    checked_lp, checked_conditional = 0, 0
    for r in bounds.loc[bounds.source.isin(["real", "heldout_run"])].itertuples():
        y = data["real_calls"] if r.source == "real" else data["run_calls"][data["run_val_indices"]]
        ci = run_ci if r.calibration == "run_first_half_q4" else cis[1]
        expected = independent_lp(y, ci, r.transfer_slack)
        np.testing.assert_allclose([r.lower, r.upper], expected, atol=2e-6, rtol=0, equal_nan=True)
        checked_lp += 1
    event_bounds = pd.read_csv(folder/"conditional_bounds.csv")
    # All observed conditional patterns, not only visually favorable events.
    for r in event_bounds.itertuples():
        ci = run_ci if r.calibration == "run_first_half_q4" else cis[1]
        expected = independent_lp(data["real_calls"], ci, r.transfer_slack, int(r.pattern))
        np.testing.assert_allclose([r.lower, r.upper], expected, atol=2e-6, rtol=0, equal_nan=True)
        checked_conditional += 1
    for condition in ("native", "eightfold", "map_drift_eightfold", "shared_assembly_eightfold"):
        multiplier = 1 if condition == "native" else 8
        for rep in range(parameters["replicates"]):
            saved = load(folder/f"{condition}_{rep}.npz")
            generated = regenerated(data, rng, parameters["target_events"], False, multiplier,
                condition == "map_drift_eightfold", .9 if condition == "shared_assembly_eightfold" else 0.)
            for k, value in generated.items():
                np.testing.assert_array_equal(saved[k], value)
            check = dense(saved["counts"], data)
            for k, value in check.items():
                np.testing.assert_allclose(saved[k], value, atol=2e-8, rtol=1e-10)
            record = scores.loc[scores.source.eq(condition) & scores.replicate.eq(rep)]
            assert len(record) == 1
            record = record.iloc[0]
            pi = float(record.latent_prevalence)
            theta = saved["latent_theta"]
            log_joint = np.column_stack([
                np.log(pi if z else 1-pi) +
                np.log(theta[z, np.arange(4)[None, :], saved["calls"]]).sum(axis=1)
                for z in (0, 1)])
            q = np.exp(log_joint[:, 1]-logsumexp(log_joint, axis=1))
            np.testing.assert_allclose(saved["latent_probability"], q, atol=1e-8, rtol=1e-8)
            for label, probabilities in (("latent", q), ("naive", saved["mass"].mean(axis=1)),
                                          ("oracle_prevalence", np.full(len(q), .30))):
                safe = np.clip(probabilities, 1e-12, 1-1e-12)
                z = saved["truth"]
                brier = np.mean((safe-z)**2)
                loss = -np.mean(z*np.log(safe)+(1-z)*np.log1p(-safe))
                np.testing.assert_allclose([record[label+"_brier"], record[label+"_log_loss"]],
                                           [brier, loss], atol=1e-8, rtol=1e-8)
            count_checks += saved["counts"].size
            posterior_checks += len(check["calls"])*4
            for r in bounds.loc[bounds.source.eq(condition) & bounds.replicate.eq(rep)].itertuples():
                expected = independent_lp(saved["calls"], cis[multiplier], r.transfer_slack)
                np.testing.assert_allclose([r.lower, r.upper], expected, atol=2e-6, rtol=0, equal_nan=True)
                assert bool(r.covers_truth) == bool(np.isfinite(expected).all() and expected[0]-1e-7 <= .30 <= expected[1]+1e-7)
                checked_lp += 1
    return {"session": row.session, "cell_window_counts": count_checks, "population_readouts": posterior_checks,
                "prevalence_intervals": checked_lp, "conditional_intervals": checked_conditional}



def audit_stage_zero(root, manifest):
    original = pd.read_csv(manifest["input_file_paths"]["matched_events"])
    original = original.loc[original.family.eq("targeted")]
    output = pd.read_csv(root/"matched_home_bayes_factor_events.csv.gz")
    home = pd.read_csv(manifest["input_file_paths"]["home"]).set_index("session")
    keys = ["animal", "session", "window_uid", "encoding", "cohort"]
    if original.duplicated(keys).any() or output.duplicated(keys+["population"]).any():
        raise ValueError("duplicate matched event keys")
    if len(output) != 2*len(original):
        raise ValueError("missing matched-event BF rows")
    decoded = 0
    old_root = Path(manifest["input_file_paths"]["matched_manifest"]).parent
    for session, group in output.groupby("session"):
        folder = old_root/session.replace("/", "_")
        check = json.loads((folder/"freeze_checkpoint.json").read_text())
        rv = load(folder/"run_validation.npz")
        coordinates = home.loc[session, ["home_x_cm", "home_y_cm"]].to_numpy(float)
        region = np.linalg.norm(rv["grid_cm"]-coordinates, axis=1) <= 20
        prior = region.mean()
        np.testing.assert_allclose(group.prior_home_mass, prior, atol=1e-12)
        arrays_path = next(p for p in check["inputs"] if "shuffle-baseline" in p)
        arrays = load(arrays_path)
        ids = {u:i for i,u in enumerate(arrays["window_uids"])}
        pair = next(p for p in check["selected"] if p["family"] == "targeted")
        for side in ("high", "low"):
            joined = group.loc[group.population.eq(side)].merge(original, on=keys, validate="one_to_one")
            q = joined[f"home_mass_{side}"].to_numpy()
            np.testing.assert_allclose(joined.posterior_home_mass, q, atol=1e-12)
            q = np.clip(q, 1e-15, 1-1e-15)
            bf = np.log(q/(1-q))-np.log(prior/(1-prior))
            np.testing.assert_allclose(joined.log_bayes_factor, bf, atol=1e-8)
            np.testing.assert_array_equal(joined.positive_evidence, bf > np.log(3))
            np.testing.assert_array_equal(joined.negative_evidence, bf < -np.log(3))
            cells = pair[side]["indices"]
            for _, part in joined.groupby(["encoding", "cohort"]):
                for row in part.iloc[np.unique(np.r_[np.arange(min(3,len(part))), len(part)-1])].itertuples():
                    event = ids[row.window_uid]
                    start = arrays["frame_offsets"][2*event]+row.fixed_endpoint_frame
                    counts = arrays["frame_counts"][int(start), cells]
                    rates = (rv["early_rates"] if row.encoding == "early_run"
                             else arrays["rates_hz"][:, arrays["support"]])[cells]
                    ll = (counts[:, None]*np.log(rates)).sum(axis=0)-.02*rates.sum(axis=0)
                    mass = np.exp(logsumexp(ll[region])-logsumexp(ll))
                    np.testing.assert_allclose(row.posterior_home_mass, mass, atol=1e-8, rtol=1e-8)
                    decoded += 1
    return {"bayes_factor_rows":len(output), "independent_posterior_reconstructions":decoded}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("refuse to overwrite audit")
    manifest_path = args.input_dir/"manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["status"] != "complete":
        raise ValueError("production not complete")
    for key, name in manifest["input_file_paths"].items():
        if file_sha256(name) != manifest["input_file_sha256"][key]:
            raise ValueError("recorded input changed: "+name)
    sessions = pd.read_csv(args.input_dir/"sessions.csv")
    rows = []
    for row in sessions.itertuples():
        rows.append(check_one(row, args, manifest))
        print(row.session, "audit complete", flush=True)
    for name in ("bounds", "conditional_bounds", "scores", "calibration"):
        pooled = pd.read_csv(args.input_dir/f"{name}.csv")
        separate = pd.concat([pd.read_csv(Path(r.folder)/f"{name}.csv") for r in sessions.itertuples()], ignore_index=True)
        pd.testing.assert_frame_equal(pooled, separate, check_exact=False, atol=1e-10)
    stage_zero = audit_stage_zero(args.input_dir, manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(status="pass", checked=rows, stage_zero=stage_zero,
        **build_script_provenance(input_paths={"manifest": manifest_path, "script": Path(__file__)}),
        result_file_sha256={str(p): file_sha256(p) for p in args.input_dir.glob("*.csv")},
        scope="All saved counts/readouts/calibration and LP bounds; conditional EM probabilities and losses, not raw spike sorting, biological truth, or global EM optimum."), indent=2)+"\n")


if __name__ == "__main__":
    main()


