#!/usr/bin/env python3
"""Simulation-only falsification of a PRE/POST conditional-order assay."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from _provenance import build_script_provenance, file_sha256

SCENARIOS = ("unchanged", "rate_only", "occupancy_dwell_only", "emission_only", "order_changed")
EMISSIONS = ("oracle_phase_map", "pooled_map")


def stochastic(value):
    a = np.asarray(value, float)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or len(a) < 3:
        raise ValueError("square transition matrix with at least three states required")
    if not np.isfinite(a).all() or (a <= 0).any() or not np.allclose(a.sum(1), 1, atol=1e-12, rtol=0):
        raise ValueError("strictly positive stochastic transition required")
    return a


def equilibrium(transition):
    a = stochastic(transition)
    lhs = a.T - np.eye(len(a))
    lhs[-1] = 1
    rhs = np.zeros(len(a))
    rhs[-1] = 1
    pi = np.linalg.solve(lhs, rhs)
    if (pi <= 0).any() or not np.allclose(pi @ a, pi, atol=1e-10, rtol=0):
        raise ValueError("invalid equilibrium")
    return pi


def transport_transition(reference, target):
    """Match target occupancy and diagonal, retaining reference off-diagonal odds."""
    reference, target = stochastic(reference), stochastic(target)
    if reference.shape != target.shape:
        raise ValueError("unaligned transitions")
    pi = equilibrium(target)
    stay = np.diag(target)
    mass = pi * (1 - stay)
    flow = reference.copy()
    np.fill_diagonal(flow, 0)
    for _ in range(20000):
        flow *= (mass / flow.sum(1))[:, None]
        flow *= (mass / flow.sum(0))[None, :]
        if max(np.max(np.abs(flow.sum(1) - mass)), np.max(np.abs(flow.sum(0) - mass))) < 1e-13:
            break
    else:
        raise ValueError("transition transport did not converge")
    a = flow / pi[:, None]
    np.fill_diagonal(a, stay)
    stochastic(a)
    if not np.allclose(pi @ a, pi, atol=1e-11, rtol=0):
        raise ValueError("transport failed equilibrium constraint")
    return a


def validate_emission(value, states):
    e = np.asarray(value, float)
    if e.ndim != 2 or e.shape[0] != states or not np.isfinite(e).all() or (e <= 0).any():
        raise ValueError("strictly positive states-by-cells emission probabilities required")
    return e / e.sum(1, keepdims=True)


def make_world(seed, scenario, states=6, cells=24):
    if scenario not in SCENARIOS or cells < states:
        raise ValueError("unknown scenario or insufficient cells")
    rng = np.random.default_rng(seed)
    forward = np.roll(np.eye(states), 1, axis=1)
    pre_t = 0.55 * np.eye(states) + 0.40 * forward + 0.05 / states
    preference = np.arange(cells) % states
    pre_e = 0.12 + 2.0 * (np.arange(states)[:, None] == preference[None, :])
    pre_e *= rng.lognormal(0, 0.25, cells)[None, :]
    pre_e /= pre_e.sum(1, keepdims=True)
    post_t, post_e, post_rate = pre_t.copy(), pre_e.copy(), 8.0
    if scenario == "rate_only":
        post_rate = 16.0
    elif scenario == "occupancy_dwell_only":
        # Sinkhorn transport preserves all off-diagonal cross-product odds.
        target_pi = np.array([0.28, 0.22, 0.18, 0.14, 0.10, 0.08])
        raw = 0.45 * np.eye(states) + 0.55 * target_pi[None, :]
        post_t = transport_transition(pre_t, raw)
    elif scenario == "emission_only":
        # Cell recruitment changes, but state order, equilibrium and dwell do not.
        post_e = pre_e * rng.lognormal(0, 1.4, pre_e.shape)
        post_e /= post_e.sum(1, keepdims=True)
    elif scenario == "order_changed":
        post_t = 0.55 * np.eye(states) + 0.40 * forward.T + 0.05 / states
    return (pre_t, pre_e, 8.0), (post_t, post_e, post_rate)


def generate(rng, world, n_events, n_bins):
    a, e, rate = world
    pi = equilibrium(a)
    result = []
    for _ in range(n_events):
        state = rng.choice(len(a), p=pi)
        counts = np.zeros((n_bins, e.shape[1]), dtype=int)
        for t in range(n_bins):
            counts[t] = rng.multinomial(rng.poisson(rate), e[state])
            state = rng.choice(len(a), p=a[state])
        result.append(counts)
    return result


def fit_transition(sequences, emission):
    from hmmlearn.hmm import MultinomialHMM

    logging.getLogger("hmmlearn.hmm").setLevel(logging.ERROR)
    e = validate_emission(emission, len(emission))
    k = len(e)
    class FixedEmissionHMM(MultinomialHMM):
        def _compute_lower_bound(self, log_probability):
            return float(log_probability + (np.log(self.startprob_).sum() + np.log(self.transmat_).sum()) / self.n_components)

    model = FixedEmissionHMM(
        n_components=k, params="st", init_params="", implementation="log",
        n_iter=100, tol=1e-4, startprob_prior=1 + 1 / k, transmat_prior=1 + 1 / k,
    )
    model.startprob_ = np.full(k, 1 / k)
    model.transmat_ = 0.5 * np.eye(k) + 0.5 / k
    model.emissionprob_ = e.copy()
    model.fit(np.concatenate(sequences), [len(s) for s in sequences])
    history = np.asarray(model.monitor_.history, float)
    # monitor_.converged includes reaching the iteration cap; record actual tolerance.
    converged = len(history) > 1 and 0 <= history[-1] - history[-2] < model.tol
    if (not np.isfinite(history).all() or (np.diff(history) < -1e-8).any()
            or not np.array_equal(model.emissionprob_, e)):
        raise ValueError("invalid EM result or supposedly fixed emission map changed")
    return stochastic(model.transmat_).copy(), bool(converged), len(history)


def predictive_gain(counts, emission, own, other, train, held, horizon=2):
    """Causal train-cell filtering; no future or held-cell observation enters it."""
    x = np.asarray(counts)
    if x.ndim != 2 or not np.isfinite(x).all() or (x < 0).any() or not np.equal(x, np.floor(x)).all():
        raise ValueError("integer nonnegative counts required")
    train, held = np.asarray(train, int), np.asarray(held, int)
    if not len(train) or not len(held) or sorted([*train, *held]) != list(range(x.shape[1])):
        raise ValueError("disjoint complete cell partition required")
    if not isinstance(horizon, int) or horizon < 1 or horizon >= len(x):
        raise ValueError("positive horizon smaller than event required")
    e = validate_emission(emission, len(own))
    if e.shape[1] != x.shape[1]:
        raise ValueError("unaligned cells")
    te = e[:, train] / e[:, train].sum(1, keepdims=True)
    he = e[:, held] / e[:, held].sum(1, keepdims=True)
    ll = x[:, train] @ np.log(te).T
    target_ll = x[horizon:, held] @ np.log(he).T
    initial = equilibrium(own)
    predictions, scores = [], []
    for transition in (own, other):
        transition = stochastic(transition)
        step = np.linalg.matrix_power(transition, horizon)
        q = initial.copy()
        predicted = []
        for t in range(len(x) - horizon):
            if t:
                q = q @ transition
            logq = np.log(q) + ll[t]
            q = np.exp(logq - logsumexp(logq))
            predicted.append(q @ step)
        predicted = np.asarray(predicted)
        predictions.append(predicted)
        # The held-out count-conditioned multinomial coefficient cancels.
        scores.append(float(logsumexp(np.log(predicted) + target_ll, axis=1).sum()))
    target_spikes = int(x[horizon:, held].sum())
    return scores[0] - scores[1], target_spikes, predictions


def replicate(task):
    seed, scenario, n_cal, n_test, n_bins = task
    world = make_world(seed, scenario)
    rng = np.random.default_rng(seed + 1000000)
    calibration = [generate(rng, w, n_cal, n_bins) for w in world]
    test = [generate(rng, w, n_test, n_bins) for w in world]
    # Every latent state has both training and held-out neurons.
    held = np.arange(18, 24)
    train = np.arange(18)
    rows = []
    for mode in EMISSIONS:
        emissions = [w[1] for w in world] if mode == "oracle_phase_map" else [(world[0][1] + world[1][1]) / 2] * 2
        fits = [fit_transition(calibration[p], emissions[p]) for p in range(2)]
        for phase in range(2):
            own = fits[phase][0]
            transported = transport_transition(fits[1 - phase][0], own)
            gains = []
            for counts in test[phase]:
                delta, spikes, _ = predictive_gain(counts, emissions[phase], own, transported, train, held)
                if spikes:
                    gains.append(delta / spikes)
            if len(gains) != n_test:
                raise ValueError("zero held-out spike event; no silent event exclusion")
            rows.append({
                "replicate_seed": seed, "scenario": scenario, "emission_mode": mode,
                "phase": ("PRE", "POST")[phase], "mean_event_gain_per_spike": float(np.mean(gains)),
                "n_events": len(gains), "own_fit_converged": fits[phase][1],
                "other_fit_converged": fits[1 - phase][1], "own_fit_iterations": fits[phase][2],
                "target_equilibrium_error": float(np.max(np.abs(equilibrium(own) @ transported - equilibrium(own)))),
                "target_dwell_error": float(np.max(np.abs(np.diag(own) - np.diag(transported)))),
            })
    return rows


def summarize(table):
    event_averages = table.groupby(["replicate_seed", "scenario", "emission_mode"]).mean(numeric_only=True).reset_index()
    rows = []
    # Even replicates set a reference threshold; odd replicates test it independently.
    for mode in EMISSIONS:
        same = event_averages[(event_averages.scenario == "unchanged") & (event_averages.emission_mode == mode)]
        threshold = float(np.quantile(same[same.replicate_seed % 2 == 0].mean_event_gain_per_spike, 0.95))
        for scenario in SCENARIOS:
            selected = event_averages[(event_averages.scenario == scenario) & (event_averages.emission_mode == mode) & (event_averages.replicate_seed % 2 == 1)]
            values = selected.mean_event_gain_per_spike.to_numpy()
            rows.append({
                "scenario": scenario, "emission_mode": mode, "n_evaluation_replicates": len(values),
                "null_calibration_p95": threshold, "median_gain_per_spike": float(np.median(values)),
                "p05_gain_per_spike": float(np.quantile(values, 0.05)),
                "p95_gain_per_spike": float(np.quantile(values, 0.95)),
                "above_null_p95_fraction": float(np.mean(values > threshold)),
                "interpretation": "positive_control" if scenario == "order_changed" else "negative_control",
            })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=64)
    parser.add_argument("--calibration-events", type=int, default=80)
    parser.add_argument("--test-events", type=int, default=40)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.replicates < 8 or args.calibration_events < 10 or args.test_events < 5 or args.workers < 1:
        raise ValueError("invalid pilot size")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    protocol = ROOT / "docs/hc11_phase_order_protocol.md"
    manifest = build_script_provenance(input_paths={"protocol": protocol})
    manifest.update(status="running", created_at_utc=datetime.now(UTC).isoformat(),
                    simulation_only=True, latent_trajectory_supplied=False,
                    emission_maps_supplied=True, parameters=vars(args) | {"output_dir": str(output)},
                    seed_start=20260924, n_bins=20, bin_width_ms=20, forecast_ms=40)
    path = output / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    tasks = [(20260924 + r, scenario, args.calibration_events, args.test_events, 20)
             for r in range(args.replicates) for scenario in SCENARIOS]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, result in enumerate(pool.map(replicate, tasks), 1):
            rows.extend(result)
            if i % 10 == 0:
                print(f"completed {i}/{len(tasks)} simulation cases", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(output / "simulation_phase_scores.csv", index=False)
    summary = summarize(table)
    summary.to_csv(output / "simulation_recovery_summary.csv", index=False)
    # This is an engineering screen, not a hypothesis test on real animals.
    gates = []
    for mode in EMISSIONS:
        part = summary[summary.emission_mode == mode]
        negative = part[part.interpretation == "negative_control"]
        positive = part[part.interpretation == "positive_control"]
        gates.extend([
            {"gate": mode + "_negative_control_screen", "passed": bool((negative.above_null_p95_fraction <= 0.15).all())},
            {"gate": mode + "_positive_control_screen", "passed": bool((positive.above_null_p95_fraction >= 0.80).all())},
        ])
    gates.extend([
        {"gate": "all_rows_present", "passed": len(table) == len(tasks) * 4},
        {"gate": "all_fits_converged", "passed": bool(table.own_fit_converged.all() and table.other_fit_converged.all())},
        {"gate": "occupancy_and_dwell_matched", "passed": bool((table.target_equilibrium_error < 1e-10).all() and (table.target_dwell_error < 1e-12).all())},
    ])
    pd.DataFrame(gates).to_csv(output / "calibration_gate_summary.csv", index=False)
    manifest.update(status="complete", completed_at_utc=datetime.now(UTC).isoformat(),
                    calibration_screen_passed=all(g["passed"] for g in gates),
                    real_data_analysis_authorized=False,
                    output_sha256={p.name: file_sha256(p) for p in output.glob("*.csv")})
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
