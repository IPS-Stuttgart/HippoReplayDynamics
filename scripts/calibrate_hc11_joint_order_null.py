#!/usr/bin/env python3
"""Joint fixed-emission PRE/POST routing null; simulation calibration only."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from _provenance import build_script_provenance, file_sha256
from simulate_hc11_phase_order_identifiability import (
    EMISSIONS,
    SCENARIOS,
    generate,
    make_world,
    predictive_gain,
    stochastic,
    summarize,
    validate_emission,
)


@dataclass(frozen=True)
class Fit:
    transition: np.ndarray
    initial: np.ndarray
    history: np.ndarray
    converged: bool
    max_mstep_gradient: float


def routing_objective(parameters, counts):
    """Shared off-diagonal affinity, with a POST destination propensity."""
    k = counts.shape[1]
    base, offset = parameters[: k * k].reshape(k, k), parameters[k * k:]
    if counts.shape != (2, k, k) or offset.shape != (k,):
        raise ValueError("two aligned phase count matrices required")
    logits = np.stack((base, base + offset[None, :]))
    for phase in range(2):
        np.fill_diagonal(logits[phase], -np.inf)
    log_routing = logits - logsumexp(logits, axis=2, keepdims=True)
    routing = np.exp(log_routing)
    off = counts.copy()
    for phase in range(2):
        np.fill_diagonal(off[phase], 0)
        np.fill_diagonal(log_routing[phase], 0)
    scale = off.sum()
    residual = off.sum(2, keepdims=True) * routing - off
    value = -np.sum(off * log_routing) / scale
    gradient = np.r_[residual.sum(0).ravel(), residual[1].sum(0)] / scale
    return float(value), gradient


def constrained_mstep(counts, start=None):
    counts = np.asarray(counts, float)
    if counts.ndim != 3 or counts.shape[0] != 2 or counts.shape[1] != counts.shape[2]:
        raise ValueError("two square transition-count matrices required")
    if not np.isfinite(counts).all() or (counts <= 0).any():
        raise ValueError("strictly positive regularized counts required")
    k = counts.shape[1]
    start = np.zeros(k * k + k) if start is None else np.asarray(start, float)
    result = minimize(routing_objective, start, args=(counts,), jac=True, method="L-BFGS-B",
                      options={"maxiter": 500, "ftol": 1e-13, "gtol": 1e-9, "maxls": 50})
    gradient = float(np.max(np.abs(result.jac)))
    if not np.isfinite(result.fun) or gradient > 1e-6:
        raise ValueError(f"joint routing M-step failed: {result.message}; gradient={gradient}")
    base = result.x[: k * k].reshape(k, k)
    offset = result.x[k * k:]
    transitions = []
    for phase in range(2):
        logits = base + phase * offset[None, :]
        logits = logits.copy()
        np.fill_diagonal(logits, -np.inf)
        routing = np.exp(logits - logsumexp(logits, axis=1, keepdims=True))
        stay = np.diag(counts[phase]) / counts[phase].sum(1)
        a = (1 - stay[:, None]) * routing
        np.fill_diagonal(a, stay)
        transitions.append(stochastic(a))
    return np.stack(transitions), result.x, gradient


def estep(likelihoods, initial, transition):
    """Use hmmlearn's exact forward/backward and expected-transition kernels."""
    from hmmlearn import _hmmc

    k = len(initial)
    starts, transitions = np.zeros(k), np.zeros((k, k))
    score = 0.
    for ll in likelihoods:
        logp, forward = _hmmc.forward_log(initial, transition, ll)
        backward = _hmmc.backward_log(initial, transition, ll)
        logpost = forward + backward
        logpost -= logsumexp(logpost, axis=1, keepdims=True)
        starts += np.exp(logpost[0])
        if len(ll) > 1:
            transitions += np.exp(_hmmc.compute_log_xi_sum(forward, transition, backward, ll))
        score += logp
    return float(score), starts, transitions


def fit_model(sequences, emissions, *, common_order, start=None, max_iter=100):
    k = len(emissions[0])
    e = [validate_emission(p, k) for p in emissions]
    if len(sequences) != 2 or len(e) != 2 or e[0].shape != e[1].shape:
        raise ValueError("two aligned phases required")
    likelihoods = []
    for phase in range(2):
        if not sequences[phase]:
            raise ValueError("empty phase")
        phase_ll = []
        for value in sequences[phase]:
            x = np.asarray(value)
            if x.ndim != 2 or x.shape[1] != e[phase].shape[1] or not len(x):
                raise ValueError("aligned nonempty event counts required")
            if not np.isfinite(x).all() or (x < 0).any() or not np.equal(x, np.floor(x)).all():
                raise ValueError("nonnegative integer counts required")
            phase_ll.append(x @ np.log(e[phase]).T)
        likelihoods.append(phase_ll)
    if start is None:
        initial = np.full((2, k), 1 / k)
        transition = np.array([0.5 * np.eye(k) + 0.5 / k] * 2)
    else:
        initial, transition = start.initial.copy(), start.transition.copy()
    history, routing_parameters, gradients = [], None, []
    for iteration in range(max_iter + 1):
        values = [estep(likelihoods[p], initial[p], transition[p]) for p in range(2)]
        objective = sum(v[0] for v in values) + (np.log(initial).sum() + np.log(transition).sum()) / k
        history.append(objective)
        if not np.isfinite(objective):
            raise ValueError("nonfinite regularized objective")
        if len(history) > 1:
            change = history[-1] - history[-2]
            if change < -1e-7:
                raise ValueError(f"regularized EM objective decreased: {change}")
            if change < 1e-4:
                return Fit(transition, initial, np.array(history), True, max(gradients, default=0.))
        if iteration == max_iter:
            break
        starts = np.stack([v[1] for v in values]) + 1 / k
        initial = starts / starts.sum(1, keepdims=True)
        counts = np.stack([v[2] for v in values]) + 1 / k
        if common_order:
            transition, routing_parameters, gradient = constrained_mstep(counts, routing_parameters)
            gradients.append(gradient)
        else:
            transition = counts / counts.sum(2, keepdims=True)
    return Fit(transition, initial, np.array(history), False, max(gradients, default=0.))


def routing_invariance_error(transitions):
    """Shared routing implies off-diagonal POST/PRE log ratios are row+column."""
    k = len(transitions[0])
    rows = [(i, j) for i in range(k) for j in range(k) if i != j]
    design = np.zeros((len(rows), 2 * k))
    target = []
    for r, (i, j) in enumerate(rows):
        design[r, i] = design[r, k + j] = 1
        target.append(np.log(transitions[1, i, j] / transitions[0, i, j]))
    coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
    return float(np.max(np.abs(design @ coefficients - target)))


def relabelling_check(seed=20260924):
    """Exact counterexample: unrestricted phase emissions can absorb reversal."""
    pre, post = make_world(seed, "order_changed")
    permutation = (-np.arange(len(pre[0]))) % len(pre[0])
    emissions_as_nuisance = post[1][permutation]
    sequences = generate(np.random.default_rng(seed), post, 20, 20)
    pi = np.full(len(pre[0]), 1 / len(pre[0]))
    errors = []
    for x in sequences:
        a = estep([x @ np.log(post[1]).T], pi, post[0])[0]
        b = estep([x @ np.log(emissions_as_nuisance).T], pi, pre[0])[0]
        errors.append(abs(a - b))
    return {"max_absolute_log_score_difference": max(errors), "n_sequences": len(errors),
            "observationally_equivalent": bool(max(errors) < 1e-10),
            "independent_state_identity_required": True}


def replicate_joint(task):
    seed, scenario, n_cal, n_test = task
    world = make_world(seed, scenario)
    rng = np.random.default_rng(seed + 1000000)
    calibration = [generate(rng, w, n_cal, 20) for w in world]
    test = [generate(rng, w, n_test, 20) for w in world]
    rows = []
    for mode in EMISSIONS:
        emissions = [w[1] for w in world] if mode == "oracle_phase_map" else [(world[0][1] + world[1][1]) / 2] * 2
        common = fit_model(calibration, emissions, common_order=True)
        separate = fit_model(calibration, emissions, common_order=False, start=common)
        if separate.history[-1] < common.history[-1] - 1e-7:
            raise ValueError("nested alternative fit worse than shared-order null")
        for phase in range(2):
            gains = []
            for x in test[phase]:
                score, spikes, _ = predictive_gain(x, emissions[phase], separate.transition[phase], common.transition[phase], np.arange(18), np.arange(18, 24))
                if not spikes:
                    raise ValueError("zero target spikes; refusing silent exclusion")
                gains.append(score / spikes)
            rows.append({
                "replicate_seed": seed, "scenario": scenario, "emission_mode": mode,
                "phase": ("PRE", "POST")[phase], "n_events": n_test,
                "mean_event_gain_per_spike": float(np.mean(gains)),
                "null_converged": common.converged, "alternative_converged": separate.converged,
                "null_iterations": len(common.history) - 1, "alternative_iterations": len(separate.history) - 1,
                "null_max_mstep_gradient": common.max_mstep_gradient,
                "null_routing_invariance_error": routing_invariance_error(common.transition),
                "training_objective_increment": float(separate.history[-1] - common.history[-1]),
            })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.replicates < 8 or args.workers < 1:
        raise ValueError("invalid calibration size")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = build_script_provenance(input_paths={"protocol": ROOT / "docs/hc11_joint_order_null_protocol.md"})
    manifest.update(status="running", created_at_utc=datetime.now(UTC).isoformat(),
                    simulation_only=True, real_data_scored=False, known_emission_maps=True,
                    known_trajectory=False, seed_start=20260924, replicates=args.replicates,
                    calibration_events_per_phase=80, test_events_per_phase=40)
    path = output / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    identity = relabelling_check()
    (output / "state_identity_counterexample.json").write_text(json.dumps(identity, indent=2) + "\n")
    tasks = [(20260924 + r, s, 80, 40) for r in range(args.replicates) for s in SCENARIOS]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, result in enumerate(pool.map(replicate_joint, tasks), 1):
            rows.extend(result)
            if i % 10 == 0:
                print(f"completed {i}/{len(tasks)} joint-fit simulation cases", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(output / "joint_order_phase_scores.csv", index=False)
    summary = summarize(table)
    summary.to_csv(output / "joint_order_recovery_summary.csv", index=False)
    gates = [
        {"gate": "all_rows_present", "passed": len(table) == len(tasks) * 4},
        {"gate": "all_fits_converged", "passed": bool(table.null_converged.all() and table.alternative_converged.all())},
        {"gate": "shared_order_constraint", "passed": bool((table.null_routing_invariance_error < 1e-10).all())},
    ]
    for mode in EMISSIONS:
        sub = summary[summary.emission_mode == mode]
        gates.extend([
            {"gate": mode + "_negative_control_screen", "passed": bool((sub[sub.interpretation.eq("negative_control")].above_null_p95_fraction <= .15).all())},
            {"gate": mode + "_positive_control_screen", "passed": bool((sub[sub.interpretation.eq("positive_control")].above_null_p95_fraction >= .8).all())},
        ])
    pd.DataFrame(gates).to_csv(output / "joint_order_gate_summary.csv", index=False)
    manifest.update(status="complete", completed_at_utc=datetime.now(UTC).isoformat(),
                    known_map_screen_passed=all(g["passed"] for g in gates),
                    real_data_analysis_authorized=False,
                    output_sha256={p.name: file_sha256(p) for p in output.iterdir() if p.name != "manifest.json"})
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
