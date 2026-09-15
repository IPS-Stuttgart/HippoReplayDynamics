#!/usr/bin/env python3
"""Nonlinear, training-only reserve allocation with integer verification."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, linear_sum_assignment, minimize

from scripts import guarded_reserve_content as previous
from scripts._provenance import build_script_provenance, file_sha256
from scripts.report_content_screening_bound import markdown

robust = previous.robust
base = previous.base
SOURCES = previous.SOURCES
MAX_EVALUATIONS = 120
MAX_ITERATIONS = 60
ROUND_SCALES = (0.0,) + (0.05,) * 4 + (0.2,) * 4


def budgets(reserve):
    maximum = len(reserve) // 2
    return sorted({b for b in (1, 2, 4, maximum) if 1 <= b <= maximum})


def membership(enc, added, budget):
    reserve = set(map(int, base.reserve_indices(enc)))
    if budget not in budgets(reserve) or set(added) != set(base.SIDES):
        raise ValueError("invalid budget or missing side")
    if any(len(added[s]) != len(set(added[s])) or len(added[s]) != budget for s in base.SIDES):
        raise ValueError("unequal or repeated additions")
    a, b = (set(added[s]) for s in base.SIDES)
    if not (a | b) <= reserve or a & b:
        raise ValueError("nonreserve or shared additions")
    return {s: sorted(set(map(int, enc[f"{s}_indices"])) | set(added[s])) for s in base.SIDES}


def random_added(session, reserve, budget, draw):
    order = np.random.default_rng(base.seed(session, draw)).permutation(reserve)
    return dict(high=sorted(map(int, order[:budget])), low=sorted(map(int, order[budget : 2 * budget])))


class NonlinearLoss:
    """Fractional likelihood powers propose cells; only binary scores certify them."""

    def __init__(self, enc, banks):
        self.enc = enc
        self.reserve = base.reserve_indices(enc)
        self.n = len(self.reserve)
        self.banks = banks
        self.rates = enc["early_run"][self.reserve]
        self.log_rates = np.log(self.rates)
        self.original = {s: sorted(map(int, enc[f"{s}_indices"])) for s in base.SIDES}
        self.offsets = {source: {s: robust.exact.likelihood(banks[source]["counts"], enc["early_run"], self.original[s]) for s in base.SIDES} for source in SOURCES}
        self.counts = {s: banks[s]["counts"][:, self.reserve] for s in SOURCES}
        self.last_x, self.last_result = None, None

    def __call__(self, weights):
        weights = np.asarray(weights, float)
        if weights.shape != (2 * self.n,) or not np.isfinite(weights).all():
            raise ValueError("invalid fractional weights")
        if self.last_x is not None and np.array_equal(weights, self.last_x):
            return self.last_result
        risks, jacobians = [], []
        objective, objective_gradient = None, None
        grid, near = self.enc["grid_cm"], self.enc["near"]
        for source in SOURCES:
            bank = self.banks[source]
            counts = self.counts[source]
            likelihoods = {}
            for index, side in enumerate(base.SIDES):
                w = weights[index * self.n : (index + 1) * self.n]
                likelihoods[side] = self.offsets[source][side] + (counts * w) @ self.log_rates - 0.02 * (w @ self.rates)
            result = robust.exact.readout(likelihoods, grid, near, bank["truth_cm"], bank["labels"])
            risks.extend(result["risks"])
            derivatives, dj = [], []
            for p, home in zip(result["probabilities"], result["homes"], strict=True):
                expected = counts * (p @ self.log_rates.T) - 0.02 * (p @ self.rates.T)
                ph = p * near
                dh = counts * (ph @ self.log_rates.T) - 0.02 * (ph @ self.rates.T) - home[:, None] * expected
                mean = p @ grid
                residual = mean - bank["truth_cm"]
                norm = np.linalg.norm(residual, axis=1)
                direction = np.divide(residual, norm[:, None], out=np.zeros_like(residual), where=norm[:, None] > 0)
                de = np.zeros_like(expected)
                for d in range(grid.shape[1]):
                    px = p * grid[:, d]
                    dm = counts * (px @ self.log_rates.T) - 0.02 * (px @ self.rates.T) - mean[:, d, None] * expected
                    de += direction[:, d, None] * dm
                db = 2 * (home - bank["labels"])[:, None] * dh
                derivatives.append(np.stack([g[bank["labels"] == k].mean(axis=0) for k in (0, 1) for g in (de, db)]))
                dj.append(sum(result["delta"][k] * dh[bank["labels"] == k].mean(axis=0) for k in (0, 1)))
            zero = np.zeros_like(derivatives[0])
            jacobians.extend(np.block([[derivatives[0], zero], [zero, derivatives[1]]]))
            if source == "run_q3":
                objective = result["objective"]
                objective_gradient = np.r_[dj[0], -dj[1]]
        value = np.asarray(risks), objective, np.asarray(jacobians), objective_gradient
        if not all(np.isfinite(v).all() for v in value):
            raise ValueError("nonfinite nonlinear state")
        self.last_x, self.last_result = weights.copy(), value
        return value


def linear_feasible(x, n, budget):
    return bool(
        np.isfinite(x).all()
        and np.min(x) >= -1e-7
        and np.max(x) <= 1 + 1e-7
        and abs(x[:n].sum() - budget) <= 1e-6
        and abs(x[n:].sum() - budget) <= 1e-6
        and np.all(x[:n] + x[n:] <= 1 + 1e-6)
    )


def rounded(weights, reserve, budget, session, tag, draw):
    n = len(reserve)
    scores = np.array(weights).reshape(2, n).copy()
    scale = ROUND_SCALES[draw]
    seed = int.from_bytes(hashlib.sha256(f"20260915|nonlinear-reserve-round|{session}|{tag}|{draw}".encode()).digest()[:8], "little")
    scores += scale * np.random.default_rng(seed).standard_normal(scores.shape)
    costs = np.vstack([np.tile(-scores[0], (budget, 1)), np.tile(-scores[1], (budget, 1)), np.zeros((n - 2 * budget, n))])
    rows, columns = linear_sum_assignment(costs)
    return dict(high=sorted(map(int, np.array(reserve)[columns[rows < budget]])), low=sorted(map(int, np.array(reserve)[columns[(rows >= budget) & (rows < 2 * budget)]])))


class EvaluationBudget(Exception):
    pass


def optimize(loss, initial, budget, mode):
    n = loss.n
    baseline_risk, baseline_j, _, _ = loss(np.zeros(2 * n))
    scale = np.maximum(abs(baseline_risk), 1e-6)
    j_scale = max(baseline_j, 1e-6)
    trace, best = [], None
    saved_x, saved_value = None, None

    def evaluate(z):
        nonlocal saved_x, saved_value, best
        x = z[: 2 * n]
        if saved_x is not None and np.array_equal(x, saved_x):
            return saved_value
        if len(trace) >= MAX_EVALUATIONS:
            raise EvaluationBudget
        risk, j, jac, dj = loss(x)
        violation = float(max(0, np.max((risk - baseline_risk - 1e-10) / scale)))
        row = dict(evaluation=len(trace), weights=x.tolist(), objective=j, risks=risk.tolist(), max_scaled_violation=violation, linear_feasible=linear_feasible(x, n, budget))
        trace.append(row)
        key = (violation, j) if mode == "minimax" else (violation > 1e-8, j if violation <= 1e-8 else violation, j)
        if row["linear_feasible"] and (best is None or key < best[0]):
            best = key, row
        saved_x, saved_value = x.copy(), (risk, j, jac, dj)
        return saved_value

    matrix = np.block([[np.ones((1, n)), np.zeros((1, n))], [np.zeros((1, n)), np.ones((1, n))], [np.eye(n), np.eye(n)]])
    lower, upper = np.r_[budget, budget, np.zeros(n)], np.r_[budget, budget, np.ones(n)]
    dimension = 2 * n + (mode == "minimax")
    if mode == "minimax":
        matrix = np.c_[matrix, np.zeros(n + 2)]
        initial_risk = loss(initial)[0]
        z0 = np.r_[initial, max(0, np.max((initial_risk - baseline_risk) / scale)) + 1e-6]
        bounds = Bounds(np.zeros(dimension), np.r_[np.ones(2 * n), np.inf])

        def fun(z):
            evaluate(z)
            return z[-1]

        def jac_fun(z):
            return np.r_[np.zeros(2 * n), 1.0]

        def constraint(z):
            r, _, _, _ = evaluate(z)
            return (baseline_risk + 1e-10 - r) / scale + z[-1]

        def jac_constraint(z):
            _, _, jac, _ = evaluate(z)
            return np.c_[-jac / scale[:, None], np.ones(24)]
    else:
        z0, bounds = initial, Bounds(np.zeros(dimension), np.ones(dimension))

        def fun(z):
            return evaluate(z)[1] / j_scale

        def jac_fun(z):
            return evaluate(z)[3] / j_scale

        def constraint(z):
            return (baseline_risk + 1e-10 - evaluate(z)[0]) / scale

        def jac_constraint(z):
            return -evaluate(z)[2] / scale[:, None]

    start = time.monotonic()
    try:
        result = minimize(
            fun,
            z0,
            jac=jac_fun,
            method="SLSQP",
            bounds=bounds,
            constraints=[
                LinearConstraint(matrix[:2], lower[:2], upper[:2]),
                LinearConstraint(matrix[2:], lower[2:], upper[2:]),
                {"type": "ineq", "fun": constraint, "jac": jac_constraint},
            ],
            options=dict(maxiter=MAX_ITERATIONS, ftol=1e-9),
        )
        evaluate(result.x)
        status, message = int(result.status), str(result.message)
    except EvaluationBudget:
        status, message = -1, "fixed nonlinear evaluation budget reached"
    if best is None:
        raise ValueError("no quota-feasible fractional iterate")
    return dict(
        mode=mode,
        budget=budget,
        initial=initial.tolist(),
        status=status,
        message=message,
        runtime_s=time.monotonic() - start,
        evaluations=len(trace),
        trace=trace,
        selected=best[1],
    )


def select(enc, banks, session):
    loss = NonlinearLoss(enc, banks)
    reserve, n = loss.reserve, loss.n
    risk0, j0, _, _ = loss(np.zeros(2 * n))
    original = loss.original
    verified_risk, verified_j = previous.measurements(enc, banks, original)
    np.testing.assert_allclose(risk0, verified_risk, atol=1e-9, rtol=1e-9)
    np.testing.assert_allclose(j0, verified_j, atol=1e-10, rtol=1e-9)
    searches, candidates, cache = [], [], {}

    def candidate(name, family, added, budget):
        pair = membership(enc, added, budget)
        key = tuple(pair["high"]), tuple(pair["low"])
        if key not in cache:
            cache[key] = previous.measurements(enc, banks, pair)
        r, j = cache[key]
        record = dict(
            name=name,
            family=family,
            budget=budget,
            added=added,
            pair=pair,
            risks=r.tolist(),
            objective=j,
            failed_risks=int(np.count_nonzero(r > risk0 + 1e-10)),
            admissible=bool(j < j0 - 1e-10 and np.all(r <= risk0 + 1e-10)),
        )
        candidates.append(record)
        return record

    for budget in budgets(reserve):
        random = [candidate(f"budget_{budget}_random_{i:02}", "random", random_added(session, reserve, budget, i), budget) for i in range(20)]
        best_random = min(random, key=lambda r: (float(np.maximum(0, (np.array(r["risks"]) - risk0) / np.maximum(abs(risk0), 1e-6)).max()), r["objective"], r["name"]))
        starts = [np.full(2 * n, budget / n), np.array([i in best_random["added"][s] for s in base.SIDES for i in reserve], float)]
        for index, initial in enumerate(starts):
            for mode in ("minimax", "guarded_objective"):
                tag = f"budget_{budget}_start_{index}_{mode}"
                record = optimize(loss, initial, budget, mode)
                record["name"] = tag
                searches.append(record)
                for draw in range(len(ROUND_SCALES)):
                    added = rounded(record["selected"]["weights"], reserve, budget, session, tag, draw)
                    candidate(f"{tag}_round_{draw}", "nonlinear_round", added, budget)
                initial = np.array(record["selected"]["weights"])
                print("nonlinear reserve search", session, tag, record["status"], record["selected"]["max_scaled_violation"], flush=True)
    allowed = [c for c in candidates if c["admissible"]]
    best = None
    if allowed:
        best_j = min(c["objective"] for c in allowed)
        best = min((c for c in allowed if c["objective"] - best_j <= 1e-12), key=lambda c: (c["budget"], tuple(c["added"]["high"]), tuple(c["added"]["low"]), c["name"]))
    return dict(
        session=session,
        original=original,
        reserve=reserve.tolist(),
        baseline_risks=risk0.tolist(),
        baseline_j=j0,
        searches=searches,
        candidates=candidates,
        chosen=None if best is None else best["name"],
        budget=0 if best is None else best["budget"],
        added=dict(high=[], low=[]) if best is None else best["added"],
        final_pair=original if best is None else best["pair"],
        final_j=j0 if best is None else best["objective"],
    )


def worker(args):
    source, session, splits, output = args
    folder = source / session.replace("/", "_")
    enc = base.read_npz(folder / "encoding.npz")
    banks = {}
    for s in SOURCES:
        bank = base.read_npz(folder / f"{s}.npz")
        previous.partition_audit.check_partition(bank, splits[s], session, s)
        banks[s] = robust.subset(bank, splits[s]["train"])
    choice = select(enc, banks, session)
    (output / f"{session.replace('/', '_')}_choice.json").write_text(json.dumps(choice, indent=2) + "\n")
    print("frozen nonlinear reserve choice", session, choice["chosen"], flush=True)
    return choice


def run(source, reference_dir, reference_audit, output, workers=4):
    reference = robust.checked.checked_manifest(reference_dir, reference_audit)
    splits = json.loads((reference_dir / "frozen_assignments.json").read_text())["splits"]
    if set(splits) != set(base.SESSIONS):
        raise ValueError("missing original cohort")
    inputs = dict(reference["input_file_sha256"])
    for p in (
        reference_dir / "manifest.json",
        reference_dir / "frozen_assignments.json",
        reference_audit,
        Path(__file__),
        ROOT / "docs/nonlinear_reserve_content_protocol.md",
        ROOT / "scripts/guarded_reserve_content.py",
    ):
        inputs[str(p)] = file_sha256(p)
    for session in base.SESSIONS:
        for name in ("encoding",) + SOURCES:
            path = source / session.replace("/", "_") / f"{name}.npz"
            if inputs.get(str(path)) != file_sha256(path):
                raise ValueError("changed source")
    output.mkdir(parents=True, exist_ok=False)
    (output / "pre_selection.json").write_text(json.dumps(dict(created_at_utc=datetime.now(timezone.utc).isoformat(), input_file_sha256=inputs, splits=splits), indent=2) + "\n")
    jobs = [(source, s, splits[s], output) for s in base.SESSIONS]
    if workers == 1:
        results = [worker(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(worker, jobs))
    choices = {c["session"]: c for c in results}
    frozen = output / "frozen_assignments.json"
    frozen.write_text(json.dumps(dict(created_at_utc=datetime.now(timezone.utc).isoformat(), choices=choices, splits=splits, input_file_sha256=inputs), indent=2) + "\n")
    frozen_hash = file_sha256(frozen)
    started = datetime.now(timezone.utc).isoformat()
    rows = []
    for session in base.SESSIONS:
        folder = source / session.replace("/", "_")
        enc = base.read_npz(folder / "encoding.npz")
        banks = {s: robust.subset(base.read_npz(folder / f"{s}.npz"), splits[session][s]["validation"]) for s in SOURCES}
        rows.extend(robust.validation(enc, banks, choices[session]))
    validation = pd.DataFrame(rows)
    validation.to_csv(output / "internal_validation.csv", index=False)
    summary = pd.DataFrame(
        [
            dict(
                session=s,
                animal=s.split("/")[0],
                searches=len(c["searches"]),
                candidates=len(c["candidates"]),
                admissible_candidates=sum(v["admissible"] for v in c["candidates"]),
                chosen=c["chosen"],
                budget=c["budget"],
                baseline_j=c["baseline_j"],
                targeted_j=c["final_j"],
            )
            for s, c in choices.items()
        ]
    )
    summary.to_csv(output / "selection_summary.csv", index=False)
    gates = pd.DataFrame([dict(gate=k, passed=v) for k, v in previous.gate_values(validation, choices).items()])
    gates.to_csv(output / "gates.csv", index=False)
    (output / "report.md").write_text(
        "# Nonlinear reserve acquisition\n\nDevelopment only. All original pairs and neurons retained. Continuous powers only propose integer additions; no fractional decoder is promoted.\n\n"
        + markdown(summary)
        + "\n\n"
        + markdown(gates)
        + "\n\nAny failure blocks Q4/test/replay/independent-recording scoring. An unchanged baseline is not a remedy.\n"
    )
    for path, sha in inputs.items():
        if sha is None or file_sha256(path) != sha:
            raise ValueError("input changed")
    if file_sha256(frozen) != frozen_hash:
        raise ValueError("choices changed after validation")
    manifest = {
        **build_script_provenance(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation_started_at_utc": started,
        "source_dir": str(source),
        "reference_dir": str(reference_dir),
        "reference_audit": str(reference_audit),
        "input_file_sha256": inputs,
        "frozen_assignments_sha256": frozen_hash,
        "max_evaluations_per_search": MAX_EVALUATIONS,
        "max_iterations_per_search": MAX_ITERATIONS,
        "round_scales": ROUND_SCALES,
        "workers": workers,
        "q4_scored": False,
        "test_banks_scored": False,
        "replay_scored": False,
        "external_validation": False,
        "validated_remedy": False,
        "output_sha256": {p.name: file_sha256(p) for p in output.iterdir()},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("source-dir", "reference-dir", "reference-audit", "output-dir"):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4, choices=range(1, 5))
    a = parser.parse_args()
    run(a.source_dir, a.reference_dir, a.reference_audit, a.output_dir, a.workers)
