#!/usr/bin/env python3
"""Risk-constrained population exchange with grouped internal validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csc_array

from scripts import exchange_content as exact
from scripts import audit_exchange_content as proof
from scripts import audit_joint_exchange_truth as checked
from scripts import joint_exchange_content as joint
from scripts._provenance import build_script_provenance, file_sha256
from scripts.report_content_screening_bound import markdown

SOURCES = ("run_q3", "cal_poisson_gain1", "cal_conditional")
BUDGETS = (1, 2, 4, 8)
OPTIONS = dict(time_limit=20.0, node_limit=64, mip_rel_gap=0.001)


def grouped_split(bank, session, source):
    groups = bank.get("parent_ids", np.arange(len(bank["counts"])))
    if len(groups) != len(bank["counts"]) or not np.isin(bank["labels"], [0, 1]).all():
        raise ValueError("invalid groups or labels")
    ids, first = np.unique(groups, return_index=True)
    hold = []
    for k in (0, 1):
        eligible = ids[bank["labels"][first] == k]
        order = sorted(map(int, eligible), key=lambda g: hashlib.sha256(f"20260915|robust-exchange|{session}|{source}|{g}".encode()).digest())
        hold.extend(order[: max(3, round(0.25 * len(order)))])
    validation = np.flatnonzero(np.isin(groups, hold))
    train = np.flatnonzero(~np.isin(groups, hold))
    for ix, minimum in ((train, 10), (validation, 3)):
        for k in (0, 1):
            if len(np.unique(groups[ix][bank["labels"][ix] == k])) < minimum:
                raise ValueError("insufficient grouped truth support")
    return dict(train=train.tolist(), validation=validation.tolist())


def subset(bank, ix):
    return {k: bank[k][ix] for k in ("counts", "truth_cm", "labels")}


def state(enc, bank, pair):
    ll = {s: exact.likelihood(bank["counts"], enc["early_run"], pair[s]) for s in ("high", "low")}
    result = exact.readout(ll, enc["grid_cm"], enc["near"], bank["truth_cm"], bank["labels"])
    _, _, risks, _, objective = proof.state_for(pair, enc, bank)
    np.testing.assert_allclose(result["risks"], risks, atol=1e-9, rtol=1e-9)
    np.testing.assert_allclose(result["objective"], objective, atol=1e-10, rtol=1e-9)
    return result


def derivatives(enc, bank, pair):
    result = state(enc, bank, pair)
    counts, rates, grid = bank["counts"], enc["early_run"], enc["grid_cm"]
    log_rates = np.log(rates)
    labels = bank["labels"]
    risk_gradients, objective_gradients = [], []
    for p, home in zip(result["probabilities"], result["homes"], strict=True):
        expected = counts * (p @ log_rates.T) - 0.02 * (p @ rates.T)
        ph = p * enc["near"]
        dh = counts * (ph @ log_rates.T) - 0.02 * (ph @ rates.T) - home[:, None] * expected
        mean = p @ grid
        residual = mean - bank["truth_cm"]
        norm = np.linalg.norm(residual, axis=1)
        direction = np.divide(residual, norm[:, None], out=np.zeros_like(residual), where=norm[:, None] > 0)
        de = np.zeros_like(expected)
        for d in range(grid.shape[1]):
            px = p * grid[:, d]
            dm = counts * (px @ log_rates.T) - 0.02 * (px @ rates.T) - mean[:, d, None] * expected
            de += direction[:, d, None] * dm
        db = 2 * (home - labels)[:, None] * dh
        risk_gradients.append(np.stack([g[labels == k].mean(axis=0) for k in (0, 1) for g in (de, db)]))
        objective_gradients.append(sum(result["delta"][k] * dh[labels == k].mean(axis=0) for k in (0, 1)))
    # Low-population Home mass enters high-minus-low J with the opposite sign.
    objective_gradients[1] *= -1
    return result, risk_gradients, objective_gradients


def coefficients(enc, bank, original):
    result, risks, objective = derivatives(enc, bank, original)
    high = sorted(set(original["high"]) - set(original["low"]))
    low = sorted(set(original["low"]) - set(original["high"]))
    grad = np.r_[-objective[0][high] + objective[1][high], objective[0][low] - objective[1][low]]
    risk = np.vstack([np.c_[-risks[0][:, high], risks[0][:, low]], np.c_[risks[1][:, high], -risks[1][:, low]]])
    return result, grad, risk, high, low


def proposals(gradient, risk_gradient, baseline_j, baseline_risks, high, low):
    n, h = len(high) + len(low), len(high)
    for budget in BUDGETS:
        if budget > min(len(high), len(low)):
            yield dict(budget=budget, index=0, status="budget_unavailable", feasible=False)
            continue
        previous = []
        for index in range(4):
            rows = [np.r_[np.ones(h), np.zeros(n - h)], np.r_[np.zeros(h), np.ones(n - h)]]
            rows.extend(risk_gradient / np.maximum(np.abs(baseline_risks), 1e-6)[:, None])
            scale = max(baseline_j, 1e-6)
            rows.append(gradient / scale)
            rows.extend(previous)
            a = csc_array(np.array(rows))
            lo = np.r_[[budget, budget], np.full(len(baseline_risks), -np.inf), -baseline_j / scale, np.full(len(previous), -np.inf)]
            up = np.r_[[budget, budget], np.zeros(len(baseline_risks)), -1e-10 / scale, np.full(len(previous), 2 * budget - 1)]
            start = time.monotonic()
            solved = milp(gradient / scale, integrality=np.ones(n), bounds=Bounds(np.zeros(n), np.ones(n)), constraints=LinearConstraint(a, lo, up), options=OPTIONS)
            x = joint.feasible_incumbent(solved, a, lo, up)
            row = dict(budget=budget, index=index, status=int(solved.status), feasible=x is not None, runtime_s=time.monotonic() - start, message=str(solved.message))
            if x is None:
                yield row
                break
            chosen_h = np.asarray(high)[x[:h].astype(bool)].tolist()
            chosen_l = np.asarray(low)[x[h:].astype(bool)].tolist()
            row.update(high=chosen_h, low=chosen_l, predicted_j_change=float(gradient @ x), predicted_risk_changes=(risk_gradient @ x).tolist())
            yield row
            previous.append(x)


def select(enc, banks, session):
    original = {s: list(map(int, enc[f"{s}_indices"])) for s in ("high", "low")}
    exact.base.reserve_indices(enc)
    baseline, risk_coefficients = {}, []
    for source in SOURCES:
        s, grad, risk, high, low = coefficients(enc, banks[source], original)
        baseline[source] = s
        risk_coefficients.append(risk)
        if source == "run_q3":
            primary_gradient = grad
    risk0 = np.concatenate([baseline[s]["risks"] for s in SOURCES])
    j0 = baseline["run_q3"]["objective"]
    records, candidates = [], []
    for row in proposals(primary_gradient, np.vstack(risk_coefficients), j0, risk0, high, low):
        records.append(row)
        if not row["feasible"]:
            continue
        pair = joint.joint_pair(original, row["high"], row["low"])
        actual = {s: state(enc, banks[s], pair) for s in SOURCES}
        risks = np.concatenate([actual[s]["risks"] for s in SOURCES])
        j = actual["run_q3"]["objective"]
        candidates.append(
            dict(
                name=f"joint_{row['budget']}_{row['index']}",
                high=row["high"],
                low=row["low"],
                pair=pair,
                objective=j,
                risks=risks.tolist(),
                admissible=bool(j < j0 - 1e-10 and np.all(risks <= risk0 + 1e-10)),
            )
        )
    best = joint.winner(candidates)
    return dict(
        session=session,
        original=original,
        baseline_j=j0,
        baseline_risks=risk0.tolist(),
        proposals=records,
        candidates=candidates,
        chosen=None if best is None else best["name"],
        final_pair=original if best is None else best["pair"],
        net_exchanged=0 if best is None else len(best["high"]),
        final_j=j0 if best is None else best["objective"],
    )


def validation(enc, banks, choice):
    rows = []
    for source in SOURCES:
        before = state(enc, banks[source], choice["original"])
        after = state(enc, banks[source], choice["final_pair"])
        for n, (a, b) in enumerate(zip(before["risks"], after["risks"], strict=True)):
            rows.append(
                dict(
                    session=choice["session"],
                    animal=choice["session"].split("/")[0],
                    source=source,
                    side="high" if n < 4 else "low",
                    true_home=(n % 4) // 2,
                    metric="error_cm" if n % 2 == 0 else "home_brier",
                    observations=int((banks[source]["labels"] == (n % 4) // 2).sum()),
                    baseline=float(a),
                    targeted=float(b),
                    change=float(b - a),
                    nonworsening=bool(b <= a + 1e-10),
                    baseline_j=before["objective"],
                    targeted_j=after["objective"],
                )
            )
    return rows


def run(source_dir, reference_dir, reference_audit, output_dir):
    reference = checked.checked_manifest(reference_dir, reference_audit)
    if Path(reference["source_dir"]) != source_dir:
        raise ValueError("source directory differs from audited reference")
    # Hashing the complete reference preserves lineage; only calibration arrays
    # are decoded below, never Q4, test simulations or replay observations.
    inputs = {
        str(p): file_sha256(p)
        for p in [
            reference_dir / "manifest.json",
            reference_audit,
            Path(__file__),
            ROOT / "docs/robust_exchange_protocol.md",
            ROOT / "scripts/exchange_content.py",
            ROOT / "scripts/audit_exchange_content.py",
            ROOT / "scripts/joint_exchange_content.py",
        ]
    }
    inputs.update(reference["input_file_sha256"])
    inputs[str(ROOT / "scripts/audit_joint_exchange_truth.py")] = file_sha256(ROOT / "scripts/audit_joint_exchange_truth.py")
    for session in exact.base.SESSIONS:
        folder = source_dir / session.replace("/", "_")
        for name in ("encoding",) + SOURCES:
            path = folder / f"{name}.npz"
            sha = file_sha256(path)
            previous = reference["input_file_sha256"].get(str(path))
            if previous != sha:
                raise ValueError("changed calibration input")
            inputs[str(path)] = sha
    output_dir.mkdir(parents=True, exist_ok=False)
    choices, splits, failure_rows = {}, {}, []
    for session in exact.base.SESSIONS:
        folder = source_dir / session.replace("/", "_")
        enc = exact.base.read_npz(folder / "encoding.npz")
        banks = {s: exact.base.read_npz(folder / f"{s}.npz") for s in SOURCES}
        try:
            split = {s: grouped_split(banks[s], session, s) for s in SOURCES}
        except ValueError as exc:
            original = {s: list(map(int, enc[f"{s}_indices"])) for s in ("high", "low")}
            choices[session] = dict(session=session, original=original, final_pair=original, net_exchanged=0, chosen=None, failure_reason=str(exc))
            failure_rows.append(dict(session=session, failure_reason=str(exc)))
            continue
        splits[session] = split
        choices[session] = select(enc, {s: subset(banks[s], split[s]["train"]) for s in SOURCES}, session)
        print("robust choice frozen", session, choices[session]["chosen"], flush=True)
    frozen = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), choices=choices, splits=splits, input_file_sha256=inputs)
    (output_dir / "frozen_assignments.json").write_text(json.dumps(frozen, indent=2) + "\n")
    frozen_hash = file_sha256(output_dir / "frozen_assignments.json")
    rows = []
    for session, split in splits.items():
        folder = source_dir / session.replace("/", "_")
        enc = exact.base.read_npz(folder / "encoding.npz")
        banks = {s: subset(exact.base.read_npz(folder / f"{s}.npz"), split[s]["validation"]) for s in SOURCES}
        rows.extend(validation(enc, banks, choices[session]))
    table = pd.DataFrame(
        rows, columns=["session", "animal", "source", "side", "true_home", "metric", "observations", "baseline", "targeted", "change", "nonworsening", "baseline_j", "targeted_j"]
    )
    table.to_csv(output_dir / "internal_validation.csv", index=False)
    pd.DataFrame(failure_rows, columns=["session", "failure_reason"]).to_csv(output_dir / "support_failures.csv", index=False)
    summary = pd.DataFrame(
        [
            dict(
                session=s,
                animal=s.split("/")[0],
                chosen=c["chosen"],
                net_exchanged=c["net_exchanged"],
                baseline_j=c.get("baseline_j", np.nan),
                final_j=c.get("final_j", np.nan),
                admissible_candidates=sum(x["admissible"] for x in c.get("candidates", [])),
            )
            for s, c in choices.items()
        ]
    )
    summary.to_csv(output_dir / "selection_summary.csv", index=False)
    all_rats = set(s.split("/")[0] for s in exact.base.SESSIONS)
    changing_rats = set(summary.loc[summary.net_exchanged.gt(0), "animal"])
    native = table[table.source.eq("run_q3")].drop_duplicates("session")
    improvements = native.assign(change_j=native.targeted_j - native.baseline_j).groupby("animal").change_j.mean()
    gates = dict(
        all_original_pairs_have_support=not failure_rows and len(choices) == 4,
        all_validation_risks_present=len(table) == 96,
        internal_accuracy_nonworsening=len(table) == 96 and bool(table.nonworsening.all()),
        changed_populations_in_all_rats=changing_rats == all_rats,
        native_validation_improves_all_rats=set(improvements.index) == all_rats and bool((improvements < -1e-10).all()),
    )
    gates["ready_for_truth_preflight"] = all(gates.values())
    pd.DataFrame([dict(gate=k, passed=v) for k, v in gates.items()]).to_csv(output_dir / "gates.csv", index=False)
    (output_dir / "report.md").write_text(
        "# Robust gradient exchange calibration\n\nDevelopment only; no Q4, test-bank or replay scores. No validated remedy or independent recording confirmation.\n\n"
        + markdown(summary)
        + "\n\n"
        + markdown(pd.read_csv(output_dir / "gates.csv"))
        + "\n"
    )
    for path, sha in inputs.items():
        if file_sha256(path) != sha:
            raise ValueError("input changed during run")
    if file_sha256(output_dir / "frozen_assignments.json") != frozen_hash:
        raise ValueError("choices changed after internal validation")
    record = build_script_provenance()
    record.update(
        input_file_sha256=inputs,
        frozen_assignments_sha256=frozen_hash,
        solver_options=OPTIONS,
        external_validation=False,
        validated_remedy=False,
        replay_scored=False,
        q4_scored=False,
        output_sha256={p.name: file_sha256(p) for p in output_dir.iterdir() if p.is_file()},
    )
    (output_dir / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("source-dir", "reference-dir", "reference-audit", "output-dir"):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    a = parser.parse_args()
    run(a.source_dir, a.reference_dir, a.reference_audit, a.output_dir)
