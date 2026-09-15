#!/usr/bin/env python3
"""Complete finite-swap calibration costs with risk-guarded joint proposals."""

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
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csc_matrix, vstack

from scripts import robust_exchange_content as robust
from scripts import audit_robust_exchange_content as audit
from scripts._provenance import build_script_provenance, file_sha256
from scripts.report_content_screening_bound import markdown

SOURCES = robust.SOURCES
RISK_COLUMNS = tuple(f"risk_{i}" for i in range(24))


def original_pair(enc):
    robust.exact.base.reserve_indices(enc)
    return {s: enc[f"{s}_indices"].astype(int).tolist() for s in ("high", "low")}


def single_costs(enc, banks, session):
    pair = original_pair(enc)
    high = sorted(set(pair["high"]) - set(pair["low"]))
    low = sorted(set(pair["low"]) - set(pair["high"]))
    rates, counts = enc["early_run"], {s: banks[s]["counts"] for s in SOURCES}
    if not np.isfinite(rates).all() or np.any(rates <= 0):
        raise ValueError("invalid encoding rates")
    for source in SOURCES:
        if not np.isfinite(counts[source]).all() or np.any(counts[source] < 0) or not np.isfinite(banks[source]["truth_cm"]).all():
            raise ValueError("invalid calibration observations")
        if set(map(int, np.unique(banks[source]["labels"]))) != {0, 1}:
            raise ValueError("missing truth class")
    baseline = {s: robust.state(enc, banks[s], pair) for s in SOURCES}
    risks0 = np.concatenate([baseline[s]["risks"] for s in SOURCES])
    j0 = baseline["run_q3"]["objective"]
    ll = {s: {side: robust.exact.likelihood(counts[s], rates, pair[side]) for side in pair} for s in SOURCES}
    logs = np.log(rates)
    rows = []
    for index, h in enumerate(high):
        removed = {s: counts[s][:, h, None] * logs[h] - 0.02 * rates[h] for s in SOURCES}
        for lo in low:
            values = []
            for s in SOURCES:
                change = counts[s][:, lo, None] * logs[lo] - 0.02 * rates[lo] - removed[s]
                values.append(
                    robust.exact.readout({"high": ll[s]["high"] + change, "low": ll[s]["low"] - change}, enc["grid_cm"], enc["near"], banks[s]["truth_cm"], banks[s]["labels"])
                )
            risks = np.concatenate([v["risks"] for v in values])
            objective = values[0]["objective"]
            if not np.isfinite(risks).all() or not np.isfinite(objective):
                raise ValueError("nonfinite finite-swap score")
            row = dict(high=h, low=lo, objective=objective, objective_improvement=j0 - objective, admissible=bool(objective < j0 - 1e-10 and np.all(risks <= risks0 + 1e-10)))
            row.update(zip(RISK_COLUMNS, risks, strict=True))
            rows.append(row)
        if index % 10 == 0 or index == len(high) - 1:
            print("finite costs", session, len(rows), "/", len(high) * len(low), flush=True)
    return pd.DataFrame(rows, columns=["high", "low", "objective", "objective_improvement", "admissible", *RISK_COLUMNS]), risks0, j0


def verify_singles(enc, banks, session, table, risks0, j0):
    pair = original_pair(enc)
    expected = {(h, lo) for h in set(pair["high"]) - set(pair["low"]) for lo in set(pair["low"]) - set(pair["high"])}
    if table.duplicated(["high", "low"]).any() or set(zip(table.high, table.low, strict=True)) != expected:
        raise ValueError("incomplete single-swap neighborhood")
    sentinels = sorted(expected, key=lambda x: hashlib.sha256(f"20260915|finite-exchange|{session}|{x[0]}|{x[1]}".encode()).digest())[:32]
    checked = set(sentinels) | set(zip(table.loc[table.admissible, "high"], table.loc[table.admissible, "low"], strict=True))
    if not table.empty:
        best = table.sort_values(["objective", "high", "low"]).iloc[0]
        checked.add((int(best.high), int(best.low)))
    indexed, maximum = table.set_index(["high", "low"]), 0.0
    for h, lo in sorted(checked):
        altered = audit.exchanged(pair, [h], [lo])
        states = [audit.measure(enc, banks[s], altered) for s in SOURCES]
        risks, j = np.concatenate([v[0] for v in states]), states[0][1]
        row = indexed.loc[(h, lo)]
        actual = row[list(RISK_COLUMNS)].to_numpy(float)
        np.testing.assert_allclose(actual, risks, atol=1e-9, rtol=1e-9)
        np.testing.assert_allclose([row.objective, row.objective_improvement], [j, j0 - j], atol=1e-10, rtol=1e-9)
        if row.admissible != bool(j < j0 - 1e-10 and np.all(risks <= risks0 + 1e-10)):
            raise ValueError("single-swap admissibility mismatch")
        maximum = max(maximum, abs(row.objective - j), float(np.max(np.abs(actual - risks))))
    return dict(
        complete_neighborhood=len(expected),
        numerically_reconstructed=len(checked),
        max_absolute_error=maximum,
        full_neighborhood_independently_reconstructed=len(checked) == len(expected),
    )


def matching_problem(table, risks0, j0, budget, previous):
    n = len(table)
    high, low = table.high.to_numpy(int), table.low.to_numpy(int)
    risk_delta = table[list(RISK_COLUMNS)].to_numpy().T - risks0[:, None]
    gradient = -table.objective_improvement.to_numpy() / max(j0, 1e-6)
    rows = [csc_matrix(np.ones((1, n)))]
    lower, upper = [float(budget)], [float(budget)]
    for cells in (high, low):
        ids = np.unique(cells)
        rows.append(csc_matrix(np.asarray([cells == cell for cell in ids], float)))
        lower.extend([-np.inf] * len(ids))
        upper.extend([1.0] * len(ids))
    rows.extend([csc_matrix(risk_delta / np.maximum(np.abs(risks0), 1e-6)[:, None]), csc_matrix(gradient[None, :])])
    lower.extend([-np.inf] * 24 + [-j0 / max(j0, 1e-6)])
    upper.extend([0.0] * 24 + [-1e-10 / max(j0, 1e-6)])
    for h, lo in previous:
        coefficients = np.isin(high, h).astype(int) + np.isin(low, lo).astype(int)
        rows.append(csc_matrix(coefficients[None, :]))
        lower.append(-np.inf)
        upper.append(2 * budget - 1)
    return gradient, vstack(rows, format="csc"), np.array(lower), np.array(upper)


def joint_proposals(table, risks0, j0):
    for budget in (2, 4, 8):
        previous = []
        if budget > min(table.high.nunique(), table.low.nunique()):
            yield dict(budget=budget, index=0, status="budget_unavailable", feasible=False)
            continue
        for index in range(4):
            c, matrix, lower, upper = matching_problem(table, risks0, j0, budget, previous)
            start = time.monotonic()
            solved = milp(c, integrality=np.ones(len(c)), bounds=Bounds(0, 1), constraints=LinearConstraint(matrix, lower, upper), options=robust.OPTIONS)
            x = robust.joint.feasible_incumbent(solved, matrix, lower, upper)
            row = dict(
                budget=budget,
                index=index,
                status=int(solved.status),
                feasible=x is not None,
                runtime_s=time.monotonic() - start,
                message=str(solved.message),
                approximation="sum_of_measured_single_swap_changes",
            )
            if x is None:
                yield row
                break
            selected = table.loc[x.astype(bool)]
            high, low = sorted(selected.high.astype(int)), sorted(selected.low.astype(int))
            row.update(
                high=high,
                low=low,
                edges=selected[["high", "low"]].astype(int).values.tolist(),
                predicted_j_change=float(-selected.objective_improvement.sum()),
                predicted_risk_changes=(selected[list(RISK_COLUMNS)].to_numpy() - risks0).sum(axis=0).tolist(),
            )
            yield row
            previous.append((high, low))


def exact_candidate(enc, banks, original, proposal, risks0, j0):
    pair = robust.joint.joint_pair(original, proposal["high"], proposal["low"])
    states = [robust.state(enc, banks[s], pair) for s in SOURCES]
    risks, j = np.concatenate([s["risks"] for s in states]), states[0]["objective"]
    return dict(
        name=f"joint_{proposal['budget']}_{proposal['index']}",
        high=proposal["high"],
        low=proposal["low"],
        pair=pair,
        objective=j,
        risks=risks.tolist(),
        admissible=bool(j < j0 - 1e-10 and np.all(risks <= risks0 + 1e-10)),
    )


def select(enc, banks, session, output_dir):
    table, risks0, j0 = single_costs(enc, banks, session)
    check = verify_singles(enc, banks, session, table, risks0, j0)
    path = output_dir / f"{session.replace('/', '_')}_single_costs.csv"
    table.to_csv(path, index=False)
    pd.testing.assert_frame_equal(pd.read_csv(path), table, check_dtype=False, atol=1e-9, rtol=1e-9)
    print("single costs checked", session, check, flush=True)
    original, records, candidates = original_pair(enc), [], []
    for index, row in enumerate(table.loc[table.admissible].sort_values(["high", "low"]).itertuples()):
        proposal = dict(
            budget=1,
            index=index,
            status=0,
            feasible=True,
            high=[int(row.high)],
            low=[int(row.low)],
            predicted_j_change=-row.objective_improvement,
            predicted_risk_changes=[getattr(row, k) - risks0[i] for i, k in enumerate(RISK_COLUMNS)],
            approximation="exact_single_swap",
        )
        records.append(proposal)
        candidates.append(exact_candidate(enc, banks, original, proposal, risks0, j0))
    for row in joint_proposals(table, risks0, j0):
        records.append(row)
        if row["feasible"]:
            candidates.append(exact_candidate(enc, banks, original, row, risks0, j0))
    best = robust.joint.winner(candidates)
    return dict(
        session=session,
        original=original,
        baseline_j=j0,
        baseline_risks=risks0.tolist(),
        proposals=records,
        candidates=candidates,
        chosen=None if best is None else best["name"],
        final_pair=original if best is None else best["pair"],
        net_exchanged=0 if best is None else len(best["high"]),
        final_j=j0 if best is None else best["objective"],
        finite_single_audit=check,
        finite_cost_sha256=file_sha256(path),
        admissible_single_swaps=int(table.admissible.sum()),
    )


def worker(args):
    source_dir, session, split, output_dir = args
    folder = source_dir / session.replace("/", "_")
    enc, banks = audit.load(folder / "encoding.npz"), {}
    for s in SOURCES:
        bank = audit.load(folder / f"{s}.npz")
        audit.check_partition(bank, split[s], session, s)
        banks[s] = robust.subset(bank, split[s]["train"])
    choice = select(enc, banks, session, output_dir)
    (output_dir / f"{session.replace('/', '_')}_choice.json").write_text(json.dumps(choice, indent=2) + "\n")
    print("finite choice frozen", session, choice["chosen"], flush=True)
    return choice


def run(source_dir, reference_dir, reference_audit, output_dir, workers=4):
    source = robust.checked.checked_manifest(reference_dir, reference_audit)
    frozen_reference = json.loads((reference_dir / "frozen_assignments.json").read_text())
    sessions = robust.exact.base.SESSIONS
    if set(frozen_reference["choices"]) != set(sessions) or set(frozen_reference["splits"]) != set(sessions):
        raise ValueError("missing original pair or grouped split")
    inputs = dict(source["input_file_sha256"])
    for path in [
        reference_dir / "manifest.json",
        reference_dir / "frozen_assignments.json",
        reference_audit,
        Path(__file__),
        ROOT / "docs/finite_exchange_protocol.md",
        ROOT / "scripts/audit_robust_exchange_content.py",
    ]:
        inputs[str(path)] = file_sha256(path)
    for session in sessions:
        folder = source_dir / session.replace("/", "_")
        for name in ("encoding",) + SOURCES:
            path = folder / f"{name}.npz"
            if file_sha256(path) != inputs.get(str(path)):
                raise ValueError("changed or unregistered source")
    output_dir.mkdir(parents=True, exist_ok=False)
    args = [(source_dir, s, frozen_reference["splits"][s], output_dir) for s in sessions]
    if workers == 1:
        selected = list(map(worker, args))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            selected = list(pool.map(worker, args))
    choices = {c["session"]: c for c in selected}
    frozen = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), choices=choices, splits=frozen_reference["splits"], input_file_sha256=inputs)
    (output_dir / "frozen_assignments.json").write_text(json.dumps(frozen, indent=2) + "\n")
    frozen_hash = file_sha256(output_dir / "frozen_assignments.json")
    rows = []
    for session in sessions:
        folder = source_dir / session.replace("/", "_")
        enc = audit.load(folder / "encoding.npz")
        banks = {s: robust.subset(audit.load(folder / f"{s}.npz"), frozen["splits"][session][s]["validation"]) for s in SOURCES}
        rows.extend(robust.validation(enc, banks, choices[session]))
    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "internal_validation.csv", index=False)
    summary = pd.DataFrame(
        [
            dict(
                session=c["session"],
                animal=c["session"].split("/")[0],
                chosen=c["chosen"],
                net_exchanged=c["net_exchanged"],
                baseline_j=c["baseline_j"],
                final_j=c["final_j"],
                legal_single_swaps=c["finite_single_audit"]["complete_neighborhood"],
                admissible_single_swaps=c["admissible_single_swaps"],
                admissible_candidates=sum(x["admissible"] for x in c["candidates"]),
            )
            for c in selected
        ]
    )
    summary.to_csv(output_dir / "selection_summary.csv", index=False)
    native = table[table.source.eq("run_q3")].drop_duplicates("session")
    changes = native.assign(change_j=native.targeted_j - native.baseline_j).groupby("animal").change_j.mean()
    rats = {s.split("/")[0] for s in sessions}
    gates = dict(
        all_original_pairs_have_support=len(choices) == 4,
        all_validation_risks_present=len(table) == 96,
        internal_accuracy_nonworsening=len(table) == 96 and bool(table.nonworsening.all()),
        changed_populations_in_all_rats=set(summary.loc[summary.net_exchanged.gt(0), "animal"]) == rats,
        native_validation_improves_all_rats=set(changes.index) == rats and bool((changes < -1e-10).all()),
    )
    gates["ready_for_truth_preflight"] = all(gates.values())
    pd.DataFrame([dict(gate=k, passed=v) for k, v in gates.items()]).to_csv(output_dir / "gates.csv", index=False)
    (output_dir / "report.md").write_text(
        "# Finite whole-cell exchange calibration\n\nDevelopment only; no Q4, test banks or replay scored. No independently validated remedy.\n\n"
        + markdown(summary)
        + "\n\n"
        + markdown(pd.read_csv(output_dir / "gates.csv"))
        + "\n\nSingle-swap costs are complete; independent numerical reconstruction samples 32 hash-selected swaps, the minimum-J swap and every admissible single. Joint coefficients are approximate sums of measured finite changes, with final exact scoring required.\n"
    )
    for path, sha in inputs.items():
        if file_sha256(path) != sha:
            raise ValueError("input changed during run")
    if file_sha256(output_dir / "frozen_assignments.json") != frozen_hash:
        raise ValueError("choices changed during validation")
    manifest = build_script_provenance()
    manifest.update(
        input_file_sha256=inputs,
        frozen_assignments_sha256=frozen_hash,
        workers=workers,
        solver_options=robust.OPTIONS,
        external_validation=False,
        validated_remedy=False,
        replay_scored=False,
        q4_scored=False,
        output_sha256={p.name: file_sha256(p) for p in output_dir.iterdir() if p.is_file()},
    )
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("source-dir", "reference-dir", "reference-audit", "output-dir"):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2, 4), default=4)
    a = parser.parse_args()
    run(a.source_dir, a.reference_dir, a.reference_audit, a.output_dir, a.workers)
