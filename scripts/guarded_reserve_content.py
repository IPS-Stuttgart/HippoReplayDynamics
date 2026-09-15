#!/usr/bin/env python3
"""Grouped, accuracy-guarded allocation of disjoint reserve neurons."""

from __future__ import annotations

import argparse
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
from scipy.sparse import csc_array

from scripts import audit_robust_exchange_content as partition_audit
from scripts import robust_exchange_content as robust
from scripts._provenance import build_script_provenance, file_sha256
from scripts.report_content_screening_bound import markdown

base = robust.exact.base
SOURCES = robust.SOURCES
RISK_COLUMNS = tuple(f"risk_{i}" for i in range(24))
OPTIONS = dict(time_limit=20.0, node_limit=64, mip_rel_gap=0.001)
MAX_PROPOSALS = 12


def augmented(enc, added):
    reserve = set(map(int, base.reserve_indices(enc)))
    quota = len(reserve) // 2
    if quota < 1 or set(added) != {"high", "low"}:
        raise ValueError("no reserve budget or missing side")
    a, b = (set(added[s]) for s in base.SIDES)
    if any(len(added[s]) != quota or len(set(added[s])) != quota for s in base.SIDES) or not (a | b) <= reserve or a & b:
        raise ValueError("addition is not disjoint, outside originals and quota-matched")
    original = {s: set(map(int, enc[f"{s}_indices"])) for s in base.SIDES}
    pair = {s: sorted(original[s] | set(added[s])) for s in base.SIDES}
    if set(pair["high"]) & set(pair["low"]) != original["high"] & original["low"]:
        raise ValueError("new shared observations")
    return pair


def measurements(enc, banks, pair):
    states = {s: robust.state(enc, banks[s], pair) for s in SOURCES}
    return np.concatenate([states[s]["risks"] for s in SOURCES]), states["run_q3"]["objective"]


def single_additions(enc, banks):
    reserve = base.reserve_indices(enc)
    if len(reserve) < 2:
        raise ValueError("insufficient reserve")
    original = {s: sorted(map(int, enc[f"{s}_indices"])) for s in base.SIDES}
    risk0, j0 = measurements(enc, banks, original)
    rows = []
    for side in base.SIDES:
        for cell in reserve:
            pair = {s: original[s] + ([int(cell)] if s == side else []) for s in base.SIDES}
            risk, j = measurements(enc, banks, pair)
            rows.append(dict(side=side, cell=int(cell), objective=j, objective_change=j - j0, **dict(zip(RISK_COLUMNS, risk.tolist(), strict=True))))
    return pd.DataFrame(rows), risk0, j0


def allocation_proposals(table, risk0, j0, reserve):
    expected = [(s, int(i)) for s in base.SIDES for i in reserve]
    if list(table[["side", "cell"]].itertuples(index=False, name=None)) != expected:
        raise ValueError("single-addition table is incomplete or reordered")
    n = len(reserve)
    quota = n // 2
    gradient = table.objective_change.to_numpy()
    risk = table[list(RISK_COLUMNS)].to_numpy().T - risk0[:, None]
    for guarded in (True, False):
        family = "guarded_finite" if guarded else "objective_finite"
        previous = []
        for index in range(MAX_PROPOSALS):
            matrix = [np.r_[np.ones(n), np.zeros(n)], np.r_[np.zeros(n), np.ones(n)]]
            lower, upper = [quota, quota], [quota, quota]
            for cell in range(n):
                row = np.zeros(2 * n)
                row[cell] = row[n + cell] = 1
                matrix.append(row)
                lower.append(0)
                upper.append(1)
            if guarded:
                matrix.extend(risk / np.maximum(abs(risk0), 1e-6)[:, None])
                lower.extend([-np.inf] * len(risk0))
                upper.extend([0] * len(risk0))
            matrix.extend(previous)
            lower.extend([-np.inf] * len(previous))
            upper.extend([2 * quota - 1] * len(previous))
            matrix, lower, upper = csc_array(np.asarray(matrix)), np.asarray(lower), np.asarray(upper)
            start = time.monotonic()
            solved = milp(
                gradient / max(j0, 1e-6),
                integrality=np.ones(2 * n),
                bounds=Bounds(np.zeros(2 * n), np.ones(2 * n)),
                constraints=LinearConstraint(matrix, lower, upper),
                options=dict(OPTIONS),
            )
            x = robust.joint.feasible_incumbent(solved, matrix, lower, upper)
            record = dict(
                name=f"{family}_{index:02}", family=family, status=int(solved.status), message=str(solved.message), runtime_s=time.monotonic() - start, feasible=x is not None
            )
            if x is not None:
                record.update(
                    added={"high": np.asarray(reserve)[x[:n].astype(bool)].tolist(), "low": np.asarray(reserve)[x[n:].astype(bool)].tolist()},
                    predicted_j_change=float(gradient @ x),
                    predicted_risk_changes=(risk @ x).tolist(),
                )
            yield record
            if x is None:
                break
            previous.append(x)


def winner(candidates):
    available = [c for c in candidates if c["admissible"]]
    if not available:
        return None
    best = min(c["objective"] for c in available)
    return min((c for c in available if c["objective"] - best <= 1e-12), key=lambda c: (tuple(c["added"]["high"]), tuple(c["added"]["low"]), c["name"]))


def select(enc, banks, session, output):
    reserve = base.reserve_indices(enc)
    quota = len(reserve) // 2
    table, risk0, j0 = single_additions(enc, banks)
    print("measured reserve additions", session, len(table), flush=True)
    table_path = output / f"{session.replace('/', '_')}_single_additions.csv"
    table.to_csv(table_path, index=False)
    proposals = list(allocation_proposals(table, risk0, j0, reserve))
    for index in range(20):
        order = np.random.default_rng(base.seed(session, index)).permutation(reserve)
        proposals.append(
            dict(name=f"random_{index:02}", family="random", feasible=True, added=dict(high=sorted(map(int, order[:quota])), low=sorted(map(int, order[quota : 2 * quota]))))
        )
    candidates, cache = [], {}
    for row in proposals:
        if not row["feasible"]:
            continue
        pair = augmented(enc, row["added"])
        key = tuple(pair["high"]), tuple(pair["low"])
        if key not in cache:
            cache[key] = measurements(enc, banks, pair)
        risk, j = cache[key]
        candidates.append(
            dict(
                name=row["name"],
                family=row["family"],
                added=row["added"],
                pair=pair,
                risks=risk.tolist(),
                objective=j,
                failed_risks=int(np.count_nonzero(risk > risk0 + 1e-10)),
                admissible=bool(j < j0 - 1e-10 and np.all(risk <= risk0 + 1e-10)),
            )
        )
    best = winner(candidates)
    original = {s: sorted(map(int, enc[f"{s}_indices"])) for s in base.SIDES}
    return dict(
        session=session,
        original=original,
        reserve=reserve.tolist(),
        quota=quota,
        baseline_risks=risk0.tolist(),
        baseline_j=j0,
        single_addition_sha256=file_sha256(table_path),
        proposals=proposals,
        candidates=candidates,
        chosen=None if best is None else best["name"],
        added={s: [] for s in base.SIDES} if best is None else best["added"],
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
        partition_audit.check_partition(bank, splits[s], session, s)
        banks[s] = robust.subset(bank, splits[s]["train"])
    choice = select(enc, banks, session, output)
    path = output / f"{session.replace('/', '_')}_choice.json"
    path.write_text(json.dumps(choice, indent=2) + "\n")
    print("frozen reserve allocation", session, choice["chosen"], flush=True)
    return choice


def gate_values(table, choices):
    sessions = set(base.SESSIONS)
    animals = {s.split("/")[0] for s in sessions}
    native = table[table.source.eq("run_q3")].drop_duplicates("session")
    changes = native.assign(delta_j=native.targeted_j - native.baseline_j).groupby("animal").delta_j.mean()
    expected = {(s, source, side, label, metric) for s in sessions for source in SOURCES for side in base.SIDES for label in (0, 1) for metric in ("error_cm", "home_brier")}
    actual = set(table[["session", "source", "side", "true_home", "metric"]].itertuples(index=False, name=None))
    complete = len(table) == len(expected) and actual == expected
    result = dict(
        all_original_pairs_present=set(choices) == sessions,
        all_validation_risks_present=complete,
        internal_accuracy_nonworsening=complete and bool(table.nonworsening.all()),
        all_pairs_augmented=all(c["chosen"] is not None for c in choices.values()) and set(choices) == sessions,
        native_validation_improves_all_rats=set(changes.index) == animals and bool((changes < -1e-10).all()),
    )
    result["ready_for_truth_preflight"] = all(result.values())
    return result


def run(source, reference_dir, reference_audit, output, workers=4):
    reference = robust.checked.checked_manifest(reference_dir, reference_audit)
    ref_freeze = reference_dir / "frozen_assignments.json"
    splits = json.loads(ref_freeze.read_text())["splits"]
    if set(splits) != set(base.SESSIONS):
        raise ValueError("missing original grouped partition")
    inputs = dict(reference["input_file_sha256"])
    for p in (
        reference_dir / "manifest.json",
        reference_audit,
        ref_freeze,
        Path(__file__),
        ROOT / "docs/guarded_reserve_content_protocol.md",
        ROOT / "scripts/robust_exchange_content.py",
        ROOT / "scripts/audit_robust_exchange_content.py",
        ROOT / "scripts/reserve_cell_content.py",
        ROOT / "scripts/audit_exchange_content.py",
        ROOT / "scripts/_provenance.py",
        ROOT / "scripts/report_content_screening_bound.py",
    ):
        inputs[str(p)] = file_sha256(p)
    if any(v is None for v in inputs.values()):
        raise ValueError("missing provenance input")
    for session in base.SESSIONS:
        for name in ("encoding",) + SOURCES:
            p = source / session.replace("/", "_") / f"{name}.npz"
            if inputs.get(str(p)) != file_sha256(p):
                raise ValueError("changed or unlinked calibration source")
    output.mkdir(parents=True, exist_ok=False)
    started = {**build_script_provenance(), "created_at_utc": datetime.now(timezone.utc).isoformat(), "input_file_sha256": inputs, "splits": splits}
    (output / "pre_selection.json").write_text(json.dumps(started, indent=2) + "\n")
    jobs = [(source, s, splits[s], output) for s in base.SESSIONS]
    if workers == 1:
        results = [worker(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(worker, jobs))
    choices = {c["session"]: c for c in results}
    frozen = output / "frozen_assignments.json"
    frozen.write_text(json.dumps(dict(created_at_utc=datetime.now(timezone.utc).isoformat(), choices=choices, splits=splits, input_file_sha256=inputs), indent=2) + "\n")
    frozen_sha = file_sha256(frozen)
    rows, random_rows = [], []
    validation_started = datetime.now(timezone.utc).isoformat()
    for session in base.SESSIONS:
        folder = source / session.replace("/", "_")
        enc = base.read_npz(folder / "encoding.npz")
        banks = {s: robust.subset(base.read_npz(folder / f"{s}.npz"), splits[session][s]["validation"]) for s in SOURCES}
        rows.extend(robust.validation(enc, banks, choices[session]))
        for candidate in choices[session]["candidates"]:
            if candidate["family"] == "random":
                state = robust.state(enc, banks["run_q3"], candidate["pair"])
                random_rows.append(dict(session=session, animal=session.split("/")[0], method=candidate["name"], validation_j=state["objective"]))
    table = pd.DataFrame(rows)
    table.to_csv(output / "internal_validation.csv", index=False)
    pd.DataFrame(random_rows).to_csv(output / "random_native_validation.csv", index=False)
    summary = pd.DataFrame(
        [
            dict(
                session=s,
                animal=s.split("/")[0],
                quota=c["quota"],
                reserve_cells=len(c["reserve"]),
                single_additions=2 * len(c["reserve"]),
                exact_candidates=len(c["candidates"]),
                admissible_candidates=sum(x["admissible"] for x in c["candidates"]),
                chosen=c["chosen"],
                baseline_j=c["baseline_j"],
                final_j=c["final_j"],
            )
            for s, c in choices.items()
        ]
    )
    summary.to_csv(output / "selection_summary.csv", index=False)
    gates = gate_values(table, choices)
    gate_table = pd.DataFrame([dict(gate=k, passed=v) for k, v in gates.items()])
    gate_table.to_csv(output / "gates.csv", index=False)
    (output / "report.md").write_text(
        "# Guarded disjoint reserve acquisition\n\nDevelopment only. No Q4, test-bank, replay or independent-recording scoring. All original populations retained; added cells are disjoint, quotas fixed, interpopulation overlap unchanged.\n\n"
        + markdown(summary)
        + "\n\n"
        + markdown(gate_table)
        + "\n\nFinite-addition costs only propose joint allocations; exact training risks decide eligibility. Every source/side/true-class physical-error and Home-Brier safeguard remains in force. A baseline fallback is not a remedy. Random allocation search is training-only; reserved native random J is descriptive, not a replay-validation claim.\n"
    )
    for path, sha in inputs.items():
        if file_sha256(path) != sha:
            raise ValueError("input changed during experiment")
    if file_sha256(frozen) != frozen_sha:
        raise ValueError("assignment changed after validation")
    manifest = {
        **build_script_provenance(),
        "source_dir": str(source),
        "reference_dir": str(reference_dir),
        "reference_audit": str(reference_audit),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation_started_at_utc": validation_started,
        "input_file_sha256": inputs,
        "frozen_assignments_sha256": frozen_sha,
        "solver_options": OPTIONS,
        "max_proposals_per_family": MAX_PROPOSALS,
        "workers": workers,
        "q4_scored": False,
        "test_banks_scored": False,
        "replay_scored": False,
        "external_validation": False,
        "validated_remedy": False,
        "output_sha256": {p.name: file_sha256(p) for p in output.iterdir() if p.is_file()},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("source-dir", "reference-dir", "reference-audit", "output-dir"):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4, choices=range(1, 5))
    args = parser.parse_args()
    run(args.source_dir, args.reference_dir, args.reference_audit, args.output_dir, args.workers)
