#!/usr/bin/env python3
"""Exhaustive one/two-cell acquisition using exact population factorization."""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np
import pandas as pd
from scipy.special import softmax

from scripts import guarded_reserve_content as previous
from scripts._provenance import build_script_provenance, file_sha256
from scripts.report_content_screening_bound import markdown

base, robust, SOURCES = previous.base, previous.robust, previous.SOURCES
SIDES = base.SIDES
RISK_COLUMNS = [f"risk_{i}" for i in range(12)]
PAIR_COLUMNS = ["high_state", "low_state", "budget", "disjoint", "objective", "admissible"]


def budgets(reserve):
    return [b for b in (1, 2) if 2 * b <= len(reserve)]


def cells(row):
    return [int(row["cell_1"])] + ([int(row["cell_2"])] if int(row["budget"]) == 2 else [])


def from_likelihood(ll, enc, bank):
    p = softmax(ll, axis=1)
    home = p[:, enc["near"]].sum(axis=1)
    error = np.linalg.norm(p @ enc["grid_cm"] - bank["truth_cm"], axis=1)
    losses = (error, (home - bank["labels"]) ** 2)
    if any(not np.any(bank["labels"] == k) for k in (0, 1)):
        raise ValueError("missing truth class")
    risk = np.array([loss[bank["labels"] == k].mean() for k in (0, 1) for loss in losses])
    means = np.array([home[bank["labels"] == k].mean() for k in (0, 1)])
    if not np.isfinite(risk).all() or not np.isfinite(means).all():
        raise ValueError("nonfinite population state")
    return risk, means


class SideScorer:
    def __init__(self, enc, banks, original):
        self.enc, self.banks = enc, banks
        self.offsets = {s: {source: robust.exact.likelihood(bank["counts"], enc["early_run"], original[s]) for source, bank in banks.items()} for s in SIDES}

    def __call__(self, side, added):
        risks, native = [], None
        for source in SOURCES:
            bank = self.banks[source]
            ll = self.offsets[side][source]
            if added:
                ll = ll + robust.exact.likelihood(bank["counts"], self.enc["early_run"], added)
            risk, means = from_likelihood(ll, self.enc, bank)
            risks.extend(risk)
            if source == "run_q3":
                native = means
        return np.asarray(risks), native


def eligible_pairs(table, baseline_j):
    rows = []
    for budget in sorted(table.budget.unique()):
        safe = {s: table[table.side.eq(s) & table.budget.eq(budget) & table.safe].to_dict("records") for s in SIDES}
        for high in safe["high"]:
            for low in safe["low"]:
                disjoint = not set(cells(high)) & set(cells(low))
                j = float(np.mean([(high[k] - low[k]) ** 2 for k in ("native_mean_nonhome", "native_mean_home")]))
                rows.append(
                    dict(
                        high_state=high["state_index"],
                        low_state=low["state_index"],
                        budget=int(budget),
                        disjoint=disjoint,
                        objective=j,
                        admissible=bool(disjoint and j < baseline_j - 1e-10),
                    )
                )
    return pd.DataFrame(rows, columns=PAIR_COLUMNS)


def choose(table, pairs):
    eligible = pairs[pairs.admissible]
    if eligible.empty:
        return None
    best = eligible.objective.min()
    tied = eligible[eligible.objective.le(best + 1e-12)]
    states = table.set_index("state_index").to_dict("index")
    return min(tied.to_dict("records"), key=lambda r: (r["budget"], tuple(cells(states[r["high_state"]])), tuple(cells(states[r["low_state"]]))))


def select(enc, banks, session):
    reserve = base.reserve_indices(enc).tolist()
    original = {s: sorted(map(int, enc[f"{s}_indices"])) for s in SIDES}
    if len(original["high"]) != len(original["low"]) or not budgets(reserve):
        raise ValueError("unequal originals or no disjoint acquisition budget")
    scorer = SideScorer(enc, banks, original)
    baseline = {s: scorer(s, []) for s in SIDES}
    baseline_j = float(np.mean((baseline["high"][1] - baseline["low"][1]) ** 2))
    pair_risk, pair_j = previous.measurements(enc, banks, original)
    np.testing.assert_allclose(
        pair_risk, np.concatenate([np.r_[baseline["high"][0][4 * i : 4 * i + 4], baseline["low"][0][4 * i : 4 * i + 4]] for i in range(3)]), atol=1e-9, rtol=1e-9
    )
    np.testing.assert_allclose(baseline_j, pair_j, atol=1e-10, rtol=1e-9)
    rows = []
    for side in SIDES:
        for budget in budgets(reserve):
            for added in combinations(reserve, budget):
                started = time.monotonic()
                risk, means = scorer(side, list(added))
                failures = int(np.count_nonzero(risk > baseline[side][0] + 1e-10))
                rows.append(
                    dict(
                        state_index=len(rows),
                        side=side,
                        budget=budget,
                        cell_1=int(added[0]),
                        cell_2=-1 if budget == 1 else int(added[1]),
                        native_mean_nonhome=float(means[0]),
                        native_mean_home=float(means[1]),
                        **dict(zip(RISK_COLUMNS, risk, strict=True)),
                        failed_risks=failures,
                        safe=failures == 0,
                        runtime_s=time.monotonic() - started,
                    )
                )
            print("exact small reserve side complete", session, side, budget, flush=True)
    table = pd.DataFrame(rows)
    pairs = eligible_pairs(table, baseline_j)
    chosen = choose(table, pairs)
    added = {s: [] for s in SIDES}
    if chosen is not None:
        states = table.set_index("state_index").to_dict("index")
        added = {s: cells(states[chosen[f"{s}_state"]]) for s in SIDES}
    final_pair = {s: sorted(set(original[s]) | set(added[s])) for s in SIDES}
    budget = 0 if chosen is None else chosen["budget"]
    if any(len(added[s]) != budget or len(set(added[s])) != budget for s in SIDES) or set(added["high"]) & set(added["low"]):
        raise ValueError("invalid disjoint additions")
    if chosen is not None:
        risk, j = previous.measurements(enc, banks, final_pair)
        if not np.all(risk <= pair_risk + 1e-10):
            raise ValueError("factorized eligibility disagrees with full pair likelihood")
        np.testing.assert_allclose(j, chosen["objective"], atol=1e-10, rtol=1e-9)
    coverage = []
    for b in budgets(reserve):
        group = pairs[pairs.budget.eq(b)]
        row = dict(budget=b, safe_pair_products=len(group), disjoint_safe_pairs=int(group.disjoint.sum()), admissible_pairs=int(group.admissible.sum()))
        row.update({f"{s}_{k}": int((table.side.eq(s) & table.budget.eq(b) & (table.safe if k == "safe" else True)).sum()) for s in SIDES for k in ("states", "safe")})
        coverage.append(row)
    choice = dict(
        session=session,
        original=original,
        reserve=reserve,
        baseline_by_side={s: dict(risks=baseline[s][0].tolist(), native_means=baseline[s][1].tolist()) for s in SIDES},
        baseline_j=baseline_j,
        coverage=coverage,
        chosen=None if chosen is None else f"high_{chosen['high_state']}_low_{chosen['low_state']}",
        chosen_pair=chosen,
        budget=budget,
        added=added,
        final_pair=final_pair,
        final_j=baseline_j if chosen is None else chosen["objective"],
    )
    return choice, table, pairs


def worker(args):
    source, session, splits, output = args
    folder = source / session.replace("/", "_")
    enc = base.read_npz(folder / "encoding.npz")
    banks = {}
    for s in SOURCES:
        bank = base.read_npz(folder / f"{s}.npz")
        previous.partition_audit.check_partition(bank, splits[s], session, s)
        banks[s] = robust.subset(bank, splits[s]["train"])
    choice, table, pairs = select(enc, banks, session)
    stem = session.replace("/", "_")
    paths = {"side_states": output / f"{stem}_side_states.csv", "safe_pair_candidates": output / f"{stem}_safe_pair_candidates.csv"}
    table.to_csv(paths["side_states"], index=False)
    pairs.to_csv(paths["safe_pair_candidates"], index=False)
    choice["tables_sha256"] = {k: file_sha256(v) for k, v in paths.items()}
    (output / f"{stem}_choice.json").write_text(json.dumps(choice, indent=2) + "\n")
    print("exact small reserve choice frozen", session, choice["chosen"], flush=True)
    return choice


def run(source, reference_dir, reference_audit, output, workers=4):
    reference = robust.checked.checked_manifest(reference_dir, reference_audit)
    splits = json.loads((reference_dir / "frozen_assignments.json").read_text())["splits"]
    if set(splits) != set(base.SESSIONS):
        raise ValueError("original cohort missing")
    inputs = dict(reference["input_file_sha256"])
    for p in (reference_dir / "manifest.json", reference_dir / "frozen_assignments.json", reference_audit, Path(__file__), ROOT / "docs/exact_small_reserve_content_protocol.md"):
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
    frozen_path = output / "frozen_assignments.json"
    frozen_path.write_text(json.dumps(dict(created_at_utc=datetime.now(timezone.utc).isoformat(), choices=choices, splits=splits, input_file_sha256=inputs), indent=2) + "\n")
    frozen_hash = file_sha256(frozen_path)
    validation_started = datetime.now(timezone.utc).isoformat()
    rows = []
    for session in base.SESSIONS:
        folder = source / session.replace("/", "_")
        enc = base.read_npz(folder / "encoding.npz")
        banks = {s: robust.subset(base.read_npz(folder / f"{s}.npz"), splits[session][s]["validation"]) for s in SOURCES}
        rows.extend(robust.validation(enc, banks, choices[session]))
    validation = pd.DataFrame(rows)
    validation.to_csv(output / "internal_validation.csv", index=False)
    coverage = pd.DataFrame([dict(session=s, animal=s.split("/")[0], **r) for s, c in choices.items() for r in c["coverage"]])
    coverage.to_csv(output / "exhaustive_coverage.csv", index=False)
    summary = pd.DataFrame(
        [
            dict(
                session=s,
                animal=s.split("/")[0],
                population_subsets=sum(r["high_states"] + r["low_states"] for r in c["coverage"]),
                admissible_pairs=sum(r["admissible_pairs"] for r in c["coverage"]),
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
        "# Exact small-budget reserve acquisition\n\nDevelopment only; exhaustive for one/two-cell additions, not larger budgets. No validated remedy.\n\n"
        + markdown(summary)
        + "\n\n"
        + markdown(coverage)
        + "\n\n"
        + markdown(gates)
        + "\n\nFailure stops before Q4/test/replay/external validation. An unchanged baseline is not a remedy.\n"
    )
    for path, sha in inputs.items():
        if sha is None or file_sha256(path) != sha:
            raise ValueError("input changed")
    if file_sha256(frozen_path) != frozen_hash:
        raise ValueError("assignment changed after validation")
    manifest = {
        **build_script_provenance(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation_started_at_utc": validation_started,
        "source_dir": str(source),
        "reference_dir": str(reference_dir),
        "reference_audit": str(reference_audit),
        "input_file_sha256": inputs,
        "frozen_assignments_sha256": frozen_hash,
        "budgets": [1, 2],
        "selection": "exhaustive_population_subsets_then_disjoint_safe_pair_cross_product",
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
        parser.add_argument(f"--{flag}", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4, choices=range(1, 5))
    args = parser.parse_args()
    run(args.source_dir, args.reference_dir, args.reference_audit, args.output_dir, args.workers)
