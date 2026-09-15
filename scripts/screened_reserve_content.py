#!/usr/bin/env python3
"""Exact three/four-cell acquisition with recorded, stage-wise risk rejection."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from itertools import combinations
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np
import pandas as pd
from scipy.special import softmax

from scripts import exact_small_reserve_content as small
from scripts._provenance import build_script_provenance, file_sha256
from scripts.report_content_screening_bound import markdown

base, robust, previous = small.base, small.robust, small.previous
SOURCES, SIDES = small.SOURCES, small.SIDES
BUDGETS = (3, 4)
STAGES = tuple((source, k) for k in (0, 1) for source in ("cal_poisson_gain1", "cal_conditional", "run_q3"))
RISKS = [f"risk_{i}" for i in range(12)]
PAIR_COLUMNS = small.PAIR_COLUMNS


def cells(row):
    return json.loads(row["added_cells"])


class Screen:
    def __init__(self, enc, banks, original):
        self.enc = enc
        self.rates = enc["early_run"]
        if not np.isfinite(self.rates).all() or np.any(self.rates <= 0):
            raise ValueError("invalid rate map")
        self.log_rates = np.log(self.rates)
        self.parts = {}
        self.offsets = {}
        for stage, (source, label) in enumerate(STAGES):
            bank = banks[source]
            if not np.isfinite(bank["counts"]).all() or np.any(bank["counts"] < 0):
                raise ValueError("invalid counts")
            mask = bank["labels"] == label
            if not mask.any():
                raise ValueError("missing class")
            self.parts[stage] = {k: bank[k][mask] for k in ("counts", "truth_cm", "labels")}
            counts = self.parts[stage]["counts"]
            for side in SIDES:
                ix = original[side]
                self.offsets[side, stage] = counts[:, ix] @ self.log_rates[ix] - 0.02 * self.rates[ix].sum(axis=0)
        self.baseline = {s: self.measure(s, [], screen=False) for s in SIDES}

    def stage(self, side, added, stage):
        bank = self.parts[stage]
        ll = self.offsets[side, stage]
        if added:
            ll = ll + bank["counts"][:, added] @ self.log_rates[added] - 0.02 * self.rates[added].sum(axis=0)
        p = softmax(ll, axis=1)
        home = p[:, self.enc["near"]].sum(axis=1)
        risk = np.array([np.linalg.norm(p @ self.enc["grid_cm"] - bank["truth_cm"], axis=1).mean(), ((home - STAGES[stage][1]) ** 2).mean()])
        if not np.isfinite(risk).all() or not np.isfinite(home).all():
            raise ValueError("nonfinite stage")
        return risk, float(home.mean())

    def measure(self, side, added, screen=True):
        risks, means = np.full(12, np.nan), np.full(2, np.nan)
        for stage, (source, label) in enumerate(STAGES):
            offset = 4 * SOURCES.index(source) + 2 * label
            risk, mean = self.stage(side, added, stage)
            risks[offset : offset + 2] = risk
            if source == "run_q3":
                means[label] = mean
            if screen and np.any(risk > self.baseline[side][0][offset : offset + 2] + 1e-10):
                return risks, means, stage + 1, False
        return risks, means, len(STAGES), True


def pairs_and_choice(table, baseline_j):
    rows = []
    for budget in BUDGETS:
        safe = {s: table[table.side.eq(s) & table.budget.eq(budget) & table.safe].to_dict("records") for s in SIDES}
        for h in safe["high"]:
            for low in safe["low"]:
                disjoint = not set(cells(h)) & set(cells(low))
                j = float(np.mean([(h[k] - low[k]) ** 2 for k in ("native_mean_nonhome", "native_mean_home")]))
                rows.append(
                    dict(
                        high_state=h["state_index"],
                        low_state=low["state_index"],
                        budget=budget,
                        disjoint=disjoint,
                        objective=j,
                        admissible=bool(disjoint and j < baseline_j - 1e-10),
                    )
                )
    pairs = pd.DataFrame(rows, columns=PAIR_COLUMNS)
    allowed = pairs[pairs.admissible]
    chosen = None
    if not allowed.empty:
        states = table.set_index("state_index").to_dict("index")
        candidates = allowed[allowed.objective.le(allowed.objective.min() + 1e-12)].to_dict("records")
        chosen = min(candidates, key=lambda p: (p["budget"], tuple(cells(states[p["high_state"]])), tuple(cells(states[p["low_state"]]))))
    return pairs, chosen


def select(enc, banks, session):
    reserve = base.reserve_indices(enc).tolist()
    original = {s: sorted(map(int, enc[f"{s}_indices"])) for s in SIDES}
    budgets = [b for b in BUDGETS if 2 * b <= len(reserve)]
    if len(original["high"]) != len(original["low"]) or not budgets:
        raise ValueError("unequal originals or unavailable budget")
    scorer = Screen(enc, banks, original)
    baseline = scorer.baseline
    j0 = float(np.mean((baseline["high"][1] - baseline["low"][1]) ** 2))
    r0, direct_j = previous.measurements(enc, banks, original)
    np.testing.assert_allclose(j0, direct_j, atol=1e-10, rtol=1e-9)
    expected_r0 = np.concatenate([np.r_[baseline["high"][0][4 * i : 4 * i + 4], baseline["low"][0][4 * i : 4 * i + 4]] for i in range(3)])
    np.testing.assert_allclose(r0, expected_r0, atol=1e-9, rtol=1e-9)
    rows = []
    for side in SIDES:
        for budget in budgets:
            for added in combinations(reserve, budget):
                start = time.monotonic()
                risks, means, evaluated, safe = scorer.measure(side, list(added))
                rows.append(
                    dict(
                        state_index=len(rows),
                        side=side,
                        budget=budget,
                        added_cells=json.dumps(list(added)),
                        stages_evaluated=evaluated,
                        safe=safe,
                        native_mean_nonhome=means[0],
                        native_mean_home=means[1],
                        **dict(zip(RISKS, risks, strict=True)),
                        runtime_s=time.monotonic() - start,
                    )
                )
                if len(rows) % 10000 == 0:
                    print("screened reserve progress", session, len(rows), flush=True)
            print("screened reserve side complete", session, side, budget, flush=True)
    table = pd.DataFrame(rows)
    pairs, chosen = pairs_and_choice(table, j0)
    added = {s: [] for s in SIDES}
    if chosen is not None:
        states = table.set_index("state_index").to_dict("index")
        added = {s: cells(states[chosen[f"{s}_state"]]) for s in SIDES}
    final = {s: sorted(original[s] + added[s]) for s in SIDES}
    budget = 0 if chosen is None else chosen["budget"]
    if set(added["high"]) & set(added["low"]) or any(len(set(added[s])) != budget or not set(added[s]) <= set(reserve) for s in SIDES):
        raise ValueError("invalid acquisition")
    if chosen is not None:
        risk, j = previous.measurements(enc, banks, final)
        if not np.all(risk <= r0 + 1e-10):
            raise ValueError("screening and full pair disagree")
        np.testing.assert_allclose(j, chosen["objective"], atol=1e-10, rtol=1e-9)
    coverage = []
    for b in budgets:
        p = pairs[pairs.budget.eq(b)]
        coverage.append(
            dict(
                budget=b,
                safe_pair_products=len(p),
                disjoint_safe_pairs=int(p.disjoint.sum()),
                admissible_pairs=int(p.admissible.sum()),
                **{f"{s}_{k}": int((table.side.eq(s) & table.budget.eq(b) & (table.safe if k == "safe" else True)).sum()) for s in SIDES for k in ("states", "safe")},
            )
        )
    choice = dict(
        session=session,
        original=original,
        reserve=reserve,
        baseline_by_side={s: dict(risks=baseline[s][0].tolist(), native_means=baseline[s][1].tolist()) for s in SIDES},
        baseline_j=j0,
        coverage=coverage,
        chosen=None if chosen is None else f"high_{chosen['high_state']}_low_{chosen['low_state']}",
        chosen_pair=chosen,
        budget=budget,
        added=added,
        final_pair=final,
        final_j=j0 if chosen is None else chosen["objective"],
    )
    return choice, table, pairs


def worker(args):
    source, session, splits, output = args
    folder, stem = source / session.replace("/", "_"), session.replace("/", "_")
    enc, banks = base.read_npz(folder / "encoding.npz"), {}
    for s in SOURCES:
        bank = base.read_npz(folder / f"{s}.npz")
        previous.partition_audit.check_partition(bank, splits[s], session, s)
        banks[s] = robust.subset(bank, splits[s]["train"])
    choice, states, pairs = select(enc, banks, session)
    paths = {k: output / f"{stem}_{k}.csv" for k in ("side_states", "safe_pair_candidates")}
    states.to_csv(paths["side_states"], index=False)
    pairs.to_csv(paths["safe_pair_candidates"], index=False)
    choice["tables_sha256"] = {k: file_sha256(p) for k, p in paths.items()}
    (output / f"{stem}_choice.json").write_text(json.dumps(choice, indent=2) + "\n")
    print("screened reserve choice frozen", session, choice["chosen"], flush=True)
    return choice


def run(source, reference_dir, reference_audit, output, workers=4):
    reference = robust.checked.checked_manifest(reference_dir, reference_audit)
    splits = json.loads((reference_dir / "frozen_assignments.json").read_text())["splits"]
    if set(splits) != set(base.SESSIONS):
        raise ValueError("original cohort missing")
    inputs = dict(reference["input_file_sha256"])
    for p in (reference_dir / "manifest.json", reference_dir / "frozen_assignments.json", reference_audit, Path(__file__), ROOT / "docs/screened_reserve_content_protocol.md"):
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
        "# Exact screened three/four-cell acquisition\n\nDevelopment only. An audit pass is not a remedy pass.\n\n"
        + markdown(summary)
        + "\n\n"
        + markdown(coverage)
        + "\n\n"
        + markdown(gates)
        + "\n\nNo Q4, test-bank, replay or external scoring. Safeguards/cohort unchanged.\n"
    )
    if any(sha is None or file_sha256(p) != sha for p, sha in inputs.items()) or file_sha256(frozen) != frozen_hash:
        raise ValueError("inputs or frozen choices changed")
    manifest = {
        **build_script_provenance(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation_started_at_utc": validation_started,
        "source_dir": str(source),
        "reference_dir": str(reference_dir),
        "reference_audit": str(reference_audit),
        "input_file_sha256": inputs,
        "frozen_assignments_sha256": frozen_hash,
        "budgets": list(BUDGETS),
        "stages": STAGES,
        "selection": "exhaustive_complete_subsets_with_first_failed_stage_certificate",
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
