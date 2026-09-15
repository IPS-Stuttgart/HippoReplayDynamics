#!/usr/bin/env python3
"""Independently verify exhaustive small acquisition and full paired likelihoods."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from itertools import combinations, product
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np
import pandas as pd

from scripts import audit_guarded_reserve_content as prior
from scripts import audit_reserve_cell_content as poisson
from scripts._provenance import build_script_provenance, file_sha256

SESSIONS, SOURCES, SIDES = prior.SESSIONS, prior.SOURCES, prior.SIDES
require, close, compare = prior.require, prior.close, prior.compare
RISK_COLUMNS = [f"risk_{i}" for i in range(12)]
PAIR_COLUMNS = ["high_state", "low_state", "budget", "disjoint", "objective", "admissible"]


def full_side(enc, banks, indices):
    risks, means = [], None
    for source in SOURCES:
        bank = banks[source]
        p = poisson.posterior(bank["counts"][:, indices], enc["early_run"][indices])
        home = p[:, enc["near"]].sum(axis=1)
        error = np.sqrt(np.sum((np.einsum("nb,bd->nd", p, enc["grid_cm"]) - bank["truth_cm"]) ** 2, axis=1))
        for k in (0, 1):
            selected = bank["labels"] == k
            require(selected.any(), "missing class")
            risks.extend([float(error[selected].mean()), float(((home[selected] - k) ** 2).mean())])
        if source == "run_q3":
            means = np.array([home[bank["labels"] == k].mean() for k in (0, 1)])
    return np.asarray(risks), means


def population_states(enc, banks, original, reserve, table):
    budgets = [b for b in (1, 2) if 2 * b <= len(reserve)]
    expected = [(s, b, cells) for s in SIDES for b in budgets for cells in combinations(reserve, b)]
    columns = ["state_index", "side", "budget", "cell_1", "cell_2", "native_mean_nonhome", "native_mean_home"] + RISK_COLUMNS + ["failed_risks", "safe", "runtime_s"]
    require(set(table.columns) == set(columns), "population schema")
    require(len(table) == len(expected) and table.state_index.tolist() == list(range(len(expected))), "exhaustive subset coverage")
    baseline = {s: full_side(enc, banks, original[s]) for s in SIDES}
    reconstructed, risk_rows = [], []
    for row, (side, budget, added) in zip(table.to_dict("records"), expected, strict=True):
        require(
            row["side"] == side and row["budget"] == budget and row["cell_1"] == added[0] and row["cell_2"] == (-1 if budget == 1 else added[1]), "subset identity/order mismatch"
        )
        require(np.isfinite(row["runtime_s"]) and row["runtime_s"] >= 0, "missing subset runtime")
        risk, means = full_side(enc, banks, sorted(original[side] + list(added)))
        close(risk, [row[k] for k in RISK_COLUMNS])
        close(means, [row["native_mean_nonhome"], row["native_mean_home"]])
        failures = int(np.count_nonzero(risk > baseline[side][0] + 1e-10))
        require(row["safe"] == (failures == 0) and row["failed_risks"] == failures, "incorrect side eligibility")
        reconstructed.append(
            dict(state_index=row["state_index"], side=side, budget=budget, added=list(added), risks=risk, native_means=means, safe=failures == 0, failed_risks=failures)
        )
        for i in range(12):
            risk_rows.append(
                dict(
                    state_index=row["state_index"],
                    side=side,
                    budget=budget,
                    source=SOURCES[i // 4],
                    true_home=(i % 4) // 2,
                    metric="error_cm" if i % 2 == 0 else "home_brier",
                    baseline=baseline[side][0][i],
                    targeted=risk[i],
                    change=risk[i] - baseline[side][0][i],
                    nonworsening=bool(risk[i] <= baseline[side][0][i] + 1e-10),
                )
            )
    return baseline, reconstructed, risk_rows


def paired_states(enc, banks, original, reserve, states, baseline, actual):
    j0 = float(np.mean((baseline["high"][1] - baseline["low"][1]) ** 2))
    r0 = np.concatenate([np.r_[baseline["high"][0][4 * i : 4 * i + 4], baseline["low"][0][4 * i : 4 * i + 4]] for i in range(3)])
    rows, eligible, coverage = [], [], []
    for budget in (b for b in (1, 2) if 2 * b <= len(reserve)):
        safe = {s: [v for v in states if v["side"] == s and v["budget"] == budget and v["safe"]] for s in SIDES}
        disjoint_count, allowed_count = 0, 0
        for high, low in product(safe["high"], safe["low"]):
            disjoint = not set(high["added"]) & set(low["added"])
            j = float(np.dot(high["native_means"] - low["native_means"], high["native_means"] - low["native_means"]) / 2)
            pair = {s: sorted(original[s] + v["added"]) for s, v in (("high", high), ("low", low))}
            if disjoint:
                direct_risk, direct_j = prior.measured(enc, banks, pair)
                expected_risk = np.concatenate([np.r_[high["risks"][4 * i : 4 * i + 4], low["risks"][4 * i : 4 * i + 4]] for i in range(3)])
                close(direct_risk, expected_risk)
                close(direct_j, j)
                require(np.all(direct_risk <= r0 + 1e-10), "full-pair risk disagrees with safe components")
                disjoint_count += 1
            allowed = bool(disjoint and j < j0 - 1e-10)
            row = dict(high_state=high["state_index"], low_state=low["state_index"], budget=budget, disjoint=disjoint, objective=j, admissible=allowed)
            rows.append(row)
            if allowed:
                eligible.append(dict(row=row, pair=pair, added=dict(high=high["added"], low=low["added"])))
                allowed_count += 1
        coverage.append(
            dict(
                budget=budget,
                safe_pair_products=len(safe["high"]) * len(safe["low"]),
                disjoint_safe_pairs=disjoint_count,
                admissible_pairs=allowed_count,
                **{f"{s}_{k}": len(safe[s]) if k == "safe" else comb(len(reserve), budget) for s in SIDES for k in ("states", "safe")},
            )
        )
    compare(actual, pd.DataFrame(rows, columns=PAIR_COLUMNS), ["high_state", "low_state"])
    best = None
    if eligible:
        minimum = min(v["row"]["objective"] for v in eligible)
        best = min((v for v in eligible if v["row"]["objective"] <= minimum + 1e-12), key=lambda v: (v["row"]["budget"], tuple(v["added"]["high"]), tuple(v["added"]["low"])))
    return j0, rows, coverage, best


def audit(source_dir, result_dir, output_dir):
    manifest_path = result_dir / "manifest.json"
    manifest_hash = file_sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    names = ("audit_guarded_reserve_content", "audit_robust_exchange_content", "audit_exchange_content", "audit_reserve_cell_content", "audit_joint_exchange_truth", "_provenance")
    own_inputs = {str(p): file_sha256(p) for p in [Path(__file__)] + [ROOT / "scripts" / f"{n}.py" for n in names]}

    def unchanged():
        for p, sha in {**manifest["input_file_sha256"], **own_inputs}.items():
            require(sha is not None and file_sha256(p) == sha, "changed source")
        for p, sha in manifest["output_sha256"].items():
            require(sha is not None and file_sha256(result_dir / p) == sha, "changed output")
        require(file_sha256(manifest_path) == manifest_hash, "changed manifest")

    unchanged()
    reference_dir, reference_audit = Path(manifest["reference_dir"]), Path(manifest["reference_audit"])
    reference = prior.checked_manifest(reference_dir, reference_audit)
    for p, sha in reference["input_file_sha256"].items():
        require(manifest["input_file_sha256"].get(p) == sha, "reference input mismatch")
    for p in (reference_dir / "manifest.json", reference_dir / "frozen_assignments.json", reference_audit):
        require(manifest["input_file_sha256"].get(str(p)) == file_sha256(p), "unlinked reference")
    frozen_path = result_dir / "frozen_assignments.json"
    frozen = json.loads(frozen_path.read_text())
    before = json.loads((result_dir / "pre_selection.json").read_text())
    require(file_sha256(frozen_path) == manifest["frozen_assignments_sha256"], "changed choices")
    require(before["input_file_sha256"] == frozen["input_file_sha256"] == manifest["input_file_sha256"], "freeze source mismatch")
    require(before["splits"] == frozen["splits"] == json.loads((reference_dir / "frozen_assignments.json").read_text())["splits"], "changed partition")
    times = [datetime.fromisoformat(t) for t in (before["created_at_utc"], frozen["created_at_utc"], manifest["validation_started_at_utc"], manifest["created_at_utc"])]
    require(times == sorted(times) and times[1] < times[2], "validation before choices frozen")
    require(set(frozen["choices"]) == set(frozen["splits"]) == set(SESSIONS), "missing original cohort")
    require(manifest["budgets"] == [1, 2] and manifest["selection"] == "exhaustive_population_subsets_then_disjoint_safe_pair_cross_product", "changed enumeration rule")
    require(all(manifest[k] is False for k in ("q4_scored", "test_banks_scored", "replay_scored", "external_validation", "validated_remedy")), "unsupported downstream claim")
    summaries, coverage_rows, state_rows, risk_rows, pair_rows, validation = [], [], [], [], [], []
    native_changes, augmented_count = {}, 0
    for session in SESSIONS:
        folder, stem = source_dir / session.replace("/", "_"), session.replace("/", "_")
        for name in ("encoding",) + SOURCES:
            p = folder / f"{name}.npz"
            require(manifest["input_file_sha256"].get(str(p)) == file_sha256(p), "unregistered input")
        enc = prior.proof.load(folder / "encoding.npz")
        original = {s: sorted(map(int, enc[f"{s}_indices"])) for s in SIDES}
        require(len(original["high"]) == len(original["low"]), "unequal original populations")
        reserve = sorted(set(range(len(enc["early_run"]))) - set(original["high"]) - set(original["low"]))
        choice = frozen["choices"][session]
        require(choice == json.loads((result_dir / f"{stem}_choice.json").read_text()), "individual freeze mismatch")
        require(choice["session"] == session and choice["original"] == original and choice["reserve"] == reserve, "changed originals/reserve")
        banks = {phase: {} for phase in ("train", "validation")}
        for source in SOURCES:
            bank = prior.proof.load(folder / f"{source}.npz")
            split = frozen["splits"][session][source]
            prior.proof.check_partition(bank, split, session, source)
            for phase in banks:
                banks[phase][source] = {k: bank[k][np.asarray(split[phase], int)] for k in ("counts", "truth_cm", "labels")}
        paths = {k: result_dir / f"{stem}_{k}.csv" for k in ("side_states", "safe_pair_candidates")}
        require(choice["tables_sha256"] == {k: file_sha256(p) for k, p in paths.items()}, "changed enumeration tables")
        baseline, states, risks = population_states(enc, banks["train"], original, reserve, pd.read_csv(paths["side_states"]))
        for side in SIDES:
            close(baseline[side][0], choice["baseline_by_side"][side]["risks"])
            close(baseline[side][1], choice["baseline_by_side"][side]["native_means"])
        j0, pairs, coverage, best = paired_states(enc, banks["train"], original, reserve, states, baseline, pd.read_csv(paths["safe_pair_candidates"]))
        close(j0, choice["baseline_j"])
        require(choice["coverage"] == coverage, "incorrect exhaustive coverage counts")
        if best is None:
            name, pair, added, budget, j = None, original, dict(high=[], low=[]), 0, j0
            require(choice["chosen_pair"] is None, "non-improving choice")
        else:
            row, pair, added = best["row"], best["pair"], best["added"]
            name, budget, j = f"high_{row['high_state']}_low_{row['low_state']}", row["budget"], row["objective"]
            for k in PAIR_COLUMNS:
                if k == "objective":
                    close(choice["chosen_pair"][k], row[k])
                else:
                    require(choice["chosen_pair"][k] == row[k], "chosen pair mismatch")
            require(prior.add(original, reserve, budget, added) == pair, "overlap or membership changed")
        require(choice["chosen"] == name and choice["final_pair"] == pair and choice["added"] == added and choice["budget"] == budget, "not exact training winner")
        close(choice["final_j"], j)
        augmented_count += name is not None
        summaries.append(
            dict(
                session=session,
                animal=session.split("/")[0],
                population_subsets=len(states),
                admissible_pairs=sum(r["admissible_pairs"] for r in coverage),
                chosen=name,
                budget=budget,
                baseline_j=j0,
                targeted_j=j,
            )
        )
        coverage_rows.extend(dict(session=session, animal=session.split("/")[0], **r) for r in coverage)
        state_rows.extend(
            dict(
                session=session,
                state_index=v["state_index"],
                side=v["side"],
                budget=v["budget"],
                cell_1=v["added"][0],
                cell_2=-1 if v["budget"] == 1 else v["added"][1],
                failed_risks=v["failed_risks"],
                safe=v["safe"],
                native_mean_nonhome=v["native_means"][0],
                native_mean_home=v["native_means"][1],
            )
            for v in states
        )
        risk_rows.extend(dict(session=session, **r) for r in risks)
        pair_rows.extend(dict(session=session, **r) for r in pairs)
        for source in SOURCES:
            bank = banks["validation"][source]
            rb, jb = prior.proof.measure(enc, bank, original)
            ra, ja = prior.proof.measure(enc, bank, pair)
            if source == "run_q3":
                native_changes[session] = ja - jb
            for i in range(8):
                label = (i % 4) // 2
                validation.append(
                    dict(
                        session=session,
                        animal=session.split("/")[0],
                        source=source,
                        side=SIDES[i // 4],
                        true_home=label,
                        metric="error_cm" if i % 2 == 0 else "home_brier",
                        observations=int(sum(bank["labels"] == label)),
                        baseline=rb[i],
                        targeted=ra[i],
                        change=ra[i] - rb[i],
                        nonworsening=bool(ra[i] <= rb[i] + 1e-10),
                        baseline_j=jb,
                        targeted_j=ja,
                    )
                )
        print("independently reconstructed exhaustive subsets and pairs", session, len(states), len(pairs), flush=True)
    compare(pd.read_csv(result_dir / "selection_summary.csv"), pd.DataFrame(summaries), ["session"])
    compare(pd.read_csv(result_dir / "exhaustive_coverage.csv"), pd.DataFrame(coverage_rows), ["session", "budget"])
    compare(pd.read_csv(result_dir / "internal_validation.csv"), pd.DataFrame(validation), ["session", "source", "side", "true_home", "metric"])
    rats = {s.split("/")[0] for s in SESSIONS}
    changes = {r: np.mean([v for s, v in native_changes.items() if s.split("/")[0] == r]) for r in rats}
    gates = dict(
        all_original_pairs_present=True,
        all_validation_risks_present=len(validation) == 96,
        internal_accuracy_nonworsening=all(v["nonworsening"] for v in validation),
        all_pairs_augmented=augmented_count == 4,
        native_validation_improves_all_rats=all(v < -1e-10 for v in changes.values()),
    )
    gates["ready_for_truth_preflight"] = all(gates.values())
    actual = pd.read_csv(result_dir / "gates.csv")
    require(not actual.gate.duplicated().any() and actual.set_index("gate").passed.to_dict() == gates, "gate mismatch")
    unchanged()
    output_dir.mkdir(parents=True, exist_ok=False)
    for name, rows in (
        ("population_state_reconstruction", state_rows),
        ("population_risk_reconstruction", risk_rows),
        ("exhaustive_coverage", coverage_rows),
        ("validation_reconstruction", validation),
    ):
        pd.DataFrame(rows).to_csv(output_dir / f"{name}.csv", index=False)
    pd.DataFrame(pair_rows, columns=["session"] + PAIR_COLUMNS).to_csv(output_dir / "safe_pair_reconstruction.csv", index=False)
    record = {
        **build_script_provenance(),
        "status": "pass",
        "manifest_sha256": manifest_hash,
        "population_subsets_reconstructed": len(state_rows),
        "population_risks_reconstructed": len(risk_rows),
        "safe_pair_cross_products_verified": len(pair_rows),
        "disjoint_safe_pairs_full_likelihood_reconstructed": sum(r["disjoint"] for r in pair_rows),
        "validation_risks_reconstructed": len(validation),
        "exhaustive_budgets": [1, 2],
        "scope": "all feasible frozen one/two-cell reserve subsets and every same-budget safe pair; no completeness claim for larger budgets or other samples",
        "ready_for_truth_preflight": gates["ready_for_truth_preflight"],
        "validated_remedy": False,
        "external_validation": False,
        "input_file_sha256": {str(manifest_path): manifest_hash, **own_inputs},
        "output_sha256": {p.name: file_sha256(p) for p in output_dir.iterdir()},
    }
    (output_dir / "independent_audit.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("source-dir", "result-dir", "output-dir"):
        parser.add_argument(f"--{flag}", required=True, type=Path)
    args = parser.parse_args()
    audit(args.source_dir, args.result_dir, args.output_dir)
