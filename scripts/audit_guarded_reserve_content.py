#!/usr/bin/env python3
"""Independently reconstruct guarded reserve allocations and all calibration risks."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np
import pandas as pd

from scripts import audit_robust_exchange_content as proof
from scripts.audit_joint_exchange_truth import checked_manifest
from scripts._provenance import build_script_provenance, file_sha256

SESSIONS = ("Rat1/Open1", "Rat1/Open2", "Rat2/Open1", "Rat4/Open2")
SOURCES = ("run_q3", "cal_poisson_gain1", "cal_conditional")
SIDES = ("high", "low")
RISK_COLUMNS = [f"risk_{i}" for i in range(24)]


def require(value, message):
    if not value:
        raise ValueError(message)


def close(a, b):
    require(np.isfinite(a).all() and np.isfinite(b).all(), "nonfinite reconstruction")
    np.testing.assert_allclose(a, b, atol=1e-9, rtol=1e-9)


def compare(actual, expected, keys):
    require(set(actual.columns) == set(expected.columns), "table schema mismatch")
    require(not actual.duplicated(keys).any() and len(actual) == len(expected), "incomplete or duplicate table rows")
    a = actual.sort_values(keys).reset_index(drop=True)[expected.columns]
    e = expected.sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, e, check_dtype=False, atol=1e-9, rtol=1e-9)


def add(original, reserve, quota, added):
    require(set(added) == set(SIDES), "missing addition side")
    require(all(len(added[s]) == len(set(added[s])) == quota for s in SIDES), "wrong reserve quota")
    a, b = (set(added[s]) for s in SIDES)
    require(not a & b and (a | b) <= set(reserve), "shared or nonreserve addition")
    result = {s: sorted(set(original[s]) | set(added[s])) for s in SIDES}
    require(set(result["high"]) & set(result["low"]) == set(original["high"]) & set(original["low"]), "overlap changed")
    return result


def measured(enc, banks, pair):
    values = [proof.measure(enc, banks[s], pair) for s in SOURCES]
    return np.concatenate([v[0] for v in values]), values[0][1]


def check_proposals(choice, table, r0, j0, original, reserve, quota):
    proposals = choice["proposals"]
    require(len({p["name"] for p in proposals}) == len(proposals), "duplicate proposal names")
    require({p["family"] for p in proposals} == {"guarded_finite", "objective_finite", "random"}, "proposal family coverage")
    changes = table[RISK_COLUMNS].to_numpy().T - r0[:, None]
    objective = table.objective_change.to_numpy()
    for family in ("guarded_finite", "objective_finite"):
        rows = [p for p in proposals if p["family"] == family]
        require(1 <= len(rows) <= 12, "proposal budget")
        if len(rows) < 12:
            require(not rows[-1]["feasible"], "premature proposal stop")
        memberships = set()
        for i, p in enumerate(rows):
            require(p["name"] == f"{family}_{i:02}", "proposal sequence")
            require(np.isfinite(p["runtime_s"]) and p["runtime_s"] >= 0, "missing solver runtime")
            if not p["feasible"]:
                require(i == len(rows) - 1, "continued after failed solver")
                continue
            require(p["status"] in (0, 1), "invalid incumbent status")
            add(original, reserve, quota, p["added"])
            key = tuple(p["added"]["high"]), tuple(p["added"]["low"])
            require(key not in memberships, "duplicate allocation within family")
            memberships.add(key)
            x = np.array([cell in p["added"][s] for s in SIDES for cell in reserve], int)
            close(changes @ x, p["predicted_risk_changes"])
            close(objective @ x, p["predicted_j_change"])
            if family == "guarded_finite":
                require(np.all((changes @ x) / np.maximum(abs(r0), 1e-6) <= 1e-7), "infeasible approximate guard")
    random = [p for p in proposals if p["family"] == "random"]
    require([p["name"] for p in random] == [f"random_{i:02}" for i in range(20)], "random coverage")
    for i, p in enumerate(random):
        seed = int.from_bytes(hashlib.sha256(f"20260915|reserve|{choice['session']}|{i}".encode()).digest()[:8], "little")
        order = np.random.default_rng(seed).permutation(reserve)
        expected = dict(high=sorted(map(int, order[:quota])), low=sorted(map(int, order[quota : 2 * quota])))
        require(p["feasible"] and p["added"] == expected, "random seed or membership mismatch")
    return {p["name"]: p for p in proposals if p["feasible"]}


def audit(source_dir, result_dir, output_dir):
    manifest_path = result_dir / "manifest.json"
    manifest_hash = file_sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    own_inputs = {
        str(p): file_sha256(p)
        for p in (
            Path(__file__),
            ROOT / "scripts/audit_robust_exchange_content.py",
            ROOT / "scripts/audit_exchange_content.py",
            ROOT / "scripts/audit_reserve_cell_content.py",
            ROOT / "scripts/audit_joint_exchange_truth.py",
            ROOT / "scripts/_provenance.py",
        )
    }

    def unchanged():
        for path, sha in manifest["input_file_sha256"].items():
            require(sha is not None and file_sha256(path) == sha, "changed producer input")
        for name, sha in manifest["output_sha256"].items():
            require(sha is not None and file_sha256(result_dir / name) == sha, "changed producer output")
        for path, sha in own_inputs.items():
            require(sha is not None and file_sha256(path) == sha, "changed auditor source")
        require(file_sha256(manifest_path) == manifest_hash, "changed manifest")

    unchanged()
    reference_dir, reference_audit = Path(manifest["reference_dir"]), Path(manifest["reference_audit"])
    reference = checked_manifest(reference_dir, reference_audit)
    for path, sha in reference["input_file_sha256"].items():
        require(manifest["input_file_sha256"].get(path) == sha, "reference source mismatch")
    for p in (reference_dir / "manifest.json", reference_dir / "frozen_assignments.json", reference_audit):
        require(manifest["input_file_sha256"].get(str(p)) == file_sha256(p), "unlinked reference")
    frozen_path = result_dir / "frozen_assignments.json"
    frozen = json.loads(frozen_path.read_text())
    before = json.loads((result_dir / "pre_selection.json").read_text())
    require(file_sha256(frozen_path) == manifest["frozen_assignments_sha256"], "changed assignments")
    require(before["input_file_sha256"] == frozen["input_file_sha256"] == manifest["input_file_sha256"], "freeze source mismatch")
    require(before["splits"] == frozen["splits"] == json.loads((reference_dir / "frozen_assignments.json").read_text())["splits"], "changed split")
    times = [datetime.fromisoformat(v) for v in (before["created_at_utc"], frozen["created_at_utc"], manifest["validation_started_at_utc"], manifest["created_at_utc"])]
    require(times == sorted(times) and times[1] < times[2], "validation before all choices frozen")
    require(set(frozen["choices"]) == set(frozen["splits"]) == set(SESSIONS), "missing original cohort")
    require(manifest["solver_options"] == dict(time_limit=20.0, node_limit=64, mip_rel_gap=0.001) and manifest["max_proposals_per_family"] == 12, "changed search protocol")
    require(
        all(manifest[k] is False for k in ("q4_scored", "test_banks_scored", "replay_scored", "external_validation", "validated_remedy")), "unsupported downstream evaluation claim"
    )
    summaries, singles, candidates, risk_rows, validation, random_rows = [], [], [], [], [], []
    native_changes = {}
    augmented_count = 0
    for session in SESSIONS:
        folder = source_dir / session.replace("/", "_")
        for name in ("encoding",) + SOURCES:
            p = folder / f"{name}.npz"
            require(manifest["input_file_sha256"].get(str(p)) == file_sha256(p), "unregistered source")
        enc = proof.load(folder / "encoding.npz")
        original = {s: sorted(map(int, enc[f"{s}_indices"])) for s in SIDES}
        reserve = sorted(set(range(len(enc["early_run"]))) - set(original["high"]) - set(original["low"]))
        quota = len(reserve) // 2
        choice = frozen["choices"][session]
        require(choice == json.loads((result_dir / f"{session.replace('/', '_')}_choice.json").read_text()), "individual choice mismatch")
        require(
            choice["session"] == session and choice["original"] == original and choice["reserve"] == reserve and choice["quota"] == quota > 0, "changed baseline or reserve budget"
        )
        banks = {phase: {} for phase in ("train", "validation")}
        for source in SOURCES:
            bank = proof.load(folder / f"{source}.npz")
            split = frozen["splits"][session][source]
            proof.check_partition(bank, split, session, source)
            for phase in banks:
                ix = np.asarray(split[phase], int)
                banks[phase][source] = {k: bank[k][ix] for k in ("counts", "truth_cm", "labels")}
        r0, j0 = measured(enc, banks["train"], original)
        close(r0, choice["baseline_risks"])
        close(j0, choice["baseline_j"])
        single_path = result_dir / f"{session.replace('/', '_')}_single_additions.csv"
        require(file_sha256(single_path) == choice["single_addition_sha256"], "changed single additions")
        table = pd.read_csv(single_path)
        require(list(table[["side", "cell"]].itertuples(index=False, name=None)) == [(s, i) for s in SIDES for i in reserve], "incomplete single additions")
        for row in table.itertuples(index=False):
            pair = {s: original[s] + ([row.cell] if s == row.side else []) for s in SIDES}
            risk, j = measured(enc, banks["train"], pair)
            recorded = [getattr(row, k) for k in RISK_COLUMNS]
            close(risk, recorded)
            close([j, j - j0], [row.objective, row.objective_change])
            singles.append(dict(session=session, side=row.side, cell=row.cell, objective=j, max_risk_reconstruction_error=float(np.max(abs(risk - recorded)))))
        proposals = check_proposals(choice, table, r0, j0, original, reserve, quota)
        require(len(choice["candidates"]) == len(proposals) and {c["name"] for c in choice["candidates"]} == set(proposals), "candidate coverage mismatch")
        eligible, cache = [], {}
        for c in choice["candidates"]:
            p = proposals[c["name"]]
            require(c["added"] == p["added"] and c["family"] == p["family"], "candidate proposal mismatch")
            pair = add(original, reserve, quota, c["added"])
            require(c["pair"] == pair, "candidate membership mismatch")
            key = tuple(pair["high"]), tuple(pair["low"])
            if key not in cache:
                cache[key] = measured(enc, banks["train"], pair)
            risk, j = cache[key]
            close(risk, c["risks"])
            close(j, c["objective"])
            failures = int(np.count_nonzero(risk > r0 + 1e-10))
            admissible = bool(j < j0 - 1e-10 and failures == 0)
            require(c["admissible"] == admissible and c["failed_risks"] == failures, "candidate accuracy or eligibility mismatch")
            if admissible:
                eligible.append({**c, "objective": j})
            candidates.append(dict(session=session, candidate=c["name"], family=c["family"], baseline_j=j0, objective=j, failed_risks=failures, admissible=admissible))
            for i in range(24):
                risk_rows.append(
                    dict(
                        session=session,
                        candidate=c["name"],
                        source=SOURCES[i // 8],
                        side=SIDES[(i % 8) // 4],
                        true_home=(i % 4) // 2,
                        metric="error_cm" if i % 2 == 0 else "home_brier",
                        baseline=r0[i],
                        targeted=risk[i],
                        change=risk[i] - r0[i],
                        nonworsening=bool(risk[i] <= r0[i] + 1e-10),
                    )
                )
        best = None
        if eligible:
            best_j = min(c["objective"] for c in eligible)
            best = min((c for c in eligible if c["objective"] - best_j <= 1e-12), key=lambda c: (tuple(c["added"]["high"]), tuple(c["added"]["low"]), c["name"]))
        name, final_pair, final_j, added = (None, original, j0, dict(high=[], low=[])) if best is None else (best["name"], best["pair"], best["objective"], best["added"])
        require(choice["chosen"] == name and choice["final_pair"] == final_pair and choice["added"] == added, "choice is not the exact training winner")
        close(choice["final_j"], final_j)
        augmented_count += name is not None
        summaries.append(
            dict(
                session=session,
                animal=session.split("/")[0],
                quota=quota,
                reserve_cells=len(reserve),
                single_additions=2 * len(reserve),
                exact_candidates=len(choice["candidates"]),
                admissible_candidates=len(eligible),
                chosen=name,
                baseline_j=j0,
                final_j=final_j,
            )
        )
        for source in SOURCES:
            bank = banks["validation"][source]
            r_before, j_before = proof.measure(enc, bank, original)
            r_after, j_after = proof.measure(enc, bank, final_pair)
            if source == "run_q3":
                native_changes[session] = j_after - j_before
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
                        observations=int(np.count_nonzero(bank["labels"] == label)),
                        baseline=r_before[i],
                        targeted=r_after[i],
                        change=r_after[i] - r_before[i],
                        nonworsening=bool(r_after[i] <= r_before[i] + 1e-10),
                        baseline_j=j_before,
                        targeted_j=j_after,
                    )
                )
        for c in choice["candidates"]:
            if c["family"] == "random":
                _, value = proof.measure(enc, banks["validation"]["run_q3"], c["pair"])
                random_rows.append(dict(session=session, animal=session.split("/")[0], method=c["name"], validation_j=value))
        print("independently reconstructed reserve calibration", session, flush=True)
    compare(pd.read_csv(result_dir / "selection_summary.csv"), pd.DataFrame(summaries), ["session"])
    compare(pd.read_csv(result_dir / "internal_validation.csv"), pd.DataFrame(validation), ["session", "source", "side", "true_home", "metric"])
    compare(pd.read_csv(result_dir / "random_native_validation.csv"), pd.DataFrame(random_rows), ["session", "method"])
    rats = {s.split("/")[0] for s in SESSIONS}
    rat_j = {r: np.mean([v for s, v in native_changes.items() if s.split("/")[0] == r]) for r in rats}
    gates = dict(
        all_original_pairs_present=True,
        all_validation_risks_present=len(validation) == 96,
        internal_accuracy_nonworsening=all(r["nonworsening"] for r in validation),
        all_pairs_augmented=augmented_count == 4,
        native_validation_improves_all_rats=all(v < -1e-10 for v in rat_j.values()),
    )
    gates["ready_for_truth_preflight"] = all(gates.values())
    actual_gates = pd.read_csv(result_dir / "gates.csv")
    require(not actual_gates.gate.duplicated().any() and actual_gates.set_index("gate").passed.to_dict() == gates, "gate mismatch")
    unchanged()
    output_dir.mkdir(parents=True, exist_ok=False)
    for name, rows in (
        ("single_addition_reconstruction", singles),
        ("candidate_reconstruction", candidates),
        ("risk_reconstruction", risk_rows),
        ("validation_reconstruction", validation),
        ("random_validation_reconstruction", random_rows),
    ):
        pd.DataFrame(rows).to_csv(output_dir / f"{name}.csv", index=False)
    record = {
        **build_script_provenance(),
        "status": "pass",
        "manifest_sha256": manifest_hash,
        "single_additions_reconstructed": len(singles),
        "joint_candidates_reconstructed": len(candidates),
        "joint_risks_reconstructed": len(risk_rows),
        "validation_risks_reconstructed": len(validation),
        "random_validation_objectives_reconstructed": len(random_rows),
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
