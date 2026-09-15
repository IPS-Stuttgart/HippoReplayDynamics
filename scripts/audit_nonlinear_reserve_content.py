#!/usr/bin/env python3
"""Reconstruct integer acquisition decisions and selected fractional proposals."""

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
from scipy.optimize import linear_sum_assignment
from scipy.special import gammaln, logsumexp

from scripts import audit_guarded_reserve_content as prior
from scripts._provenance import build_script_provenance, file_sha256

require, close, compare = prior.require, prior.close, prior.compare
SESSIONS, SOURCES, SIDES = prior.SESSIONS, prior.SOURCES, prior.SIDES
SCALES = [0.0] + [0.05] * 4 + [0.2] * 4


def possible_budgets(reserve):
    maximum = len(reserve) // 2
    return sorted(set(b for b in (1, 2, 4, maximum) if 0 < b <= maximum))


def quota_feasible(weights, n, budget):
    x = np.asarray(weights, float)
    return bool(
        x.shape == (2 * n,)
        and np.isfinite(x).all()
        and (x >= -1e-7).all()
        and (x <= 1 + 1e-7).all()
        and abs(sum(x[:n]) - budget) <= 1e-6
        and abs(sum(x[n:]) - budget) <= 1e-6
        and (x[:n] + x[n:] <= 1 + 1e-6).all()
    )


def full_power_scores(enc, banks, original, reserve, weights):
    """Independent full Poisson, including every weighted count constant."""
    rates, grid, near = enc["early_run"], enc["grid_cm"], enc["near"]
    powers = np.zeros((2, len(rates)))
    for i, side in enumerate(SIDES):
        powers[i, original[side]] = 1
        powers[i, reserve] = np.asarray(weights)[i * len(reserve) : (i + 1) * len(reserve)]
    risks, objective = [], None
    for source in SOURCES:
        bank, homes = banks[source], []
        counts, labels = bank["counts"], bank["labels"]
        for w in powers:
            logp = (counts * w) @ np.log(0.02 * rates) - 0.02 * (w @ rates) - (gammaln(counts + 1) @ w)[:, None]
            probabilities = np.exp(logp - logsumexp(logp, axis=1, keepdims=True))
            home = probabilities[:, near].sum(axis=1)
            error = np.linalg.norm(probabilities @ grid - bank["truth_cm"], axis=1)
            brier = (home - labels) ** 2
            for label in (0, 1):
                for loss in (error, brier):
                    risks.append(float(loss[labels == label].mean()))
            homes.append(home)
        if source == "run_q3":
            objective = float(np.mean([np.mean((homes[0] - homes[1])[labels == k]) ** 2 for k in (0, 1)]))
    return np.asarray(risks), objective


def seeded_random(session, reserve, budget, draw):
    seed = int.from_bytes(hashlib.sha256(f"20260915|reserve|{session}|{draw}".encode()).digest()[:8], "little")
    order = np.random.default_rng(seed).permutation(reserve)
    return dict(high=sorted(map(int, order[:budget])), low=sorted(map(int, order[budget : 2 * budget])))


def seeded_round(session, reserve, budget, name, weights, draw):
    seed = int.from_bytes(hashlib.sha256(f"20260915|nonlinear-reserve-round|{session}|{name}|{draw}".encode()).digest()[:8], "little")
    scores = np.asarray(weights, float).reshape(2, len(reserve)) + SCALES[draw] * np.random.default_rng(seed).normal(size=(2, len(reserve)))
    slots = [0] * budget + [1] * budget + [2] * (len(reserve) - 2 * budget)
    utility = np.vstack((scores, np.zeros(len(reserve))))
    rows, cells = linear_sum_assignment(-utility[slots])
    return {s: sorted(int(reserve[c]) for row, c in zip(rows, cells, strict=True) if slots[row] == i) for i, s in enumerate(SIDES)}


def check_searches(choice, enc, banks, risk0, original, reserve):
    expected, fractional = {}, []
    budgets = possible_budgets(reserve)
    searches = choice["searches"]
    names = [f"budget_{b}_start_{i}_{mode}" for b in budgets for i in (0, 1) for mode in ("minimax", "guarded_objective")]
    require([s["name"] for s in searches] == names, "search coverage/order mismatch")
    named = {c["name"]: c for c in choice["candidates"]}
    scale = np.maximum(abs(risk0), 1e-6)
    for budget in budgets:
        random = []
        for draw in range(20):
            name = f"budget_{budget}_random_{draw:02}"
            expected[name] = ("random", budget, seeded_random(choice["session"], reserve, budget, draw))
            require(name in named, "missing random candidate")
            random.append(named[name])
        best_random = min(random, key=lambda c: (float(np.maximum(0, (np.asarray(c["risks"]) - risk0) / scale).max()), c["objective"], c["name"]))
        starts = [np.full(2 * len(reserve), budget / len(reserve)), np.array([c in best_random["added"][side] for side in SIDES for c in reserve], float)]
        for index, initial in enumerate(starts):
            for mode in ("minimax", "guarded_objective"):
                name = f"budget_{budget}_start_{index}_{mode}"
                record = searches[names.index(name)]
                require(record["mode"] == mode and record["budget"] == budget, "changed solve protocol")
                close(record["initial"], initial)
                require(
                    isinstance(record["status"], int) and bool(record["message"]) and np.isfinite(record["runtime_s"]) and record["runtime_s"] >= 0, "missing solve termination"
                )
                trace = record["trace"]
                require(0 < len(trace) == record["evaluations"] <= 120, "loss evaluation budget")
                feasible = []
                for i, row in enumerate(trace):
                    require(row["evaluation"] == i, "trace order")
                    w, r = np.asarray(row["weights"]), np.asarray(row["risks"])
                    require(
                        w.shape == (2 * len(reserve),) and r.shape == (24,) and np.isfinite(w).all() and np.isfinite(r).all() and np.isfinite(row["objective"]), "invalid trace"
                    )
                    violation = float(max(0, ((r - risk0 - 1e-10) / scale).max()))
                    close(violation, row["max_scaled_violation"])
                    linear = quota_feasible(w, len(reserve), budget)
                    require(row["linear_feasible"] == linear, "incorrect fractional quota flag")
                    if linear:
                        key = (violation, row["objective"]) if mode == "minimax" else (violation > 1e-8, row["objective"] if violation <= 1e-8 else violation, row["objective"])
                        feasible.append((key, row))
                require(bool(feasible), "no feasible trace entry")
                selected = min(feasible, key=lambda item: item[0])[1]
                require(record["selected"] == selected, "selected iterate not frozen trace winner")
                r, j = full_power_scores(enc, banks, original, reserve, selected["weights"])
                close(r, selected["risks"])
                close(j, selected["objective"])
                fractional.append(
                    dict(
                        session=choice["session"],
                        search=name,
                        budget=budget,
                        status=record["status"],
                        evaluations=len(trace),
                        objective=j,
                        max_risk_reconstruction_error=float(max(abs(r - selected["risks"]))),
                    )
                )
                for draw in range(9):
                    expected[f"{name}_round_{draw}"] = ("nonlinear_round", budget, seeded_round(choice["session"], reserve, budget, name, selected["weights"], draw))
                initial = np.array(selected["weights"])
    require(len(named) == len(choice["candidates"]) == 56 * len(budgets) and set(named) == set(expected), "integer candidate coverage")
    for name, (family, budget, added) in expected.items():
        c = named[name]
        require(c["family"] == family and c["budget"] == budget and c["added"] == added, "integer rounding or random seed mismatch")
    return fractional


def audit(source_dir, result_dir, output_dir):
    manifest_path = result_dir / "manifest.json"
    manifest_hash = file_sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    own_files = [Path(__file__)] + [
        ROOT / "scripts" / f"{s}.py"
        for s in (
            "audit_guarded_reserve_content",
            "audit_robust_exchange_content",
            "audit_exchange_content",
            "audit_reserve_cell_content",
            "audit_joint_exchange_truth",
            "_provenance",
        )
    ]
    own_inputs = {str(p): file_sha256(p) for p in own_files}

    def unchanged():
        for path, sha in {**manifest["input_file_sha256"], **own_inputs}.items():
            require(sha is not None and file_sha256(path) == sha, "changed source")
        for name, sha in manifest["output_sha256"].items():
            require(sha is not None and file_sha256(result_dir / name) == sha, "changed output")
        require(file_sha256(manifest_path) == manifest_hash, "changed manifest")

    unchanged()
    reference_dir, reference_audit = Path(manifest["reference_dir"]), Path(manifest["reference_audit"])
    reference = prior.checked_manifest(reference_dir, reference_audit)
    for path, sha in reference["input_file_sha256"].items():
        require(manifest["input_file_sha256"].get(path) == sha, "reference input mismatch")
    for p in (reference_dir / "manifest.json", reference_dir / "frozen_assignments.json", reference_audit):
        require(manifest["input_file_sha256"].get(str(p)) == file_sha256(p), "unlinked reference")
    frozen_path = result_dir / "frozen_assignments.json"
    frozen = json.loads(frozen_path.read_text())
    before = json.loads((result_dir / "pre_selection.json").read_text())
    require(file_sha256(frozen_path) == manifest["frozen_assignments_sha256"], "changed assignments")
    require(before["input_file_sha256"] == frozen["input_file_sha256"] == manifest["input_file_sha256"], "freeze source mismatch")
    require(before["splits"] == frozen["splits"] == json.loads((reference_dir / "frozen_assignments.json").read_text())["splits"], "changed split")
    times = [datetime.fromisoformat(t) for t in (before["created_at_utc"], frozen["created_at_utc"], manifest["validation_started_at_utc"], manifest["created_at_utc"])]
    require(times == sorted(times) and times[1] < times[2], "validation before all choices frozen")
    require(set(frozen["choices"]) == set(frozen["splits"]) == set(SESSIONS), "missing original cohort")
    require(manifest["max_evaluations_per_search"] == 120 and manifest["max_iterations_per_search"] == 60 and manifest["round_scales"] == SCALES, "changed search protocol")
    require(all(manifest[k] is False for k in ("q4_scored", "test_banks_scored", "replay_scored", "external_validation", "validated_remedy")), "unsupported downstream claim")
    summaries, candidate_rows, risks, fractional, validation = [], [], [], [], []
    native_changes, augmented_count = {}, 0
    for session in SESSIONS:
        folder = source_dir / session.replace("/", "_")
        for name in ("encoding",) + SOURCES:
            path = folder / f"{name}.npz"
            require(manifest["input_file_sha256"].get(str(path)) == file_sha256(path), "unregistered source")
        enc = prior.proof.load(folder / "encoding.npz")
        original = {s: sorted(map(int, enc[f"{s}_indices"])) for s in SIDES}
        require(len(original["high"]) == len(original["low"]), "unequal original counts")
        reserve = sorted(set(range(len(enc["early_run"]))) - set(original["high"]) - set(original["low"]))
        choice = frozen["choices"][session]
        require(choice == json.loads((result_dir / f"{session.replace('/', '_')}_choice.json").read_text()), "individual choice mismatch")
        require(choice["session"] == session and choice["original"] == original and choice["reserve"] == reserve and len(reserve) >= 2, "changed original/reserve cells")
        banks = {phase: {} for phase in ("train", "validation")}
        for source in SOURCES:
            bank = prior.proof.load(folder / f"{source}.npz")
            split = frozen["splits"][session][source]
            prior.proof.check_partition(bank, split, session, source)
            for phase in banks:
                ix = np.asarray(split[phase], int)
                banks[phase][source] = {k: bank[k][ix] for k in ("counts", "truth_cm", "labels")}
        r0, j0 = prior.measured(enc, banks["train"], original)
        close(r0, choice["baseline_risks"])
        close(j0, choice["baseline_j"])
        eligible, cache = [], {}
        for c in choice["candidates"]:
            require(c["budget"] in possible_budgets(reserve), "unfrozen budget")
            pair = prior.add(original, reserve, c["budget"], c["added"])
            require(pair == c["pair"], "integer membership mismatch")
            key = tuple(pair["high"]), tuple(pair["low"])
            if key not in cache:
                cache[key] = prior.measured(enc, banks["train"], pair)
            r, j = cache[key]
            close(r, c["risks"])
            close(j, c["objective"])
            failures = int(np.count_nonzero(r > r0 + 1e-10))
            admissible = bool(j < j0 - 1e-10 and failures == 0)
            require(c["failed_risks"] == failures and c["admissible"] == admissible, "incorrect integer eligibility")
            if admissible:
                eligible.append({**c, "objective": j})
            candidate_rows.append(
                dict(session=session, candidate=c["name"], family=c["family"], budget=c["budget"], baseline_j=j0, objective=j, failed_risks=failures, admissible=admissible)
            )
            for i in range(24):
                risks.append(
                    dict(
                        session=session,
                        candidate=c["name"],
                        source=SOURCES[i // 8],
                        side=SIDES[(i % 8) // 4],
                        true_home=(i % 4) // 2,
                        metric="error_cm" if i % 2 == 0 else "home_brier",
                        baseline=r0[i],
                        targeted=r[i],
                        change=r[i] - r0[i],
                        nonworsening=bool(r[i] <= r0[i] + 1e-10),
                    )
                )
        fractional.extend(check_searches(choice, enc, banks["train"], r0, original, reserve))
        best = None
        if eligible:
            best_j = min(c["objective"] for c in eligible)
            best = min((c for c in eligible if c["objective"] - best_j <= 1e-12), key=lambda c: (c["budget"], tuple(c["added"]["high"]), tuple(c["added"]["low"]), c["name"]))
        name, final_pair, final_j, added, budget = (
            (None, original, j0, dict(high=[], low=[]), 0) if best is None else (best["name"], best["pair"], best["objective"], best["added"], best["budget"])
        )
        require(choice["chosen"] == name and choice["final_pair"] == final_pair and choice["added"] == added and choice["budget"] == budget, "choice is not exact training winner")
        close(choice["final_j"], final_j)
        augmented_count += name is not None
        summaries.append(
            dict(
                session=session,
                animal=session.split("/")[0],
                searches=len(choice["searches"]),
                candidates=len(choice["candidates"]),
                admissible_candidates=len(eligible),
                chosen=name,
                budget=budget,
                baseline_j=j0,
                targeted_j=final_j,
            )
        )
        for source in SOURCES:
            bank = banks["validation"][source]
            rb, jb = prior.proof.measure(enc, bank, original)
            ra, ja = prior.proof.measure(enc, bank, final_pair)
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
        print("independently reconstructed nonlinear acquisition", session, flush=True)
    compare(pd.read_csv(result_dir / "selection_summary.csv"), pd.DataFrame(summaries), ["session"])
    compare(pd.read_csv(result_dir / "internal_validation.csv"), pd.DataFrame(validation), ["session", "source", "side", "true_home", "metric"])
    rats = {s.split("/")[0] for s in SESSIONS}
    rat_changes = {r: np.mean([v for s, v in native_changes.items() if s.split("/")[0] == r]) for r in rats}
    gates = dict(
        all_original_pairs_present=True,
        all_validation_risks_present=len(validation) == 96,
        internal_accuracy_nonworsening=all(r["nonworsening"] for r in validation),
        all_pairs_augmented=augmented_count == 4,
        native_validation_improves_all_rats=all(v < -1e-10 for v in rat_changes.values()),
    )
    gates["ready_for_truth_preflight"] = all(gates.values())
    actual = pd.read_csv(result_dir / "gates.csv")
    require(not actual.gate.duplicated().any() and actual.set_index("gate").passed.to_dict() == gates, "gate mismatch")
    unchanged()
    output_dir.mkdir(parents=True, exist_ok=False)
    for name, rows in (
        ("candidate_reconstruction", candidate_rows),
        ("risk_reconstruction", risks),
        ("selected_fractional_reconstruction", fractional),
        ("validation_reconstruction", validation),
    ):
        pd.DataFrame(rows).to_csv(output_dir / f"{name}.csv", index=False)
    record = {
        **build_script_provenance(),
        "status": "pass",
        "manifest_sha256": manifest_hash,
        "integer_candidates_reconstructed": len(candidate_rows),
        "integer_risks_reconstructed": len(risks),
        "selected_fractional_states_reconstructed": len(fractional),
        "validation_risks_reconstructed": len(validation),
        "fractional_scope": "all selected proposal states; all trace allocation/selection rules; intermediate unselected trace losses not numerically reconstructed",
        "solver_optimality_certified": False,
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
