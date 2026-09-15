#!/usr/bin/env python3
"""Verify every finite subset certificate without trusting producer likelihoods."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from itertools import combinations, product
import json
from math import comb
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np
import pandas as pd
from scipy.special import gammaln, logsumexp

from scripts import audit_exact_small_reserve_content as full
from scripts._provenance import build_script_provenance, file_sha256

prior = full.prior
SESSIONS, SOURCES, SIDES = full.SESSIONS, full.SOURCES, full.SIDES
require, close, compare = full.require, full.close, full.compare
BUDGETS = (3, 4)
STAGES = tuple((source, k) for k in (0, 1) for source in ("cal_poisson_gain1", "cal_conditional", "run_q3"))
RISKS, PAIR_COLUMNS = full.RISK_COLUMNS, full.PAIR_COLUMNS


class IndependentStages:
    def __init__(self, enc, banks, original):
        self.enc, self.parts, self.offsets = enc, {}, {}
        rates = enc["early_run"]
        require(np.isfinite(rates).all() and np.all(rates > 0), "invalid rate map")
        self.mu, self.logmu = 0.02 * rates, np.log(0.02 * rates)
        for i, (source, label) in enumerate(STAGES):
            bank = banks[source]
            require(np.isfinite(bank["counts"]).all() and np.all(bank["counts"] >= 0), "invalid counts")
            mask = bank["labels"] == label
            require(mask.any(), "missing stage class")
            part = {k: bank[k][mask] for k in ("counts", "truth_cm", "labels")}
            self.parts[i] = part
            for side in SIDES:
                ix = original[side]
                count = part["counts"][:, ix]
                ll = np.einsum("nc,cb->nb", count, self.logmu[ix], optimize=True) - self.mu[ix].sum(axis=0)
                self.offsets[side, i] = ll - gammaln(count + 1).sum(axis=1)[:, None]

    def stage(self, side, added, stage):
        part = self.parts[stage]
        ll = self.offsets[side, stage].copy()
        if added:
            count = part["counts"][:, added]
            ll += np.einsum("nc,cb->nb", count, self.logmu[added], optimize=True) - self.mu[added].sum(axis=0)
            ll -= gammaln(count + 1).sum(axis=1)[:, None]
        p = np.exp(ll - logsumexp(ll, axis=1)[:, None])
        home = p[:, self.enc["near"]].sum(axis=1)
        error = np.sqrt(np.sum((np.einsum("nb,bd->nd", p, self.enc["grid_cm"]) - part["truth_cm"]) ** 2, axis=1))
        values = np.array([error.mean(), ((home - STAGES[stage][1]) ** 2).mean()])
        require(np.isfinite(values).all() and np.isfinite(home).all(), "nonfinite reconstruction")
        return values, float(home.mean())


def states_for(enc, banks, original, reserve, table):
    columns = ["state_index", "side", "budget", "added_cells", "stages_evaluated", "safe", "native_mean_nonhome", "native_mean_home"] + RISKS + ["runtime_s"]
    require(set(table.columns) == set(columns), "state schema")
    budgets = [b for b in BUDGETS if 2 * b <= len(reserve)]
    expected = [(s, b, c) for s in SIDES for b in budgets for c in combinations(reserve, b)]
    require(len(table) == len(expected) and table.state_index.tolist() == list(range(len(expected))), "incomplete subset enumeration")
    scorer = IndependentStages(enc, banks, original)
    baseline = {s: full.full_side(enc, banks, original[s]) for s in SIDES}
    for side in SIDES:
        for i, (source, label) in enumerate(STAGES):
            risk, mean = scorer.stage(side, [], i)
            ix = 4 * SOURCES.index(source) + 2 * label
            close(risk, baseline[side][0][ix : ix + 2])
            if source == "run_q3":
                close(mean, baseline[side][1][label])
    states, audit_rows, checked = [], [], 0
    for row, (side, budget, added) in zip(table.to_dict("records"), expected, strict=True):
        require(row["side"] == side and row["budget"] == budget and json.loads(row["added_cells"]) == list(added), "subset identity mismatch")
        require(np.isfinite(row["runtime_s"]) and row["runtime_s"] >= 0, "missing runtime")
        risks, means = np.full(12, np.nan), np.full(2, np.nan)
        safe, error_max = True, 0.0
        for i, (source, label) in enumerate(STAGES):
            ix = 4 * SOURCES.index(source) + 2 * label
            risk, mean = scorer.stage(side, list(added), i)
            actual = np.array([row[k] for k in RISKS[ix : ix + 2]])
            close(risk, actual)
            error_max = max(error_max, float(np.max(abs(risk - actual))))
            risks[ix : ix + 2] = risk
            checked += 2
            if source == "run_q3":
                means[label] = mean
            if np.any(risk > baseline[side][0][ix : ix + 2] + 1e-10):
                safe = False
                break
        require(row["safe"] == safe and row["stages_evaluated"] == i + 1, "incorrect first-failure certificate")
        actual_risks = np.array([row[k] for k in RISKS])
        require(np.array_equal(np.isnan(actual_risks), np.isnan(risks)), "uncomputed risks hidden or fabricated")
        actual_means = np.array([row["native_mean_nonhome"], row["native_mean_home"]])
        require(np.array_equal(np.isnan(actual_means), np.isnan(means)), "missing/fabricated native mean")
        finite = np.isfinite(means)
        close(means[finite], actual_means[finite])
        if safe:
            # Recompute every survivor from the full original-plus-added count matrix.
            direct, direct_means = full.full_side(enc, banks, sorted(original[side] + list(added)))
            close(direct, risks)
            close(direct_means, means)
        states.append(dict(state_index=row["state_index"], side=side, budget=budget, added=list(added), safe=safe, risks=risks, native_means=means))
        audit_rows.append(
            dict(
                state_index=row["state_index"],
                side=side,
                budget=budget,
                added_cells=json.dumps(list(added)),
                stages_evaluated=i + 1,
                safe=safe,
                max_risk_reconstruction_error=error_max,
            )
        )
    return baseline, states, audit_rows, checked


def pairs_for(enc, banks, original, reserve, states, baseline, actual):
    j0 = float(np.mean((baseline["high"][1] - baseline["low"][1]) ** 2))
    r0, _ = prior.measured(enc, banks, original)
    rows, allowed, coverage = [], [], []
    for budget in (b for b in BUDGETS if 2 * b <= len(reserve)):
        safe = {s: [r for r in states if r["side"] == s and r["budget"] == budget and r["safe"]] for s in SIDES}
        disjoint_n, allowed_n = 0, 0
        for h, low in product(safe["high"], safe["low"]):
            disjoint = not set(h["added"]) & set(low["added"])
            j = float(np.mean((h["native_means"] - low["native_means"]) ** 2))
            pair = {s: sorted(original[s] + v["added"]) for s, v in (("high", h), ("low", low))}
            if disjoint:
                direct_risk, direct_j = prior.measured(enc, banks, pair)
                expected = np.concatenate([np.r_[h["risks"][4 * i : 4 * i + 4], low["risks"][4 * i : 4 * i + 4]] for i in range(3)])
                close(direct_risk, expected)
                close(direct_j, j)
                require(np.all(direct_risk <= r0 + 1e-10), "full pair fails risk")
                disjoint_n += 1
            admissible = bool(disjoint and j < j0 - 1e-10)
            row = dict(high_state=h["state_index"], low_state=low["state_index"], budget=budget, disjoint=disjoint, objective=j, admissible=admissible)
            rows.append(row)
            if admissible:
                allowed.append(dict(row=row, pair=pair, added=dict(high=h["added"], low=low["added"])))
                allowed_n += 1
        coverage.append(
            dict(
                budget=budget,
                safe_pair_products=len(safe["high"]) * len(safe["low"]),
                disjoint_safe_pairs=disjoint_n,
                admissible_pairs=allowed_n,
                **{f"{s}_{k}": len(safe[s]) if k == "safe" else comb(len(reserve), budget) for s in SIDES for k in ("states", "safe")},
            )
        )
    compare(actual, pd.DataFrame(rows, columns=PAIR_COLUMNS), ["high_state", "low_state"])
    best = None
    if allowed:
        minimum = min(r["row"]["objective"] for r in allowed)
        best = min((r for r in allowed if r["row"]["objective"] <= minimum + 1e-12), key=lambda r: (r["row"]["budget"], tuple(r["added"]["high"]), tuple(r["added"]["low"])))
    return j0, rows, coverage, best


def worker(args):
    source, result, output, session, frozen = args
    stem = session.replace("/", "_")
    folder = source / stem
    enc = prior.proof.load(folder / "encoding.npz")
    original = {s: sorted(map(int, enc[f"{s}_indices"])) for s in SIDES}
    require(len(original["high"]) == len(original["low"]), "unequal originals")
    reserve = sorted(set(range(len(enc["early_run"]))) - set(original["high"]) - set(original["low"]))
    c = frozen["choices"][session]
    require(c == json.loads((result / f"{stem}_choice.json").read_text()), "individual freeze mismatch")
    require(c["session"] == session and c["original"] == original and c["reserve"] == reserve, "changed population")
    banks = {p: {} for p in ("train", "validation")}
    for name in ("encoding",) + SOURCES:
        path = folder / f"{name}.npz"
        require(frozen["input_file_sha256"].get(str(path)) == file_sha256(path), "unregistered source")
    for name in SOURCES:
        bank = prior.proof.load(folder / f"{name}.npz")
        split = frozen["splits"][session][name]
        prior.proof.check_partition(bank, split, session, name)
        for phase in banks:
            banks[phase][name] = {k: bank[k][np.array(split[phase], int)] for k in ("counts", "truth_cm", "labels")}
    paths = {k: result / f"{stem}_{k}.csv" for k in ("side_states", "safe_pair_candidates")}
    require(c["tables_sha256"] == {k: file_sha256(p) for k, p in paths.items()}, "unlinked tables")
    baseline, states, certificates, checked = states_for(enc, banks["train"], original, reserve, pd.read_csv(paths["side_states"]))
    for s in SIDES:
        close(baseline[s][0], c["baseline_by_side"][s]["risks"])
        close(baseline[s][1], c["baseline_by_side"][s]["native_means"])
    j0, pairs, coverage, best = pairs_for(enc, banks["train"], original, reserve, states, baseline, pd.read_csv(paths["safe_pair_candidates"]))
    close(j0, c["baseline_j"])
    require(c["coverage"] == coverage, "incorrect coverage")
    if best is None:
        name, pair, added, budget, j = None, original, dict(high=[], low=[]), 0, j0
        require(c["chosen_pair"] is None, "non-improving choice")
    else:
        row, pair, added = best["row"], best["pair"], best["added"]
        name, budget, j = f"high_{row['high_state']}_low_{row['low_state']}", row["budget"], row["objective"]
        for k in PAIR_COLUMNS:
            if k == "objective":
                close(c["chosen_pair"][k], row[k])
            else:
                require(c["chosen_pair"][k] == row[k], "winner mismatch")
        require(prior.add(original, reserve, budget, added) == pair, "invalid additions")
    require(c["chosen"] == name and c["final_pair"] == pair and c["added"] == added and c["budget"] == budget, "not training winner")
    close(c["final_j"], j)
    validation, native_change = [], None
    for source_name in SOURCES:
        bank = banks["validation"][source_name]
        rb, jb = prior.proof.measure(enc, bank, original)
        ra, ja = prior.proof.measure(enc, bank, pair)
        if source_name == "run_q3":
            native_change = ja - jb
        for i in range(8):
            k = (i % 4) // 2
            validation.append(
                dict(
                    session=session,
                    animal=session.split("/")[0],
                    source=source_name,
                    side=SIDES[i // 4],
                    true_home=k,
                    metric="error_cm" if i % 2 == 0 else "home_brier",
                    observations=int(sum(bank["labels"] == k)),
                    baseline=rb[i],
                    targeted=ra[i],
                    change=ra[i] - rb[i],
                    nonworsening=bool(ra[i] <= rb[i] + 1e-10),
                    baseline_j=jb,
                    targeted_j=ja,
                )
            )
    pd.DataFrame(certificates).to_csv(output / f"{stem}_certificate_reconstruction.csv", index=False)
    pd.DataFrame(pairs, columns=PAIR_COLUMNS).to_csv(output / f"{stem}_pair_reconstruction.csv", index=False)
    print("screened reserve independently verified", session, len(states), checked, len(pairs), flush=True)
    return dict(
        summary=dict(
            session=session,
            animal=session.split("/")[0],
            population_subsets=len(states),
            admissible_pairs=sum(r["admissible_pairs"] for r in coverage),
            chosen=name,
            budget=budget,
            baseline_j=j0,
            targeted_j=j,
        ),
        coverage=[dict(session=session, animal=session.split("/")[0], **r) for r in coverage],
        validation=validation,
        native_change=native_change,
        checked_risks=checked,
        safe_subsets=sum(r["safe"] for r in states),
        safe_products=len(pairs),
        full_pairs=sum(r["disjoint"] for r in pairs),
    )


def audit(source, result, output, workers=4):
    manifest_path = result / "manifest.json"
    manifest_hash = file_sha256(manifest_path)
    m = json.loads(manifest_path.read_text())
    names = (
        "audit_exact_small_reserve_content",
        "audit_guarded_reserve_content",
        "audit_robust_exchange_content",
        "audit_exchange_content",
        "audit_reserve_cell_content",
        "audit_joint_exchange_truth",
        "_provenance",
    )
    own = {str(p): file_sha256(p) for p in [Path(__file__)] + [ROOT / "scripts" / f"{n}.py" for n in names]}

    def unchanged():
        for path, sha in {**m["input_file_sha256"], **own}.items():
            require(sha is not None and file_sha256(path) == sha, "changed input")
        for path, sha in m["output_sha256"].items():
            require(sha is not None and file_sha256(result / path) == sha, "changed output")
        require(file_sha256(manifest_path) == manifest_hash, "changed manifest")

    unchanged()
    ref, ref_audit = Path(m["reference_dir"]), Path(m["reference_audit"])
    reference = prior.checked_manifest(ref, ref_audit)
    require(all(m["input_file_sha256"].get(p) == sha for p, sha in reference["input_file_sha256"].items()), "reference source mismatch")
    for p in (ref / "manifest.json", ref / "frozen_assignments.json", ref_audit):
        require(m["input_file_sha256"].get(str(p)) == file_sha256(p), "unlinked reference")
    frozen_path = result / "frozen_assignments.json"
    frozen = json.loads(frozen_path.read_text())
    before = json.loads((result / "pre_selection.json").read_text())
    require(file_sha256(frozen_path) == m["frozen_assignments_sha256"], "changed freeze")
    require(before["input_file_sha256"] == frozen["input_file_sha256"] == m["input_file_sha256"], "input freeze mismatch")
    require(before["splits"] == frozen["splits"] == json.loads((ref / "frozen_assignments.json").read_text())["splits"], "partition changed")
    times = [datetime.fromisoformat(t) for t in (before["created_at_utc"], frozen["created_at_utc"], m["validation_started_at_utc"], m["created_at_utc"])]
    require(times == sorted(times) and times[1] < times[2], "validation before freeze")
    require(set(frozen["choices"]) == set(frozen["splits"]) == set(SESSIONS), "missing original cohort")
    require(
        m["budgets"] == list(BUDGETS) and m["stages"] == [list(s) for s in STAGES] and m["selection"] == "exhaustive_complete_subsets_with_first_failed_stage_certificate",
        "changed rule",
    )
    require(all(m[k] is False for k in ("q4_scored", "test_banks_scored", "replay_scored", "external_validation", "validated_remedy")), "unsupported downstream claim")
    output.mkdir(parents=True, exist_ok=False)
    jobs = [(source, result, output, session, frozen) for session in SESSIONS]
    if workers == 1:
        results = [worker(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(worker, jobs))
    summaries = pd.DataFrame([v["summary"] for v in results])
    coverage = pd.DataFrame([r for v in results for r in v["coverage"]])
    validation = pd.DataFrame([r for v in results for r in v["validation"]])
    compare(pd.read_csv(result / "selection_summary.csv"), summaries, ["session"])
    compare(pd.read_csv(result / "exhaustive_coverage.csv"), coverage, ["session", "budget"])
    compare(pd.read_csv(result / "internal_validation.csv"), validation, ["session", "source", "side", "true_home", "metric"])
    changes = {s: v["native_change"] for s, v in zip(SESSIONS, results, strict=True)}
    rats = {s.split("/")[0] for s in SESSIONS}
    gates = dict(
        all_original_pairs_present=True,
        all_validation_risks_present=len(validation) == 96,
        internal_accuracy_nonworsening=bool(validation.nonworsening.all()),
        all_pairs_augmented=bool(summaries.budget.gt(0).all()),
        native_validation_improves_all_rats=all(np.mean([v for s, v in changes.items() if s.split("/")[0] == r]) < -1e-10 for r in rats),
    )
    gates["ready_for_truth_preflight"] = all(gates.values())
    actual = pd.read_csv(result / "gates.csv")
    require(not actual.gate.duplicated().any() and actual.set_index("gate").passed.to_dict() == gates, "gate mismatch")
    unchanged()
    coverage.to_csv(output / "exhaustive_coverage.csv", index=False)
    validation.to_csv(output / "validation_reconstruction.csv", index=False)
    record = {
        **build_script_provenance(),
        "status": "pass",
        "manifest_sha256": manifest_hash,
        "population_subsets_verified": int(summaries.population_subsets.sum()),
        "checked_risks_reconstructed": sum(v["checked_risks"] for v in results),
        "safe_subsets_full_poisson_reconstructed": sum(v["safe_subsets"] for v in results),
        "safe_pair_cross_products_verified": sum(v["safe_products"] for v in results),
        "disjoint_safe_pairs_full_likelihood_reconstructed": sum(v["full_pairs"] for v in results),
        "validation_risks_reconstructed": len(validation),
        "exhaustive_budgets": list(BUDGETS),
        "workers": workers,
        "scope": "every three/four-cell subset, every computed stage and every first-failure certificate; full uncomputed risks of rejected subsets are not reconstructed; every safe subset and disjoint safe pair is independently fully scored",
        "ready_for_truth_preflight": gates["ready_for_truth_preflight"],
        "validated_remedy": False,
        "external_validation": False,
        "input_file_sha256": {str(manifest_path): manifest_hash, **own},
        "output_sha256": {p.name: file_sha256(p) for p in output.iterdir()},
    }
    (output / "independent_audit.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ("source-dir", "result-dir", "output-dir"):
        p.add_argument(f"--{flag}", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4, choices=range(1, 5))
    a = p.parse_args()
    audit(a.source_dir, a.result_dir, a.output_dir, a.workers)
