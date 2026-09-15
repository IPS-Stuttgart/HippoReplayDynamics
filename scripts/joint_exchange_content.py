#!/usr/bin/env python3
"""RUN-only joint exchange proposals with exact, independently checked risks."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csc_matrix, vstack

from scripts import exchange_content as exact
from scripts import audit_exchange_content as proof
from scripts import audit_exchange_search_coverage as coverage
from scripts._provenance import build_script_provenance, file_sha256
from scripts.report_content_screening_bound import markdown

BUDGETS = (2, 4, 8)
PROPOSALS_PER_BUDGET = 4
OPTIONS = dict(node_limit=64, time_limit=20.0, mip_rel_gap=0.001)


def joint_pair(original, high, low):
    a, b = set(high), set(low)
    if len(a) != len(high) or len(b) != len(low) or len(a) != len(b):
        raise ValueError("invalid exchanged counts")
    high_set, low_set = set(original["high"]), set(original["low"])
    if not a <= high_set - low_set or not b <= low_set - high_set:
        raise ValueError("shared, foreign or wrong-side cell exchanged")
    result = dict(high=sorted(high_set - a | b), low=sorted(low_set - b | a))
    assert set(result["high"]) | set(result["low"]) == high_set | low_set
    assert set(result["high"]) & set(result["low"]) == high_set & low_set
    return result


def problem(table, original_risks, original_j, budget, previous):
    high, low = table.high.to_numpy(int), table.low.to_numpy(int)
    risk_scale = np.maximum(np.abs(original_risks), 1e-6)
    risk = np.array([table[f"change_{r}"].to_numpy() for r in coverage.RISK_NAMES]) / risk_scale[:, None]
    d_j = -table.objective_improvement.to_numpy() / max(original_j, 1e-6)
    rows = [csc_matrix(np.ones((1, len(table))))]
    lower, upper = [float(budget)], [float(budget)]
    for endpoint in (high, low):
        unique = np.unique(endpoint)
        rows.append(csc_matrix(np.array([endpoint == v for v in unique], float)))
        lower.extend([-np.inf] * len(unique))
        upper.extend([1.0] * len(unique))
    rows.extend([csc_matrix(risk), csc_matrix(d_j[None, :])])
    lower.extend([-np.inf] * 8 + [-original_j / max(original_j, 1e-6)])
    upper.extend([0.0] * 8 + [-1e-10 / max(original_j, 1e-6)])
    for h, lo in previous:
        coefficients = np.isin(high, h).astype(int) + np.isin(low, lo).astype(int)
        rows.append(csc_matrix(coefficients[None, :]))
        lower.append(-np.inf)
        upper.append(float(2 * budget - 1))
    return d_j, vstack(rows, format="csc"), np.array(lower), np.array(upper)


def feasible_incumbent(result, a, lower, upper):
    if result.status not in (0, 1) or result.x is None or not np.isfinite(result.x).all():
        return None
    x = np.rint(result.x)
    if np.max(np.abs(x - result.x)) > 1e-7 or np.any((x < 0) | (x > 1)):
        return None
    value = a @ x
    if np.any(value < lower - 1e-7) or np.any(value > upper + 1e-7):
        return None
    return x.astype(int)


def propose(table, risks, objective):
    table = table.sort_values(["high", "low"]).reset_index(drop=True)
    records = []
    for budget in BUDGETS:
        if budget > min(table.high.nunique(), table.low.nunique()):
            records.append(dict(budget=budget, proposal_index=0, status="budget_unavailable", high=[], low=[], edges=[]))
            continue
        previous = []
        for iteration in range(PROPOSALS_PER_BUDGET):
            c, a, lower, upper = problem(table, risks, objective, budget, previous)
            start = time.monotonic()
            result = milp(c, integrality=np.ones(len(c)), bounds=Bounds(0, 1), constraints=LinearConstraint(a, lower, upper), options=OPTIONS)
            x = feasible_incumbent(result, a, lower, upper)
            record = dict(
                budget=budget,
                proposal_index=iteration,
                status=int(result.status),
                message=str(result.message),
                runtime_s=time.monotonic() - start,
                high=[],
                low=[],
                edges=[],
                feasible_incumbent=x is not None,
            )
            for key in ("mip_gap", "mip_dual_bound", "mip_node_count", "fun"):
                v = getattr(result, key, None)
                record[key] = None if v is None or not np.isfinite(v) else float(v)
            if x is None:
                records.append(record)
                break
            selected = table.loc[x.astype(bool)]
            h, lo = sorted(selected.high.astype(int)), sorted(selected.low.astype(int))
            record.update(
                high=h,
                low=lo,
                edges=selected[["high", "low"]].astype(int).values.tolist(),
                approximate_j=float(objective + np.dot(-table.objective_improvement, x)),
                approximate_risks=(risks + np.array([table[f"change_{r}"].to_numpy() @ x for r in coverage.RISK_NAMES])).tolist(),
            )
            records.append(record)
            previous.append((h, lo))
    return records


def exact_candidate(enc, bank, original, high, low, initial, name):
    pair = joint_pair(original, high, low)
    ll = {s: exact.likelihood(bank["counts"], enc["early_run"], pair[s]) for s in original}
    state = exact.readout(ll, enc["grid_cm"], enc["near"], bank["truth_cm"], bank["labels"])
    allowed = bool(state["objective"] < initial["objective"] - 1e-10 and np.all(state["risks"] <= initial["risks"] + 1e-10))
    return dict(name=name, high=sorted(high), low=sorted(low), pair=pair, objective=state["objective"], risks=state["risks"].tolist(), admissible=allowed)


def winner(candidates):
    valid = [c for c in candidates if c["admissible"]]
    if not valid:
        return None
    minimum = min(c["objective"] for c in valid)
    return min((c for c in valid if c["objective"] - minimum <= 1e-12), key=lambda c: (len(c["high"]), c["high"], c["low"]))


def select(enc, q3, singles, session):
    bank, original, options = coverage.calibration(enc, q3)
    if set(zip(singles.high, singles.low, strict=True)) != set(options) or singles.duplicated(["high", "low"]).any():
        raise ValueError("incomplete or repeated single-swap source")
    ll = {s: exact.likelihood(bank["counts"], enc["early_run"], original[s]) for s in original}
    initial = exact.readout(ll, enc["grid_cm"], enc["near"], bank["truth_cm"], bank["labels"])
    records = propose(singles, initial["risks"], initial["objective"])
    candidates = []
    for row in singles.loc[singles.admissible].sort_values(["high", "low"]).itertuples():
        candidates.append(exact_candidate(enc, bank, original, [row.high], [row.low], initial, f"single_{row.high}_{row.low}"))
    for r in records:
        if r.get("feasible_incumbent"):
            candidates.append(exact_candidate(enc, bank, original, r["high"], r["low"], initial, f"joint_{r['budget']}_{r['proposal_index']}"))
    best = winner(candidates)
    return dict(
        session=session,
        original=original,
        baseline_j=initial["objective"],
        baseline_risks=initial["risks"].tolist(),
        proposals=records,
        candidates=candidates,
        chosen=None if best is None else best["name"],
        final_pair=original if best is None else best["pair"],
        net_exchanged=0 if best is None else len(best["high"]),
        final_j=initial["objective"] if best is None else best["objective"],
    )


def verify(enc, q3, singles, choice):
    bank, original, options = coverage.calibration(enc, q3)
    assert choice["original"] == original
    assert set(zip(singles.high, singles.low, strict=True)) == set(options) and not singles.duplicated(["high", "low"]).any()
    _, _, initial_risks, _, initial_j = proof.state_for(original, enc, bank)
    np.testing.assert_allclose(choice["baseline_risks"], initial_risks, atol=1e-9)
    np.testing.assert_allclose(choice["baseline_j"], initial_j, atol=1e-10)
    valid_names, audited = [], 0
    expected_members = {f"single_{r.high}_{r.low}": ([r.high], [r.low]) for r in singles.loc[singles.admissible].itertuples()}
    indexed = singles.set_index(["high", "low"])
    assert set(r["budget"] for r in choice["proposals"]) == set(BUDGETS)
    for budget in BUDGETS:
        group = [r for r in choice["proposals"] if r["budget"] == budget]
        assert 1 <= len(group) <= PROPOSALS_PER_BUDGET
        previous = []
        for n, r in enumerate(group):
            assert r["proposal_index"] == n
            if r["status"] == "budget_unavailable":
                assert n == 0 and len(group) == 1 and budget > min(singles.high.nunique(), singles.low.nunique())
                continue
            if not r["feasible_incumbent"]:
                assert n == len(group) - 1 and not r["edges"]
                continue
            assert r["status"] in (0, 1)
            h, lo = sorted(e[0] for e in r["edges"]), sorted(e[1] for e in r["edges"])
            assert h == r["high"] and lo == r["low"] and len(set(h)) == len(set(lo)) == budget
            joint_pair(original, h, lo)
            assert (h, lo) not in previous
            rows = indexed.loc[[tuple(e) for e in r["edges"]]]
            dj = -rows.objective_improvement.sum()
            risk_changes = np.array([rows[f"change_{key}"].sum() for key in coverage.RISK_NAMES])
            assert np.all(risk_changes / np.maximum(np.abs(initial_risks), 1e-6) <= 1e-7)
            assert -initial_j - 1e-7 * max(initial_j, 1e-6) <= dj <= -1e-10 + 1e-7 * max(initial_j, 1e-6)
            np.testing.assert_allclose(r["approximate_j"], initial_j + dj, atol=1e-10)
            np.testing.assert_allclose(r["approximate_risks"], initial_risks + risk_changes, atol=1e-9)
            np.testing.assert_allclose(r["fun"], dj / max(initial_j, 1e-6), atol=1e-7)
            previous.append((h, lo))
            expected_members[f"joint_{budget}_{n}"] = (h, lo)
    assert {c["name"] for c in choice["candidates"]} == set(expected_members) and len(choice["candidates"]) == len(expected_members)
    for c in choice["candidates"]:
        assert (c["high"], c["low"]) == expected_members[c["name"]]
        high_members, low_members = list(original["high"]), list(original["low"])
        for i, j in zip(c["high"], c["low"], strict=True):
            high_members.remove(i)
            high_members.append(j)
            low_members.remove(j)
            low_members.append(i)
        pair = dict(high=sorted(high_members), low=sorted(low_members))
        assert c["pair"] == pair
        _, _, risks, _, value = proof.state_for(pair, enc, bank)
        np.testing.assert_allclose(c["risks"], risks, atol=1e-9, rtol=1e-9)
        np.testing.assert_allclose(c["objective"], value, atol=1e-10, rtol=1e-9)
        allowed = bool(value < initial_j - 1e-10 and (risks <= initial_risks + 1e-10).all())
        assert c["admissible"] == allowed
        if allowed:
            valid_names.append((value, len(c["high"]), c["high"], c["low"], c["name"], pair))
        audited += 1
    if valid_names:
        minimum = min(r[0] for r in valid_names)
        best = min((r for r in valid_names if r[0] - minimum <= 1e-12), key=lambda r: (r[1], r[2], r[3]))
        assert choice["chosen"] == best[4] and choice["final_pair"] == best[5] and choice["net_exchanged"] == best[1]
        np.testing.assert_allclose(choice["final_j"], best[0], atol=1e-10)
    else:
        assert choice["chosen"] is None and choice["final_pair"] == original and choice["net_exchanged"] == 0
        np.testing.assert_allclose(choice["final_j"], initial_j, atol=1e-10)
    return audited


def run(result_dir, output_dir):
    manifest_path = result_dir / "manifest.json"
    source = json.loads(manifest_path.read_text())
    audit = json.loads((result_dir / "independent_audit.json").read_text())
    if audit.get("status") != "pass" or audit.get("manifest_sha256") != file_sha256(manifest_path):
        raise ValueError("source requires matching independent reconstruction")
    for name, sha in source["output_sha256"].items():
        if file_sha256(result_dir / name) != sha:
            raise ValueError(f"single-swap artifact changed: {name}")
    for path, sha in source["input_file_sha256"].items():
        if file_sha256(path) != sha:
            raise ValueError(f"source changed: {path}")
    sources = source["input_file_sha256"]
    inputs = {str(p): file_sha256(p) for p in (manifest_path, result_dir / "independent_audit.json", Path(__file__), ROOT / "docs/joint_exchange_protocol.md")}
    inputs.update(source["input_file_sha256"])
    inputs.update({str(result_dir / name): sha for name, sha in source["output_sha256"].items()})
    output_dir.mkdir(parents=True, exist_ok=False)
    choices, summaries, candidate_rows, checks = {}, [], [], {}
    for session in exact.base.SESSIONS:
        folder = session.replace("/", "_")
        enc_paths = [Path(p) for p in sources if p.endswith(f"/{folder}/encoding.npz")]
        if len(enc_paths) != 1:
            raise ValueError("ambiguous encoding source")
        enc_path = enc_paths[0]
        q3_path = enc_path.with_name("run_q3.npz")
        assert str(q3_path) in sources
        enc, q3 = exact.base.read_npz(enc_path), exact.base.read_npz(q3_path)
        singles = pd.read_csv(result_dir / f"{folder}_proposals.csv")
        choice = select(enc, q3, singles, session)
        checks[session] = verify(enc, q3, singles, choice)
        choices[session] = choice
        valid = [c for c in choice["candidates"] if c["admissible"]]
        summaries.append(
            dict(
                session=session,
                animal=session.split("/")[0],
                proposals=len(choice["proposals"]),
                exact_joint_candidates=sum(c["name"].startswith("joint") for c in choice["candidates"]),
                admissible_joint_candidates=sum(c["name"].startswith("joint") for c in valid),
                chosen=choice["chosen"],
                net_exchanged=choice["net_exchanged"],
                baseline_j=choice["baseline_j"],
                final_j=choice["final_j"],
            )
        )
        for c in choice["candidates"]:
            row = dict(session=session, name=c["name"], cells_exchanged=len(c["high"]), objective=c["objective"], admissible=c["admissible"])
            row.update({f"risk_change_{key}": c["risks"][n] - choice["baseline_risks"][n] for n, key in enumerate(coverage.RISK_NAMES)})
            candidate_rows.append(row)
        print("joint calibration", session, "admissible", len(valid), "chosen", choice["chosen"], flush=True)
    (output_dir / "selection.json").write_text(json.dumps(choices, indent=2) + "\n")
    summary = pd.DataFrame(summaries)
    summary.to_csv(output_dir / "summary.csv", index=False)
    columns = ["session", "name", "cells_exchanged", "objective", "admissible"] + [f"risk_change_{key}" for key in coverage.RISK_NAMES]
    pd.DataFrame(candidate_rows, columns=columns).to_csv(output_dir / "proposals.csv", index=False)
    report = "# Joint exchange RUN-calibration experiment\n\n" + markdown(summary)
    report += "\n\nSelection only. No replay or held-out test observations were scored. Approximate MILP feasibility does not imply exact accuracy safety. "
    report += "Every admitted candidate passed the exact full-likelihood classwise guards. No global optimality or independent-data remedy claim. "
    report += "All original pairs remain in the experiment; unchanged pairs are not dropped.\n"
    (output_dir / "report.md").write_text(report)
    for path, sha in inputs.items():
        if file_sha256(path) != sha:
            raise ValueError(f"changed input: {path}")
    provenance = build_script_provenance()
    provenance.update(
        input_file_sha256=inputs,
        scipy_version=scipy.__version__,
        milp_options=OPTIONS,
        budgets=BUDGETS,
        proposals_per_budget=PROPOSALS_PER_BUDGET,
        replay_scored=False,
        validated_remedy=False,
        external_validation=False,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        output_sha256={p.name: file_sha256(p) for p in output_dir.iterdir() if p.is_file()},
    )
    (output_dir / "manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (output_dir / "independent_audit.json").write_text(
        json.dumps(
            dict(
                status="pass",
                manifest_sha256=file_sha256(output_dir / "manifest.json"),
                independently_reconstructed_candidates=checks,
                validated_remedy=False,
                external_validation=False,
                scope="Exact outcomes, feasible proposal memberships/constraints and selection; not proof of MILP optimality or exhaustive joint search.",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    run(args.result_dir, args.output_dir)
