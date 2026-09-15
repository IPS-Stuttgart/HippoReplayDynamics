#!/usr/bin/env python3
"""Optimistic, non-rescoring bounds; not a deployable event selector."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from scripts._provenance import build_script_provenance, file_sha256

SESSIONS = ("Rat1/Open1", "Rat1/Open2", "Rat2/Open1", "Rat4/Open2")
REAL = ("all_fixed_candidates", "full_accepted_segment")
TRUTH = ("run_q4", "test_poisson_gain1", "test_poisson_gain4", "test_conditional", "test_conditional_map_drift", "test_conditional_shared_assembly")
COVERAGES = (0.25, 0.5, 0.75)
LOSSES = ("high_error", "low_error", "high_brier", "low_brier")


def npz(path):
    with np.load(path, allow_pickle=False) as z:
        return dict(z)


def groups(counts, oracle):
    x = np.asarray(counts)
    if x.ndim != 2 or not len(x) or not x.shape[1] or not np.isfinite(x).all() or np.any(x < 0) or np.any(x != np.floor(x)):
        raise ValueError("nonempty integer spike vectors required")
    _, inverse = np.unique(x, axis=0, return_inverse=True)
    unique = int(inverse.max() + 1)
    if oracle == "free_event":
        return np.arange(len(x)), unique
    if oracle == "count_pattern":
        return inverse, unique
    raise ValueError("unknown oracle")


def build_problem(data, counts, coverage, oracle, guard):
    if not 0 < coverage < 1 or guard not in ("agreement_only", "truth_guarded"):
        raise ValueError("invalid screening case")
    if len(data) != len(counts):
        raise ValueError("count/readout mismatch")
    labels = data.true_home.to_numpy(float)
    known = np.isfinite(labels).all()
    if np.isfinite(labels).any() != known or (known and not np.isin(labels, (0, 1)).all()):
        raise ValueError("incomplete or invalid truth labels")
    if guard == "truth_guarded" and not known:
        raise ValueError("cannot impose unknown biological truth")
    required = ["high_home", "low_home", "separation", "regional_tv", "high_entropy", "low_entropy"]
    if known:
        required += list(LOSSES)
        if min(np.sum(labels == k) for k in (0, 1)) < 1:
            raise ValueError("both true regions required")
    if not np.isfinite(data[required]).all().all():
        raise ValueError("nonfinite source readout")
    group, unique = groups(counts, oracle)
    n, g = len(data), int(group.max()) + 1
    rows, upper, names, eq, rhs = [], [], [], [], []

    def compress(value):
        return np.bincount(group, weights=np.asarray(value, float), minlength=g)

    def constrain(name, values, progress, bound):
        rows.append(np.r_[compress(values), progress])
        upper.append(bound)
        names.append(name)

    d = data.high_home.to_numpy() - data.low_home.to_numpy()
    gap = abs(float(d.mean()))
    scale = max(0.05, gap)
    for sign in (1, -1):
        constrain(f"home_{sign}", sign * d / (n * scale), coverage * 0.2 * gap / scale, coverage * gap / scale)
    for key, slope in (("separation", 0.1), ("regional_tv", 0.1), ("high_entropy", 0), ("low_entropy", 0)):
        value = data[key].to_numpy(float)
        scale = max(float(value.mean()), 1e-12)
        constrain(key, value / (n * scale), coverage * slope * value.mean() / scale, coverage * value.mean() / scale)
    if known:
        balanced = sum((labels == k) / np.sum(labels == k) for k in (0, 1)) / 2
        for k in (0, 1):
            eq.append(np.r_[compress((labels == k) / np.sum(labels == k)), 0])
            rhs.append(coverage)
        if guard == "truth_guarded":
            for key in LOSSES:
                value = data[key].to_numpy(float) * balanced
                scale = max(float(value.sum()), 1e-12)
                constrain(key, value / scale, 0, coverage * value.sum() / scale)
    else:
        eq.append(np.r_[compress(np.ones(n) / n), 0])
        rhs.append(coverage)
    result = dict(c=np.r_[np.zeros(g), -1.0], A=np.array(rows), b=np.array(upper), E=np.array(eq), f=np.array(rhs), lo=np.zeros(g + 1), hi=np.r_[np.ones(g), 5.0], group=group)
    constant = np.r_[np.full(g, coverage), 0]
    if np.max(result["A"] @ constant - result["b"]) > 1e-10 or np.max(abs(result["E"] @ constant - result["f"])) > 1e-10:
        raise ValueError("constant zero-improvement screen should be feasible")
    return result, names, unique


def metrics(data, weights):
    w = np.asarray(weights, float)
    denominator = w.sum()
    if not np.isfinite(w).all() or denominator <= 0:
        raise ValueError("nonvacuous finite weights required")
    row = {k: float(np.dot(w, data[k]) / denominator) for k in ("high_home", "low_home", "separation", "regional_tv", "high_entropy", "low_entropy")}
    row["home_gap"] = abs(row["high_home"] - row["low_home"])
    for k in (0, 1):
        ix = data.true_home.eq(k).to_numpy()
        row[f"class{k}_retention"] = float(w[ix].mean()) if ix.any() else np.nan
    for key in LOSSES:
        terms = []
        for k in (0, 1):
            ix = data.true_home.eq(k).to_numpy()
            terms.append(np.dot(w[ix], data.loc[ix, key]) / w[ix].sum() if w[ix].sum() else np.nan)
        row[f"balanced_{key}"] = float(np.mean(terms))
    return row


def solve_program(problem):
    # Preserve collectively material small Brier coefficients in HiGHS's matrix.
    row_scale = 1e4
    result = linprog(
        problem["c"],
        A_ub=row_scale * problem["A"],
        b_ub=row_scale * problem["b"],
        A_eq=row_scale * problem["E"],
        b_eq=row_scale * problem["f"],
        bounds=list(zip(problem["lo"], problem["hi"], strict=True)),
        method="highs",
        options=dict(time_limit=120, primal_feasibility_tolerance=1e-9, dual_feasibility_tolerance=1e-9),
    )
    if result.status == 0:
        result.ineqlin.marginals *= row_scale
        result.eqlin.marginals *= row_scale
    return result


def solve_case(data, counts, coverage, oracle, guard, path):
    problem, names, unique = build_problem(data, counts, coverage, oracle, guard)
    result = solve_program(problem)
    status = dict(solver_status=int(result.status), message=result.message)
    path.with_suffix(".json").write_text(json.dumps(status, indent=2) + "\n")
    if result.status != 0:
        raise RuntimeError(f"uncertified LP case {path.name}: {result.message}")
    np.savez_compressed(
        path,
        x=result.x,
        inequality_dual=result.ineqlin.marginals,
        equality_dual=result.eqlin.marginals,
        lower_dual=result.lower.marginals,
        upper_dual=result.upper.marginals,
        group=problem["group"],
        inequality_names=np.array(names),
    )
    w = result.x[problem["group"]]
    row = dict(
        coverage=coverage,
        oracle=oracle,
        guard=guard,
        observations=len(data),
        unique_patterns=unique,
        retention_variables=len(result.x) - 1,
        max_progress=float(result.x[-1]),
        targets_attainable=bool(result.x[-1] >= 1 - 1e-7),
        retained_weight=float(w.sum()),
        fractional_probabilities=int(np.sum((result.x[:-1] > 1e-8) & (result.x[:-1] < 1 - 1e-8))),
        **status,
    )
    row.update({f"baseline_{k}": v for k, v in metrics(data, np.ones(len(data))).items()})
    row.update({f"retained_{k}": v for k, v in metrics(data, w).items()})
    row["baseline_target_nonzero"] = all(row[f"baseline_{k}"] > 1e-12 for k in ("home_gap", "separation", "regional_tv"))
    row["targets_attainable"] = row["targets_attainable"] and row["baseline_target_nonzero"]
    return row


def measure(args):
    source_manifest = json.loads((args.result_dir / "manifest.json").read_text())
    source_audit = json.loads(args.audit.read_text())
    if source_audit.get("status") != "pass" or source_audit["manifest_sha256"] != file_sha256(args.result_dir / "manifest.json"):
        raise ValueError("source is not bound to passing independent reconstruction")
    inputs = {str(args.result_dir / "manifest.json"): file_sha256(args.result_dir / "manifest.json"), str(args.audit): file_sha256(args.audit)}
    for key, value in source_manifest["input_file_sha256"].items():
        if file_sha256(key) != value:
            raise ValueError(f"source input changed: {key}")
        inputs[key] = value
    for name, value in source_manifest["output_sha256"].items():
        path = args.result_dir / name
        if file_sha256(path) != value:
            raise ValueError(f"source result changed: {path}")
        inputs[str(path)] = value
    for path in (Path(__file__), ROOT / "docs/content_screening_bound_protocol.md"):
        inputs[str(path)] = file_sha256(path)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "certificates").mkdir()
    freeze = {
        **build_script_provenance(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_file_sha256": inputs,
        "non_rescoring": True,
        "oracle_not_deployable": True,
    }
    (args.output_dir / "pre_solve.json").write_text(json.dumps(freeze, indent=2) + "\n")
    summary = []
    for session in SESSIONS:
        slug = session.replace("/", "_")
        enc = npz(Path(source_manifest["source_dir"]) / slug / "encoding.npz")
        columns = np.union1d(enc["high_indices"], enc["low_indices"])
        events = pd.read_csv(args.result_dir / f"{slug}_events.csv.gz", dtype={"event_id": str})
        events = events[events.method.eq("baseline")]
        for source in REAL + TRUTH:
            bank = npz(Path(source_manifest["source_dir"]) / slug / f"{source}.npz")
            counts = bank["counts"][:, columns]
            for encoding in ("early_run", "full_run") if source in REAL else ("early_run",):
                frame = events[events.source.eq(source) & events.encoding.eq(encoding)].sort_values("observation_index").reset_index(drop=True)
                np.testing.assert_array_equal(frame.observation_index, np.arange(len(counts)))
                np.testing.assert_array_equal(frame.event_id, bank.get("event_ids", np.arange(len(counts))).astype(str))
                for coverage in COVERAGES:
                    for oracle in ("free_event", "count_pattern"):
                        for guard in ("agreement_only", "truth_guarded") if source in TRUTH else ("agreement_only",):
                            key = f"{slug}__{source}__{encoding}__{coverage:g}__{oracle}__{guard}"
                            path = args.output_dir / "certificates" / f"{key}.npz"
                            row = solve_case(frame, counts, coverage, oracle, guard, path)
                            summary.append(dict(case_id=key, session=session, animal=session.split("/")[0], source=source, encoding=encoding, **row))
        print(f"solved {session}", flush=True)
    pd.DataFrame(summary).to_csv(args.output_dir / "bounds.csv", index=False)
    for path, value in inputs.items():
        if file_sha256(path) != value:
            raise ValueError(f"input changed while solving: {path}")
    manifest = {
        **build_script_provenance(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_result_dir": str(args.result_dir),
        "source_dir": source_manifest["source_dir"],
        "input_file_sha256": inputs,
        "non_rescoring": True,
        "external_validation": False,
        "oracle_not_deployable": True,
        "output_sha256": {str(p.relative_to(args.output_dir)): file_sha256(p) for p in args.output_dir.rglob("*") if p.is_file()},
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    measure(parser.parse_args())
