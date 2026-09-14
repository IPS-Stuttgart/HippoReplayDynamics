#!/usr/bin/env python3
"""Rebuild LP constraints and certify optimality without importing the solver."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

SESSIONS = ("Rat1/Open1", "Rat1/Open2", "Rat2/Open1", "Rat4/Open2")
REAL = ("all_fixed_candidates", "full_accepted_segment")
TRUTH = ("run_q4", "test_poisson_gain1", "test_poisson_gain4", "test_conditional", "test_conditional_map_drift", "test_conditional_shared_assembly")
LOSSES = ("high_error", "low_error", "high_brier", "low_brier")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while data := f.read(1048576):
            h.update(data)
    return h.hexdigest()


def load(path):
    with np.load(path, allow_pickle=False) as f:
        return dict(f)


def rebuild(frame, counts, coverage, oracle, guard):
    tuples = list(map(tuple, counts.tolist()))
    ordering = {key: index for index, key in enumerate(sorted(set(tuples)))}
    ids = np.arange(len(frame)) if oracle == "free_event" else np.array([ordering[k] for k in tuples])
    g, n = int(ids.max()) + 1, len(frame)
    labels = frame.true_home.to_numpy()
    known = np.isfinite(labels).all()
    A, b, E, f = [], [], [], []

    def coefficients(values, slope):
        result = np.zeros(g + 1)
        np.add.at(result, ids, values)
        result[-1] = slope
        return result

    delta = frame.high_home.to_numpy() - frame.low_home.to_numpy()
    base_gap = abs(delta.sum() / n)
    norm = max(base_gap, 0.05)
    for sign in (1, -1):
        A.append(coefficients(sign * delta / n / norm, coverage * base_gap * 0.2 / norm))
        b.append(coverage * base_gap / norm)
    for name, fraction in (("separation", 0.1), ("regional_tv", 0.1), ("high_entropy", 0), ("low_entropy", 0)):
        val = frame[name].to_numpy(float)
        average = sum(val) / n
        norm = max(average, 1e-12)
        A.append(coefficients(val / n / norm, coverage * fraction * average / norm))
        b.append(coverage * average / norm)
    if known:
        weights = np.zeros(n)
        for label in (0, 1):
            ix = labels == label
            assert ix.any()
            v = ix.astype(float) / ix.sum()
            weights += v / 2
            E.append(coefficients(v, 0))
            f.append(coverage)
        if guard == "truth_guarded":
            for name in LOSSES:
                val = frame[name].to_numpy(float) * weights
                norm = max(float(val.sum()), 1e-12)
                A.append(coefficients(val / norm, 0))
                b.append(coverage * val.sum() / norm)
    else:
        assert guard == "agreement_only"
        E.append(coefficients(np.ones(n) / n, 0))
        f.append(coverage)
    return dict(
        A=np.array(A), b=np.array(b), E=np.array(E), f=np.array(f), c=np.r_[np.zeros(g), -1.0], lo=np.zeros(g + 1), hi=np.r_[np.ones(g), 5.0], group=ids, unique=len(ordering)
    )


def verify_certificate(problem, certificate):
    x = certificate["x"]
    y, v, lo, hi = (certificate[name] for name in ("inequality_dual", "equality_dual", "lower_dual", "upper_dual"))
    A, b, E, f, c = (problem[name] for name in ("A", "b", "E", "f", "c"))
    assert all(np.isfinite(z).all() for z in (x, y, v, lo, hi))
    np.testing.assert_array_equal(certificate["group"], problem["group"])
    primal = max(float(np.max(A @ x - b)), float(np.max(abs(E @ x - f))), float(np.max(problem["lo"] - x)), float(np.max(x - problem["hi"])), 0)
    signs = max(float(y.max()), float((-lo).max()), float(hi.max()), 0)
    stationarity = float(np.max(abs(c - A.T @ y - E.T @ v - lo - hi)))
    dual_value = b @ y + f @ v + problem["lo"] @ lo + problem["hi"] @ hi
    gap = abs(float(c @ x - dual_value))
    complementarity = max(float(np.max(abs(y * (b - A @ x)))), float(np.max(abs(lo * (x - problem["lo"])))), float(np.max(abs(hi * (problem["hi"] - x)))))
    assert primal <= 1e-7, ("primal", primal)
    assert signs <= 1e-7, ("dual signs", signs)
    assert stationarity <= 1e-6, ("stationarity", stationarity)
    assert gap <= 1e-6, ("duality gap", gap)
    assert complementarity <= 1e-6, ("complementarity", complementarity)
    return dict(primal_residual=primal, dual_sign_residual=signs, stationarity_residual=stationarity, duality_gap=gap, complementarity_residual=complementarity)


def summarize(frame, weight):
    result = {}
    for name in ("high_home", "low_home", "separation", "regional_tv", "high_entropy", "low_entropy"):
        result[name] = float(np.average(frame[name], weights=weight))
    result["home_gap"] = abs(result["high_home"] - result["low_home"])
    for k in (0, 1):
        ix = frame.true_home.to_numpy() == k
        result[f"class{k}_retention"] = weight[ix].sum() / ix.sum() if ix.any() else np.nan
    for name in LOSSES:
        pieces = []
        for k in (0, 1):
            ix = frame.true_home.to_numpy() == k
            pieces.append(float(np.average(frame.loc[ix, name], weights=weight[ix])) if weight[ix].sum() > 0 else np.nan)
        result[f"balanced_{name}"] = np.mean(pieces)
    return result


def audit(root, output):
    manifest = json.loads((root / "manifest.json").read_text())
    for path, value in manifest["input_file_sha256"].items():
        assert sha(path) == value, path
    for path, value in manifest["output_sha256"].items():
        assert sha(root / path) == value, path
    freeze = json.loads((root / "pre_solve.json").read_text())
    assert freeze["input_file_sha256"] == manifest["input_file_sha256"] and freeze["created_at_utc"] < manifest["created_at_utc"]
    table = pd.read_csv(root / "bounds.csv")
    keys = ["session", "source", "encoding", "coverage", "oracle", "guard"]
    expected = {
        (s, src, enc, q, o, guard)
        for s in SESSIONS
        for src in REAL + TRUTH
        for enc in (("early_run", "full_run") if src in REAL else ("early_run",))
        for q in (0.25, 0.5, 0.75)
        for o in ("free_event", "count_pattern")
        for guard in (("agreement_only", "truth_guarded") if src in TRUTH else ("agreement_only",))
    }
    assert len(table) == len(expected) and set(table[keys].itertuples(index=False, name=None)) == expected
    cache, residuals = {}, []
    for row in table.itertuples(index=False):
        key = (row.session, row.source, row.encoding)
        if key not in cache:
            slug = row.session.replace("/", "_")
            folder = Path(manifest["source_dir"]) / slug
            enc = load(folder / "encoding.npz")
            bank = load(folder / f"{row.source}.npz")
            ix = sorted(set(enc["high_indices"]) | set(enc["low_indices"]))
            counts = bank["counts"][:, ix]
            events = pd.read_csv(Path(manifest["source_result_dir"]) / f"{slug}_events.csv.gz", dtype={"event_id": str})
            frame = (
                events[(events.method == "baseline") & (events.source == row.source) & (events.encoding == row.encoding)].sort_values("observation_index").reset_index(drop=True)
            )
            np.testing.assert_array_equal(frame.observation_index, np.arange(len(counts)))
            np.testing.assert_array_equal(frame.event_id, bank.get("event_ids", np.arange(len(counts))).astype(str))
            cache[key] = (frame, counts)
        frame, counts = cache[key]
        problem = rebuild(frame, counts, row.coverage, row.oracle, row.guard)
        cert = load(root / "certificates" / f"{row.case_id}.npz")
        status = json.loads((root / "certificates" / f"{row.case_id}.json").read_text())
        assert status["solver_status"] == row.solver_status == 0
        assert status["message"] == row.message
        checked = verify_certificate(problem, cert)
        w, t = cert["x"][problem["group"]], float(cert["x"][-1])
        base, selected = summarize(frame, np.ones(len(frame))), summarize(frame, w)
        np.testing.assert_allclose([row.max_progress, row.retained_weight], [t, w.sum()], atol=1e-8)
        assert row.unique_patterns == problem["unique"] and row.observations == len(frame) and row.retention_variables == len(cert["x"]) - 1
        assert row.fractional_probabilities == np.sum((cert["x"][:-1] > 1e-8) & (cert["x"][:-1] < 1 - 1e-8))
        applicable = all(base[k] > 1e-12 for k in ("home_gap", "separation", "regional_tv"))
        assert bool(row.baseline_target_nonzero) == applicable and bool(row.targets_attainable) == (applicable and t >= 1 - 1e-7)
        for prefix, values in (("baseline", base), ("retained", selected)):
            for name, value in values.items():
                np.testing.assert_allclose(getattr(row, f"{prefix}_{name}"), value, atol=1e-8, rtol=1e-9)
        residuals.append(dict(case_id=row.case_id, **checked))
    residuals = pd.DataFrame(residuals)
    result = dict(
        status="pass",
        manifest_sha256=sha(root / "manifest.json"),
        cases=len(table),
        maximum_certificate_residuals=residuals.drop(columns="case_id").max().to_dict(),
        scope="independent linear constraints, count equivalence, full primal-dual certificates, original observation identities and all reported metrics; no usable predictor or external validation",
        audit_script_sha256=sha(__file__),
    )
    output.write_text(json.dumps(result, indent=2) + "\n")
    residuals.to_csv(output.with_suffix(".csv"), index=False)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.result_dir, args.output)
