#!/usr/bin/env python3
"""RUN-only exchange of exclusive cells with invariant population overlap."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np
import pandas as pd
from scipy.special import softmax

from scripts import reserve_cell_content as base
from scripts import local_content_screen as regional
from scripts._provenance import build_script_provenance, file_sha256


def likelihood(counts, rates, indices):
    return counts[:, indices] @ np.log(rates[indices]) - 0.02 * rates[indices].sum(axis=0)


def readout(ll, grid, near, truth, labels):
    probabilities, homes, risks = [], [], []
    for side in base.SIDES:
        p = softmax(ll[side], axis=1)
        home = p[:, near].sum(axis=1)
        error = np.linalg.norm(p @ grid - truth, axis=1)
        brier = (home - labels) ** 2
        risks.extend(value[labels == k].mean() for k in (0, 1) for value in (error, brier))
        probabilities.append(p)
        homes.append(home)
    delta = np.array([(homes[0] - homes[1])[labels == k].mean() for k in (0, 1)])
    return dict(probabilities=probabilities, homes=homes, risks=np.array(risks), delta=delta, objective=float(np.mean(delta**2)))


def derivative(state, counts, rates, near, labels):
    coefficient = np.array([state["delta"][int(k)] / np.sum(labels == k) for k in labels])
    gradient = np.zeros(len(rates))
    for p, home in zip(state["probabilities"], state["homes"], strict=True):
        h = coefficient[:, None] * p * (near[None, :] - home[:, None])
        gradient += np.sum((h.T @ counts) * np.log(rates).T, axis=0) - 0.02 * (h.sum(axis=0) @ rates.T)
    return gradient


def swap(pair, high, low):
    return dict(high=sorted(set(pair["high"]) - {high} | {low}), low=sorted(set(pair["low"]) - {low} | {high}))


def random_pair(original, net, session, draw):
    seed = int.from_bytes(hashlib.sha256(f"20260915|exchange|{session}|{draw}".encode()).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    high = rng.choice(sorted(set(original["high"]) - set(original["low"])), net, replace=False)
    low = rng.choice(sorted(set(original["low"]) - set(original["high"])), net, replace=False)
    return dict(high=sorted(set(original["high"]) - set(high) | set(low)), low=sorted(set(original["low"]) - set(low) | set(high)))


def validate(enc, assignment):
    base.reserve_indices(enc)
    original = {s: set(map(int, enc[f"{s}_indices"])) for s in base.SIDES}
    union = original["high"] | original["low"]
    shared = original["high"] & original["low"]
    if set(assignment["methods"]) != set(base.METHODS):
        raise ValueError("required assignments missing")
    for name, pair in assignment["methods"].items():
        a, b = (set(map(int, pair[s])) for s in base.SIDES)
        if any(len(pair[s]) != len(set(pair[s])) or len(pair[s]) != len(original[s]) for s in base.SIDES):
            raise ValueError("population count or uniqueness changed")
        if a | b != union or a & b != shared:
            raise ValueError("population union or shared observations changed")
        expected = 0 if name == "baseline" else assignment["net_exchanged"]
        if len(original["high"] - a) != expected or len(original["low"] - b) != expected:
            raise ValueError("net exchange budget differs")


def select_cells(enc, q3, session):
    base.reserve_indices(enc)
    rows = base.calibration_rows(q3)
    counts, rates = q3["counts"][rows], enc["early_run"]
    truth, labels, grid, near = q3["truth_cm"][rows], q3["labels"][rows].astype(int), enc["grid_cm"], enc["near"]
    if not np.isfinite(truth).all() or np.any(rates <= 0) or not np.isfinite(rates).all() or np.any(counts < 0) or not np.isfinite(counts).all():
        raise ValueError("invalid calibration observations")
    original = {s: sorted(map(int, enc[f"{s}_indices"])) for s in base.SIDES}
    current = original
    ll = {s: likelihood(counts, rates, current[s]) for s in base.SIDES}
    state = readout(ll, grid, near, truth, labels)
    baseline_risks, baseline_objective = state["risks"].copy(), state["objective"]
    trace, stop = [], "accepted_step_budget"
    for step in range(10):
        gradient = derivative(state, counts, rates, near, labels)
        pairs = [(i, j) for i in sorted(set(current["high"]) - set(current["low"])) for j in sorted(set(current["low"]) - set(current["high"]))]
        pairs.sort(key=lambda ij: (round(float(gradient[ij[1]] - gradient[ij[0]]), 10), ij[0], ij[1]))
        proposals = []
        for i, j in pairs[:32]:
            proposed = swap(current, i, j)
            candidate_ll = {s: likelihood(counts, rates, proposed[s]) for s in base.SIDES}
            result = readout(candidate_ll, grid, near, truth, labels)
            allowed = bool(result["objective"] < state["objective"] - 1e-10 and np.all(result["risks"] <= baseline_risks + 1e-10))
            proposals.append(dict(high=i, low=j, gradient=float(gradient[j] - gradient[i]), objective=result["objective"], risks=result["risks"].tolist(), admissible=allowed))
        available = [p for p in proposals if p["admissible"]]
        winner = None
        if available:
            best = min(p["objective"] for p in available)
            winner = min((p for p in available if p["objective"] - best <= 1e-12), key=lambda p: (p["high"], p["low"]))
        trace.append(dict(step=step, before=state["objective"], proposals=proposals, chosen=None if winner is None else [winner["high"], winner["low"]]))
        if winner is None:
            stop = "no_admissible_proposal"
            break
        current = swap(current, winner["high"], winner["low"])
        ll = {s: likelihood(counts, rates, current[s]) for s in base.SIDES}
        state = readout(ll, grid, near, truth, labels)
    net = len(set(original["high"]) - set(current["high"]))
    methods = {"baseline": original, "targeted": current}
    methods.update({f"random_{j:02}": random_pair(original, net, session, j) for j in range(20)})
    result = dict(
        session=session,
        calibration_rows=rows.tolist(),
        methods=methods,
        net_exchanged=net,
        accepted_steps=sum(t["chosen"] is not None for t in trace),
        stop_reason=stop,
        trace=trace,
        baseline_objective=baseline_objective,
        final_objective=state["objective"],
        baseline_risks=baseline_risks.tolist(),
        final_risks=state["risks"].tolist(),
    )
    validate(enc, result)
    return result


def assigned_encoding(enc, pair):
    return {**enc, **{f"{side}_indices": np.array(pair[side], int) for side in base.SIDES}}


def class_rows(values, session, source, method):
    rows = []
    for label in (0, 1):
        ix = values["true_home"] == label
        row = dict(animal=session.split("/")[0], session=session, source=source, method=method, true_home=label, observations=int(ix.sum()))
        row.update({k: float(np.mean(values[k][ix])) for k in regional.CLASS_METRICS})
        rows.append(row)
    return rows


def gates(draws, animal, summary, classes):
    flags = base.gates(draws, animal, summary)
    flags = flags[flags.gate.ne("development_numerical_screen")].copy()
    primary = summary[summary.source.eq(base.REAL[0]) & summary.encoding.eq("early_run")].set_index("method")
    for metric in ("home_gap", "separation", "regional_tv"):
        # Identical zero-budget controls must not win on rounding of draw means.
        key = f"beats_equal_budget_random_{metric}"
        flags.loc[flags.gate.eq(key), "passed"] &= primary.loc["targeted", metric] < primary.loc["random_mean", metric] - 1e-10
    table = classes[classes.method.isin(("baseline", "targeted"))].copy()
    table["method"] = table.method.replace({"baseline": "all", "targeted": "local_half"})
    extra = regional.regional_flags(table)
    flags = pd.concat([flags, pd.DataFrame([dict(gate=k, passed=v) for k, v in extra.items()])], ignore_index=True)
    flags.loc[len(flags)] = ["development_numerical_screen", bool(flags.passed.all())]
    return flags


def measure(args):
    inputs = regional.checked_source(args.result_dir, args.audit)
    previous = json.loads((args.result_dir / "manifest.json").read_text())
    source_dir = Path(previous["source_dir"])
    for path in (
        Path(__file__),
        ROOT / "scripts/reserve_cell_content.py",
        ROOT / "scripts/spatial_predictive_content.py",
        ROOT / "scripts/local_content_screen.py",
        ROOT / "docs/exchange_content_protocol.md",
    ):
        inputs[str(path)] = file_sha256(path)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    assignments = {}
    for session in base.SESSIONS:
        folder = source_dir / session.replace("/", "_")
        assignments[session] = select_cells(base.read_npz(folder / "encoding.npz"), base.read_npz(folder / "run_q3.npz"), session)
        print("selected", session, assignments[session]["net_exchanged"], flush=True)
    frozen = {**build_script_provenance(), "assignments": assignments, "input_file_sha256": inputs, "created_at_utc": datetime.now(timezone.utc).isoformat()}
    (args.output_dir / "pre_scoring.json").write_text(json.dumps(frozen, indent=2) + "\n")
    freeze_hash = file_sha256(args.output_dir / "pre_scoring.json")
    scoring_started = datetime.now(timezone.utc).isoformat()
    summaries, classes, count = [], [], 0
    for session in base.SESSIONS:
        folder = source_dir / session.replace("/", "_")
        enc = base.read_npz(folder / "encoding.npz")
        primary = []
        for source in base.REAL + base.TRUTH:
            bank = base.read_npz(folder / f"{source}.npz")
            for encoding in ("early_run", "full_run") if source in base.REAL else ("early_run",):
                for method, pair in assignments[session]["methods"].items():
                    values = base.score_pair(bank, assigned_encoding(enc, pair), encoding, dict(high=[], low=[]))
                    if method == "baseline":
                        np.testing.assert_allclose(np.column_stack([values[f"{s}_home"] for s in base.SIDES]), bank[f"{encoding}_scores"], atol=1e-9, rtol=1e-9)
                    summaries.append(base.summarize_pair(values, session, source, encoding, method))
                    if source in base.TRUTH:
                        classes.extend(class_rows(values, session, source, method))
                    if method in ("baseline", "targeted"):
                        primary.append(
                            pd.DataFrame(values).assign(
                                observation_index=np.arange(len(bank["counts"])),
                                event_id=bank.get("event_ids", np.arange(len(bank["counts"]))).astype(str),
                                session=session,
                                source=source,
                                encoding=encoding,
                                method=method,
                            )
                        )
                    count += len(bank["counts"])
        pd.concat(primary, ignore_index=True).to_csv(args.output_dir / f"{session.replace('/', '_')}_events.csv.gz", index=False)
        print("evaluated", session, flush=True)
    draws, classes = pd.DataFrame(summaries), pd.DataFrame(classes)
    combined, animal, summary = base.aggregate(draws)
    for name, table in dict(
        session_draws=draws, session_summary=combined, animal_summary=animal, summary=summary, truth_by_class=classes, gates=gates(draws, animal, summary, classes)
    ).items():
        table.to_csv(args.output_dir / f"{name}.csv", index=False)
    for path, value in inputs.items():
        if file_sha256(path) != value:
            raise ValueError(f"input changed: {path}")
    if file_sha256(args.output_dir / "pre_scoring.json") != freeze_hash:
        raise ValueError("assignments changed")
    manifest = {
        **build_script_provenance(),
        "source_dir": str(source_dir),
        "source_result_dir": str(args.result_dir),
        "source_audit": str(args.audit),
        "input_file_sha256": inputs,
        "assignments_sha256": freeze_hash,
        "scoring_started_at_utc": scoring_started,
        "evaluated_event_method_rows": count,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "external_validation": False,
        "validated_remedy": False,
        "output_sha256": {p.name: file_sha256(p) for p in args.output_dir.iterdir() if p.is_file()},
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("result-dir", "audit", "output-dir"):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    measure(parser.parse_args())
