#!/usr/bin/env python3
"""Test frozen RUN-only acquisition of disjoint reserve cells."""

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

from scripts._provenance import build_script_provenance, file_sha256
from scripts.spatial_predictive_content import REAL, SESSIONS, TRUTH, decode, read_npz

SIDES = ("high", "low")
METHODS = ("baseline", "targeted") + tuple(f"random_{j:02}" for j in range(20))
METRICS = ("home_gap", "high_home", "low_home", "separation", "regional_tv", "high_entropy", "low_entropy") + tuple(f"balanced_{s}_{m}" for s in SIDES for m in ("error", "brier"))


def seed(session, draw):
    return int.from_bytes(hashlib.sha256(f"20260915|reserve|{session}|{draw}".encode()).digest()[:8], "little")


def reserve_indices(enc):
    n = len(enc["cell_ids"])
    if len(np.unique(enc["cell_ids"])) != n:
        raise ValueError("nonunique cell IDs")
    originals = [np.asarray(enc[f"{s}_indices"], int) for s in SIDES]
    for ix in originals:
        if ix.ndim != 1 or len(np.unique(ix)) != len(ix) or np.any(ix < 0) or np.any(ix >= n):
            raise ValueError("invalid original population")
    if len(originals[0]) != len(originals[1]):
        raise ValueError("original populations differ in size")
    return np.setdiff1d(np.arange(n), np.union1d(*originals))


def calibration_rows(bank):
    _, rows = np.unique(bank["parent_ids"], return_index=True)
    rows = np.sort(rows)
    labels = bank["labels"][rows]
    if not np.isin(labels, [0, 1]).all() or min(np.sum(labels == k) for k in (0, 1)) < 10:
        raise ValueError("fewer than ten native RUN parents per truth class")
    return rows


def balanced_loss(log_likelihood, near, labels):
    score = softmax(log_likelihood, axis=1)[:, near].sum(axis=1)
    loss = (score - labels) ** 2
    return float(np.mean([loss[labels == k].mean() for k in (0, 1)]))


def select_cells(enc, q3, session):
    reserve = reserve_indices(enc)
    quota = len(reserve) // 2
    if quota < 1:
        raise ValueError("no disjoint reserve-cell budget")
    rows = calibration_rows(q3)
    counts, rates = q3["counts"][rows], enc["early_run"]
    if counts.shape[1] != len(rates) or not np.isfinite(rates).all() or np.any(rates <= 0) or np.any(counts < 0) or not np.isfinite(counts).all():
        raise ValueError("invalid calibration observations")
    near, labels = enc["near"], q3["labels"][rows]
    if not np.any(near) or np.all(near):
        raise ValueError("both spatial regions required")
    current = {}
    for side in SIDES:
        ix = enc[f"{side}_indices"]
        current[side] = counts[:, ix] @ np.log(rates[ix]) - 0.02 * rates[ix].sum(axis=0)
    contributions = {int(i): counts[:, i, None] * np.log(rates[i]) - 0.02 * rates[i] for i in reserve}
    choices, trace = {s: [] for s in SIDES}, []
    available = set(map(int, reserve))
    for step in range(2 * quota):
        candidates = []
        for side in SIDES:
            if len(choices[side]) == quota:
                continue
            before = balanced_loss(current[side], near, labels)
            for cell in sorted(available):
                after = balanced_loss(current[side] + contributions[cell], near, labels)
                candidates.append((before - after, side, cell, before, after))
        best_gain = max(x[0] for x in candidates)
        best = next(x for x in candidates if best_gain - x[0] <= 1e-12)
        gain, side, cell, before, after = best
        choices[side].append(cell)
        available.remove(cell)
        current[side] += contributions[cell]
        trace.append(dict(step=step, side=side, cell_index=cell, before=before, after=after, gain=gain))
    methods = {"baseline": {s: [] for s in SIDES}, "targeted": choices}
    for j in range(20):
        perm = np.random.default_rng(seed(session, j)).permutation(reserve)
        methods[f"random_{j:02}"] = dict(high=perm[:quota].tolist(), low=perm[quota : 2 * quota].tolist())
    result = dict(session=session, calibration_rows=rows.tolist(), reserve=reserve.tolist(), quota=quota, methods=methods, trace=trace)
    validate_assignments(enc, result)
    return result


def validate_assignments(enc, assignment):
    reserve = set(map(int, reserve_indices(enc)))
    quota = len(reserve) // 2
    if assignment["quota"] != quota or set(assignment["reserve"]) != reserve or set(assignment["methods"]) != set(METHODS):
        raise ValueError("wrong reserve budget or methods")
    original_overlap = set(enc["high_indices"]) & set(enc["low_indices"])
    for method, added in assignment["methods"].items():
        n = 0 if method == "baseline" else quota
        a, b = (set(added[s]) for s in SIDES)
        if any(len(added[s]) != n or len(set(added[s])) != n for s in SIDES) or not (a | b) <= reserve or a & b:
            raise ValueError("added populations overlap, repeat or change budget")
        new_overlap = (set(enc["high_indices"]) | a) & (set(enc["low_indices"]) | b)
        if new_overlap != original_overlap:
            raise ValueError("shared original observations changed")


def score_pair(bank, enc, encoding, added):
    outputs, values = [], {}
    labels = bank.get("labels", np.full(len(bank["counts"]), np.nan)).astype(float)
    for side in SIDES:
        ix = np.r_[enc[f"{side}_indices"], added[side]].astype(int)
        out = decode(bank["counts"][:, ix], enc[encoding][ix], enc["grid_cm"], enc["near"], bank["truth_cm"])
        for m in ("home", "entropy", "error"):
            values[f"{side}_{m}"] = out[m]
        values[f"{side}_brier"] = (out["home"] - labels) ** 2
        values[f"{side}_spikes"] = bank["counts"][:, ix].sum(axis=1)
        values[f"{side}_active"] = (bank["counts"][:, ix] > 0).sum(axis=1)
        outputs.append(out)
    values["separation"] = np.linalg.norm(outputs[0]["mean"] - outputs[1]["mean"], axis=1)
    values["regional_tv"] = np.abs(outputs[0]["tiles"] - outputs[1]["tiles"]).sum(axis=1) / 2
    values["true_home"] = labels
    return values


def summarize_pair(values, session, source, encoding, method):
    row = dict(session=session, animal=session.split("/")[0], source=source, encoding=encoding, method=method, events=len(values["true_home"]))
    for m in ("high_home", "low_home", "separation", "regional_tv", "high_entropy", "low_entropy", "high_spikes", "low_spikes", "high_active", "low_active"):
        row[m] = float(np.mean(values[m]))
    row["home_gap"] = abs(row["high_home"] - row["low_home"])
    for side in SIDES:
        for m in ("error", "brier"):
            row[f"balanced_{side}_{m}"] = float(np.mean([np.mean(values[f"{side}_{m}"][values["true_home"] == k]) for k in (0, 1)])) if source in TRUTH else np.nan
    return row


def aggregate(session):
    keys = ["animal", "session", "source", "encoding"]
    random = session[session.method.str.startswith("random_")].groupby(keys, as_index=False)[list(METRICS)].mean()
    random["method"] = "random_mean"
    combined = pd.concat([session[session.method.isin(("baseline", "targeted"))], random], ignore_index=True)
    animal = combined.groupby(["animal", "source", "encoding", "method"], as_index=False)[list(METRICS)].mean()
    summary = animal.groupby(["source", "encoding", "method"], as_index=False)[list(METRICS)].mean()
    return combined, animal, summary


def gates(session, animal, summary):
    result = []

    def add(name, passed):
        result.append(dict(gate=name, passed=bool(passed)))

    expected = {(s, src, e, m) for s in SESSIONS for src in REAL + TRUTH for e in (("early_run", "full_run") if src in REAL else ("early_run",)) for m in METHODS}
    actual = set(session[["session", "source", "encoding", "method"]].itertuples(index=False, name=None))
    add("source_and_random_coverage", actual == expected and len(session) == len(expected))
    for source, n in zip(REAL, (1836, 513), strict=True):
        counts = session[session.source.eq(source) & session.encoding.eq("early_run")]
        add(f"fixed_{source}", len(counts) == len(SESSIONS) * len(METHODS) and counts.groupby("method").events.sum().eq(n).all())

    def read(source, enc, metric):
        table = animal[animal.source.eq(source) & animal.encoding.eq(enc)].pivot(index="animal", columns="method", values=metric)
        pooled = summary[summary.source.eq(source) & summary.encoding.eq(enc)].set_index("method")[metric]
        valid = len(table) == 3 and np.isfinite(table[["baseline", "targeted", "random_mean"]]).all().all()
        return table, pooled, valid

    for source in REAL:
        for enc in ("early_run", "full_run"):
            v, p, ok = read(source, enc, "home_gap")
            factor = 0.8 if source == REAL[0] else 1.0
            add(f"home_{source}_{enc}", ok and (v.targeted <= v.baseline + 1e-10).all() and p.targeted <= factor * p.baseline + 1e-10)
    for m in ("separation", "regional_tv", "high_entropy", "low_entropy"):
        v, p, ok = read(REAL[0], "early_run", m)
        if m in ("separation", "regional_tv"):
            add(f"real_{m}", ok and (v.targeted < v.baseline).all() and p.targeted <= 0.9 * p.baseline)
        else:
            add(f"real_{m}", ok and p.targeted <= p.baseline + 1e-10)
    for m in ("home_gap", "separation", "regional_tv"):
        _, p, ok = read(REAL[0], "early_run", m)
        add(f"beats_equal_budget_random_{m}", ok and p.targeted < p.random_mean)
    for source in TRUTH:
        for m in (x for x in METRICS if x.startswith("balanced_")):
            v, _, ok = read(source, "early_run", m)
            add(f"truth_{source}_{m}", ok and (v.targeted <= v.baseline + 1e-10).all())
    add("development_numerical_screen", all(x["passed"] for x in result))
    return pd.DataFrame(result)


def measure(args):
    snapshot = json.loads(args.source_snapshot.read_text())
    audit = json.loads(args.source_audit.read_text())
    if audit.get("status") != "pass" or audit["input_file_sha256"]["source"] != snapshot[str(args.source_dir / "manifest.json")]:
        raise ValueError("unverified source snapshot")

    def verify():
        for path, sha in snapshot.items():
            if file_sha256(path) != sha:
                raise ValueError(f"source changed: {path}")

    verify()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    inputs = {**snapshot, str(args.source_audit): file_sha256(args.source_audit), str(args.source_snapshot): file_sha256(args.source_snapshot)}
    for path in (Path(__file__), ROOT / "scripts/spatial_predictive_content.py", ROOT / "docs/reserve_cell_content_protocol.md"):
        inputs[str(path)] = file_sha256(path)
    frozen = {}
    for session in SESSIONS:
        folder = args.source_dir / session.replace("/", "_")
        frozen[session] = select_cells(read_npz(folder / "encoding.npz"), read_npz(folder / "run_q3.npz"), session)
        print(f"allocated {session}", flush=True)
    freeze = {**build_script_provenance(), "created_at_utc": datetime.now(timezone.utc).isoformat(), "assignments": frozen, "input_file_sha256": inputs}
    # This file is committed to disk before opening any target outcome array.
    (args.output_dir / "pre_scoring.json").write_text(json.dumps(freeze, indent=2) + "\n")
    freeze_hash = file_sha256(args.output_dir / "pre_scoring.json")
    summaries, event_count = [], 0
    for session in SESSIONS:
        folder = args.source_dir / session.replace("/", "_")
        enc = read_npz(folder / "encoding.npz")
        primary_events = []
        for source in REAL + TRUTH:
            bank = read_npz(folder / f"{source}.npz")
            for encoding in ("early_run", "full_run") if source in REAL else ("early_run",):
                for method, added in frozen[session]["methods"].items():
                    values = score_pair(bank, enc, encoding, added)
                    if method == "baseline":
                        np.testing.assert_allclose(np.column_stack([values[f"{s}_home"] for s in SIDES]), bank[f"{encoding}_scores"], atol=1e-9, rtol=1e-9)
                    summaries.append(summarize_pair(values, session, source, encoding, method))
                    event_count += len(bank["counts"])
                    if method in ("baseline", "targeted"):
                        frame = pd.DataFrame(values)
                        frame.insert(0, "observation_index", np.arange(len(frame)))
                        frame["event_id"] = bank.get("event_ids", np.arange(len(frame))).astype(str)
                        frame = frame.assign(session=session, source=source, encoding=encoding, method=method)
                        primary_events.append(frame)
        pd.concat(primary_events, ignore_index=True).to_csv(args.output_dir / f"{session.replace('/', '_')}_events.csv.gz", index=False)
        print(f"evaluated {session}", flush=True)
    session = pd.DataFrame(summaries)
    combined, animal, summary = aggregate(session)
    for name, frame in dict(session_draws=session, session_summary=combined, animal_summary=animal, summary=summary, gates=gates(session, animal, summary)).items():
        frame.to_csv(args.output_dir / f"{name}.csv", index=False)
    verify()
    for path, sha in inputs.items():
        if file_sha256(path) != sha:
            raise ValueError(f"input/code changed during run: {path}")
    if file_sha256(args.output_dir / "pre_scoring.json") != freeze_hash:
        raise ValueError("assignments changed during run")
    manifest = {
        **build_script_provenance(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_dir": str(args.source_dir),
        "input_file_sha256": inputs,
        "evaluated_event_method_rows": event_count,
        "assignments_sha256": freeze_hash,
        "external_validation": False,
        "output_sha256": {p.name: file_sha256(p) for p in args.output_dir.iterdir() if p.is_file()},
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("source-dir", "source-audit", "source-snapshot", "output-dir"):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    args = parser.parse_args()
    args.source_dir = args.source_dir.resolve()
    measure(args)


if __name__ == "__main__":
    main()
