#!/usr/bin/env python3
"""Independent exchange-gradient, sampling and full-likelihood audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np
import pandas as pd

from scripts import audit_reserve_cell_content as base
from scripts.audit_local_content_screen import class_flags


def state_for(pair, enc, bank):
    probabilities, homes, risks = [], [], []
    labels = bank["labels"].astype(int)
    for side in ("high", "low"):
        ix = pair[side]
        p = base.posterior(bank["counts"][:, ix], enc["early_run"][ix])
        home = p[:, enc["near"]].sum(axis=1)
        error = np.sqrt(np.sum((np.einsum("nb,bd->nd", p, enc["grid_cm"]) - bank["truth_cm"]) ** 2, axis=1))
        for label in (0, 1):
            risks.extend([error[labels == label].mean(), ((home[labels == label] - label) ** 2).mean()])
        probabilities.append(p)
        homes.append(home)
    delta = np.array([(homes[0] - homes[1])[labels == k].mean() for k in (0, 1)])
    return probabilities, homes, np.array(risks), delta, float((delta @ delta) / 2)


def direct_derivative(probabilities, homes, delta, enc, bank):
    labels = bank["labels"].astype(int)
    coefficient = np.array([delta[k] / np.count_nonzero(labels == k) for k in labels])
    gradient = []
    for cell, rates in enumerate(enc["early_run"]):
        contribution = bank["counts"][:, cell, None] * np.log(rates) - 0.02 * rates
        value = 0.0
        for p, home in zip(probabilities, homes, strict=True):
            expected_home = (p[:, enc["near"]] * contribution[:, enc["near"]]).sum(axis=1)
            expected = (p * contribution).sum(axis=1)
            value += np.dot(coefficient, expected_home - home * expected)
        gradient.append(value)
    return np.array(gradient)


def exchanged(pair, i, j):
    assert i in pair["high"] and i not in pair["low"] and j in pair["low"] and j not in pair["high"]
    result = {side: list(values) for side, values in pair.items()}
    result["high"].remove(i)
    result["high"].append(j)
    result["low"].remove(j)
    result["low"].append(i)
    return {side: sorted(values) for side, values in result.items()}


def check_assignment(enc, q3, assignment):
    index = np.sort(np.unique(q3["parent_ids"], return_index=True)[1])
    assert assignment["calibration_rows"] == index.tolist()
    bank = {k: v[index] for k, v in q3.items()}
    assert all(np.count_nonzero(bank["labels"] == k) >= 10 for k in (0, 1))
    original = {s: sorted(map(int, enc[f"{s}_indices"])) for s in ("high", "low")}
    pair = original
    p, h, risks, delta, objective = state_for(pair, enc, bank)
    initial_risks = risks.copy()
    np.testing.assert_allclose(assignment["baseline_objective"], objective, atol=1e-10)
    np.testing.assert_allclose(assignment["baseline_risks"], risks, atol=1e-9)
    steps = 0
    assert 1 <= len(assignment["trace"]) <= 10
    for n, trace in enumerate(assignment["trace"]):
        assert trace["step"] == n
        np.testing.assert_allclose(trace["before"], objective, atol=1e-10)
        gradient = direct_derivative(p, h, delta, enc, bank)
        options = [(i, j) for i in pair["high"] if i not in pair["low"] for j in pair["low"] if j not in pair["high"]]
        options.sort(key=lambda ij: (round(float(gradient[ij[1]] - gradient[ij[0]]), 10), *ij))
        assert [(r["high"], r["low"]) for r in trace["proposals"]] == options[:32]
        valid = []
        for (i, j), row in zip(options[:32], trace["proposals"], strict=True):
            _, _, loss, _, value = state_for(exchanged(pair, i, j), enc, bank)
            allowed = bool(value < objective - 1e-10 and (loss <= initial_risks + 1e-10).all())
            assert row["admissible"] == allowed
            np.testing.assert_allclose(row["risks"], loss, atol=1e-9, rtol=1e-9)
            np.testing.assert_allclose([row["gradient"], row["objective"]], [gradient[j] - gradient[i], value], atol=1e-10)
            if allowed:
                valid.append((value, i, j))
        if not valid:
            assert trace["chosen"] is None and n == len(assignment["trace"]) - 1
            assert assignment["stop_reason"] == "no_admissible_proposal"
            break
        minimum = min(v[0] for v in valid)
        _, i, j = min((v for v in valid if v[0] - minimum <= 1e-12), key=lambda v: (v[1], v[2]))
        assert trace["chosen"] == [i, j]
        pair = exchanged(pair, i, j)
        p, h, risks, delta, objective = state_for(pair, enc, bank)
        steps += 1
    if steps == 10:
        assert assignment["stop_reason"] == "accepted_step_budget"
    assert assignment["methods"]["baseline"] == original and assignment["methods"]["targeted"] == pair
    assert assignment["accepted_steps"] == steps
    net = len(set(original["high"]) - set(pair["high"]))
    assert assignment["net_exchanged"] == net
    np.testing.assert_allclose(assignment["final_objective"], objective, atol=1e-10)
    np.testing.assert_allclose(assignment["final_risks"], risks, atol=1e-9)
    assert set(assignment["methods"]) == set(base.METHODS)
    union, shared = set(original["high"]) | set(original["low"]), set(original["high"]) & set(original["low"])
    for method, choice in assignment["methods"].items():
        high, low = set(choice["high"]), set(choice["low"])
        assert high | low == union and high & low == shared
        assert len(high) == len(choice["high"]) == len(original["high"]) == len(original["low"]) == len(choice["low"]) == len(low)
        if method.startswith("random_"):
            seed = int.from_bytes(hashlib.sha256(f"20260915|exchange|{assignment['session']}|{int(method.split('_')[1])}".encode()).digest()[:8], "little")
            rng = np.random.default_rng(seed)
            a = set(rng.choice(sorted(set(original["high"]) - shared), net, replace=False))
            b = set(rng.choice(sorted(set(original["low"]) - shared), net, replace=False))
            assert high == set(original["high"]) - a | b and low == set(original["low"]) - b | a
        assert len(set(original["high"]) - high) == (0 if method == "baseline" else net)
        assert len(set(original["low"]) - low) == (0 if method == "baseline" else net)


def classes_for(values, session, source, method):
    output = []
    for label in (0, 1):
        ix = values["true_home"] == label
        row = dict(animal=session.split("/")[0], session=session, source=source, method=method, true_home=label, observations=int(ix.sum()))
        for key in ("high_error", "low_error", "high_brier", "low_brier"):
            row[key] = np.mean(values[key][ix])
        output.append(row)
    return output


def audit(root, output):
    manifest = json.loads((root / "manifest.json").read_text())
    for path, h in manifest["input_file_sha256"].items():
        assert base.sha(path) == h, path
    for name, h in manifest["output_sha256"].items():
        assert base.sha(root / name) == h, name
    prior = Path(manifest["source_result_dir"]) / "manifest.json"
    prior_manifest = json.loads(prior.read_text())
    prior_audit = json.loads(Path(manifest["source_audit"]).read_text())
    assert prior_audit["status"] == "pass" and prior_audit["manifest_sha256"] == base.sha(prior)
    for path, h in prior_manifest["input_file_sha256"].items():
        assert manifest["input_file_sha256"][path] == h
    frozen = json.loads((root / "pre_scoring.json").read_text())
    assert base.sha(root / "pre_scoring.json") == manifest["assignments_sha256"]
    assert frozen["input_file_sha256"] == manifest["input_file_sha256"]
    assert frozen["created_at_utc"] < manifest["scoring_started_at_utc"] < manifest["created_at_utc"]
    assert set(frozen["assignments"]) == set(base.SESSIONS)
    summaries, classes, count = [], [], 0
    for session in base.SESSIONS:
        folder = Path(manifest["source_dir"]) / session.replace("/", "_")
        enc = base.load(folder / "encoding.npz")
        check_assignment(enc, base.load(folder / "run_q3.npz"), frozen["assignments"][session])
        primary = []
        for source in base.REAL + base.TRUTH:
            bank = base.load(folder / f"{source}.npz")
            for encoding in ("early_run", "full_run") if source in base.REAL else ("early_run",):
                for method, pair in frozen["assignments"][session]["methods"].items():
                    assigned = {**enc, "high_indices": np.array(pair["high"]), "low_indices": np.array(pair["low"])}
                    values = base.values_for(bank, assigned, encoding, dict(high=[], low=[]))
                    if method == "baseline":
                        np.testing.assert_allclose(np.array([values[f"{s}_home"] for s in ("high", "low")]).T, bank[f"{encoding}_scores"], atol=1e-9, rtol=1e-9)
                    summaries.append(base.mean_row(values, session, source, encoding, method))
                    if source in base.TRUTH:
                        classes.extend(classes_for(values, session, source, method))
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
        actual = pd.read_csv(root / f"{session.replace('/', '_')}_events.csv.gz", dtype={"event_id": str})
        base.compare_frame(actual, pd.concat(primary, ignore_index=True), ["session", "source", "encoding", "method", "observation_index"])
        print("independently reconstructed", session, flush=True)
    tables, flags = base.rebuild_tables(summaries)
    flags.pop("development_numerical_screen")
    primary = tables["summary"]
    primary = primary[primary.source.eq(base.REAL[0]) & primary.encoding.eq("early_run")].set_index("method")
    for metric in ("home_gap", "separation", "regional_tv"):
        flags[f"beats_equal_budget_random_{metric}"] = bool(primary.loc["random_mean", metric] - primary.loc["targeted", metric] > 1e-10)
    classes = pd.DataFrame(classes)
    table = classes[classes.method.isin(("baseline", "targeted"))].copy()
    table["method"] = table.method.replace({"baseline": "all", "targeted": "local_half"})
    flags.update(class_flags(table))
    flags["development_numerical_screen"] = all(flags.values())
    for name, table in tables.items():
        base.compare_frame(pd.read_csv(root / f"{name}.csv"), table, [k for k in ("animal", "session", "source", "encoding", "method") if k in table])
    base.compare_frame(pd.read_csv(root / "truth_by_class.csv"), classes, ["session", "source", "method", "true_home"])
    gates = pd.read_csv(root / "gates.csv")
    recorded = dict(zip(gates.gate, gates.passed, strict=True))
    assert not gates.gate.duplicated().any() and recorded == flags, {k: (recorded.get(k), v) for k, v in flags.items() if recorded.get(k) != v}
    assert count == manifest["evaluated_event_method_rows"]
    result = dict(
        status="pass",
        manifest_sha256=base.sha(root / "manifest.json"),
        reconstructed_event_method_rows=count,
        reconstructed_population_posteriors=2 * count,
        gates=flags,
        audit_script_sha256=base.sha(__file__),
        external_validation=False,
        validated_remedy=False,
        scope="Independent gradients/proposals/choices, exact membership/budgets, all full-likelihood outcomes, source hashes, summaries and regional gates; no biological truth.",
    )
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.result_dir, args.output)
