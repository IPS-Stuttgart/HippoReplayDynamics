#!/usr/bin/env python3
"""Reconstruct frozen gradient-exchange risks without scoring replay events."""

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

from scripts import audit_exchange_content as proof
from scripts._provenance import build_script_provenance, file_sha256

SESSIONS = ("Rat1/Open1", "Rat1/Open2", "Rat2/Open1", "Rat4/Open2")
SOURCES = ("run_q3", "cal_poisson_gain1", "cal_conditional")


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def check_partition(bank, split, session, source):
    count = len(bank["counts"])
    group = bank.get("parent_ids", np.arange(count))
    labels = bank["labels"]
    if len(group) != count or not np.isin(labels, [0, 1]).all():
        raise ValueError("invalid grouping")
    first = {}
    for i, g in enumerate(group):
        first.setdefault(int(g), i)
    hold = set()
    for k in (0, 1):
        ids = [g for g, i in first.items() if labels[i] == k]
        ids.sort(key=lambda g: hashlib.sha256(f"20260915|robust-exchange|{session}|{source}|{g}".encode()).digest())
        hold.update(ids[: max(3, round(len(ids) * 0.25))])
    expected = {"train": [], "validation": []}
    for i, g in enumerate(group):
        expected["validation" if int(g) in hold else "train"].append(i)
    if split != expected:
        raise ValueError("partition differs from frozen grouped rule")
    for name, minimum in (("train", 10), ("validation", 3)):
        ix = np.array(split[name], dtype=int)
        for k in (0, 1):
            if len(set(group[ix][labels[ix] == k])) < minimum:
                raise ValueError("insufficient grouped truth support")


def exchanged(original, high, low):
    a, b = set(original["high"]), set(original["low"])
    h, low_set = set(high), set(low)
    if not (len(h) == len(high) == len(low_set) == len(low)) or not h <= a - b or not low_set <= b - a:
        raise ValueError("invalid exclusive exchange")
    return {"high": sorted(a - h | low_set), "low": sorted(b - low_set | h)}


def measure(enc, bank, pair):
    # The full-Poisson auditor includes the constants omitted by the producer.
    _, _, risks, _, objective = proof.state_for(pair, enc, bank)
    if not np.isfinite(risks).all() or not np.isfinite(objective):
        raise ValueError("nonfinite reconstructed metrics")
    return risks, objective


def audit(source_dir, result_dir, output_dir):
    manifest_path = result_dir / "manifest.json"
    manifest_hash = file_sha256(manifest_path)
    audit_hash = file_sha256(__file__)
    manifest = json.loads(manifest_path.read_text())
    for path, sha in manifest["input_file_sha256"].items():
        if file_sha256(path) != sha:
            raise ValueError("changed producer input")
    for name, sha in manifest["output_sha256"].items():
        if file_sha256(result_dir / name) != sha:
            raise ValueError("changed producer output")
    frozen = json.loads((result_dir / "frozen_assignments.json").read_text())
    if file_sha256(result_dir / "frozen_assignments.json") != manifest["frozen_assignments_sha256"]:
        raise ValueError("changed frozen assignments")
    if set(frozen["choices"]) != set(SESSIONS) or set(frozen["splits"]) != set(SESSIONS):
        raise ValueError("incomplete original cohort or split support")
    validation = pd.read_csv(result_dir / "internal_validation.csv")
    key = ["session", "source", "side", "true_home", "metric"]
    if len(validation) != 96 or validation.duplicated(key).any():
        raise ValueError("incomplete validation table")
    risks_rows, candidates_rows, support = [], [], []
    chosen_rats, native_changes = set(), {}
    all_nonworsening = True
    for session in SESSIONS:
        folder = source_dir / session.replace("/", "_")
        for name in ("encoding",) + SOURCES:
            path = folder / f"{name}.npz"
            if file_sha256(path) != manifest["input_file_sha256"].get(str(path)):
                raise ValueError("unregistered calibration source")
        enc = load(folder / "encoding.npz")
        choice = frozen["choices"][session]
        original = {s: enc[f"{s}_indices"].tolist() for s in ("high", "low")}
        if original != choice["original"]:
            raise ValueError("changed baseline population")
        banks = {}
        for source in SOURCES:
            bank = load(folder / f"{source}.npz")
            split = frozen["splits"][session][source]
            check_partition(bank, split, session, source)
            for phase in ("train", "validation"):
                ix = np.asarray(split[phase], dtype=int)
                banks[(source, phase)] = {k: bank[k][ix] for k in ("counts", "truth_cm", "labels")}
                groups = bank.get("parent_ids", np.arange(len(bank["counts"])))
                support.append(dict(session=session, source=source, phase=phase, observations=len(ix), groups=len(np.unique(groups[ix])), true_home=int(bank["labels"][ix].sum())))
        baseline = [measure(enc, banks[(s, "train")], original) for s in SOURCES]
        r0 = np.concatenate([v[0] for v in baseline])
        j0 = baseline[0][1]
        np.testing.assert_allclose(r0, choice["baseline_risks"], atol=1e-9, rtol=1e-9)
        np.testing.assert_allclose(j0, choice["baseline_j"], atol=1e-10, rtol=1e-9)
        eligible = []
        for c in choice["candidates"]:
            pair = exchanged(original, c["high"], c["low"])
            if pair != c["pair"]:
                raise ValueError("candidate membership mismatch")
            values = [measure(enc, banks[(s, "train")], pair) for s in SOURCES]
            risks, j = np.concatenate([v[0] for v in values]), values[0][1]
            np.testing.assert_allclose(risks, c["risks"], atol=1e-9, rtol=1e-9)
            np.testing.assert_allclose(j, c["objective"], atol=1e-10, rtol=1e-9)
            admissible = bool(j < j0 - 1e-10 and np.all(risks <= r0 + 1e-10))
            if admissible != c["admissible"]:
                raise ValueError("exact admissibility mismatch")
            if admissible:
                eligible.append(c)
            proposals = [p for p in choice["proposals"] if p.get("feasible") and f"joint_{p['budget']}_{p['index']}" == c["name"]]
            if len(proposals) != 1:
                raise ValueError("missing unique proposal")
            predicted = np.asarray(proposals[0]["predicted_risk_changes"])
            if len(predicted) != 24:
                raise ValueError("missing proposal risks")
            candidates_rows.append(
                dict(
                    session=session,
                    candidate=c["name"],
                    exchanged=len(c["high"]),
                    baseline_j=j0,
                    targeted_j=j,
                    objective_improves=j < j0 - 1e-10,
                    failed_risks=int((risks > r0 + 1e-10).sum()),
                    admissible=admissible,
                )
            )
            for i in range(24):
                risks_rows.append(
                    dict(
                        session=session,
                        candidate=c["name"],
                        source=SOURCES[i // 8],
                        side="high" if i % 8 < 4 else "low",
                        true_home=(i % 4) // 2,
                        metric="error_cm" if i % 2 == 0 else "home_brier",
                        baseline=r0[i],
                        targeted=risks[i],
                        actual_change=risks[i] - r0[i],
                        predicted_change=predicted[i],
                        failed=risks[i] > r0[i] + 1e-10,
                    )
                )
        if eligible:
            best_j = min(c["objective"] for c in eligible)
            best = min([c for c in eligible if abs(c["objective"] - best_j) <= 1e-12], key=lambda c: (len(c["high"]), c["high"], c["low"]))
            name, pair, net = best["name"], best["pair"], len(best["high"])
        else:
            name, pair, net = None, original, 0
        if choice["chosen"] != name or choice["final_pair"] != pair or choice["net_exchanged"] != net:
            raise ValueError("frozen choice is not the admissible training winner")
        if net:
            chosen_rats.add(session.split("/")[0])
        for source in SOURCES:
            before, before_j = measure(enc, banks[(source, "validation")], original)
            after, after_j = measure(enc, banks[(source, "validation")], pair)
            if source == "run_q3":
                native_changes[session] = after_j - before_j
            for i in range(8):
                v = validation[
                    (validation.session == session)
                    & (validation.source == source)
                    & (validation.side == ("high" if i < 4 else "low"))
                    & (validation.true_home == (i % 4) // 2)
                    & (validation.metric == ("error_cm" if i % 2 == 0 else "home_brier"))
                ].iloc[0]
                np.testing.assert_allclose(
                    [v.baseline, v.targeted, v.change, v.baseline_j, v.targeted_j], [before[i], after[i], after[i] - before[i], before_j, after_j], atol=1e-9, rtol=1e-9
                )
                nonworsening = bool(after[i] <= before[i] + 1e-10)
                if nonworsening != bool(v.nonworsening):
                    raise ValueError("validation guard mismatch")
                all_nonworsening &= nonworsening
        print("independently reconstructed", session, flush=True)
    rat_changes = {r: np.mean([v for s, v in native_changes.items() if s.split("/")[0] == r]) for r in ("Rat1", "Rat2", "Rat4")}
    expected_gates = dict(
        all_original_pairs_have_support=True,
        all_validation_risks_present=True,
        internal_accuracy_nonworsening=all_nonworsening,
        changed_populations_in_all_rats=chosen_rats == set(rat_changes),
        native_validation_improves_all_rats=all(v < -1e-10 for v in rat_changes.values()),
    )
    expected_gates["ready_for_truth_preflight"] = all(expected_gates.values())
    if pd.read_csv(result_dir / "gates.csv").set_index("gate").passed.to_dict() != expected_gates:
        raise ValueError("gate mismatch")
    if any(manifest[k] for k in ("external_validation", "validated_remedy", "replay_scored", "q4_scored")):
        raise ValueError("unsupported evaluation claim")
    output_dir.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(candidates_rows, columns=["session", "candidate", "exchanged", "baseline_j", "targeted_j", "objective_improves", "failed_risks", "admissible"]).to_csv(
        output_dir / "candidate_reconstruction.csv", index=False
    )
    pd.DataFrame(
        risks_rows, columns=["session", "candidate", "source", "side", "true_home", "metric", "baseline", "targeted", "actual_change", "predicted_change", "failed"]
    ).to_csv(output_dir / "risk_reconstruction.csv", index=False)
    pd.DataFrame(support).to_csv(output_dir / "partition_support.csv", index=False)
    for path, sha in manifest["input_file_sha256"].items():
        if file_sha256(path) != sha:
            raise ValueError("producer input changed during audit")
    for name, sha in manifest["output_sha256"].items():
        if file_sha256(result_dir / name) != sha:
            raise ValueError("producer output changed during audit")
    if file_sha256(manifest_path) != manifest_hash or file_sha256(__file__) != audit_hash:
        raise ValueError("manifest or auditor changed during audit")
    record = build_script_provenance()
    record.update(
        status="pass",
        manifest_sha256=manifest_hash,
        producer_directory=str(result_dir),
        candidate_reconstructions=len(candidates_rows),
        exact_risk_reconstructions=len(risks_rows),
        validation_risks_reconstructed=96,
        ready_for_truth_preflight=expected_gates["ready_for_truth_preflight"],
        validated_remedy=False,
        external_validation=False,
        input_file_sha256={
            str(manifest_path): manifest_hash,
            str(Path(__file__)): audit_hash,
            str(ROOT / "scripts/audit_exchange_content.py"): file_sha256(ROOT / "scripts/audit_exchange_content.py"),
        },
        output_sha256={p.name: file_sha256(p) for p in output_dir.iterdir()},
    )
    (output_dir / "independent_audit.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("source-dir", "result-dir", "output-dir"):
        parser.add_argument(f"--{flag}", required=True, type=Path)
    a = parser.parse_args()
    audit(a.source_dir, a.result_dir, a.output_dir)
