#!/usr/bin/env python3
"""Post-hoc fixed-choice audit of RUN calibration transfer; no replay scoring."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np
import pandas as pd

from scripts import audit_joint_exchange_truth as truth
from scripts._provenance import build_script_provenance, file_sha256
from scripts.report_content_screening_bound import markdown


def phase_indices(bank):
    n = len(bank["counts"])
    parents = bank["parent_ids"]
    if len(parents) != n or not n:
        raise ValueError("missing native parent observations")
    starts, ends = bank["starts_s"], bank["ends_s"]
    if len(starts) != n or len(ends) != n or not np.isfinite(starts).all() or not np.isfinite(ends).all():
        raise ValueError("missing native time coordinates")
    if np.any(np.diff(starts) <= 0) or not np.allclose(ends - starts, 0.02, atol=1e-8, rtol=0):
        raise ValueError("nonmonotone native times or changed 20ms duration")
    _, first, counts = np.unique(parents, return_index=True, return_counts=True)
    if np.any(counts < 2):
        raise ValueError("no remaining observations for a native parent")
    first = np.sort(first)
    rest = np.setdiff1d(np.arange(n), first)
    return dict(first=np.asarray(first), remaining=rest, all=np.arange(n))


def objective(values):
    labels = values["true_home"]
    delta = values["high_home"] - values["low_home"]
    return float(np.mean([delta[labels == k].mean() ** 2 for k in (0, 1)]))


def inspect_bank(enc, bank, choice, session, source):
    truth.validate_pair(enc, choice)
    phases = phase_indices(bank)
    original, _ = truth.score(bank, enc, choice["original"])
    final, _ = truth.score(bank, enc, choice["final_pair"])
    rows, summaries = [], []
    for phase, ix in phases.items():
        before = {k: v[ix] for k, v in original.items()}
        after = {k: v[ix] for k, v in final.items()}
        classes = truth.class_table(before, after, session, source, bank["parent_ids"][ix])
        before_j, after_j = objective(before), objective(after)
        if source == "run_q3" and phase == "first":
            np.testing.assert_allclose([before_j, after_j], [choice["baseline_j"], choice["final_j"]], atol=1e-9)
            risks = {"baseline": choice["baseline_risks"]}
            risks["targeted"] = choice["baseline_risks"] if choice["chosen"] is None else next(c["risks"] for c in choice["candidates"] if c["name"] == choice["chosen"])
            for method, expected in risks.items():
                actual = [
                    next(r[f"{method}_{metric}"] for r in classes if r["side"] == s and r["true_home"] == k)
                    for s in ("high", "low")
                    for k in (0, 1)
                    for metric in ("error", "brier")
                ]
                np.testing.assert_allclose(actual, expected, atol=1e-9)
        rows.extend([{**r, "phase": phase} for r in classes])
        summaries.append(
            dict(
                session=session,
                animal=session.split("/")[0],
                source=source,
                phase=phase,
                observations=len(ix),
                parents=len(np.unique(bank["parent_ids"][ix])),
                true_home_observations=int((before["true_home"] == 1).sum()),
                baseline_j=before_j,
                targeted_j=after_j,
                objective_improves=after_j < before_j - 1e-10,
                errors_nonworsening=sum(r["error_nonworsening"] for r in classes),
                briers_nonworsening=sum(r["brier_nonworsening"] for r in classes),
                all_eight_guards_pass=all(r[f"{k}_nonworsening"] for r in classes for k in ("error", "brier")),
            )
        )
    return rows, summaries


def run(selection_dir, source_dir, output_dir):
    manifest = truth.checked_manifest(selection_dir, selection_dir / "independent_audit.json")
    choices = json.loads((selection_dir / "selection.json").read_text())
    if set(choices) != set(truth.exact.base.SESSIONS):
        raise ValueError("missing original pair")
    inputs = {
        **manifest["input_file_sha256"],
        **{
            str(p): file_sha256(p)
            for p in (
                selection_dir / "manifest.json",
                selection_dir / "selection.json",
                selection_dir / "independent_audit.json",
                Path(__file__),
                ROOT / "scripts/audit_joint_exchange_truth.py",
                ROOT / "docs/joint_exchange_transfer_protocol.md",
            )
        },
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    rows, summaries = [], []
    for session in truth.exact.base.SESSIONS:
        folder = source_dir / session.replace("/", "_")
        enc = truth.exact.base.read_npz(folder / "encoding.npz")
        for source in ("run_q3", "run_q4"):
            bank_path = folder / f"{source}.npz"
            for path in (folder / "encoding.npz", bank_path):
                if file_sha256(path) != manifest["input_file_sha256"].get(str(path)):
                    raise ValueError("changed native input")
            r, s = inspect_bank(enc, truth.exact.base.read_npz(bank_path), choices[session], session, source)
            rows.extend(r)
            summaries.extend(s)
        print("native transfer checked", session, flush=True)
    table, summary = pd.DataFrame(rows), pd.DataFrame(summaries)
    if len(table) != 96 or len(summary) != 24:
        raise ValueError("incomplete session/phase coverage")
    table.to_csv(output_dir / "classwise_native_transfer.csv", index=False)
    summary.to_csv(output_dir / "native_transfer_summary.csv", index=False)
    text = "# Joint exchange native calibration transfer\n\nPost-hoc failure diagnosis, not an independently validated diagnostic or remedy.\n\n"
    text += markdown(summary) + "\n\nFirst: the first20-ms sample in each native250-ms parent. Remaining: its other20-ms samples. "
    text += "Only Q3-first entered selection. Q3-remaining is unselected but temporally adjacent and NOT independent recording validation. "
    text += "Q4 is the later RUN block. All includes first+remaining and is not an additional independent comparison.\n\n"
    text += "Same frozen final choices in every row; no alternative selection, replay rescoring, dropped pairs, or changed accuracy safeguards. "
    text += "Objectives and all eight original selection risks are independently reconstructed for Q3-first.\n"
    (output_dir / "report.md").write_text(text)
    for path, sha in inputs.items():
        if file_sha256(path) != sha:
            raise ValueError(f"changed input: {path}")
    record = build_script_provenance()
    record.update(
        input_file_sha256=inputs,
        posthoc_diagnostic=True,
        validated_remedy=False,
        external_validation=False,
        replay_scored=False,
        output_sha256={p.name: file_sha256(p) for p in output_dir.iterdir() if p.is_file()},
    )
    (output_dir / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("selection-dir", "source-dir", "output-dir"):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    args = parser.parse_args()
    run(args.selection_dir, args.source_dir, args.output_dir)
