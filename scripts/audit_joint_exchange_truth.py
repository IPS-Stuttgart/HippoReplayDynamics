#!/usr/bin/env python3
"""Known-position preflight for frozen joint exchange; no replay scoring."""

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

from scripts import exchange_content as exact
from scripts import audit_exchange_content as proof
from scripts._provenance import build_script_provenance, file_sha256
from scripts.report_content_screening_bound import markdown


def checked_manifest(root, audit_path):
    manifest = json.loads((root / "manifest.json").read_text())
    audit = json.loads(audit_path.read_text())
    if audit.get("status") != "pass" or audit.get("manifest_sha256") != file_sha256(root / "manifest.json"):
        raise ValueError("matching independent audit required")
    for name, sha in manifest["output_sha256"].items():
        if file_sha256(root / name) != sha:
            raise ValueError(f"changed artifact: {name}")
    for path, sha in manifest["input_file_sha256"].items():
        if file_sha256(path) != sha:
            raise ValueError(f"changed source: {path}")
    return manifest


def validate_pair(enc, choice):
    original = {s: set(map(int, enc[f"{s}_indices"])) for s in ("high", "low")}
    pair = choice["final_pair"]
    for s in original:
        if set(choice["original"][s]) != original[s] or len(pair[s]) != len(original[s]) or len(set(pair[s])) != len(pair[s]):
            raise ValueError("changed original or final population count")
    a, b = set(pair["high"]), set(pair["low"])
    if a | b != original["high"] | original["low"] or a & b != original["high"] & original["low"]:
        raise ValueError("changed union or shared cells")
    if any(len(original[s] - set(pair[s])) != choice["net_exchanged"] for s in original):
        raise ValueError("changed exchange count")


def score(bank, enc, pair):
    assigned = {**enc, **{f"{s}_indices": np.asarray(pair[s], int) for s in ("high", "low")}}
    values = exact.base.score_pair(bank, assigned, "early_run", dict(high=[], low=[]))
    independent = proof.base.values_for(bank, assigned, "early_run", dict(high=[], low=[]))
    for key in values:
        np.testing.assert_allclose(values[key], independent[key], atol=1e-9, rtol=1e-9)
    return values, independent


def class_table(before, after, session, source, parent_ids=None):
    labels = before["true_home"]
    if not np.array_equal(labels, after["true_home"]) or not np.isin(labels, [0, 1]).all():
        raise ValueError("truth labels changed or missing")
    rows = []
    for label in (0, 1):
        ix = labels == label
        if not ix.any():
            raise ValueError("empty truth class")
        for side in ("high", "low"):
            row = dict(
                session=session,
                animal=session.split("/")[0],
                source=source,
                true_home=label,
                side=side,
                observations=int(ix.sum()),
                parent_groups=None if parent_ids is None else len(np.unique(parent_ids[ix])),
            )
            for metric in ("error", "brier"):
                x, y = float(before[f"{side}_{metric}"][ix].mean()), float(after[f"{side}_{metric}"][ix].mean())
                if not np.isfinite([x, y]).all():
                    raise ValueError("nonfinite known-position risk")
                row.update({f"baseline_{metric}": x, f"targeted_{metric}": y, f"change_{metric}": y - x, f"{metric}_nonworsening": y <= x + 1e-10})
            rows.append(row)
    return rows


def run(selection_dir, reference_dir, reference_audit, output_dir):
    selected = checked_manifest(selection_dir, selection_dir / "independent_audit.json")
    reference = checked_manifest(reference_dir, reference_audit)
    ref_path = reference_dir / "manifest.json"
    if selected["input_file_sha256"].get(str(ref_path)) != file_sha256(ref_path):
        raise ValueError("selection not linked to this reference")
    choices = json.loads((selection_dir / "selection.json").read_text())
    if set(choices) != set(exact.base.SESSIONS):
        raise ValueError("missing original pair")
    inputs = {
        str(p): file_sha256(p)
        for p in (
            selection_dir / "manifest.json",
            selection_dir / "selection.json",
            selection_dir / "independent_audit.json",
            ref_path,
            reference_audit,
            Path(__file__),
            ROOT / "docs/joint_exchange_truth_protocol.md",
        )
    }
    inputs.update(selected["input_file_sha256"])
    inputs.update(reference["input_file_sha256"])
    output_dir.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    if selected["created_at_utc"] >= started:
        raise ValueError("selection must precede evaluation")
    rows, independent_rows, count = [], [], 0
    for session in exact.base.SESSIONS:
        folder = Path(reference["source_dir"]) / session.replace("/", "_")
        enc_path = folder / "encoding.npz"
        if selected["input_file_sha256"].get(str(enc_path)) != reference["input_file_sha256"].get(str(enc_path)):
            raise ValueError("encoding identity differs")
        enc = exact.base.read_npz(enc_path)
        choice = choices[session]
        validate_pair(enc, choice)
        for source in exact.base.TRUTH:
            bank_path = folder / f"{source}.npz"
            if file_sha256(bank_path) != reference["input_file_sha256"].get(str(bank_path)):
                raise ValueError("test bank differs")
            bank = exact.base.read_npz(bank_path)
            before, before_check = score(bank, enc, choice["original"])
            after, after_check = score(bank, enc, choice["final_pair"])
            np.testing.assert_allclose(np.column_stack([before[f"{s}_home"] for s in ("high", "low")]), bank["early_run_scores"], atol=1e-9, rtol=1e-9)
            rows.extend(class_table(before, after, session, source, bank.get("parent_ids")))
            independent_rows.extend(class_table(before_check, after_check, session, source, bank.get("parent_ids")))
            count += len(bank["counts"])
        print("held-out truth checked", session, flush=True)
    table, check = pd.DataFrame(rows), pd.DataFrame(independent_rows)
    pd.testing.assert_frame_equal(table, check, atol=1e-9, rtol=1e-9)
    keys = ["session", "source", "true_home", "side"]
    if table.duplicated(keys).any() or len(table) != len(exact.base.SESSIONS) * len(exact.base.TRUTH) * 4:
        raise ValueError("incomplete truth comparison coverage")
    table.to_csv(output_dir / "truth_by_class.csv", index=False)
    failures = []
    for row in rows:
        for metric in ("error", "brier"):
            if not row[f"{metric}_nonworsening"]:
                failures.append(
                    {k: row[k] for k in keys} | dict(metric=metric, baseline=row[f"baseline_{metric}"], targeted=row[f"targeted_{metric}"], change=row[f"change_{metric}"])
                )
    failed = pd.DataFrame(failures, columns=keys + ["metric", "baseline", "targeted", "change"])
    failed.to_csv(output_dir / "failed_guards.csv", index=False)
    gates = pd.DataFrame(
        [
            dict(gate="all_pairs_sources_classes_present", passed=True),
            dict(gate="independent_likelihoods_match", passed=True),
            dict(gate="all_region_specific_risks_nonworsening", passed=failed.empty),
            dict(gate="truth_preflight", passed=failed.empty),
        ]
    )
    gates.to_csv(output_dir / "gates.csv", index=False)
    summary = table.groupby(["source"], as_index=False).agg(
        error_guards_passed=("error_nonworsening", "sum"), brier_guards_passed=("brier_nonworsening", "sum"), comparisons_per_metric=("error_nonworsening", "size")
    )
    summary.to_csv(output_dir / "summary.csv", index=False)
    text = "# Joint exchange held-out truth preflight\n\n"
    text += f"Known-position accuracy preflight: {'PASS' if failed.empty else 'FAIL'}. Independent likelihood reconstruction: PASS.\n\n"
    text += markdown(summary) + "\n\n## Failed classwise guards\n\n" + markdown(failed)
    text += "\n\nNo replay evidence was scored. These are point-estimate safeguards, not independent statistical tests or proof of population harm. "
    text += "No independent recording validation and no validated remedy. Failed guards cannot be repaired by selecting another candidate after inspecting these outcomes.\n"
    (output_dir / "report.md").write_text(text)
    for path, sha in inputs.items():
        if file_sha256(path) != sha:
            raise ValueError(f"changed input: {path}")
    record = build_script_provenance()
    record.update(
        input_file_sha256=inputs,
        evaluation_started_at_utc=started,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        paired_truth_observations=count,
        regional_loss_comparisons=len(table) * 2,
        failed_comparisons=len(failed),
        replay_scored=False,
        external_validation=False,
        validated_remedy=False,
        output_sha256={p.name: file_sha256(p) for p in output_dir.iterdir() if p.is_file()},
    )
    (output_dir / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    (output_dir / "independent_audit.json").write_text(
        json.dumps(
            dict(
                status="pass",
                manifest_sha256=file_sha256(output_dir / "manifest.json"),
                population_posteriors_reconstructed=4 * count,
                original_cached_baseline_verified=True,
                truth_preflight=failed.empty,
                external_validation=False,
                validated_remedy=False,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("selection-dir", "reference-dir", "reference-audit", "output-dir"):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    args = parser.parse_args()
    run(args.selection_dir, args.reference_dir, args.reference_audit, args.output_dir)
