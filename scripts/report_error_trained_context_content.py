#!/usr/bin/env python3
"""Non-rescoring truth and stability checks for the error-trained context gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import pandas as pd

from scripts._provenance import build_script_provenance, file_sha256
from scripts.error_trained_context_content import METHODS, PRIMARY
from scripts.report_predictive_context_content import combined_gates, diagnostic_summary
from scripts.report_temporal_endpoint_content import plot_summary, summaries


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--measurement-dir", type=Path, required=True)
    p.add_argument("--audit-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    inputs = dict(
        producer=args.measurement_dir / "manifest.json",
        audit=args.audit_dir / "independent_audit.json",
        reporter=Path(__file__),
        diagnostics=ROOT / "scripts/report_predictive_context_content.py",
        common=ROOT / "scripts/report_temporal_endpoint_content.py",
        protocol=ROOT / "docs/error_trained_context_protocol.md",
    )
    producer, audit = (json.loads(inputs[name].read_text()) for name in ("producer", "audit"))
    if (
        producer["status"] != "complete"
        or producer["primary_method"] != PRIMARY
        or audit["status"] != "passed"
        or audit["input_file_sha256"]["producer"] != file_sha256(inputs["producer"])
    ):
        raise ValueError("complete matching independent audit required")
    frames = []
    for row in producer["results"]:
        path = Path(row["artifact_dir"]) / "event_readouts.csv.gz"
        verified = [a for a in audit["results"] if (a["animal"], a["session"]) == (row["animal"], row["session"])]
        if len(verified) != 1 or verified[0]["readout_sha256"] != file_sha256(path):
            raise ValueError("changed or unaudited readout")
        inputs[row["session"]] = path
        frames.append(pd.read_csv(path, float_precision="round_trip"))
    frame = pd.concat(frames, ignore_index=True)
    session, animal, summary = summaries(frame, methods=METHODS)
    diag_session, diag_animal = diagnostic_summary(frame, prediction_suffix="_predicted_physical_gain")
    gates = combined_gates(summary, frame, diag_animal, audited=True, primary=PRIMARY)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = build_script_provenance(input_paths=inputs, cwd=ROOT)
    for name, table in (
        ("by_session", session),
        ("by_animal", animal),
        ("summary", summary),
        ("diagnostic_by_session", diag_session),
        ("diagnostic_by_animal", diag_animal),
        ("gate_summary", gates),
    ):
        table.to_csv(args.output_dir / f"{name}.csv", index=False)
    cohort = "PF leave-one-rat-out development" if frame.dataset.iloc[0] == "pfeiffer_foster" else str(frame.dataset.iloc[0]) + " independent evaluation"
    plot_summary(animal, summary, args.output_dir, primary_method=PRIMARY, primary_label="Own-population error-trained choice", cohort_label=cohort)
    failed = gates.loc[~gates.passed, "gate"].tolist()
    lines = [
        "# Error-trained temporal-context diagnostic",
        "",
        cohort,
        "",
        "Known-truth training excludes all sessions and splits of the evaluated PF rat. No real replay labels or A/B agreement train the predictor.",
        "All original event IDs and endpoint times retained. Independent-data and targeted Home validation are still required.",
        "",
        "| Source | Metric | Independent | Error-trained choice | Reduction |",
        "|---|---|---:|---:|---:|",
    ]
    for row in summary.loc[(summary.split == 0) & (summary.method == PRIMARY)].itertuples(index=False):
        if row.metric in ("separation_cm", "regional_tv", "a_error", "b_error"):
            lines.append(f"| {row.source} | {row.metric} | {row.baseline:.5f} | {row.value:.5f} | {row.reduction:+.5f} |")
    lines += [
        "",
        "## Frozen decision",
        "",
        "Advancement: " + ("FAIL" if failed else "PASS"),
        "",
        *["- " + name for name in failed],
        "",
        "No replacement of the primary method, threshold tuning or removal of late-jump controls is permitted.",
    ]
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n")
    if any(file_sha256(path) != manifest["input_file_sha256"][key] for key, path in inputs.items()):
        raise ValueError("changed report input")
    manifest.update(
        status="complete",
        non_rescoring=True,
        primary_method=PRIMARY,
        primary_advanced=not failed,
        scientific_goal_achieved=False,
        output_sha256={p.name: file_sha256(p) for p in args.output_dir.iterdir()},
    )
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
