#!/usr/bin/env python3
"""Verify and index the completed recording-coverage evidence, without rescoring."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from scripts._provenance import build_script_provenance, file_sha256

# Exact production directories; failed and small smoke runs are not substituted.
STUDIES = [
    ("likelihood", "replay-coverage-likelihood-audit-full-20260905", "coverage_likelihood_manifest.json", None, "coverage_likelihood_gate_summary.csv", None),
    ("inputs", "replay-coverage-real-inputs-all33-20260905", "coverage_input_manifest.json", None, "coverage_input_gate_summary.csv", None),
    ("subsampling", "replay-coverage-subsampling-all33-20260905", "coverage_subsampling_manifest.json", None, "coverage_subsampling_gate_summary.csv", None),
    ("RUN_validation", "replay-coverage-RUN-validation-all33-20260906", "coverage_RUN_validation_manifest.json", "coverage_RUN_validation_independent_audit.json", "coverage_RUN_validation_gate_summary.csv", None),
    ("recovery", "replay-coverage-recovery-all33-20260905", "coverage_recovery_manifest.json", "coverage_recovery_reconstruction_audit.json", "coverage_recovery_gate_summary.csv", "coverage_recovery_paired_report_manifest.json"),
    ("counterfactual", "replay-coverage-counterfactual-all33-20260905", "coverage_counterfactual_manifest.json", "coverage_counterfactual_reconstruction_audit.json", "coverage_counterfactual_gate_summary.csv", "coverage_counterfactual_report_manifest.json"),
    ("geometry", "replay-coverage-geometry-full-20260905", "geometry_manifest.json", "geometry_reconstruction_audit.json", "geometry_gate_summary.csv", "report-final/geometry_report_manifest.json"),
    ("map_mismatch", "replay-coverage-map-mismatch-all33-20260905", "coverage_map_mismatch_manifest.json", "coverage_map_mismatch_reconstruction_audit.json", "coverage_map_mismatch_gate_summary.csv", "report/coverage_map_mismatch_report_manifest.json"),
    ("speed_identifiability", "replay-speed-identifiability-all33-20260905", "speed_identifiability_manifest.json", "speed_identifiability_reconstruction_audit.json", "speed_identifiability_gate_summary.csv", "report/speed_identifiability_report_manifest.json"),
    ("event_definitions", "replay-coverage-event-definitions-all33-20260905-v2", "coverage_event_definition_manifest.json", "coverage_event_definition_reconstruction_audit.json", "coverage_event_definition_gate_summary.csv", None),
    ("detector_sensitivity", "replay-coverage-detector-decoding-all33-20260905", "coverage_detector_sensitivity_manifest.json", "coverage_detector_sensitivity_reconstruction_audit.json", "coverage_detector_sensitivity_gate_summary.csv", "report-v2/coverage_detector_sensitivity_report_manifest.json"),
    ("shuffle_baseline", "replay-coverage-shuffle-baseline-all33-20260905", "coverage_shuffle_baseline_manifest.json", "coverage_shuffle_baseline_reconstruction_audit.json", "coverage_shuffle_baseline_gate_summary.csv", "report-v2/coverage_shuffle_baseline_report_manifest.json"),
    ("population_transfer", "replay-speed-population-transfer-all33-20260905", "speed_population_transfer_manifest.json", "speed_population_transfer_reconstruction_audit.json", "speed_population_transfer_gate_summary.csv", "report-v2/speed_population_transfer_report_manifest.json"),
    ("panel_size", "replay-speed-panel-size-all33-20260905", "speed_panel_size_manifest.json", "speed_panel_size_reconstruction_audit.json", "speed_panel_size_gate_summary.csv", "report/speed_panel_size_report_manifest.json"),
    ("event_figures", "replay-coverage-matched-event-figures-20260905-v2", "coverage_matched_event_figures_manifest.json", None, None, None),
]


def linked(meta, manifest):
    digest = file_sha256(manifest)
    if any(meta.get(k) == digest for k in ["validation_manifest_sha256", "scoring_manifest_sha256"]):
        return True
    return any(Path(path).resolve() == manifest.resolve() and meta.get("input_file_sha256", {}).get(k) == digest
        for k, path in meta.get("input_file_paths", {}).items())


def check_gates(data):
    columns = [c for c in ["passed", "pass"] if c in data]
    if data.empty or len(columns) != 1:
        raise ValueError("nonempty recognized technical gate schema required")
    result = data[columns[0]].map(lambda x: str(x).lower() in ["true", "1"])
    required = data.required_for_preparation.eq(True) if "required_for_preparation" in data else pd.Series(True, index=data.index)
    if not required.any() or not result[required].all():
        raise ValueError("nonempty passing required technical gates required")
    return ";".join(data.loc[~required & ~result, "gate"].astype(str))


def verify_entry(root, spec):
    name, directory, manifest_name, audit_name, gate_name, report_name = spec
    folder = root / directory
    manifest = folder / manifest_name
    meta = json.loads(manifest.read_text())
    if "status" in meta and meta["status"] not in ["complete", "complete_with_ripple_unavailable"]:
        raise ValueError(f"incomplete {name}")
    files = []
    def record(path, role, digest=None):
        actual = file_sha256(path)
        if digest is not None and actual != digest:
            raise ValueError(f"changed {role}: {path}")
        files.append({"study": name, "role": role, "path": str(path), "sha256": actual,
            "bytes": path.stat().st_size, "matches_recorded_hash": digest is not None})
    record(manifest, "manifest")
    optional_gate_failures = ""
    if gate_name:
        gate = folder / gate_name
        data = pd.read_csv(gate)
        optional_gate_failures = check_gates(data)
        record(gate, "gates", meta.get("output_sha256", {}).get(gate_name))
    elif meta.get("status") != "complete":
        raise ValueError("entry needs explicit completion or gates")
    audit_scope = "technical_gates_only_no_separate_reconstruction_audit"
    if audit_name:
        path = folder / audit_name
        audit = json.loads(path.read_text())
        passed = audit.get("status") == "pass" or audit.get("passed") is True or (
            audit.get("all_output_hashes_verified") is True and audit.get("all_cache_hashes_verified") is True)
        if not passed or not linked(audit, manifest):
            raise ValueError(f"passing manifest-linked audit required: {name}")
        record(path, "reconstruction_audit")
        audit_scope = audit.get("verification_scope", audit.get("audit_scope", audit.get("scope", "see linked audit")))
    for filename, digest in meta.get("output_sha256", {}).items():
        record(folder / filename, "scoring_output", digest)
    if report_name:
        path = folder / report_name
        report = json.loads(path.read_text())
        if not linked(report, manifest):
            raise ValueError(f"report is not linked to scoring: {name}")
        record(path, "report_manifest")
        for filename, digest in report.get("output_sha256", {}).items():
            record(path.parent / filename, "report_output", digest)
    row = {"study": name, "manifest": str(manifest), "manifest_sha256": file_sha256(manifest),
        "code_commit": meta.get("code_commit", "unavailable"), "git_dirty": meta.get("git_dirty"),
        "audit": str(folder / audit_name) if audit_name else "", "audit_scope": audit_scope,
        "optional_gate_failures": optional_gate_failures,
        "report": str(folder / report_name) if report_name else "", "indexed_files": len(files),
        "historical_output_hashes_verified": sum(f["matches_recorded_hash"] for f in files), "registry_status": "pass"}
    return row, files


def run(root, output, studies=STUDIES):
    if not studies or len({s[0] for s in studies}) != len(studies):
        raise ValueError("nonempty unique study registry required")
    rows, files = [], []
    for spec in studies:
        row, indexed = verify_entry(root, spec)
        rows.append(row)
        files.extend(indexed)
        print(json.dumps({"study": spec[0], "status": "pass", "files": len(indexed)}), flush=True)
    output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(rows).to_csv(output / "replay_coverage_artifact_registry.csv", index=False)
    pd.DataFrame(files).to_csv(output / "replay_coverage_artifact_files.csv", index=False)
    lines = ["# Recording-Coverage Artifact Index", "", "This non-rescoring index verifies production completion/technical gates, links existing reconstruction audits to their exact scoring manifest, and checks recorded top-level scoring/report output hashes. It does not repeat those scientific audits, verify all raw inputs again, or establish biological truth. Early input/subsampling/likelihood artifacts have no separate reconstruction audit and are explicitly labeled; missing historical output hashes are not invented.", "",
        "|Study|Registry check|Scoring commit|Files indexed|Historically recorded output hashes verified|", "|---|---|---|---:|---:|"]
    for r in rows:
        lines.append(f"|{r['study']}|{r['registry_status']}|{r['code_commit'][:12]}|{r['indexed_files']}|{r['historical_output_hashes_verified']}|")
    lines += ["", "Absolute server paths, hashes and each audit's actual scope are in the CSV files. The evidence matrix and integrated manuscript interpret the results; registry pass means artifact integrity, not a positive scientific claim or guaranteed publication."]
    (output / "replay_coverage_artifact_index.md").write_text("\n".join(lines) + "\n")
    provenance = build_script_provenance(input_paths={"script": Path(__file__), **{f"manifest_{r['study']}": Path(r["manifest"]) for r in rows}}, cwd=ROOT)
    provenance.update(status="complete", studies=len(rows), indexed_files=len(files),
        verification_scope="existing technical gates, linked audit records and recorded top-level output hashes; no new scientific reconstruction",
        output_sha256={p.name: file_sha256(p) for p in output.iterdir() if p.is_file()})
    (output / "replay_coverage_artifact_index_manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.artifact_root, args.output_dir)
