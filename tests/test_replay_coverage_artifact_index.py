import json

import pandas as pd
import pytest
from scripts._provenance import file_sha256
from scripts.index_replay_coverage_artifacts import check_gates, run, verify_entry


def test_registry_checks_audit_link_output_hash_and_nonempty_gates(tmp_path):
    d = tmp_path / "study"
    d.mkdir()
    table = d / "summary.csv"
    table.write_text("value\n1\n")
    gate = d / "gate.csv"
    pd.DataFrame([{"gate": "overall", "passed": True}]).to_csv(gate, index=False)
    manifest = d / "manifest.json"
    manifest.write_text(json.dumps({"status": "complete", "code_commit": "abc", "output_sha256": {table.name: file_sha256(table)}}))
    audit = d / "audit.json"
    audit.write_text(json.dumps({"status": "pass", "input_file_paths": {"manifest": str(manifest)}, "input_file_sha256": {"manifest": file_sha256(manifest)}}))
    spec = ("example", "study", manifest.name, audit.name, gate.name, None)
    row, _ = verify_entry(tmp_path, spec)
    assert row["historical_output_hashes_verified"] == 1
    run(tmp_path, tmp_path / "report", [spec])
    assert (tmp_path / "report/replay_coverage_artifact_index_manifest.json").exists()
    table.write_text("value\n2\n")
    with pytest.raises(ValueError, match="changed"):
        verify_entry(tmp_path, spec)
    table.write_text("value\n1\n")
    audit.write_text(json.dumps({"status": "pass"}))
    with pytest.raises(ValueError, match="linked"):
        verify_entry(tmp_path, spec)
    with pytest.raises(ValueError, match="nonempty"):
        run(tmp_path, tmp_path / "empty", [])


def test_gate_alias_and_explicit_optional_unavailability():
    assert check_gates(pd.DataFrame([{"gate": "overall", "pass": True}])) == ""
    gates = pd.DataFrame([{"gate": "complete", "passed": True, "required_for_preparation": True},
        {"gate": "all_lfp_available", "passed": False, "required_for_preparation": False}])
    assert check_gates(gates) == "all_lfp_available"
    gates.loc[1, "required_for_preparation"] = True
    with pytest.raises(ValueError, match="required"):
        check_gates(gates)
