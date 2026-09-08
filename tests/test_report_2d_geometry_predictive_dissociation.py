import json

import pytest

from scripts.report_2d_geometry_predictive_dissociation import file_sha256, validate


@pytest.mark.parametrize("problem", [None, "fail", "wrong_hash", "missing_coverage", "tamper"])
def test_report_requires_matching_complete_audit(tmp_path, problem):
    root, out = tmp_path / "run", tmp_path / "audit"
    root.mkdir()
    out.mkdir()
    payload = root / "table.csv"
    payload.write_text("x\n1\n")
    manifest = root / "geometry_predictive_manifest.json"
    manifest.write_text(json.dumps({"status": "complete", "output_sha256": {payload.name: file_sha256(payload)}}))
    check = out / "checks.csv"
    check.write_text("passed\n")
    audit = {
        "status": "pass",
        "input_file_sha256": {"run_manifest": file_sha256(manifest)},
        "sessions": 33,
        "events": 9225,
        "labels": 184500,
        "predictions": 46125,
        "summary_rows": 840,
        "primary_interval_rows": 42,
        "output_sha256": {check.name: file_sha256(check)},
    }
    if problem == "fail":
        audit["status"] = "fail"
    elif problem == "wrong_hash":
        audit["input_file_sha256"]["run_manifest"] = "wrong"
    elif problem == "missing_coverage":
        audit["labels"] -= 1
    elif problem == "tamper":
        payload.write_text("changed\n")
    path = out / "audit.json"
    path.write_text(json.dumps(audit))
    if problem:
        with pytest.raises(ValueError):
            validate(root, path)
    else:
        validate(root, path)
