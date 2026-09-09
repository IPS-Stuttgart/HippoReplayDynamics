import json

import pandas as pd
import pytest

from scripts._provenance import file_sha256
from scripts.report_2d_physical_neural_metric import ORACLE, VALUE, report


def fixture(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    animals = pd.DataFrame(
        [{"dataset": d, "animal": a, "horizon": 2, VALUE: value, ORACLE: 0.05} for d, a, value in [("pfeiffer_foster", "Rat1", 0.02), ("tanni2022", "R2470", -0.01)]]
    )
    summary = animals.drop(columns="animal").assign(animals=1, positive_animals=[1, 0])
    decision = pd.DataFrame(
        [
            {
                "horizon_ms": 40,
                "necessary_oracle_screen_pass": False,
                "real_event_scoring_performed": False,
                "real_replay_mechanism_established": False,
                "high_importance_discovery_established": False,
            }
        ]
    )
    for name, frame in (("animals", animals), ("summary", summary), ("decision", decision)):
        frame.to_csv(run / f"physical_neural_metric_{name}.csv", index=False)
    m = {"status": "complete", "code_commit": "fixture", "output_sha256": {p.name: file_sha256(p) for p in run.iterdir()}}
    mp = run / "physical_neural_metric_manifest.json"
    mp.write_text(json.dumps(m))
    audit = tmp_path / "audit.json"
    audit.write_text(
        json.dumps(
            {
                "status": "pass",
                "input_file_sha256": {"run_manifest": file_sha256(mp)},
                "kernels_checked": 14,
                "expected_scores_checked": 120,
                "max_absolute_score_error": 1e-12,
                "scope": "Synthetic reporter fixture only.",
            }
        )
    )
    return run, audit


def test_report_is_read_only_and_preserves_failure(tmp_path):
    run, audit = fixture(tmp_path)
    before = {p.name: file_sha256(p) for p in run.iterdir()}
    out = tmp_path / "report"
    result = report(run, audit, out)
    assert "fails" in result["conclusion"]
    assert {p.name: file_sha256(p) for p in run.iterdir()} == before
    assert (out / "physical_neural_metric_screen.png").stat().st_size > 10000
    text = (out / "physical_neural_metric_report.md").read_text()
    assert "not a mathematical" in text
    assert "not biological significance" in text
    for name, digest in result["output_sha256"].items():
        assert file_sha256(out / name) == digest


def test_report_rejects_changed_scores_and_wrong_audit(tmp_path):
    run, audit = fixture(tmp_path)
    a = json.loads(audit.read_text())
    a["status"] = "fail"
    audit.write_text(json.dumps(a))
    with pytest.raises(ValueError):
        report(run, audit, tmp_path / "bad_audit")
    a["status"] = "pass"
    audit.write_text(json.dumps(a))
    (run / "physical_neural_metric_summary.csv").write_text("corrupt")
    with pytest.raises(ValueError):
        report(run, audit, tmp_path / "bad_data")
