import json

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from test_reversible_neural_forecasts import fixture_rows

from scripts._provenance import file_sha256
from scripts.audit_2d_reversible_neural_forecasts import aggregate, decision
from scripts.report_2d_reversible_neural_forecasts import interpret, report


def fixture(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    rows = fixture_rows()
    tables = aggregate(rows)
    for name, t in zip(("splits", "events", "sessions", "animals", "summary"), tables, strict=True):
        t.to_csv(run / f"reversible_forecasts_{name}.csv.gz", index=False)
    pd.DataFrame([decision(tables[-1])]).to_csv(run / "reversible_forecasts_decision.csv", index=False)
    m = {
        "status": "complete",
        "events": 27,
        "rows": len(rows),
        "completed": list(range(9)),
        "code_commit": "synthetic-fixture",
        "output_sha256": {p.name: file_sha256(p) for p in run.iterdir()},
    }
    mp = run / "reversible_forecasts_manifest.json"
    mp.write_text(json.dumps(m))
    audit = tmp_path / "audit.json"
    audit.write_text(
        json.dumps(
            {
                "status": "pass",
                "input_file_sha256": {"run_manifest": file_sha256(mp)},
                "independent_scores": 10,
                "max_score_error": 1e-13,
                "scope": "Synthetic reporter fixture only.",
            }
        )
    )
    return run, audit


def test_report_is_read_only_and_does_not_promote_exploration(tmp_path):
    run, audit = fixture(tmp_path)
    before = {p.name: file_sha256(p) for p in run.iterdir()}
    out = tmp_path / "report"
    p = report(run, audit, out)
    assert before == {p.name: file_sha256(p) for p in run.iterdir()}
    assert p["non_rescoring"] and not p["high_importance_discovery_established"]
    text = (out / "reversible_forecasts_report.md").read_text()
    assert "requires independent confirmation" in text and "not all validated replay" in text
    assert "not substitutes" in text
    primary = pd.read_csv(out / "reversible_forecasts_primary_table.csv")
    assert len(primary) == 30
    with Image.open(out / "reversible_forecasts_primary.png") as im:
        assert min(im.size) >= 1000 and np.asarray(im.convert("RGB")).std() > 5
    assert len(p["output_sha256"]) == 6
    for name, digest in p["output_sha256"].items():
        assert file_sha256(out / name) == digest


def test_report_rejects_altered_scores_or_wrong_audit(tmp_path):
    run, audit = fixture(tmp_path)
    text = audit.read_text()
    a = json.loads(text)
    a["input_file_sha256"]["run_manifest"] = "wrong"
    audit.write_text(json.dumps(a))
    with pytest.raises(ValueError, match="matching passing"):
        report(run, audit, tmp_path / "bad")
    audit.write_text(text)
    with (run / "reversible_forecasts_decision.csv").open("a") as stream:
        stream.write("corrupt\n")
    with pytest.raises(ValueError, match="changed scored"):
        report(run, audit, tmp_path / "bad2")


def test_primary_model_cannot_be_replaced_by_spatial_diagnostic(tmp_path):
    run, _ = fixture(tmp_path)
    d = pd.read_csv(run / "reversible_forecasts_decision.csv")
    d.directional_predictive_lead = False
    assert "not met" in interpret(d)
    wrong = d.copy()
    wrong.primary_model = "spatial_imm_real"
    with pytest.raises(ValueError, match="fixed primary"):
        interpret(wrong)
    d.independent_confirmation = True
    with pytest.raises(ValueError, match="cannot claim"):
        interpret(d)
