import json

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from scripts._provenance import file_sha256
from scripts.report_2d_lagged_neural_prediction import BASELINES, DATASETS, MODELS, interpret, report


def decisions():
    return pd.DataFrame(
        [
            {
                "model": m,
                "replicated_forecasting_lead": False,
                "spatial_route_specific_lead": False,
                "independent_confirmation": False,
                "high_importance_discovery_established": False,
            }
            for m in MODELS
        ]
    )


def test_interpretation_cannot_promote_an_exploratory_lead():
    d = decisions()
    assert interpret(d).startswith("No model")
    d.loc[0, "replicated_forecasting_lead"] = True
    assert "requiring independent confirmation" in interpret(d)
    d.loc[0, "independent_confirmation"] = True
    with pytest.raises(ValueError, match="exploratory"):
        interpret(d)
    with pytest.raises(ValueError, match="complete unique"):
        interpret(d.iloc[0:0])


def fixture(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    rows, coverage = [], []
    for dataset in DATASETS:
        for h in (1, 2, 4):
            coverage.append(
                {"dataset": dataset, "horizon": h, "animal": "A", "session": "S", "selected_events": 5, "eligible_events": 5, "target_bins": 50 - h, "discarded_partial_spikes": 2}
            )
            for model in MODELS:
                for b in BASELINES:
                    rows.append(
                        {
                            "dataset": dataset,
                            "horizon": h,
                            "model": model,
                            "contrast": model + "__dynamic_minus_" + b,
                            "metric": "delta_per_spike",
                            "mean": 0.03 / h,
                            "ci_low": -0.04,
                            "ci_high": 0.08,
                            "positive_animals": 2,
                            "animals": 4,
                        }
                    )
    s = pd.DataFrame(rows)
    s.to_csv(run / "lagged_prediction_summary.csv.gz", index=False)
    s.assign(animal="A").to_csv(run / "lagged_prediction_animals.csv.gz", index=False)
    pd.DataFrame(coverage).to_csv(run / "lagged_prediction_coverage.csv", index=False)
    decisions().to_csv(run / "lagged_prediction_decisions.csv", index=False)
    m = {"status": "complete", "code_commit": "synthetic-fixture", "output_sha256": {p.name: file_sha256(p) for p in run.iterdir()}}
    mp = run / "lagged_prediction_manifest.json"
    mp.write_text(json.dumps(m))
    audit = tmp_path / "audit.json"
    audit.write_text(
        json.dumps(
            {
                "status": "pass",
                "input_file_sha256": {"run_manifest": file_sha256(mp)},
                "independent_score_reconstructions": 30,
                "sampled_events": 1,
                "maximum_absolute_score_error": 1e-13,
                "scope": "Synthetic test only.",
            }
        )
    )
    return run, audit


def test_non_rescoring_report_outputs_and_figures(tmp_path):
    run, audit = fixture(tmp_path)
    before = {p.name: file_sha256(p) for p in run.iterdir()}
    out = tmp_path / "report"
    report(run, audit, out)
    assert before == {p.name: file_sha256(p) for p in run.iterdir()}
    text = (out / "lagged_prediction_report.md").read_text()
    assert "not all validated continuous replay" in text
    assert "No model met" in text
    for path in out.glob("*.png"):
        with Image.open(path) as image:
            pixels = np.asarray(image.convert("RGB"))
            assert min(image.size) > 1000 and pixels.std() > 5
    provenance = json.loads((out / "lagged_prediction_report_manifest.json").read_text())
    assert provenance["non_rescoring"] and not provenance["high_importance_discovery_established"]
    assert len(provenance["output_sha256"]) == 8
    for name, digest in provenance["output_sha256"].items():
        assert file_sha256(out / name) == digest


def test_report_rejects_wrong_audit(tmp_path):
    run, audit = fixture(tmp_path)
    a = json.loads(audit.read_text())
    a["input_file_sha256"]["run_manifest"] = "mismatched"
    audit.write_text(json.dumps(a))
    with pytest.raises(ValueError, match="matching passing"):
        report(run, audit, tmp_path / "bad")
