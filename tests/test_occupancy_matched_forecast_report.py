import json

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from scripts._provenance import file_sha256
from scripts.report_2d_occupancy_matched_forecasts import DATASETS, MODELS, interpret, report


def fixture(tmp_path):
    run, old = tmp_path / "run", tmp_path / "old"
    run.mkdir()
    old.mkdir()
    summary, previous, animals, events, coverage = [], [], [], [], []
    for dataset, n in zip(DATASETS, (4, 5), strict=True):
        for h in (1, 2, 4):
            for animal in range(n):
                coverage.append({"dataset": dataset, "horizon": h, "animal": f"A{animal}", "session": f"S{animal}", "selected_events": 3, "eligible_events": 3})
                for event in range(3):
                    events.append(
                        {
                            "dataset": dataset,
                            "horizon": h,
                            "event_id": event,
                            "animal": f"A{animal}",
                            "session": f"S{animal}",
                            "contrast": "learned_hmm__dynamic_minus_matched",
                            "valid_neural_splits": 0 if event == 2 else 5,
                        }
                    )
            for model in MODELS:
                row = {
                    "dataset": dataset,
                    "horizon": h,
                    "contrast": model + "__dynamic_minus_matched",
                    "metric": "delta_per_spike",
                    "mean": 0.02,
                    "ci_low": -0.01,
                    "ci_high": 0.05,
                    "positive_animals": n - 1,
                    "animals": n,
                }
                summary.append(row)
                previous.append(row | {"contrast": model + "__dynamic_minus_dwell_only", "mean": 0.025})
                for animal in range(n):
                    animals.append(row | {"animal": f"A{animal}", "delta_per_spike": 0.03 if animal else -0.01})
    pd.DataFrame(summary).to_csv(run / "occupancy_matched_summary.csv.gz", index=False)
    pd.DataFrame(animals).to_csv(run / "occupancy_matched_animals.csv.gz", index=False)
    pd.DataFrame(events).to_csv(run / "occupancy_matched_events.csv.gz", index=False)
    pd.DataFrame(previous).to_csv(old / "lagged_prediction_summary.csv.gz", index=False)
    pd.DataFrame(coverage).to_csv(old / "lagged_prediction_coverage.csv", index=False)
    decisions = pd.DataFrame(
        [
            {
                "model": model,
                "matched_destination_structure_lead": False,
                "other_original_controls_pass": True,
                "all_updated_controls_pass": False,
                "original_compound_verdict_changed": False,
                "independent_confirmation": False,
                "high_importance_discovery_established": False,
            }
            for model in MODELS
        ]
    )
    decisions.to_csv(run / "occupancy_matched_decisions.csv", index=False)
    op = old / "lagged_prediction_manifest.json"
    op.write_text(json.dumps({"status": "complete", "output_sha256": {p.name: file_sha256(p) for p in old.iterdir()}}))
    mp = run / "occupancy_matched_manifest.json"
    mp.write_text(
        json.dumps(
            {
                "status": "complete",
                "code_commit": "synthetic-fixture",
                "input_file_paths": {"lagged_manifest": str(op)},
                "input_file_sha256": {"lagged_manifest": file_sha256(op)},
                "output_sha256": {p.name: file_sha256(p) for p in run.iterdir()},
            }
        )
    )
    audit = tmp_path / "audit.json"
    audit.write_text(
        json.dumps(
            {
                "status": "pass",
                "input_file_sha256": {"run_manifest": file_sha256(mp)},
                "independent_scores": 10,
                "parameter_sets": 3,
                "max_score_error": 1e-13,
                "max_equilibrium_error": 1e-14,
                "scope": "Synthetic reporter fixture only.",
            }
        )
    )
    return run, audit, decisions


def test_report_preserves_inputs_and_exposes_negative_animals(tmp_path):
    run, audit, _ = fixture(tmp_path)
    before = {p.name: file_sha256(p) for p in run.iterdir()}
    out = tmp_path / "report"
    provenance = report(run, audit, out)
    assert before == {p.name: file_sha256(p) for p in run.iterdir()}
    assert provenance["non_rescoring"] and not provenance["high_importance_discovery_established"]
    assert "No model passes" in provenance["classification"]
    text = (out / "occupancy_matched_report.md").read_text()
    assert "not all validated replay" in text and "No change to the earlier failed" in text
    table = pd.read_csv(out / "occupancy_matched_primary_table.csv")
    assert len(table) == 12
    assert table[table.control.eq("Occupancy + dwell null")]["mean"].eq(0.02).all()
    coverage = pd.read_csv(out / "occupancy_matched_coverage.csv")
    assert coverage.per_spike_supported_events.eq(coverage.animals * 2).all()
    assert pd.read_csv(out / "occupancy_matched_by_animal.csv").delta_per_spike.lt(0).any()
    images = list(out.glob("*.png"))
    assert len(images) == 2
    for path in images:
        with Image.open(path) as image:
            assert min(image.size) >= 1000 and np.asarray(image.convert("RGB")).std() > 5
    assert len(provenance["output_sha256"]) == 8
    for name, digest in provenance["output_sha256"].items():
        assert file_sha256(out / name) == digest


def test_interpretation_keeps_original_verdict_and_confirmation_separate(tmp_path):
    _, _, d = fixture(tmp_path)
    with pytest.raises(ValueError, match="complete unique"):
        interpret(d.iloc[:0])
    d.loc[0, "matched_destination_structure_lead"] = True
    with pytest.raises(ValueError, match="inconsistent"):
        interpret(d)
    d.loc[0, "all_updated_controls_pass"] = True
    assert "requiring independent confirmation" in interpret(d)
    for column in ("original_compound_verdict_changed", "independent_confirmation", "high_importance_discovery_established"):
        changed = d.copy()
        changed.loc[0, column] = True
        with pytest.raises(ValueError, match="cannot promote"):
            interpret(changed)


def test_report_rejects_mismatched_audit_and_changed_scored_outputs(tmp_path):
    run, audit, _ = fixture(tmp_path)
    original = audit.read_text()
    a = json.loads(original)
    a["input_file_sha256"]["run_manifest"] = "wrong"
    audit.write_text(json.dumps(a))
    with pytest.raises(ValueError, match="matching passing"):
        report(run, audit, tmp_path / "bad")
    audit.write_text(original)
    with (run / "occupancy_matched_decisions.csv").open("a") as stream:
        stream.write("corrupted\n")
    with pytest.raises(ValueError, match="changed scored"):
        report(run, audit, tmp_path / "bad2")
