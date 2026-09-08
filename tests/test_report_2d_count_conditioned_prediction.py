import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts.report_2d_count_conditioned_prediction import DATASETS, PRIMARY, SECONDARY, decision, file_sha256, readout, report_text, run, validate


def fixture_summary():
    rows = []
    for dataset, (_, events, sessions, animals) in DATASETS.items():
        for contrast in PRIMARY | SECONDARY:
            rows.append(
                {
                    "dataset": dataset,
                    "contrast": contrast,
                    "events": events,
                    "sessions": sessions,
                    "animals": animals,
                    "positive_animals": animals,
                    "mean": 1.0,
                    "ci_low": 0.2,
                    "ci_high": 2.0,
                    "mean_per_heldout_spike": 0.1,
                    "per_spike_ci_low": 0.02,
                    "per_spike_ci_high": 0.2,
                }
            )
    return pd.DataFrame(rows)


def test_all_four_gates_needed():
    table = fixture_summary()
    assert decision(table).all_four_primary_gates_passed.all()
    mask = table.dataset.eq("tanni2022") & table.contrast.eq("imm_minus_event_global")
    table.loc[mask, "ci_low"] = -0.2
    verdict = decision(table).set_index("dataset").loc["tanni2022"]
    assert not verdict.bounded_external_replication_established
    assert verdict.temporal_vs_independent_supported
    assert verdict.failed_primary_contrasts == "imm_minus_event_global"
    assert readout(table).loc[mask, "per_spike_interval_positive"].all()


def test_animal_uniformity_is_not_optional():
    table = fixture_summary()
    table.loc[table.dataset.eq("tanni2022") & table.contrast.eq("real_minus_wrong_imm"), "positive_animals"] = 4
    assert not decision(table).set_index("dataset").loc["tanni2022", "all_four_primary_gates_passed"]


@pytest.mark.parametrize("damage", ["empty", "missing", "duplicate", "zero", "nonfinite"])
def test_malformed_readout_fails(damage):
    table = fixture_summary()
    if damage == "empty":
        table = table.iloc[:0]
    elif damage == "missing":
        table = table.iloc[1:]
    elif damage == "duplicate":
        table = pd.concat([table, table.iloc[:1]])
    elif damage == "zero":
        table.loc[0, "events"] = 0
    else:
        table.loc[0, "ci_low"] = np.nan
    with pytest.raises(ValueError):
        readout(table)


def artifacts(tmp_path):
    root, audit_dir = tmp_path / "scores", tmp_path / "audit"
    root.mkdir()
    audit_dir.mkdir()
    table = fixture_summary()
    table.to_csv(root / "conditional_2d_summary.csv", index=False)
    rows = []
    for r in table.itertuples(index=False):
        for index in range(r.animals):
            rows.append({"dataset": r.dataset, "animal": f"animal{index}", "contrast": r.contrast, "delta": 1.0, "delta_per_heldout_spike": 0.1})
    pd.DataFrame(rows).to_csv(root / "conditional_2d_by_animal.csv", index=False)
    pd.DataFrame(rows).assign(session="fixture").to_csv(root / "conditional_2d_by_session.csv", index=False)
    pd.DataFrame({"gate": ["complete_nonleaking_proper_prediction", "overall"], "passed": [True, True]}).to_csv(root / "conditional_2d_gates.csv", index=False)
    manifest = {"status": "complete", "code_commit": "synthetic_fixture", "output_sha256": {p.name: file_sha256(p) for p in root.iterdir()}}
    path = root / "conditional_2d_manifest.json"
    path.write_text(json.dumps(manifest))
    audit = {
        "status": "pass",
        "code_commit": "synthetic_fixture",
        "input_file_sha256": {"run_manifest": file_sha256(path)},
        "output_sha256": {},
        "raw_events": 9225,
        "sessions": 33,
        "analytic_scores": 184500,
        "global_scores": 184500,
        "dynamic_scores": 792,
        "split_contrasts": 415125,
        "event_contrasts": 83025,
        "bootstrap_panels": 18,
        "max_dynamic_error": 1e-12,
    }
    audit_path = audit_dir / "conditional_2d_audit.json"
    audit_path.write_text(json.dumps(audit))
    return root, audit_path, manifest, audit


def test_full_non_rescoring_report(tmp_path):
    root, audit_path, _, _ = artifacts(tmp_path)
    output = tmp_path / "report"
    run(SimpleNamespace(run_dir=root, audit_manifest=audit_path, output_dir=output))
    assert (output / "conditional_2d_prediction.png").stat().st_size > 10000
    manifest = json.loads((output / "conditional_2d_report_manifest.json").read_text())
    assert manifest["non_rescoring"]
    assert not any(d["new_biological_mechanism_established"] for d in manifest["decisions"])
    with pytest.raises(ValueError, match="overwrite"):
        run(SimpleNamespace(run_dir=root, audit_manifest=audit_path, output_dir=output))


@pytest.mark.parametrize("damage", ["wrong_run", "incomplete", "tamper", "failed"])
def test_report_requires_matching_complete_audit(tmp_path, damage):
    root, audit_path, _, audit = artifacts(tmp_path)
    if damage == "wrong_run":
        audit["input_file_sha256"]["run_manifest"] = "wrong"
    elif damage == "incomplete":
        audit["raw_events"] = 0
    elif damage == "failed":
        audit["status"] = "fail"
    else:
        (root / "conditional_2d_summary.csv").write_text("changed")
    audit_path.write_text(json.dumps(audit))
    with pytest.raises(ValueError):
        validate(root, audit_path)


def test_report_does_not_hardcode_positive_conclusion():
    table = fixture_summary()
    table.loc[table.dataset.eq("tanni2022"), "ci_low"] = -1
    table = readout(table)
    audit = {
        "code_commit": "fixture",
        "raw_events": 9225,
        "sessions": 33,
        "analytic_scores": 184500,
        "global_scores": 184500,
        "dynamic_scores": 792,
        "max_dynamic_error": 1e-12,
        "split_contrasts": 415125,
        "event_contrasts": 83025,
        "bootstrap_panels": 18,
    }
    text = report_text(table, decision(table), {"code_commit": "fixture"}, audit)
    assert "Tanni: `report_partial_predictive_result_without_full_replication`" in text
    assert "not replaced by a favorable per-spike sensitivity" in text
