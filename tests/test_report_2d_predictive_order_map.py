import json

import numpy as np
import pandas as pd
import pytest

from scripts.report_2d_predictive_order_map import CONTRASTS, DATASETS, FACTORS, SENSITIVITIES, decision, file_sha256, readout, validate


def summary():
    return pd.DataFrame(
        [
            {
                "dataset": d,
                "contrast": m + "__" + c,
                "events": n,
                "sessions": s,
                "animals": a,
                "positive_animals": a,
                "mean": 2.0,
                "ci_low": 1.0,
                "ci_high": 3.0,
                "mean_per_heldout_spike": 0.2,
                "per_spike_ci_low": 0.1,
                "per_spike_ci_high": 0.3,
            }
            for d, (_, n, s, a) in DATASETS.items()
            for m in ("diffusion", "first_order_imm")
            for c in (FACTORS | CONTRASTS | SENSITIVITIES)
        ]
    )


def test_positive_is_bounded_not_parent_rescue():
    rows = decision(summary())
    assert rows.order_and_adjacency_gate_passed.all()
    assert not rows.parent_replication_gate_changed.any()
    assert not rows.new_mechanism_established.any()


@pytest.mark.parametrize("field,value", [("ci_low", -0.01), ("positive_animals", 4), ("mean", -0.1)])
def test_raw_primary_failure_not_rescued_by_other_contrasts(field, value):
    table = summary()
    table.loc[table.dataset.eq("tanni2022") & table.contrast.eq("first_order_imm__order_map_interaction"), field] = value
    rows = decision(table).set_index("dataset")
    assert rows.loc["pfeiffer_foster", "order_and_adjacency_gate_passed"]
    assert not rows.loc["tanni2022", "order_and_adjacency_gate_passed"]


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "denominator", "nan", "interval"])
def test_bad_tables_fail(mutation):
    table = summary()
    if mutation == "missing":
        table = table.iloc[1:]
    elif mutation == "duplicate":
        table = pd.concat([table, table.iloc[:1]], ignore_index=True)
    elif mutation == "denominator":
        table.loc[0, "events"] -= 1
    elif mutation == "nan":
        table.loc[0, "mean"] = np.nan
    else:
        table.loc[0, "ci_low"] = 10
    with pytest.raises(ValueError):
        readout(table)


def audit_fixture(tmp_path):
    raw, checked = tmp_path / "raw", tmp_path / "audit"
    raw.mkdir()
    checked.mkdir()
    (raw / "scores.csv").write_text("fixture\n")
    (checked / "checks.csv").write_text("fixture\n")
    manifest = raw / "predictive_order_map_manifest.json"
    manifest.write_text(json.dumps({"status": "complete", "k": 20, "output_sha256": {"scores.csv": file_sha256(raw / "scores.csv")}}))
    audit = {
        "status": "pass",
        "input_file_sha256": {"run_manifest": file_sha256(manifest)},
        "sessions": 33,
        "events": 9225,
        "score_rows": 1845000,
        "invariant_scores": 3690000,
        "permutations": 184500,
        "independent_predictions": 1584,
        "split_contrasts": 830250,
        "event_contrasts": 166050,
        "bootstrap_panels": 36,
        "max_prediction_error": 1e-10,
        "output_sha256": {"checks.csv": file_sha256(checked / "checks.csv")},
    }
    return raw, checked / "audit.json", audit


@pytest.mark.parametrize("mutation", [None, "failed", "count", "mismatch", "changed_output", "changed_audit", "error"])
def test_audit_gate(tmp_path, mutation):
    root, path, audit = audit_fixture(tmp_path)
    if mutation == "failed":
        audit["status"] = "fail"
    elif mutation == "count":
        audit["permutations"] -= 1
    elif mutation == "mismatch":
        audit["input_file_sha256"]["run_manifest"] = "wrong"
    elif mutation == "changed_output":
        (root / "scores.csv").write_text("changed\n")
    elif mutation == "changed_audit":
        (path.parent / "checks.csv").write_text("changed\n")
    elif mutation == "error":
        audit["max_prediction_error"] = 0.1
    path.write_text(json.dumps(audit))
    if mutation:
        with pytest.raises(ValueError):
            validate(root, path)
    else:
        validate(root, path)
