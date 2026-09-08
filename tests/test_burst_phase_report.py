import json

import pandas as pd
import pytest

from scripts.report_2d_burst_phase_prediction import DATASETS, LABELS, classifications, load_verified, validate_summary


def fixture():
    rows = []
    for d in DATASETS:
        for c in LABELS:
            for n in (5,) if c == "phase_order_advantage" else (3, 5, 10):
                for metric in ("delta", "delta_per_spike"):
                    animals = 4 if d == "pfeiffer_foster" else 5
                    rows.append(
                        {"dataset": d, "contrast": c, "n_knots": n, "metric": metric, "mean": 0.1, "ci_low": 0.01, "ci_high": 0.2, "positive_animals": animals, "animals": animals}
                    )
    return pd.DataFrame(rows)


def test_positive_lead_is_not_unique_mechanism():
    result = classifications(fixture())
    assert result.recruitment_lead.all()
    assert not result.unique_spatial_mechanism_established.any()
    assert not result.independent_confirmation.any()


def test_shuffle_pass_cannot_rescue_failed_predictive_comparator():
    x = fixture()
    selected = x.dataset.eq("tanni2022") & x.contrast.eq("phase_minus_phase_averaged") & x.n_knots.eq(5) & x.metric.eq("delta_per_spike")
    x.loc[selected, ["mean", "ci_low", "ci_high", "positive_animals"]] = [-0.1, -0.2, -0.01, 0]
    result = classifications(x).set_index("dataset")
    assert not result.loc["tanni2022", "recruitment_lead"]
    assert result.loc["tanni2022", "status"] == "order_sensitive_without_complete_predictive_support"
    assert result.loc["pfeiffer_foster", "recruitment_lead"]


@pytest.mark.parametrize("mutation", ["empty", "duplicate", "missing", "nan", "animals"])
def test_malformed_summaries_fail(mutation):
    x = fixture()
    if mutation == "empty":
        x = x.iloc[:0]
    elif mutation == "duplicate":
        x = pd.concat([x, x.iloc[:1]])
    elif mutation == "missing":
        x = x.iloc[1:]
    elif mutation == "nan":
        x.loc[0, "mean"] = float("nan")
    else:
        x.loc[0, "animals"] = 0
    with pytest.raises(ValueError):
        validate_summary(x)


def test_mismatched_audit_cannot_authorize_report(tmp_path):
    (tmp_path / "burst_phase_manifest.json").write_text(json.dumps({"status": "complete", "gates": {"all_events_present": True}}))
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({"status": "pass", "input_file_sha256": {"source_manifest": "wrong"}}))
    with pytest.raises(ValueError, match="matching passing"):
        load_verified(tmp_path, audit)
