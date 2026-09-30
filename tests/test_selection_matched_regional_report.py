import numpy as np
import pandas as pd

from scripts.report_selection_matched_regional_calibration import paired_checks, real_status, summarize_validation


def fixture():
    rows = []
    for phase, prevalence in (("null", .30), ("validation", .30), ("validation", .38)):
        for replica in range(20):
            for side in ("high", "low"):
                rows.append({"animal": "Rat1", "session": "S1", "family": "targeted", "side": side,
                    "population": "targeted_"+side, "phase": phase, "requested_prevalence": prevalence,
                    "generator": "mix", "perturbation": "none", "replica": replica, "prevalence": prevalence,
                    "true_prevalence": prevalence, "absolute_error": 0., "signed_error": 0., "flagged": False,
                    "events": 200, "status": "fit"})
    return pd.DataFrame(rows)


def test_same_truth_agreement_and_known_eight_point_power():
    pair, agreement, power = paired_checks(fixture())
    assert len(pair) == 60
    assert agreement.agreement_pass.all() and power.power_pass.all()
    np.testing.assert_allclose(power.mean_true_retained_difference, .08)
    assert power.positive_difference_detection_fraction.eq(1).all()


def test_bad_budget_is_not_hidden_by_mean_across_conditions():
    f = fixture()
    mask = f.phase.eq("validation") & f.side.eq("low")
    f.loc[mask, "absolute_error"] = .12
    result = summarize_validation(f)
    assert not result.loc[result.side.eq("low"), "budget_pass"].any()
    assert result.loc[result.side.eq("high"), "budget_pass"].all()


def test_no_real_calibrated_claim_after_failed_validation_or_real_flag():
    data = pd.DataFrame({"fit_tv": [.01, .3, .01], "tv_cutoff": [.1]*3,
        "calibrated_estimate_available": [False, True, True]})
    result = real_status(data)
    assert result.calibrated_claim_available.tolist() == [False, False, True]
    assert result.enrichment_claim.eq("not_established").all()


def test_csv_stage_null_is_not_missing(tmp_path):
    from scripts.measure_selection_matched_regional_calibration import read_csv
    path = tmp_path/"stages.csv"
    pd.DataFrame({"phase": ["null", "validation"], "value": [1, 2]}).to_csv(path, index=False)
    loaded = read_csv(path)
    assert loaded.phase.tolist() == ["null", "validation"]
