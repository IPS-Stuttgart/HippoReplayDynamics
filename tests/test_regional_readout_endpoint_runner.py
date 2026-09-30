import numpy as np
import pandas as pd

from scripts.diagnose_regional_readout_endpoint import analyze_bank, read_csv, summarize, validation_mask


def test_literal_null_stage_and_seed_preserved(tmp_path):
    path = tmp_path/"stages.csv"
    path.write_text("phase,seed\nnull,18446744073709551123\nvalidation,12\n")
    result = read_csv(path)
    assert result.phase.iloc[0] == "null"
    assert result.seed.iloc[0] == "18446744073709551123"


def test_validation_selector_excludes_null_and_power_panel():
    frame = pd.DataFrame({"phase": ["calibration", "null", "validation", "validation"],
                          "requested_prevalence": [.5, .3, .3, .38]})
    np.testing.assert_array_equal(validation_mask(frame), [False, False, True, False])


def test_missing_estimates_fail_budget_without_dropping_denominator():
    row = {"animal": "Rat1", "session": "one", "population": "full", "family": "full", "window_ms": 20,
               "readout": "ternary", "calibration_mode": "pooled", "generator": "stationary", "requested_prevalence": .3,
               "job": 1, "estimate": .3, "absolute_error": 0., "within_5pp": True, "raw_bf_auc": .9,
               "calibrated_readout_auc": .9, "mean_spikes": 2., "silent_fraction": .1}
    missing = dict(row, job=2, estimate=np.nan, absolute_error=np.nan, within_5pp=False)
    result = summarize(pd.DataFrame([row, missing])).iloc[0]
    assert result.replicas == 2
    assert result.finite_fits == 1
    assert result.within_5pp_fraction == .5
    assert not result.budget_pass


def test_bank_uses_calibration_only_and_retains_silent_windows():
    meta = pd.DataFrame({"phase": ["calibration", "validation"], "requested_prevalence": [.5, .3],
                             "job": [0, 1], "generator": ["stationary", "stationary"], "replica": [0, 0]})
    labels = np.array([[0, 0, 1, 1], [0, 0, 0, 1]])
    bf = np.array([[-5., -4., 4., 5.], [-5., -4., 0., 5.]])
    counts = np.array([[1, 1, 1, 1], [1, 1, 0, 1]])
    calls = np.array([[0, 0, 2, 2], [0, 0, 1, 2]])
    result = analyze_bank(meta, labels, np.ones((2, 4), bool), np.full((2, 4), "stationary"),
        bf, calls, counts, {"animal": "rat", "session": "session", "population": "full", "family": "full", "window_ms": 20}, ("pooled",))
    assert len(result) == 3
    assert all(row["events"] == 4 and row["silent_fraction"] == .25 for row in result)
    changed = labels.copy()
    changed[1] = 1-changed[1]
    again = analyze_bank(meta, changed, np.ones((2, 4), bool), np.full((2, 4), "stationary"),
        bf, calls, counts, {"animal": "rat", "session": "session", "population": "full", "family": "full", "window_ms": 20}, ("pooled",))
    np.testing.assert_array_equal([r["estimate"] for r in result], [r["estimate"] for r in again])
