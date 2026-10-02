"""A support comparison must not silently change the measured cohort."""
import copy

import numpy as np
import pandas as pd
import pytest

from scripts.report_run_pair_training_support import compare_frames


def fixture():
    keys = [{"animal": "A", "session": "s", "pause_id": "p"},
            {"animal": "B", "session": "t", "pause_id": "q"}]
    frames = {"pauses.csv": pd.DataFrame([{**k, "status": "candidate_endpoint_measured", "pair_endpoints": 1} for k in keys]),
              "pairs.csv": pd.DataFrame([{**k, "unit_a": 1, "unit_b": 2, "independent_biological_subject": False,
                                          "association_fit": False} for k in keys]),
              "period_quality.csv": pd.DataFrame([{**k, "period": period, "status": "measured", "source_bins": 50,
                    "usable_bins": 40, "predicted_bins": 40, "total_bins": 40, "physical_lag_opportunities": 100,
                    "heldout_poisson_improvement_over_global": gain} for k in keys for period, gain in (("pre", -2), ("post", 3))])}
    return frames, copy.deepcopy(frames)


def test_paired_report_retains_unfavorable_animal_and_no_association_claim():
    old, new = fixture()
    new["period_quality.csv"].loc[:1, "heldout_poisson_improvement_over_global"] = [1, 4]
    periods, animals = compare_frames(old, new)
    values = periods.set_index(["animal", "period"]).paired_predictive_gain_change
    np.testing.assert_array_equal(values.loc[[("A", "pre"), ("A", "post"), ("B", "pre"), ("B", "post")]], [3, 1, 0, 0])
    assert animals.positive_periods_expanded.tolist() == [2, 1]
    assert animals.pauses_positive_in_both_periods.tolist() == [1, 0]
    assert not animals.association_tested.any() and not animals.biological_calibration_complete.any()


@pytest.mark.parametrize("name", ["source_bins", "usable_bins", "predicted_bins", "total_bins", "physical_lag_opportunities"])
def test_changed_target_support_is_rejected(name):
    old, new = fixture()
    new["period_quality.csv"].loc[0, name] += 1
    with pytest.raises(ValueError, match="Changed endpoint support"):
        compare_frames(old, new)


def test_changed_pair_population_is_rejected():
    old, new = fixture()
    new["pairs.csv"].loc[0, "unit_b"] = 3
    with pytest.raises(AssertionError):
        compare_frames(old, new)


def test_missing_period_is_rejected():
    old, new = fixture()
    new["period_quality.csv"] = new["period_quality.csv"].iloc[1:]
    with pytest.raises(ValueError, match="period denominator"):
        compare_frames(old, new)


def test_nonfinite_gain_is_rejected():
    old, new = fixture()
    new["period_quality.csv"].loc[0, "heldout_poisson_improvement_over_global"] = np.nan
    with pytest.raises(ValueError, match="Nonfinite"):
        compare_frames(old, new)


def test_failed_theta_pause_remains_in_inventory():
    old, new = fixture()
    for frames in (old, new):
        frames["pauses.csv"].loc[2] = ["C", "u", "r", "frozen_theta_screen_failed", 0]
    periods, animals = compare_frames(old, new)
    assert len(old["pauses.csv"]) == len(new["pauses.csv"]) == 3
    assert len(periods) == 4 and len(animals) == 2
