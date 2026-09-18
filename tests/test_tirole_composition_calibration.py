import numpy as np
import pandas as pd
import pytest
from scipy.special import logit

from scripts.report_tirole_composition_calibration import known_weight_shift, probabilities, selected_composition


@pytest.mark.parametrize("retention", [(0.5, 0.5), (0.6, 0.4), (0.4, 0.6), (0.75, 0.25), (0.25, 0.75)])
def test_perfect_readout_recovers_known_composition(retention):
    x = known_weight_shift(np.zeros((4, 5)), np.ones((4, 5)), retention)
    np.testing.assert_allclose(x["observed_shift"], x["true_shift"], atol=1e-15)
    np.testing.assert_allclose(x["recovery_slope"], 1.0)


def test_uninformative_mass_does_not_recover_true_selection():
    x = known_weight_shift(np.full((4, 5), 0.5), np.full((4, 5), 0.5), (0.25, 0.75))
    assert x["true_shift"] == 0.25
    np.testing.assert_allclose(x["observed_shift"], 0.0)
    np.testing.assert_allclose(x["recovery_slope"], 0.0)


def test_soft_mass_shrinkage_is_not_misreported_as_no_true_bias():
    x = known_weight_shift(np.full((4, 5), 0.3), np.full((4, 5), 0.7), (0.25, 0.75))
    np.testing.assert_allclose(x["observed_shift"], 0.1)
    np.testing.assert_allclose(x["recovery_slope"], 0.4)


def test_bootstrap_shares_anchor_weights_across_tracks_and_splits():
    q1 = np.array([[0.1, 0.1], [0.3, 0.3]])
    q2 = 1 - q1
    weights = np.array([[2, 0], [0, 2], [1, 1]])
    a = known_weight_shift(q1, q2, (0.25, 0.75), weights)
    b = known_weight_shift(np.tile(q1, (1, 3)), np.tile(q2, (1, 3)), (0.25, 0.75), weights)
    np.testing.assert_allclose(a["observed_shift"], [0.2, 0.1, 0.15])
    np.testing.assert_allclose(a["observed_shift"], b["observed_shift"])


def test_empty_selected_groups_are_missing_not_zero():
    x = selected_composition([0, 1], [0.1, 0.9], [True, True], [False, False])
    assert np.isnan(x["readout_loss_shift"])
    assert np.isnan(x["true_loss_shift_supported"])
    assert x["n_retained"] == 0


def test_gains_are_separate_from_selective_losses():
    x = selected_composition([0, 1, 1], [0.0, 1.0, 1.0], [True, True, False], [False, True, True])
    assert x["true_loss_shift_all"] == 0.5
    assert x["true_total_shift_all"] == 0.5
    assert x["readout_gain_shift"] == 0.0
    assert x["n_gained"] == 1


def test_missing_readout_does_not_change_all_event_truth_denominator():
    x = selected_composition([0, 1], [np.nan, 0.9], [True, True], [True, True])
    assert x["true_full_all"] == 0.5
    assert x["true_full_supported"] == 1.0
    assert x["n_full"] == 2 and x["n_full_supported"] == 1


def test_signed_log_odds_orientation_and_stored_mass_validation():
    q = np.array([0.1, 0.9])
    sign = np.array([1, -1])
    data = pd.DataFrame(
        {"truth_track": [1, 2], "evaluation_true_signed_log_odds": -sign * logit(q), "conditional_true_signed_log_odds": -sign * logit(q), "evaluation_track2_probability": q}
    )
    np.testing.assert_allclose(probabilities(data).conditional_count, q)
    data.evaluation_track2_probability = 0.5
    with pytest.raises(ValueError, match="differ"):
        probabilities(data)


def test_missing_pair_is_not_zero_and_invalid_weights_fail():
    x = known_weight_shift(np.full((2, 5), np.nan), np.ones((2, 5)), (0.25, 0.75))
    assert np.isnan(x["observed_shift"]).all()
    assert x["n_estimable_splits"][0] == 0
    with pytest.raises(ValueError):
        known_weight_shift(np.zeros((2, 5)), np.ones((2, 5)), (0.0, 0.0))
