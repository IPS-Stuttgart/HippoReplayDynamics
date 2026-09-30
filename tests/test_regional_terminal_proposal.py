import numpy as np
import pytest

from scripts.audit_regional_terminal_proposal import anywhere_label, conditional_identity_probability


@pytest.mark.parametrize("duration", [20, 40, 60, 100])
def test_opposite_region_late_jumps_are_all_anywhere_positive(duration):
    endpoint = np.array([False, True, True, False])
    np.testing.assert_array_equal(anywhere_label(~endpoint, endpoint, duration), np.ones(4, bool))


def test_same_region_paths_supply_non_degenerate_truth():
    state = np.array([False, True])
    np.testing.assert_array_equal(anywhere_label(state, state, 20), state)


def test_last_five_ms_do_not_include_pre_jump_origin():
    endpoint = np.array([False, True])
    np.testing.assert_array_equal(anywhere_label(~endpoint, endpoint, 5), endpoint)


def test_anywhere_and_endpoint_are_not_interchangeable():
    assert anywhere_label(np.array([True]), np.array([False]), 20)[0]


def test_absolute_gain_and_additive_scale_are_not_identified_at_fixed_counts():
    rates = np.array([[1., 2., 5.], [4., 8., 1.]])
    additive = np.array([2., 0., .5])
    np.testing.assert_allclose(conditional_identity_probability(rates, 1., additive),
                               conditional_identity_probability(rates, 4., 4*additive))


def test_relative_additive_change_is_observable():
    rates = np.array([[1., 2., 5.]])
    assert not np.allclose(conditional_identity_probability(rates, 1., np.zeros(3)),
                           conditional_identity_probability(rates, 1., np.array([2., 0., 0.])))


def test_invalid_window_rejected():
    with pytest.raises(ValueError):
        anywhere_label(np.array([True]), np.array([False]), 0)
