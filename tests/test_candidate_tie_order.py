import numpy as np

import hipporeplayimm.models as models
import hipporeplayimm.state_space_model as state_space_model
import hipporeplayimm.state_space_utils as state_space_utils


def test_fixed_top_k_candidate_ties_keep_original_bin_order() -> None:
    log_emission = np.array([3.0, 2.0, 2.0, 2.0, 1.0])
    expected = np.array([0, 1])

    np.testing.assert_array_equal(
        state_space_utils._top_candidate_indices(log_emission, 2),
        expected,
    )
    np.testing.assert_array_equal(
        state_space_model._top_candidate_indices(log_emission, 2),
        expected,
    )
    np.testing.assert_array_equal(
        models._top_candidate_indices(log_emission, 2),
        expected,
    )


def test_flat_fixed_top_k_uses_lowest_index_bins() -> None:
    log_emission = np.zeros(6)
    expected = np.array([0, 1, 2])

    np.testing.assert_array_equal(
        state_space_utils._top_candidate_indices(log_emission, 3),
        expected,
    )
    np.testing.assert_array_equal(
        state_space_model._top_candidate_indices(log_emission, 3),
        expected,
    )
    np.testing.assert_array_equal(
        models._top_candidate_indices(log_emission, 3),
        expected,
    )


def test_disabled_mass_threshold_uses_same_stable_top_k_order() -> None:
    log_emission = np.zeros(6)

    np.testing.assert_array_equal(
        state_space_utils._mass_retaining_candidate_indices(
            log_emission,
            mass_threshold=None,
            top_k=3,
        ),
        np.array([0, 1, 2]),
    )


def test_state_space_imported_selector_alias_is_refreshed() -> None:
    assert state_space_model._top_candidate_indices is state_space_utils._top_candidate_indices
