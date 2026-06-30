import numpy as np
import pytest

from hipporeplayimm import state_space_first_order, state_space_model
from hipporeplayimm.state_space_utils import (
    _coerce_valid_bin_mask,
    _mass_retaining_candidate_indices,
    _top_candidate_indices,
    _uniform_log_prior,
    _uniform_probabilities,
    _valid_bin_count,
)


@pytest.mark.parametrize("bad_n_bins", [0, -1, True, np.bool_(True), 1.5, "2.5", np.array([2])])
def test_state_space_uniform_helpers_reject_invalid_support_size_without_mask(bad_n_bins):
    with pytest.raises(ValueError, match="n_bins must be a positive integer"):
        _uniform_log_prior(bad_n_bins)
    with pytest.raises(ValueError, match="n_bins must be a positive integer"):
        _uniform_probabilities(bad_n_bins)
    with pytest.raises(ValueError, match="n_bins must be a positive integer"):
        _valid_bin_count(bad_n_bins)
    with pytest.raises(ValueError, match="n_bins must be a positive integer"):
        _coerce_valid_bin_mask(None, bad_n_bins)


@pytest.mark.parametrize("bad_top_k", [True, np.bool_(True), 1.5, "2.5", np.array([2])])
def test_state_space_top_candidates_reject_non_integer_counts(bad_top_k):
    scores = np.array([0.0, 2.0, 1.0], dtype=float)

    with pytest.raises(TypeError, match="top_k must be an integer"):
        _top_candidate_indices(scores, bad_top_k)
    with pytest.raises(TypeError, match="top_k must be an integer"):
        state_space_model._top_candidate_indices(scores, bad_top_k)


def test_state_space_top_candidates_reject_negative_counts():
    with pytest.raises(ValueError, match="top_k must be nonnegative"):
        _top_candidate_indices(np.array([0.0, 1.0], dtype=float), -1)


@pytest.mark.parametrize(
    ("kwargs", "error_type", "message"),
    [
        ({"mass_threshold": None, "min_k": np.array([1])}, TypeError, "min_k must be an integer"),
        ({"mass_threshold": None, "max_k": np.array([2])}, TypeError, "max_k must be an integer"),
        ({"mass_threshold": 0.0, "min_k": -1}, ValueError, "min_k must be nonnegative"),
        ({"mass_threshold": 0.0, "max_k": -1}, ValueError, "max_k must be nonnegative"),
    ],
)
def test_mass_retaining_candidates_reject_bad_counts_when_threshold_is_disabled(kwargs, error_type, message):
    scores = np.array([0.0, -1.0, -2.0], dtype=float)

    with pytest.raises(error_type, match=message):
        _mass_retaining_candidate_indices(scores, **kwargs)


@pytest.mark.parametrize(
    ("kwargs", "error_type", "message"),
    [
        ({"mass_threshold": None, "min_k": np.array([1])}, TypeError, "min_k must be an integer"),
        ({"mass_threshold": None, "max_k": np.array([2])}, TypeError, "max_k must be an integer"),
        ({"mass_threshold": 0.0, "min_k": -1}, ValueError, "min_k must be nonnegative"),
        ({"mass_threshold": 0.0, "max_k": -1}, ValueError, "max_k must be nonnegative"),
    ],
)
def test_mass_retaining_candidate_aliases_reject_bad_counts_when_threshold_is_disabled(kwargs, error_type, message):
    scores = np.array([0.0, -1.0, -2.0], dtype=float)

    with pytest.raises(error_type, match=message):
        state_space_model._mass_retaining_candidate_indices(scores, **kwargs)


def test_state_space_top_candidates_preserve_zero_as_full_support():
    np.testing.assert_array_equal(
        _top_candidate_indices(np.array([0.0, 1.0], dtype=float), 0),
        np.array([0, 1], dtype=int),
    )


def test_state_space_uniform_aliases_reject_empty_support():
    with pytest.raises(ValueError, match="n_bins must be a positive integer"):
        state_space_first_order._uniform_log_prior(0)
    with pytest.raises(ValueError, match="n_bins must be a positive integer"):
        state_space_first_order._uniform_probabilities(0)


def test_state_space_uniform_helpers_preserve_masked_support():
    mask = np.array([True, False, True], dtype=bool)

    np.testing.assert_allclose(_uniform_probabilities(3, mask), [0.5, 0.0, 0.5])
    np.testing.assert_allclose(
        _uniform_log_prior(3, mask),
        [-np.log(2.0), -1.0e300, -np.log(2.0)],
    )
    assert _valid_bin_count(3, mask) == 2
    assert _valid_bin_count("3.0", mask) == 2
    np.testing.assert_array_equal(_coerce_valid_bin_mask(mask, 3), mask)


def test_state_space_uniform_helpers_accept_binary_numeric_masks():
    mask = np.array([1.0, 0.0, 1.0], dtype=float)
    expected = np.array([True, False, True], dtype=bool)

    np.testing.assert_array_equal(_coerce_valid_bin_mask(mask, 3), expected)
    np.testing.assert_allclose(_uniform_probabilities(3, mask), [0.5, 0.0, 0.5])
    assert _valid_bin_count(3, mask) == 2


@pytest.mark.parametrize(
    "bad_mask",
    [
        np.array([1.0, np.nan, 0.0]),
        np.array([1.0, np.inf, 0.0]),
        np.array([1.0, 0.5, 0.0]),
        np.array(["yes", "no", "yes"], dtype=object),
    ],
)
def test_state_space_uniform_helpers_reject_non_boolean_mask_values(bad_mask):
    with pytest.raises(ValueError, match="valid_bin_mask must contain"):
        _coerce_valid_bin_mask(bad_mask, 3)
    with pytest.raises(ValueError, match="valid_bin_mask must contain"):
        _uniform_probabilities(3, bad_mask)


def test_state_space_uniform_aliases_reject_nan_mask_values():
    bad_mask = np.array([1.0, np.nan, 0.0])

    with pytest.raises(ValueError, match="valid_bin_mask must contain"):
        state_space_first_order._uniform_probabilities(3, bad_mask)
