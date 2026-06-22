import numpy as np
import pytest

from hipporeplayimm import state_space_first_order
from hipporeplayimm.state_space_utils import (
    _coerce_valid_bin_mask,
    _uniform_log_prior,
    _uniform_probabilities,
    _valid_bin_count,
)


def test_state_space_uniform_helpers_reject_empty_support_without_mask():
    with pytest.raises(ValueError, match="n_bins must be positive"):
        _uniform_log_prior(0)
    with pytest.raises(ValueError, match="n_bins must be positive"):
        _uniform_probabilities(0)
    with pytest.raises(ValueError, match="n_bins must be positive"):
        _valid_bin_count(0)
    with pytest.raises(ValueError, match="n_bins must be positive"):
        _coerce_valid_bin_mask(None, 0)


def test_state_space_uniform_aliases_reject_empty_support():
    with pytest.raises(ValueError, match="n_bins must be positive"):
        state_space_first_order._uniform_log_prior(0)
    with pytest.raises(ValueError, match="n_bins must be positive"):
        state_space_first_order._uniform_probabilities(0)


def test_state_space_uniform_helpers_preserve_masked_support():
    mask = np.array([True, False, True], dtype=bool)

    np.testing.assert_allclose(_uniform_probabilities(3, mask), [0.5, 0.0, 0.5])
    np.testing.assert_allclose(
        _uniform_log_prior(3, mask),
        [-np.log(2.0), -1.0e300, -np.log(2.0)],
    )
    assert _valid_bin_count(3, mask) == 2
    np.testing.assert_array_equal(_coerce_valid_bin_mask(mask, 3), mask)
