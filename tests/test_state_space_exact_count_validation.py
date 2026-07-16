from decimal import Decimal

import numpy as np
import pytest

from hipporeplayimm.state_space_bin_count_validation import _integer_count, _positive_bin_count
from hipporeplayimm.state_space_utils import _top_candidate_indices


@pytest.mark.parametrize(
    "exact_count",
    [
        2**53 + 1,
        np.uint64(2**53 + 1),
        Decimal("9007199254740993"),
        "9007199254740993",
    ],
)
def test_state_space_count_validation_preserves_exact_large_integers(exact_count):
    assert _integer_count("top_k", exact_count) == 2**53 + 1
    assert _positive_bin_count(exact_count) == 2**53 + 1


@pytest.mark.parametrize(
    "fractional_count",
    [
        Decimal("9007199254740992.5"),
        "9007199254740992.5",
    ],
)
def test_state_space_count_validation_rejects_fractional_large_decimals(fractional_count):
    with pytest.raises(TypeError, match="top_k must be an integer"):
        _integer_count("top_k", fractional_count)
    with pytest.raises(TypeError, match="top_k must be an integer"):
        _top_candidate_indices(np.array([0.0, 1.0]), fractional_count)
    with pytest.raises(ValueError, match="n_bins must be a positive integer"):
        _positive_bin_count(fractional_count)
