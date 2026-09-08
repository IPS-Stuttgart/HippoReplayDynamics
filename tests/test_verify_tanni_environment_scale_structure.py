import numpy as np
import pytest

from scripts.verify_tanni_environment_scale_structure import independent_slope, reference_probability, require_close


def test_direct_regression_slope():
    assert independent_slope(np.array([1, 2, 3]), np.array([9, 6, 3])) == pytest.approx(-3)


def test_enumerated_reference_ties_and_reversal():
    y = np.tile(np.arange(3), (5, 1))
    assert reference_probability(y) == pytest.approx(2 / 7776)
    assert reference_probability(-y) == pytest.approx(reference_probability(y))
    assert reference_probability(y * 0) == 1


def test_reconstruction_failure_and_missing_values():
    require_close([1, np.nan], [1, np.nan])
    with pytest.raises(ValueError):
        require_close([1, 0], [1, np.nan])
    with pytest.raises(ValueError):
        reference_probability(np.zeros((5, 2)))
