import numpy as np
import pytest

from scripts.verify_tirole_first_pair_preflight import clock_check


def test_clock_coverage_is_not_synchronization():
    result = clock_check(np.arange(0, 5, 0.001), 1000, 0, 4)
    assert result["required_interval_covered"]
    assert result["clock_offset_applied_s"] == 0
    assert not result["independent_spike_lfp_synchronization_established"]


def test_no_guessed_offset_or_missing_gap():
    t = np.arange(0, 5, 0.001)
    assert not clock_check(t + 100, 1000, 0, 4)["required_interval_covered"]
    assert not clock_check(t[(t < 2) | (t > 2.1)], 1000, 0, 4)["required_interval_covered"]


@pytest.mark.parametrize("times,sr", [([0, 1, 0.5], 1), ([0, float("nan"), 2], 1), ([0, 1, 2], 1000), ([0], 1), ([0, 1], 0)])
def test_invalid_clocks_fail(times, sr):
    with pytest.raises(ValueError):
        clock_check(times, sr, 0, 1)
