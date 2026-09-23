import numpy as np
import pytest

from scripts.calibrate_roscow_task_arm_readout import calibrated_session
from scripts.verify_roscow_nested_calibration import independent_nested


@pytest.mark.parametrize("seed", [1, 4, 9])
def test_naive_nested_folds_match_cached_producer(seed):
    rng = np.random.default_rng(seed)
    y = rng.permutation(np.repeat(np.arange(3), 4))
    x = rng.poisson(rng.uniform(0.1, 5, (len(y), 7)))
    x[np.arange(len(y)), y] += 3
    produced, temperature = calibrated_session(x, y, 1.75)
    verified, chosen = independent_nested(x, y, 1.75)
    for kind in verified:
        np.testing.assert_allclose(verified[kind], produced[kind, "nested_temperature"])
        np.testing.assert_array_equal(chosen[kind], temperature[kind])
