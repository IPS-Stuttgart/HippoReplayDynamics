from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.sweep_seed_validation import _seed_value
from hipporeplayimm.sweeps import PyRecEstSweepConfig, pyrecest_parameter_grid


def test_sweep_seed_preserves_extended_precision_numpy_scalar() -> None:
    if np.finfo(np.longdouble).nmant <= np.finfo(np.float64).nmant:
        pytest.skip("platform longdouble does not exceed float64 precision")
    expected = 2**53 + 1
    value = np.longdouble(2**53) + np.longdouble(1)

    assert _seed_value(value, "random_seed") == expected

    rows = pyrecest_parameter_grid(PyRecEstSweepConfig(random_seed=value))
    assert rows[0]["random_seed"] == expected


@pytest.mark.parametrize(
    "value",
    (
        np.complex64(3 + 0j),
        np.complex128(3 + 2j),
    ),
)
def test_sweep_seed_rejects_numpy_complex_scalars(value: np.complexfloating) -> None:
    with pytest.raises(TypeError, match="real integer, not complex"):
        _seed_value(value, "random_seed")

    with pytest.raises(TypeError, match="real integer, not complex"):
        pyrecest_parameter_grid(PyRecEstSweepConfig(random_seed=value))
