from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.encoding import _poisson_log_emissions


def _nested_scalar(value: object) -> np.ndarray:
    inner = np.empty((), dtype=object)
    inner[()] = value
    outer = np.empty((), dtype=object)
    outer[()] = inner
    return outer


@pytest.mark.parametrize("imaginary", [0.0, 2.0])
def test_poisson_log_emissions_rejects_complex_rate_arrays(imaginary: float) -> None:
    rates = np.array([[1.0 + imaginary * 1j, 3.0 + 0.0j]], dtype=complex)

    with pytest.raises(ValueError, match="rates_hz.*complex"):
        _poisson_log_emissions(
            np.array([[1]], dtype=int),
            rates,
            0.02,
        )


def test_poisson_log_emissions_rejects_nested_complex_rates() -> None:
    rates = np.ones((1, 2), dtype=object)
    rates[0, 1] = _nested_scalar(np.complex128(2.0 + 1.0j))

    with pytest.raises(ValueError, match="rates_hz.*complex"):
        _poisson_log_emissions(
            np.array([[1]], dtype=int),
            rates,
            0.02,
        )
