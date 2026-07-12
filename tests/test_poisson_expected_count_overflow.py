from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.encoding import _poisson_log_emissions


@pytest.mark.parametrize(
    "dt",
    [2.0, np.array([2.0], dtype=float)],
    ids=["scalar-duration", "per-bin-duration"],
)
@pytest.mark.parametrize("overdispersion", [0.0, 0.25], ids=["poisson", "negative-binomial"])
def test_poisson_emissions_reject_expected_count_overflow(
    dt: float | np.ndarray,
    overdispersion: float,
) -> None:
    spike_counts = np.array([[1.0]], dtype=float)
    rates_hz = np.array([[np.finfo(float).max]], dtype=float)

    with pytest.raises(ValueError, match="scaled expected spike counts must be finite"):
        _poisson_log_emissions(
            spike_counts,
            rates_hz,
            dt,
            negative_binomial_overdispersion=overdispersion,
        )


def test_poisson_emissions_keep_large_representable_expected_counts_finite() -> None:
    spike_counts = np.array([[1.0]], dtype=float)
    rates_hz = np.array([[np.finfo(float).max / 4.0]], dtype=float)

    result = _poisson_log_emissions(spike_counts, rates_hz, 2.0)

    assert result.shape == (1, 1)
    assert np.all(np.isfinite(result))
