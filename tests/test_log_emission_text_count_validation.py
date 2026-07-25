from __future__ import annotations

from decimal import Decimal

import numpy as np
import pytest

from hipporeplayimm.encoding import LogEmissionTensor


_LARGE_EXACT_COUNT = 2**53 + 1


def _make_emissions(*, spike_counts: object, n_spikes: object) -> LogEmissionTensor:
    return LogEmissionTensor(
        log_likelihood=np.zeros((1, 1), dtype=float),
        spike_counts=spike_counts,
        times=np.array([0.0]),
        dt=0.02,
        cell_ids=np.array([1]),
        n_spikes=n_spikes,
    )


@pytest.mark.parametrize(
    "text_count",
    [
        str(_LARGE_EXACT_COUNT),
        f"{_LARGE_EXACT_COUNT}.0",
        str(_LARGE_EXACT_COUNT).encode("utf-8"),
        np.str_(str(_LARGE_EXACT_COUNT)),
        np.bytes_(str(_LARGE_EXACT_COUNT)),
    ],
)
def test_log_emission_preserves_large_text_counts_exactly(text_count: object) -> None:
    emissions = _make_emissions(
        spike_counts=np.array([[text_count]], dtype=object),
        n_spikes=text_count,
    )

    np.testing.assert_array_equal(
        emissions.spike_counts,
        np.array([[_LARGE_EXACT_COUNT]], dtype=int),
    )
    assert emissions.n_spikes == _LARGE_EXACT_COUNT
    assert isinstance(emissions.n_spikes, int)


def test_log_emission_preserves_decimal_counts_exactly() -> None:
    count = Decimal(_LARGE_EXACT_COUNT)
    emissions = _make_emissions(
        spike_counts=np.array([[count]], dtype=object),
        n_spikes=count,
    )

    assert int(emissions.spike_counts[0, 0]) == _LARGE_EXACT_COUNT
    assert emissions.n_spikes == _LARGE_EXACT_COUNT


@pytest.mark.parametrize("fractional", ["1.5", b"1.5", Decimal("1.5")])
def test_log_emission_rejects_fractional_text_or_decimal_counts(fractional: object) -> None:
    with pytest.raises(ValueError, match="spike_counts.*integer-valued"):
        _make_emissions(
            spike_counts=np.array([[fractional]], dtype=object),
            n_spikes=1,
        )

    with pytest.raises(ValueError, match="n_spikes.*integer-valued"):
        _make_emissions(
            spike_counts=np.array([[1]], dtype=int),
            n_spikes=fractional,
        )


def test_log_emission_rejects_text_count_outside_platform_integer_range() -> None:
    oversized = str(int(np.iinfo(np.dtype(int)).max) + 1)

    with pytest.raises(ValueError, match="fit into integer count range"):
        _make_emissions(
            spike_counts=np.array([[oversized]], dtype=object),
            n_spikes=oversized,
        )


def test_log_emission_keeps_integral_numeric_count_coercion() -> None:
    emissions = LogEmissionTensor(
        log_likelihood=np.zeros((2, 1), dtype=float),
        spike_counts=np.array([[1.0], [2.0]]),
        times=np.array([0.0, 0.02]),
        dt=0.02,
        cell_ids=np.array([1]),
        n_spikes=np.float64(3.0),
    )

    np.testing.assert_array_equal(emissions.spike_counts, np.array([[1], [2]], dtype=int))
    assert np.issubdtype(emissions.spike_counts.dtype, np.integer)
    assert emissions.n_spikes == 3
    assert isinstance(emissions.n_spikes, int)
