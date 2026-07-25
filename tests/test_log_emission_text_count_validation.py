from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.encoding import LogEmissionTensor


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
    "spike_counts",
    [
        np.array([["1"]]),
        np.array([[b"1"]]),
        np.array([["1"]], dtype=object),
        np.array([[np.str_("1")]], dtype=object),
        np.array([[np.bytes_("1")]], dtype=object),
    ],
)
def test_log_emission_rejects_text_backed_spike_counts(spike_counts: object) -> None:
    with pytest.raises(ValueError, match="spike_counts.*not text"):
        _make_emissions(spike_counts=spike_counts, n_spikes=1)


@pytest.mark.parametrize(
    "n_spikes",
    [
        "1",
        b"1",
        np.str_("1"),
        np.bytes_("1"),
        np.asarray("1"),
        np.asarray(b"1"),
        np.asarray("1", dtype=object),
    ],
)
def test_log_emission_rejects_text_backed_total_spike_count(n_spikes: object) -> None:
    with pytest.raises(ValueError, match="n_spikes.*not text"):
        _make_emissions(spike_counts=np.array([[1]]), n_spikes=n_spikes)


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
