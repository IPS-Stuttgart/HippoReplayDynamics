from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.encoding import EncodingConfig, EncodingModel
from hipporeplayimm.simulation_recovery import emissions_from_counts


def _encoding() -> EncodingModel:
    return EncodingModel(
        x_edges=np.array([0.0, 1.0]),
        y_edges=np.array([0.0, 1.0]),
        bin_centers=np.array([[0.5, 0.5]], dtype=float),
        rates_hz=np.array([[10.0], [5.0]], dtype=float),
        occupancy_s=np.array([1.0], dtype=float),
        cell_ids=np.array([1, 2], dtype=int),
        config=EncodingConfig(bin_size_cm=1.0),
    )


def test_emissions_from_counts_rejects_fractional_counts_before_cast() -> None:
    with pytest.raises(ValueError, match="integer-valued"):
        emissions_from_counts(
            _encoding(),
            np.array([[1.5, 0.0]], dtype=float),
            dt=0.02,
        )


def test_emissions_from_counts_rejects_negative_counts_before_cast() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        emissions_from_counts(
            _encoding(),
            np.array([[-0.5, 1.0]], dtype=float),
            dt=0.02,
        )


def test_emissions_from_counts_accepts_integer_float_counts() -> None:
    emissions = emissions_from_counts(
        _encoding(),
        np.array([[1.0, 2.0]], dtype=float),
        dt=0.02,
    )

    assert emissions.n_spikes == 3
    np.testing.assert_array_equal(emissions.spike_counts, np.array([[1, 2]], dtype=int))
