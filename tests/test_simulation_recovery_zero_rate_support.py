from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.encoding import EncodingConfig, EncodingModel
from hipporeplayimm.simulation_recovery import emissions_from_counts


def _two_bin_encoding() -> EncodingModel:
    return EncodingModel(
        x_edges=np.array([0.0, 1.0, 2.0], dtype=float),
        y_edges=np.array([0.0, 1.0], dtype=float),
        bin_centers=np.array([[0.5, 0.5], [1.5, 0.5]], dtype=float),
        rates_hz=np.array([[0.0, 2.0]], dtype=float),
        occupancy_s=np.ones(2, dtype=float),
        cell_ids=np.array([7], dtype=int),
        config=EncodingConfig(),
    )


@pytest.mark.parametrize(
    "overdispersion",
    [0.0, 0.5],
    ids=["poisson", "negative-binomial"],
)
def test_emissions_from_counts_preserves_zero_rate_support(
    overdispersion: float,
) -> None:
    emissions = emissions_from_counts(
        _two_bin_encoding(),
        np.array([[1]], dtype=int),
        dt=1.0,
        negative_binomial_overdispersion=overdispersion,
    )

    assert np.isneginf(emissions.log_likelihood[0, 0])
    assert np.isfinite(emissions.log_likelihood[0, 1])
