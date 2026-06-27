from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.encoding import _poisson_log_emissions
from hipporeplayimm.kd_reference import poisson_log_emissions as kd_poisson_log_emissions


@pytest.mark.parametrize(
    "builder",
    [_poisson_log_emissions, kd_poisson_log_emissions],
    ids=["sorted", "kd"],
)
def test_poisson_emissions_reject_boolean_spike_counts(builder):
    with pytest.raises(ValueError, match="spike_counts.*boolean"):
        builder(
            np.array([[True]], dtype=bool),
            np.ones((1, 2), dtype=float),
            0.02,
        )


@pytest.mark.parametrize(
    "builder",
    [_poisson_log_emissions, kd_poisson_log_emissions],
    ids=["sorted", "kd"],
)
def test_poisson_emissions_reject_object_boolean_rates(builder):
    with pytest.raises(ValueError, match="rates_hz.*boolean"):
        builder(
            np.array([[0]], dtype=int),
            np.array([[np.bool_(True), 0.5]], dtype=object),
            0.02,
        )
