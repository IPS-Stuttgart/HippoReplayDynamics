from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import hipporeplayimm.simulation_recovery as recovery


def test_emissions_from_counts_normalizes_arbitrary_precision_overflow() -> None:
    encoding = SimpleNamespace(n_cells=1)

    with pytest.raises(ValueError, match="counts must contain numeric values"):
        recovery.emissions_from_counts(
            encoding,
            np.array([[10**400]], dtype=object),
            dt=0.003,
        )


def test_simulate_latent_path_normalizes_n_time_overflow() -> None:
    with pytest.raises(ValueError, match="n_time must be a positive integer"):
        recovery.simulate_latent_path(
            SimpleNamespace(),
            true_model="stationary",
            n_time=10**400,
            dt=0.003,
            rng=np.random.default_rng(0),
        )


def test_simulate_latent_path_normalizes_occupancy_overflow() -> None:
    encoding = SimpleNamespace(
        n_bins=1,
        occupancy_s=np.array([10**400], dtype=object),
    )

    with pytest.raises(
        ValueError,
        match="occupancy_s must contain finite nonnegative values",
    ):
        recovery.simulate_latent_path(
            encoding,
            true_model="stationary",
            n_time=1,
            dt=0.003,
            rng=np.random.default_rng(0),
        )
