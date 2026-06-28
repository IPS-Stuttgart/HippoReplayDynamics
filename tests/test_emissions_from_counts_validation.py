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
        rates_hz=np.ones((1, 1), dtype=float),
        occupancy_s=np.array([1.0], dtype=float),
        cell_ids=np.array([1], dtype=int),
        config=EncodingConfig(),
    )


def test_emissions_from_counts_rejects_empty_time_axis() -> None:
    with pytest.raises(ValueError, match="at least one time bin"):
        emissions_from_counts(_encoding(), np.zeros((0, 1), dtype=int), dt=0.02)


@pytest.mark.parametrize(
    "counts",
    [
        np.array([[True]], dtype=bool),
        np.array([[False]], dtype=object),
    ],
)
def test_emissions_from_counts_rejects_boolean_count_values(counts: np.ndarray) -> None:
    with pytest.raises(ValueError, match="boolean"):
        emissions_from_counts(_encoding(), counts, dt=0.02)
