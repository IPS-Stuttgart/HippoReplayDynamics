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


def test_emissions_from_counts_normalizes_arbitrary_precision_overflow() -> None:
    with pytest.raises(ValueError, match="counts must contain numeric values"):
        emissions_from_counts(
            _encoding(),
            np.array([[10**400]], dtype=object),
            dt=0.02,
        )


def test_emissions_from_counts_rejects_finite_values_outside_integer_range() -> None:
    outside_integer_range = np.nextafter(float(np.iinfo(np.dtype(int)).max), np.inf)

    with pytest.raises(ValueError, match="counts must fit into integer count range"):
        emissions_from_counts(
            _encoding(),
            np.array([[outside_integer_range]], dtype=float),
            dt=0.02,
        )
