from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.encoding import _poisson_log_emissions


def test_poisson_log_emissions_rejects_boolean_spike_counts() -> None:
    with pytest.raises(ValueError, match="spike_counts.*boolean"):
        _poisson_log_emissions(
            np.array([[True, False]], dtype=bool),
            np.ones((2, 3), dtype=float),
            0.02,
        )


def test_poisson_log_emissions_rejects_boolean_rates() -> None:
    with pytest.raises(ValueError, match="rates_hz.*boolean"):
        _poisson_log_emissions(
            np.array([[0, 1]], dtype=int),
            np.array(
                [
                    [True, False, True],
                    [False, True, False],
                ],
                dtype=bool,
            ),
            0.02,
        )


def test_poisson_log_emissions_rejects_object_boolean_inputs() -> None:
    with pytest.raises(ValueError, match="spike_counts.*boolean"):
        _poisson_log_emissions(
            np.array([[True, 0]], dtype=object),
            np.ones((2, 3), dtype=float),
            0.02,
        )
