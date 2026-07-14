from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.pyrecest_models import _event_seed


def _single_bin_emissions() -> LogEmissionTensor:
    return LogEmissionTensor(
        log_likelihood=np.zeros((1, 2), dtype=float),
        spike_counts=np.zeros((1, 1), dtype=int),
        times=np.array([0.0], dtype=float),
        dt=0.02,
        cell_ids=np.array([1], dtype=int),
        n_spikes=0,
    )


def test_pyrecest_event_seed_keeps_numpy_upper_bound_distinct_from_zero() -> None:
    emissions = _single_bin_emissions()

    zero_seed = _event_seed(0, emissions)
    upper_bound_seed = _event_seed(2**32 - 1, emissions)

    assert zero_seed == 1009
    assert upper_bound_seed == 1008
    assert upper_bound_seed != zero_seed


@pytest.mark.parametrize(
    "random_seed",
    [True, False, -1, 1.5, "1", b"1", np.array([1])],
)
def test_pyrecest_event_seed_rejects_invalid_seed_values(random_seed: object) -> None:
    with pytest.raises(ValueError, match="random_seed"):
        _event_seed(random_seed, _single_bin_emissions())


def test_pyrecest_event_seed_accepts_exact_integer_valued_scalar() -> None:
    emissions = _single_bin_emissions()

    assert _event_seed(1.0, emissions) == _event_seed(1, emissions)
