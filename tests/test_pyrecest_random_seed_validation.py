from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.pyrecest_models import (
    PyRecEstGoalParticleIMMModel,
    PyRecEstGoalParticleModel,
    _event_seed,
)


def _emissions() -> LogEmissionTensor:
    return LogEmissionTensor(
        log_likelihood=np.zeros((2, 3), dtype=float),
        spike_counts=np.zeros((2, 1), dtype=int),
        times=np.asarray([0.1, 0.2], dtype=float),
        dt=0.1,
        cell_ids=np.asarray([7], dtype=int),
        n_spikes=0,
    )


@pytest.mark.parametrize(
    'constructor',
    [PyRecEstGoalParticleModel, PyRecEstGoalParticleIMMModel],
)
@pytest.mark.parametrize(
    'value',
    [True, False, 1.0, '1', np.asarray('1'), np.asarray([1]), None],
)
def test_pyrecest_models_reject_non_integer_random_seed(constructor, value) -> None:
    with pytest.raises(ValueError, match='random_seed.*integer scalar'):
        constructor(random_seed=value)


def test_pyrecest_event_seed_rejects_mutated_non_integer_seed() -> None:
    with pytest.raises(ValueError, match='random_seed.*integer scalar'):
        _event_seed('1', _emissions())


def test_pyrecest_event_seed_accepts_integer_scalars() -> None:
    assert _event_seed(np.int64(5), _emissions()) == _event_seed(5, _emissions())
