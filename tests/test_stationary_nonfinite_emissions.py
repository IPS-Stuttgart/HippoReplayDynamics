from __future__ import annotations

import numpy as np
import pytest
from scipy.special import logsumexp

from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.state_space_first_order import _score_stationary


def test_stationary_ignores_impossible_bins_without_poisoning_evidence() -> None:
    emissions = LogEmissionTensor(
        log_likelihood=np.array(
            [
                [np.log(0.6), -np.inf],
                [np.log(0.7), np.log(0.3)],
            ],
            dtype=float,
        ),
        spike_counts=np.zeros((2, 1), dtype=int),
        times=np.array([0.0, 1.0]),
        dt=1.0,
        cell_ids=np.array([1]),
        n_spikes=0,
    )

    logp, trajectory = _score_stationary(emissions)

    assert logp == pytest.approx(np.log(0.5) + np.log(0.6) + np.log(0.7))
    assert np.allclose(logsumexp(trajectory, axis=1), 0.0)
    assert np.allclose(np.exp(trajectory[:, 0]), 1.0)
    assert np.all(np.isneginf(trajectory[:, 1]))
