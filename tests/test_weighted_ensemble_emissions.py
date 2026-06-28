from __future__ import annotations

import numpy as np

import hipporeplayimm
from hipporeplayimm.accuracy_upgrades import weighted_ensemble_emissions
from hipporeplayimm.encoding import LogEmissionTensor


def _emissions(offset: float) -> LogEmissionTensor:
    return LogEmissionTensor(
        log_likelihood=np.array(
            [
                [offset, offset - 1.0],
                [offset - 0.5, offset - 0.25],
                [offset - 0.2, offset - 0.4],
            ],
            dtype=float,
        ),
        spike_counts=np.array([[1], [0], [2]], dtype=int),
        times=np.array([0.01, 0.03, 0.07], dtype=float),
        dt=0.02,
        cell_ids=np.array([11], dtype=int),
        n_spikes=3,
        bin_durations=np.array([0.01, 0.02, 0.04], dtype=float),
        transition_durations=np.array([0.02, 0.04], dtype=float),
    )


def test_weighted_ensemble_preserves_observation_count_and_durations() -> None:
    hipporeplayimm.apply_runtime_patches()
    left = _emissions(0.0)
    right = _emissions(1.0)

    out = weighted_ensemble_emissions(left, right, alpha=0.25)

    np.testing.assert_allclose(
        out.log_likelihood,
        0.25 * left.log_likelihood + 0.75 * right.log_likelihood,
    )
    assert out.n_spikes == left.n_spikes
    assert out.n_spikes == int(np.asarray(out.spike_counts).sum())
    np.testing.assert_allclose(out.bin_durations, left.bin_durations)
    np.testing.assert_allclose(out.transition_durations, left.transition_durations)
