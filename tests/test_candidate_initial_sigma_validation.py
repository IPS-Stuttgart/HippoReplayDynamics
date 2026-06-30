from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.state_space_candidates_momentum import _score_momentum_candidates


def _candidate_inputs() -> tuple[LogEmissionTensor, np.ndarray, list[np.ndarray]]:
    emissions = LogEmissionTensor(
        log_likelihood=np.log(
            np.array(
                [
                    [0.55, 0.45],
                    [0.45, 0.55],
                    [0.40, 0.60],
                ],
                dtype=float,
            )
        ),
        spike_counts=np.zeros((3, 1), dtype=int),
        times=np.array([0.0, 1.0, 2.0], dtype=float),
        dt=1.0,
        cell_ids=np.array([1], dtype=int),
        n_spikes=0,
    )
    bin_centers = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)
    candidates = [np.array([0, 1], dtype=int) for _ in range(emissions.n_time)]
    return emissions, bin_centers, candidates


@pytest.mark.parametrize("bad_initial_sigma", [0.0, -1.0, float("nan"), float("inf")])
def test_momentum_candidates_reject_invalid_initial_sigma(bad_initial_sigma: float) -> None:
    emissions, bin_centers, candidates = _candidate_inputs()

    with pytest.raises(ValueError, match="initial_sigma_cm"):
        _score_momentum_candidates(
            emissions,
            bin_centers,
            candidates,
            sigma_cm=1.0,
            initial_sigma_cm=bad_initial_sigma,
            velocity_decay=0.95,
        )


def test_momentum_candidates_accept_positive_initial_sigma() -> None:
    emissions, bin_centers, candidates = _candidate_inputs()

    logp, trajectory, masses = _score_momentum_candidates(
        emissions,
        bin_centers,
        candidates,
        sigma_cm=1.0,
        initial_sigma_cm=1.0,
        velocity_decay=0.95,
    )

    assert np.isfinite(logp)
    assert trajectory.shape == emissions.log_likelihood.shape
    assert len(masses) == emissions.n_time
