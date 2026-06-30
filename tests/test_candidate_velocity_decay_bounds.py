from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.state_space_candidates import _score_imm_candidates
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


def test_momentum_candidates_reject_amplifying_scalar_velocity_decay() -> None:
    emissions, bin_centers, candidates = _candidate_inputs()

    with pytest.raises(ValueError, match="velocity_decays values must be <= 1"):
        _score_momentum_candidates(
            emissions,
            bin_centers,
            candidates,
            sigma_cm=1.0,
            initial_sigma_cm=1.0,
            velocity_decay=1.01,
        )


def test_momentum_candidates_reject_amplifying_velocity_decay_series() -> None:
    emissions, bin_centers, candidates = _candidate_inputs()

    with pytest.raises(ValueError, match="velocity_decays values must be <= 1"):
        _score_momentum_candidates(
            emissions,
            bin_centers,
            candidates,
            sigma_cm=1.0,
            initial_sigma_cm=1.0,
            velocity_decay=0.95,
            velocity_decays=np.array([0.95, 1.01], dtype=float),
        )


def test_imm_candidates_reject_amplifying_velocity_decay_series() -> None:
    emissions, bin_centers, candidates = _candidate_inputs()

    with pytest.raises(ValueError, match="velocity_decays values must be <= 1"):
        _score_imm_candidates(
            emissions,
            bin_centers,
            stationary_sigma_cm=1.0,
            diffusion_sigma_cm=1.0,
            momentum_sigma_cm=1.0,
            momentum_initial_sigma_cm=1.0,
            velocity_decay=0.95,
            mode_stickiness=0.95,
            candidate_indices=candidates,
            velocity_decays=np.array([1.0, 1.01], dtype=float),
        )
