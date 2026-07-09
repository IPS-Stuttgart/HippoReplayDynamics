from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.state_space import StateSpaceDecoderConfig, StateSpaceReplayModel


@pytest.mark.parametrize("mode", ["momentum", "imm", "momentum-exact-sparse"])
def test_duration_state_space_rejects_velocity_decay_above_one(mode: str) -> None:
    emissions = _tiny_emissions()
    config = _config(mode, momentum_velocity_decay=1.01)
    model = StateSpaceReplayModel(mode=mode, config=config)

    with pytest.raises(ValueError, match=r"momentum_velocity_decay must be finite and in \[0, 1\]"):
        model.score(emissions, _centers(emissions.n_bins))


@pytest.mark.parametrize("mode", ["momentum", "imm", "momentum-exact-sparse"])
def test_duration_state_space_accepts_unit_velocity_decay(mode: str) -> None:
    emissions = _tiny_emissions()
    config = _config(mode, momentum_velocity_decay=1.0)
    model = StateSpaceReplayModel(mode=mode, config=config)

    score = model.score(emissions, _centers(emissions.n_bins), return_trajectory=False)

    assert np.isfinite(score.log_likelihood)
    assert score.terminal_log_posterior is not None
    assert score.terminal_log_posterior.shape == (emissions.n_bins,)


def _tiny_emissions() -> LogEmissionTensor:
    return LogEmissionTensor(
        log_likelihood=np.log(
            np.array(
                [
                    [0.65, 0.25, 0.08, 0.02],
                    [0.20, 0.60, 0.15, 0.05],
                    [0.05, 0.18, 0.62, 0.15],
                    [0.03, 0.10, 0.24, 0.63],
                ],
                dtype=float,
            )
        ),
        spike_counts=np.zeros((4, 1), dtype=int),
        times=np.array([0.0, 1.0, 2.5, 4.0]),
        dt=1.0,
        cell_ids=np.array([1]),
        n_spikes=0,
    )


def _centers(n_bins: int) -> np.ndarray:
    return np.arange(float(n_bins), dtype=float).reshape(-1, 1)


def _config(mode: str, *, momentum_velocity_decay: float) -> StateSpaceDecoderConfig:
    return StateSpaceDecoderConfig(
        mode=mode,
        stationary_sigma_cm=1.0,
        diffusion_sigma_cm_sqrt_s=1.0,
        momentum_sigma_cm_sqrt_s=1.0,
        momentum_initial_sigma_cm_sqrt_s=1.0,
        momentum_velocity_decay=momentum_velocity_decay,
        momentum_candidate_top_k=0,
        momentum_predicted_candidate_top_k=0,
        max_step_sigma=10.0,
        imm_mode_stickiness=0.8,
    )
