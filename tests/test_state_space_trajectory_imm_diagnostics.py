from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.state_space import StateSpaceDecoderConfig, StateSpaceReplayModel


def test_trajectory_imm_reports_first_transition_initial_sigma_for_partial_bins() -> None:
    transition_durations = np.array([0.25, 4.0], dtype=float)
    emissions = LogEmissionTensor(
        log_likelihood=np.log(
            np.array(
                [
                    [0.90, 0.09, 0.01],
                    [0.05, 0.90, 0.05],
                    [0.01, 0.09, 0.90],
                ],
                dtype=float,
            )
        ),
        spike_counts=np.zeros((3, 1), dtype=int),
        times=np.array([0.0, transition_durations[0], transition_durations.sum()]),
        dt=1.0,
        cell_ids=np.array([1], dtype=int),
        n_spikes=0,
    )
    centers = np.array(
        [[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]],
        dtype=float,
    )
    config = StateSpaceDecoderConfig(
        mode="trajectory-imm-exact-sparse",
        diffusion_sigma_cm_sqrt_s=1.0,
        momentum_sigma_cm_sqrt_s=1.0,
        momentum_initial_sigma_cm_sqrt_s=2.0,
        max_step_sigma=20.0,
    )

    score = StateSpaceReplayModel(mode="trajectory-imm-exact-sparse", config=config).score(
        emissions,
        centers,
        return_trajectory=False,
    )

    expected_first_sigma = config.momentum_initial_sigma_cm_sqrt_s * np.sqrt(transition_durations[0])
    previous_median_sigma = float(np.median(config.momentum_initial_sigma_cm_sqrt_s * np.sqrt(transition_durations)))
    assert score.diagnostics["state_space_momentum_initial_transition_sigma_cm"] == pytest.approx(expected_first_sigma)
    assert abs(float(score.diagnostics["state_space_momentum_initial_transition_sigma_cm"]) - previous_median_sigma) > 1e-6
    assert score.diagnostics["state_space_momentum_initial_transition_sigma_cm_per_step"] == "1,4"
