from __future__ import annotations

import numpy as np

import hipporeplayimm.duration_occupancy as duration_occupancy
from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.state_space import StateSpaceDecoderConfig, StateSpaceReplayModel


def test_first_order_imm_content_diagnostics_respect_transition_durations(monkeypatch) -> None:
    transition_durations = np.array([0.25, 0.5], dtype=float)
    emissions = LogEmissionTensor(
        log_likelihood=np.zeros((3, 2), dtype=float),
        spike_counts=np.zeros((3, 1), dtype=int),
        times=np.array([0.0, 0.25, 0.75], dtype=float),
        dt=1.0,
        cell_ids=np.array([1]),
        n_spikes=0,
    )
    bin_centers = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)
    trajectory_log_posterior = np.array(
        [
            [0.0, -np.inf],
            [-np.inf, 0.0],
            [-np.inf, 0.0],
        ],
        dtype=float,
    )
    mode_posterior = np.array(
        [
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )

    def score_first_order_imm_variable(*args, **kwargs):
        return 0.0, trajectory_log_posterior, mode_posterior

    monkeypatch.setattr(
        duration_occupancy,
        "_score_first_order_imm_variable",
        score_first_order_imm_variable,
    )

    score = StateSpaceReplayModel(
        mode="first-order-imm",
        config=StateSpaceDecoderConfig(mode="first-order-imm"),
    ).score(emissions, bin_centers)

    diagnostics = score.diagnostics
    expected_longest_bout_s = float(np.median(transition_durations) + transition_durations.sum())
    scalar_dt_fallback_bout_s = 3.0
    assert np.isclose(
        diagnostics["state_space_imm_longest_nonstationary_bout_s"],
        expected_longest_bout_s,
    )
    assert not np.isclose(
        diagnostics["state_space_imm_longest_nonstationary_bout_s"],
        scalar_dt_fallback_bout_s,
    )
    assert np.isclose(
        diagnostics["state_space_imm_posterior_path_speed_cm_s"],
        1.0 / float(transition_durations.sum()),
    )
