from __future__ import annotations

import numpy as np

from hipporeplayimm import duration_occupancy as duration_module
from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.state_space import StateSpaceDecoderConfig, StateSpaceReplayModel


def test_state_space_score_records_partial_bin_transition_durations() -> None:
    emissions = LogEmissionTensor(
        log_likelihood=np.zeros((3, 4), dtype=float),
        spike_counts=np.zeros((3, 1), dtype=int),
        times=np.array([0.005, 0.020, 0.055], dtype=float),
        dt=0.020,
        cell_ids=np.array([1], dtype=int),
        n_spikes=0,
    )
    bin_centers = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=float,
    )
    model = StateSpaceReplayModel(
        mode="diffusion",
        config=StateSpaceDecoderConfig(mode="diffusion", diffusion_sigma_cm_sqrt_s=10.0),
    )

    score = model.score(emissions, bin_centers)

    assert np.isfinite(score.log_likelihood)
    assert score.diagnostics["state_space_transition_durations"] == "0.015,0.035"


def test_first_order_imm_content_diagnostics_receive_partial_transition_durations(monkeypatch) -> None:
    emissions = LogEmissionTensor(
        log_likelihood=np.zeros((3, 4), dtype=float),
        spike_counts=np.zeros((3, 1), dtype=int),
        times=np.array([0.000, 0.010, 0.050], dtype=float),
        dt=0.020,
        cell_ids=np.array([1], dtype=int),
        n_spikes=0,
    )
    bin_centers = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=float,
    )
    captured: dict[str, object] = {}

    def fake_content_diagnostics(mode_post, trajectory, centers, dt_s):
        captured["dt"] = dt_s
        captured["transition_durations"] = getattr(dt_s, "transition_durations", None)
        return {"state_space_imm_duration_metadata_seen": 1}

    monkeypatch.setattr(duration_module, "_first_order_imm_content_diagnostics", fake_content_diagnostics)
    model = StateSpaceReplayModel(
        mode="first-order-imm",
        config=StateSpaceDecoderConfig(mode="first-order-imm", diffusion_sigma_cm_sqrt_s=10.0),
    )

    score = model.score(emissions, bin_centers)

    assert score.diagnostics["state_space_imm_duration_metadata_seen"] == 1
    assert np.isclose(float(captured["dt"]), 0.020)
    np.testing.assert_allclose(captured["transition_durations"], np.array([0.010, 0.040]))
