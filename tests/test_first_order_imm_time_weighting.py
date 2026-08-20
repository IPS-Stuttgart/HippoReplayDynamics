from __future__ import annotations

from functools import wraps

import numpy as np

import hipporeplayimm.duration_occupancy as duration_occupancy
from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.first_order_imm_time_weighting import (
    _duration_weighted_mode_summary,
    _has_time_weighting_patch,
    apply_first_order_imm_time_weighting_patch,
)
from hipporeplayimm.state_space import (
    StateSpaceDecoderConfig,
    StateSpaceReplayModel,
)


def test_duration_weighted_mode_summary_respects_partial_final_bin() -> None:
    mode_posterior = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )

    summary = _duration_weighted_mode_summary(
        mode_posterior,
        np.asarray([0.020, 0.005]),
    )

    np.testing.assert_allclose(summary["event_probability"], [0.8, 0.2, 0.0])
    np.testing.assert_allclose(summary["fraction_time_map_stationary"], 0.8)
    np.testing.assert_allclose(summary["fraction_time_map_nonstationary"], 0.2)
    np.testing.assert_allclose(summary["longest_nonstationary_bout_s"], 0.005)


def test_uniform_bin_durations_preserve_row_mean_semantics() -> None:
    mode_posterior = np.asarray(
        [
            [0.8, 0.2, 0.0],
            [0.2, 0.3, 0.5],
        ]
    )

    summary = _duration_weighted_mode_summary(
        mode_posterior,
        np.asarray([0.020, 0.020]),
    )

    np.testing.assert_allclose(
        summary["event_probability"],
        np.mean(mode_posterior, axis=0),
    )
    np.testing.assert_allclose(summary["fraction_time_map_stationary"], 0.5)
    np.testing.assert_allclose(summary["fraction_time_map_nonstationary"], 0.5)
    np.testing.assert_allclose(summary["longest_nonstationary_bout_s"], 0.020)


def test_public_first_order_imm_diagnostics_use_bin_exposure(
    monkeypatch,
) -> None:
    mode_posterior = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    trajectory = np.log(
        np.asarray(
            [
                [0.75, 0.25],
                [0.25, 0.75],
            ]
        )
    )

    def fake_score_first_order_imm_variable(*args, **kwargs):
        return 0.0, trajectory, mode_posterior

    monkeypatch.setattr(
        duration_occupancy,
        "_score_first_order_imm_variable",
        fake_score_first_order_imm_variable,
    )
    emissions = LogEmissionTensor(
        log_likelihood=np.zeros((2, 2), dtype=float),
        spike_counts=np.zeros((2, 1), dtype=int),
        times=np.asarray([0.0, 0.020]),
        dt=0.020,
        cell_ids=np.asarray([1]),
        n_spikes=0,
        bin_durations=np.asarray([0.020, 0.005]),
        transition_durations=np.asarray([0.020]),
    )
    model = StateSpaceReplayModel(
        mode="first-order-imm",
        config=StateSpaceDecoderConfig(mode="first-order-imm"),
    )

    result = model.score(
        emissions,
        np.asarray([[0.0, 0.0], [1.0, 0.0]]),
    )

    diagnostics = result.diagnostics
    np.testing.assert_allclose(diagnostics["state_space_mode_stationary_event_probability"], 0.8)
    np.testing.assert_allclose(diagnostics["state_space_mode_diffusion_event_probability"], 0.2)
    np.testing.assert_allclose(diagnostics["state_space_mode_fragmented_event_probability"], 0.0)
    np.testing.assert_allclose(diagnostics["state_space_imm_nonstationary_event_probability"], 0.2)
    np.testing.assert_allclose(diagnostics["state_space_imm_fraction_time_map_stationary"], 0.8)
    np.testing.assert_allclose(diagnostics["state_space_imm_fraction_time_map_nonstationary"], 0.2)
    np.testing.assert_allclose(diagnostics["state_space_imm_longest_nonstationary_bout_s"], 0.005)
    assert diagnostics["state_space_imm_time_weighting"] == "bin_duration"


def test_time_weighting_patch_detects_marker_below_outer_wrapper(monkeypatch) -> None:
    def already_patched_score(*args, **kwargs):
        return None

    setattr(
        already_patched_score,
        "_first_order_imm_time_weighting_aware",
        True,
    )

    @wraps(already_patched_score)
    def unrelated_outer_wrapper(*args, **kwargs):
        return already_patched_score(*args, **kwargs)

    monkeypatch.setattr(
        duration_occupancy,
        "_score_state_space_duration_with_occupancy",
        unrelated_outer_wrapper,
    )

    assert _has_time_weighting_patch(unrelated_outer_wrapper)
    apply_first_order_imm_time_weighting_patch()

    assert (
        duration_occupancy._score_state_space_duration_with_occupancy
        is unrelated_outer_wrapper
    )
