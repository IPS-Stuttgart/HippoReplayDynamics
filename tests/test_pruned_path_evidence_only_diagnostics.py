from __future__ import annotations

import numpy as np
import pytest

import hipporeplayimm.duration_occupancy as duration_occupancy
from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.state_space_model import StateSpaceDecoderConfig, StateSpaceReplayModel


def _synthetic_emissions() -> LogEmissionTensor:
    return LogEmissionTensor(
        log_likelihood=np.log(
            np.array(
                [
                    [0.70, 0.20, 0.08, 0.02],
                    [0.15, 0.65, 0.15, 0.05],
                    [0.05, 0.15, 0.65, 0.15],
                ],
                dtype=float,
            )
        ),
        spike_counts=np.zeros((3, 1), dtype=int),
        times=np.array([0.0, 0.003, 0.006], dtype=float),
        dt=0.003,
        cell_ids=np.array([1], dtype=int),
        n_spikes=0,
    )


def _bin_centers() -> np.ndarray:
    return np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], dtype=float)


def test_duration_occupancy_score_method_uses_duration_aware_runtime_wrapper() -> None:
    assert getattr(StateSpaceReplayModel.score, "_native_duration_occupancy_aware", False)
    assert callable(duration_occupancy._score_state_space_duration_with_occupancy)


@pytest.mark.parametrize(
    ("mode", "diagnostic_key"),
    [
        ("momentum", "state_space_momentum_trajectory_posterior"),
        ("imm", "state_space_imm_trajectory_posterior"),
    ],
)
def test_pruned_path_models_report_evidence_only_trajectory_state(
    mode: str,
    diagnostic_key: str,
) -> None:
    emissions = _synthetic_emissions()
    centers = _bin_centers()
    config = StateSpaceDecoderConfig(
        mode=mode,
        momentum_candidate_top_k=4,
        momentum_predicted_candidate_top_k=0,
    )
    model = StateSpaceReplayModel(mode=mode, config=config)

    full = model.score(emissions, centers, return_trajectory=True)
    evidence_only = model.score(emissions, centers, return_trajectory=False)

    assert np.isfinite(evidence_only.log_likelihood)
    assert evidence_only.log_likelihood == pytest.approx(full.log_likelihood, abs=1e-12)
    assert full.trajectory_log_posterior is not None
    assert full.terminal_log_posterior is not None
    assert evidence_only.trajectory_log_posterior is None
    assert evidence_only.terminal_log_posterior is not None
    np.testing.assert_allclose(
        evidence_only.terminal_log_posterior,
        full.terminal_log_posterior,
        atol=1e-12,
    )
    assert evidence_only.diagnostics["state_space_trajectory_posterior"] == 0
    assert evidence_only.diagnostics[diagnostic_key] == "not_returned_evidence_only"
