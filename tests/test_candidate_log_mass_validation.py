import numpy as np
import pytest

from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.models import CandidateKinematicModel
from hipporeplayimm.state_space import (
    StateSpaceDecoderConfig,
    StateSpaceReplayModel,
    _candidate_log_masses,
)


def _emissions(log_likelihood: np.ndarray) -> LogEmissionTensor:
    n_time = int(log_likelihood.shape[0])
    return LogEmissionTensor(
        log_likelihood=log_likelihood,
        spike_counts=np.zeros((n_time, 1), dtype=int),
        times=np.arange(n_time, dtype=float),
        dt=1.0,
        cell_ids=np.array([1]),
        n_spikes=0,
    )


def test_candidate_log_masses_reject_all_negative_infinity_rows():
    with pytest.raises(ValueError, match="row 1"):
        _candidate_log_masses(
            np.array([[0.0, -np.inf], [-np.inf, -np.inf]]),
            [np.array([0]), np.array([0])],
        )


def test_state_space_candidate_paths_reject_support_without_finite_mass():
    emissions = _emissions(
        np.array(
            [
                [0.0, -np.inf],
                [0.0, -np.inf],
                [0.0, -np.inf],
            ]
        )
    )
    centers = np.array([[0.0, 0.0], [1.0, 0.0]])
    candidates = [np.array([0]), np.array([1]), np.array([0])]
    model = StateSpaceReplayModel(
        mode="momentum",
        config=StateSpaceDecoderConfig(mode="momentum", momentum_candidate_top_k=2),
    )

    with pytest.raises(ValueError, match=r"candidate_indices\[1\].*finite"):
        model.score(emissions, centers, candidate_indices=candidates)


@pytest.mark.parametrize("mode", ["momentum", "imm"])
def test_occupancy_candidate_log_mass_diagnostics_use_active_support(mode: str):
    emissions = _emissions(
        np.log(
            np.array(
                [
                    [1.0e-6, 1.0],
                    [1.0e-6, 1.0],
                    [1.0e-6, 1.0],
                ],
                dtype=float,
            )
        )
    )
    centers = np.array([[0.0, 0.0], [1.0, 0.0]])
    candidates = [np.array([0]), np.array([0]), np.array([0])]
    model = StateSpaceReplayModel(
        mode=mode,
        config=StateSpaceDecoderConfig(
            mode=mode,
            momentum_candidate_top_k=2,
            momentum_predicted_candidate_top_k=0,
            valid_occupancy_threshold_s=0.5,
        ),
    )

    score = model.score(
        emissions,
        centers,
        candidate_indices=candidates,
        occupancy_s=np.array([1.0, 0.0]),
    )

    assert np.isclose(score.diagnostics["mean_candidate_log_mass"], 0.0)
    assert np.isclose(score.diagnostics["min_candidate_log_mass"], 0.0)


def test_legacy_candidate_model_rejects_support_without_finite_mass():
    emissions = _emissions(
        np.array(
            [
                [0.0, -np.inf],
                [0.0, -np.inf],
                [0.0, -np.inf],
            ]
        )
    )
    centers = np.array([[0.0, 0.0], [1.0, 0.0]])
    candidates = [np.array([0]), np.array([1]), np.array([0])]
    model = CandidateKinematicModel(mode="diffusion", top_k=2, diffusion_sigma_cm=1.0)

    with pytest.raises(ValueError, match=r"candidate_indices\[1\].*finite"):
        model.score(emissions, centers, candidate_indices=candidates)
