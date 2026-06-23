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
