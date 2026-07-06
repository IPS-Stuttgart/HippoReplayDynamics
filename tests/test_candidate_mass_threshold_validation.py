import numpy as np
import pytest

from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.state_space import StateSpaceDecoderConfig, StateSpaceReplayModel
from hipporeplayimm.state_space_utils import _mass_retaining_candidate_indices


def _toy_emissions() -> LogEmissionTensor:
    return LogEmissionTensor(
        log_likelihood=np.array(
            [
                [0.0, -1.0, -2.0],
                [-2.0, 0.0, -1.0],
                [-1.0, -2.0, 0.0],
            ],
            dtype=float,
        ),
        spike_counts=np.zeros((3, 1), dtype=int),
        times=np.array([0.0, 1.0, 2.0]),
        dt=1.0,
        cell_ids=np.array([1]),
        n_spikes=0,
    )


def _toy_centers() -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
        ],
        dtype=float,
    )


@pytest.mark.parametrize("mass_threshold", [True, np.bool_(True)])
def test_mass_retaining_candidate_indices_rejects_boolean_threshold(mass_threshold) -> None:
    with pytest.raises(TypeError, match="not boolean"):
        _mass_retaining_candidate_indices(np.array([0.0, -1.0, -2.0]), mass_threshold)


@pytest.mark.parametrize("mass_threshold", [True, np.bool_(True)])
def test_state_space_candidate_indices_rejects_boolean_mass_threshold_config(mass_threshold) -> None:
    model = StateSpaceReplayModel(
        mode="momentum",
        config=StateSpaceDecoderConfig(
            mode="momentum",
            momentum_candidate_mass_threshold=mass_threshold,
        ),
    )

    with pytest.raises(TypeError, match="not boolean"):
        model.candidate_indices(_toy_emissions(), _toy_centers())


def test_state_space_candidate_indices_rejects_vector_mass_threshold_config() -> None:
    model = StateSpaceReplayModel(
        mode="momentum",
        config=StateSpaceDecoderConfig(
            mode="momentum",
            momentum_candidate_mass_threshold=np.array([0.5, 0.75]),
        ),
    )

    with pytest.raises(TypeError, match="numeric scalar"):
        model.candidate_indices(_toy_emissions(), _toy_centers())
