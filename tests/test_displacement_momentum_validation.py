import numpy as np
import pytest

from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.sorted_spike_state_space import SortedSpikeStateSpaceReplayModel
from hipporeplayimm.state_space import StateSpaceDecoderConfig, _displacement_lattice


def _tiny_emissions() -> LogEmissionTensor:
    return LogEmissionTensor(
        log_likelihood=np.log(np.array([[0.6, 0.4], [0.4, 0.6]], dtype=float)),
        spike_counts=np.zeros((2, 1), dtype=int),
        times=np.array([0.0, 0.003]),
        dt=0.003,
        cell_ids=np.array([1]),
        n_spikes=0,
    )


@pytest.mark.parametrize(
    "centers,match",
    [
        (np.array([[0.0, 0.0], [np.nan, 1.0]], dtype=float), "finite"),
        (np.empty((0, 2), dtype=float), "shape"),
    ],
)
def test_displacement_lattice_rejects_invalid_bin_centers(centers, match):
    with pytest.raises(ValueError, match=match):
        _displacement_lattice(centers, radius_bins=1)


@pytest.mark.parametrize(
    "field",
    [
        "displacement_position_sigma_cm",
        "displacement_prior_sigma_cm",
        "displacement_transition_sigma_cm_sqrt_s",
    ],
)
def test_displacement_model_rejects_invalid_explicit_scale(field):
    config = StateSpaceDecoderConfig(
        mode="displacement-momentum",
        displacement_radius_bins=0,
        **{field: -1.0},
    )
    model = SortedSpikeStateSpaceReplayModel(mode="displacement-momentum", config=config)

    with pytest.raises(ValueError, match=field):
        model.score(_tiny_emissions(), np.array([[0.0], [1.0]], dtype=float))
