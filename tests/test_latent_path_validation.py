import numpy as np
import pytest

from hipporeplayimm.encoding import EncodingConfig, EncodingModel
from hipporeplayimm.simulation_recovery import simulate_latent_path


def _small_encoding() -> EncodingModel:
    return EncodingModel(
        x_edges=np.array([0.0, 1.0]),
        y_edges=np.array([0.0, 1.0]),
        bin_centers=np.array([[0.5, 0.5]]),
        rates_hz=np.array([[10.0]]),
        occupancy_s=np.ones(1),
        cell_ids=np.array([1]),
        config=EncodingConfig(bin_size_cm=1.0),
    )


@pytest.mark.parametrize("bad_length", [0, -1, 1.5, np.nan])
def test_simulate_latent_path_rejects_invalid_length(bad_length):
    generator = np.random.default_rng(1)
    with pytest.raises(ValueError, match="n_time must be positive"):
        simulate_latent_path(
            _small_encoding(),
            true_model="stationary",
            n_time=bad_length,
            dt=0.02,
            rng=generator,
        )
