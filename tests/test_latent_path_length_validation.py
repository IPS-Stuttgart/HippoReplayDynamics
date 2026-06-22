import numpy as np

from hipporeplayimm.encoding import EncodingConfig, EncodingModel
from hipporeplayimm.simulation_recovery import simulate_latent_path


def _encoding_for_validation() -> EncodingModel:
    return EncodingModel(
        x_edges=np.array([0.0, 1.0]),
        y_edges=np.array([0.0, 1.0]),
        bin_centers=np.array([[0.5, 0.5]]),
        rates_hz=np.array([[10.0]]),
        occupancy_s=np.ones(1),
        cell_ids=np.array([1]),
        config=EncodingConfig(bin_size_cm=1.0),
    )


def test_simulate_latent_path_validates_empty_length():
    try:
        simulate_latent_path(
            _encoding_for_validation(),
            true_model="stationary",
            n_time=int(False),
            dt=0.02,
            rng=np.random.default_rng(1),
        )
    except ValueError as exc:
        assert "n_time must be positive" in str(exc)
    else:
        raise AssertionError("simulate_latent_path accepted an empty path length")
