import numpy as np
import pytest

from hipporeplayimm.encoding import EncodingConfig, EncodingModel
from hipporeplayimm.simulation_recovery import simulate_replay_event


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


def test_simulate_replay_event_rejects_fractional_length():
    with pytest.raises(ValueError, match="n_time must be positive integer-valued"):
        simulate_replay_event(
            _small_encoding(),
            true_model="stationary",
            n_time=1.5,
            dt=0.02,
            rng=np.random.default_rng(1),
        )
