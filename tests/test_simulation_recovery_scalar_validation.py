from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.encoding import EncodingConfig, EncodingModel
from hipporeplayimm.simulation_recovery import (
    emissions_from_counts,
    simulate_latent_path,
    simulate_replay_event,
)


def _minimal_encoding() -> EncodingModel:
    return EncodingModel(
        x_edges=np.asarray([0.0, 1.0, 2.0], dtype=float),
        y_edges=np.asarray([0.0, 1.0], dtype=float),
        bin_centers=np.asarray([[0.5, 0.5], [1.5, 0.5]], dtype=float),
        rates_hz=np.ones((1, 2), dtype=float),
        occupancy_s=np.ones(2, dtype=float),
        cell_ids=np.asarray([1], dtype=int),
        config=EncodingConfig(),
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"dt": True}, "dt"),
        ({"dt": np.bool_(True)}, "dt"),
        ({"dt": np.asarray(True, dtype=object)}, "dt"),
        ({"spike_rate_scale": True}, "spike_rate_scale"),
        ({"spike_rate_scale": np.asarray([True], dtype=object)}, "spike_rate_scale"),
    ],
)
def test_emissions_from_counts_rejects_boolean_scalars(kwargs: dict[str, object], message: str) -> None:
    call_kwargs: dict[str, object] = {"dt": 0.02, "spike_rate_scale": 1.0}
    call_kwargs.update(kwargs)

    with pytest.raises(TypeError, match=message):
        emissions_from_counts(
            _minimal_encoding(),
            np.zeros((1, 1), dtype=int),
            **call_kwargs,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"dt": True}, "dt"),
        ({"spike_rate_scale": np.bool_(True)}, "spike_rate_scale"),
    ],
)
def test_simulate_replay_event_rejects_boolean_scalars(kwargs: dict[str, object], message: str) -> None:
    call_kwargs: dict[str, object] = {
        "true_model": "stationary",
        "n_time": 2,
        "dt": 0.02,
        "rng": np.random.default_rng(0),
        "spike_rate_scale": 1.0,
    }
    call_kwargs.update(kwargs)

    with pytest.raises(TypeError, match=message):
        simulate_replay_event(
            _minimal_encoding(),
            **call_kwargs,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("function", [simulate_latent_path, simulate_replay_event])
def test_simulation_recovery_rejects_boolean_n_time(function) -> None:
    with pytest.raises(TypeError, match="n_time"):
        function(
            _minimal_encoding(),
            true_model="stationary",
            n_time=True,
            dt=0.02,
            rng=np.random.default_rng(0),
        )


@pytest.mark.parametrize("function", [simulate_latent_path, simulate_replay_event])
@pytest.mark.parametrize("n_time", [0, 1.5])
def test_simulation_recovery_rejects_invalid_n_time(function, n_time: object) -> None:
    with pytest.raises(ValueError, match="n_time"):
        function(
            _minimal_encoding(),
            true_model="stationary",
            n_time=n_time,
            dt=0.02,
            rng=np.random.default_rng(0),
        )
