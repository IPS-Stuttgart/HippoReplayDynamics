from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.encoding import EncodingConfig, EncodingModel
from hipporeplayimm.simulation_recovery import emissions_from_counts, simulate_latent_path, simulate_replay_event


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


@pytest.mark.parametrize("function", [simulate_latent_path, simulate_replay_event])
@pytest.mark.parametrize("n_time", ["2", b"2", np.asarray("2", dtype=object)])
def test_simulation_recovery_rejects_text_n_time(function, n_time: object) -> None:
    with pytest.raises(ValueError, match="n_time.*text"):
        function(
            _minimal_encoding(),
            true_model="stationary",
            n_time=n_time,
            dt=0.02,
            rng=np.random.default_rng(0),
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"dt": "0.02"}, "dt.*text"),
        ({"spike_rate_scale": np.asarray("1.0", dtype=object)}, "spike_rate_scale.*text"),
    ],
)
def test_emissions_from_counts_rejects_text_scalar_parameters(kwargs: dict[str, object], message: str) -> None:
    call_kwargs: dict[str, object] = {"dt": 0.02, "spike_rate_scale": 1.0}
    call_kwargs.update(kwargs)

    with pytest.raises(TypeError, match=message):
        emissions_from_counts(
            _minimal_encoding(),
            np.zeros((1, 1), dtype=int),
            **call_kwargs,  # type: ignore[arg-type]
        )


def test_emissions_from_counts_rejects_text_count_values() -> None:
    with pytest.raises(ValueError, match="counts.*text"):
        emissions_from_counts(
            _minimal_encoding(),
            np.asarray([["0"]], dtype=object),
            dt=0.02,
        )


def test_simulate_latent_path_rejects_text_occupancy_prior() -> None:
    encoding = _minimal_encoding()
    encoding.occupancy_s = np.asarray(["1.0", "1.0"], dtype=object)

    with pytest.raises(ValueError, match="occupancy_s.*text"):
        simulate_latent_path(
            encoding,
            true_model="stationary",
            n_time=2,
            dt=0.02,
            rng=np.random.default_rng(0),
        )
