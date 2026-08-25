from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.encoding import EncodingConfig, EncodingModel
from hipporeplayimm.simulation_recovery import (
    emissions_from_counts,
    simulate_latent_path,
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


def _nested_scalar(value: object) -> np.ndarray:
    inner = np.empty((), dtype=object)
    inner[()] = np.asarray(value)
    outer = np.empty((), dtype=object)
    outer[()] = inner
    return outer


def _nested_matrix(value: object) -> np.ndarray:
    matrix = np.empty((1, 1), dtype=object)
    matrix[0, 0] = _nested_scalar(value)
    return matrix


def _nested_vector(*values: object) -> np.ndarray:
    vector = np.empty(len(values), dtype=object)
    for index, value in enumerate(values):
        vector[index] = _nested_scalar(value)
    return vector


@pytest.mark.parametrize(
    ("count", "message"),
    [
        (np.bool_(True), "boolean"),
        (np.complex128(3.0 + 2.0j), "complex"),
        ("3", "text"),
    ],
)
def test_emissions_from_counts_rejects_nested_lossy_count_scalars(
    count: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        emissions_from_counts(
            _minimal_encoding(),
            _nested_matrix(count),
            dt=0.02,
        )


@pytest.mark.parametrize(
    ("occupancy", "message"),
    [
        (_nested_vector(np.bool_(True), 1.0), "finite nonnegative"),
        (_nested_vector("1.0", 1.0), "text"),
        (_nested_vector(np.complex128(1.0 + 2.0j), 1.0), "real"),
    ],
)
def test_simulate_latent_path_rejects_nested_lossy_occupancy_scalars(
    occupancy: np.ndarray,
    message: str,
) -> None:
    encoding = _minimal_encoding()
    encoding.occupancy_s = occupancy

    with pytest.raises(ValueError, match=message):
        simulate_latent_path(
            encoding,
            true_model="stationary",
            n_time=2,
            dt=0.02,
            rng=np.random.default_rng(0),
        )


def test_simulate_latent_path_rejects_nested_boolean_n_time() -> None:
    with pytest.raises(TypeError, match="n_time.*boolean"):
        simulate_latent_path(
            _minimal_encoding(),
            true_model="stationary",
            n_time=_nested_scalar(np.bool_(True)),
            dt=0.02,
            rng=np.random.default_rng(0),
        )


def test_simulate_latent_path_rejects_nested_boolean_dt() -> None:
    with pytest.raises(TypeError, match="dt.*boolean"):
        simulate_latent_path(
            _minimal_encoding(),
            true_model="stationary",
            n_time=2,
            dt=_nested_scalar(np.bool_(True)),
            rng=np.random.default_rng(0),
        )


def test_simulation_recovery_accepts_nested_real_scalars() -> None:
    encoding = _minimal_encoding()
    encoding.occupancy_s = _nested_vector(1.0, 1.0)

    emissions = emissions_from_counts(
        encoding,
        _nested_matrix(3.0),
        dt=_nested_scalar(0.02),
    )
    path = simulate_latent_path(
        encoding,
        true_model="stationary",
        n_time=_nested_scalar(2),
        dt=_nested_scalar(0.02),
        rng=np.random.default_rng(0),
    )

    assert int(emissions.spike_counts[0, 0]) == 3
    assert path.shape == (2,)
