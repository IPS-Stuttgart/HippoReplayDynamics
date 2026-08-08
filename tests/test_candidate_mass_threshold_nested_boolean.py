from __future__ import annotations

import warnings

import numpy as np
import pytest

from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.state_space import (
    StateSpaceDecoderConfig,
    StateSpaceReplayModel,
    _mass_retaining_candidate_indices,
)


def _nested_object_scalar(value: object) -> np.ndarray:
    inner = np.empty((), dtype=object)
    inner[()] = value
    outer = np.empty((), dtype=object)
    outer[()] = inner
    return outer


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


@pytest.mark.parametrize("boolean", [True, np.bool_(True), False, np.bool_(False)])
def test_mass_retaining_helper_rejects_nested_boolean_threshold(boolean: object) -> None:
    threshold = _nested_object_scalar(np.array(boolean))
    log_emission = np.log(np.array([0.60, 0.25, 0.15]))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(TypeError, match="mass_threshold.*not boolean"):
            _mass_retaining_candidate_indices(
                log_emission,
                mass_threshold=threshold,
                top_k=1,
                min_k=0,
                max_k=0,
            )


def test_mass_retaining_helper_rejects_nested_array_scalar_surrogate() -> None:
    threshold = _nested_object_scalar(np.array([0.75]))
    log_emission = np.log(np.array([0.60, 0.25, 0.15]))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(TypeError, match="mass_threshold.*numeric scalar"):
            _mass_retaining_candidate_indices(
                log_emission,
                mass_threshold=threshold,
                top_k=1,
                min_k=0,
                max_k=0,
            )


def test_state_space_config_rejects_nested_boolean_mass_threshold() -> None:
    model = StateSpaceReplayModel(
        mode="momentum",
        config=StateSpaceDecoderConfig(
            mode="momentum",
            momentum_candidate_mass_threshold=_nested_object_scalar(np.array(True)),
            momentum_candidate_top_k=1,
        ),
    )

    with pytest.raises(
        TypeError,
        match="momentum_candidate_mass_threshold.*not boolean",
    ):
        model.candidate_indices(_toy_emissions(), _toy_centers())


def test_nested_real_mass_threshold_remains_supported() -> None:
    threshold = _nested_object_scalar(np.float64(0.75))
    log_emission = np.log(np.array([0.60, 0.25, 0.15]))

    selected = _mass_retaining_candidate_indices(
        log_emission,
        mass_threshold=threshold,
        top_k=1,
        min_k=0,
        max_k=0,
    )

    assert selected.tolist() == [0, 1]
