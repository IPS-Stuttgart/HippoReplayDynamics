from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.state_space import (
    StateSpaceDecoderConfig,
    StateSpaceReplayModel,
    _mass_retaining_candidate_indices,
    _top_candidate_indices,
)


def _emissions() -> LogEmissionTensor:
    log_likelihood = np.log(
        np.array(
            [
                [0.70, 0.20, 0.10],
                [0.10, 0.80, 0.10],
                [0.15, 0.25, 0.60],
            ],
            dtype=float,
        )
    )
    return LogEmissionTensor(
        log_likelihood=log_likelihood,
        spike_counts=np.zeros((log_likelihood.shape[0], 1), dtype=int),
        times=np.arange(log_likelihood.shape[0], dtype=float),
        dt=1.0,
        cell_ids=np.array([1]),
        n_spikes=0,
    )


@pytest.mark.parametrize("invalid", [1.5, np.float64(2.0), np.asarray(2.0)])
def test_top_candidate_indices_rejects_float_counts(invalid: object) -> None:
    with pytest.raises(TypeError, match="top_k.*integer scalar"):
        _top_candidate_indices(np.log(np.array([0.6, 0.3, 0.1])), invalid)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"mass_threshold": 0.75, "top_k": 2.0}, "top_k"),
        ({"mass_threshold": 0.75, "min_k": 1.5}, "min_k"),
        ({"mass_threshold": 0.75, "max_k": np.float64(2.0)}, "max_k"),
    ],
)
def test_mass_retaining_candidates_rejects_float_counts(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(TypeError, match=rf"{message}.*integer scalar"):
        _mass_retaining_candidate_indices(
            np.log(np.array([0.6, 0.3, 0.1])),
            **kwargs,
        )


@pytest.mark.parametrize(
    "field",
    [
        "momentum_candidate_top_k",
        "momentum_candidate_min_k",
        "momentum_candidate_max_k",
        "momentum_predicted_candidate_top_k",
    ],
)
def test_state_space_model_rejects_float_candidate_counts_before_coercion(field: str) -> None:
    config_values: dict[str, object] = {
        "mode": "momentum",
        "momentum_candidate_mass_threshold": 0.75,
        "momentum_candidate_top_k": 2,
        "momentum_candidate_min_k": 1,
        "momentum_candidate_max_k": 0,
        "momentum_predicted_candidate_top_k": 0,
    }
    config_values[field] = 1.5
    model = StateSpaceReplayModel(
        mode="momentum",
        config=StateSpaceDecoderConfig(**config_values),
    )

    with pytest.raises(TypeError, match=rf"{field}.*integer scalar"):
        model.candidate_indices(_emissions())


def test_state_space_candidate_counts_accept_numpy_integers() -> None:
    model = StateSpaceReplayModel(
        mode="momentum",
        config=StateSpaceDecoderConfig(
            mode="momentum",
            momentum_candidate_mass_threshold=0.75,
            momentum_candidate_top_k=np.int64(2),
            momentum_candidate_min_k=np.int64(1),
            momentum_candidate_max_k=np.int64(0),
            momentum_predicted_candidate_top_k=np.int64(0),
        ),
    )

    candidates = model.candidate_indices(_emissions())

    assert len(candidates) == 3
    assert all(candidate.dtype.kind in "iu" for candidate in candidates)
