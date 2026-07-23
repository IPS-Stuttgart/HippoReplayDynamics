from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.state_space import StateSpaceDecoderConfig, StateSpaceReplayModel


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


def _model_with_inactive_bound(field: str, value: object) -> StateSpaceReplayModel:
    config_values: dict[str, object] = {
        "mode": "momentum",
        "momentum_candidate_mass_threshold": None,
        "momentum_candidate_top_k": 2,
        "momentum_candidate_min_k": 1,
        "momentum_candidate_max_k": 0,
        "momentum_predicted_candidate_top_k": 0,
    }
    config_values[field] = value
    return StateSpaceReplayModel(
        mode="momentum",
        config=StateSpaceDecoderConfig(**config_values),
    )


@pytest.mark.parametrize(
    "field",
    ["momentum_candidate_min_k", "momentum_candidate_max_k"],
)
def test_state_space_model_rejects_fractional_inactive_candidate_bounds(field: str) -> None:
    model = _model_with_inactive_bound(field, 1.5)

    with pytest.raises(TypeError, match=rf"{field}.*integer"):
        model.candidate_indices(_emissions())


@pytest.mark.parametrize(
    "field",
    ["momentum_candidate_min_k", "momentum_candidate_max_k"],
)
def test_state_space_model_rejects_negative_inactive_candidate_bounds(field: str) -> None:
    model = _model_with_inactive_bound(field, -1)

    with pytest.raises(ValueError, match=rf"{field}.*nonnegative"):
        model.candidate_indices(_emissions())


def test_state_space_candidate_config_accepts_integer_valued_float_counts() -> None:
    model = StateSpaceReplayModel(
        mode="momentum",
        config=StateSpaceDecoderConfig(
            mode="momentum",
            momentum_candidate_mass_threshold=None,
            momentum_candidate_top_k=2.0,
            momentum_candidate_min_k=1.0,
            momentum_candidate_max_k=0.0,
            momentum_predicted_candidate_top_k=0.0,
        ),
    )

    candidates = model.candidate_indices(_emissions())

    assert len(candidates) == 3
    assert all(candidate.dtype.kind in "iu" for candidate in candidates)
