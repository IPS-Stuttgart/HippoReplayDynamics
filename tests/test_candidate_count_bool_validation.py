from __future__ import annotations

import numpy as np
import pytest

import hipporeplayimm
from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.state_space_model import StateSpaceDecoderConfig, StateSpaceReplayModel
from hipporeplayimm.state_space_utils import (
    _mass_retaining_candidate_indices,
    _top_candidate_indices,
)


@pytest.mark.parametrize("value", [True, np.bool_(False), np.asarray(True, dtype=object)])
def test_top_candidate_indices_rejects_boolean_top_k(value: object) -> None:
    with pytest.raises(TypeError, match="top_k.*not boolean"):
        _top_candidate_indices(np.array([0.0, -1.0, -2.0], dtype=float), value)


@pytest.mark.parametrize("name", ["top_k", "min_k", "max_k"])
@pytest.mark.parametrize("value", [True, np.bool_(False), np.asarray(True, dtype=object)])
def test_mass_retaining_candidate_indices_rejects_boolean_counts(name: str, value: object) -> None:
    kwargs: dict[str, object] = {"mass_threshold": 0.8, name: value}
    with pytest.raises(TypeError, match=f"{name}.*not boolean"):
        _mass_retaining_candidate_indices(
            np.log(np.array([0.7, 0.2, 0.1], dtype=float)),
            **kwargs,
        )


@pytest.mark.parametrize("value", [True, np.bool_(False), np.asarray(True, dtype=object)])
def test_state_space_predicted_candidate_top_k_rejects_boolean_count(value: object) -> None:
    hipporeplayimm.apply_runtime_patches()
    emissions = LogEmissionTensor(
        log_likelihood=np.log(
            np.array(
                [
                    [0.70, 0.20, 0.10],
                    [0.10, 0.80, 0.10],
                    [0.15, 0.25, 0.60],
                ],
                dtype=float,
            )
        ),
        spike_counts=np.zeros((3, 0), dtype=int),
        times=np.array([0.0, 0.02, 0.04], dtype=float),
        dt=0.02,
        cell_ids=np.empty(0, dtype=int),
        n_spikes=0,
    )
    config = StateSpaceDecoderConfig(momentum_predicted_candidate_top_k=value)  # type: ignore[arg-type]
    model = StateSpaceReplayModel(mode="momentum", config=config)

    with pytest.raises(TypeError, match="momentum_predicted_candidate_top_k.*not boolean"):
        model.candidate_indices(
            emissions,
            np.array([[0.0], [1.0], [2.0]], dtype=float),
        )
