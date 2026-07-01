from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.state_space_model import StateSpaceDecoderConfig
from hipporeplayimm.state_space_sparse_momentum import _score_sparse_momentum_exact
from hipporeplayimm.state_space_trajectory_imm import _score_trajectory_imm_exact_sparse


def _tiny_emissions() -> LogEmissionTensor:
    return LogEmissionTensor(
        log_likelihood=np.log(
            np.array(
                [
                    [0.60, 0.40],
                    [0.45, 0.55],
                ],
                dtype=float,
            )
        ),
        spike_counts=np.zeros((2, 1), dtype=int),
        times=np.array([0.0, 0.01], dtype=float),
        dt=0.01,
        cell_ids=np.array([1], dtype=int),
        n_spikes=0,
    )


def _tiny_centers() -> np.ndarray:
    return np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)


@pytest.mark.parametrize(
    "field",
    [
        "max_step_sigma",
        "momentum_sigma_cm_sqrt_s",
        "momentum_initial_sigma_cm_sqrt_s",
    ],
)
def test_sparse_momentum_exact_rejects_boolean_numeric_config(field: str) -> None:
    emissions = _tiny_emissions()
    config = StateSpaceDecoderConfig(**{field: np.bool_(True)})

    with pytest.raises(TypeError, match=f"{field}.*not boolean"):
        _score_sparse_momentum_exact(
            emissions,
            _tiny_centers(),
            config,
            emissions.transition_durations,
        )


@pytest.mark.parametrize(
    "field",
    [
        "max_step_sigma",
        "diffusion_sigma_cm_sqrt_s",
        "momentum_sigma_cm_sqrt_s",
        "momentum_initial_sigma_cm_sqrt_s",
    ],
)
def test_trajectory_imm_exact_sparse_rejects_boolean_numeric_config(field: str) -> None:
    emissions = _tiny_emissions()
    config = StateSpaceDecoderConfig(**{field: np.bool_(True)})

    with pytest.raises(TypeError, match=f"{field}.*not boolean"):
        _score_trajectory_imm_exact_sparse(
            emissions,
            _tiny_centers(),
            config,
            emissions.transition_durations,
        )
