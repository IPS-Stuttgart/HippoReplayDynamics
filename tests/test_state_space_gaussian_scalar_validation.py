from __future__ import annotations

import numpy as np
import pytest

import hipporeplayimm
import hipporeplayimm.state_space as state_space
from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.state_space_model import StateSpaceDecoderConfig, StateSpaceReplayModel


def _uniform_emissions() -> LogEmissionTensor:
    return LogEmissionTensor(
        log_likelihood=np.zeros((2, 2), dtype=float),
        spike_counts=np.empty((2, 0), dtype=int),
        times=np.array([0.0, 1.0]),
        dt=1.0,
        cell_ids=np.empty(0, dtype=int),
        n_spikes=0,
    )


@pytest.mark.parametrize(
    ("value", "exception", "message"),
    [
        (True, TypeError, "not boolean"),
        (np.bool_(True), TypeError, "not boolean"),
        (np.array([1.0]), TypeError, "real numeric scalar"),
        ("1.0", TypeError, "not string"),
        (1.0 + 0.5j, TypeError, "real-valued, not complex"),
        (10**10000, ValueError, "finite and positive"),
    ],
)
def test_pairwise_gaussian_rejects_invalid_sigma_scalars(
    value: object,
    exception: type[Exception],
    message: str,
) -> None:
    centers = np.array([[0.0], [1.0]], dtype=float)

    with pytest.raises(exception, match=message):
        state_space._full_grid_normalized_pairwise_gaussian_log_prob(
            centers[:1],
            centers,
            centers,
            value,
        )


def test_dense_gaussian_transition_rejects_boolean_max_step() -> None:
    centers = np.array([[0.0], [1.0]], dtype=float)

    with pytest.raises(TypeError, match="max_step_sigma.*not boolean"):
        state_space._gaussian_transition_matrix(
            centers,
            1.0,
            True,
        )


def test_state_space_imm_rejects_boolean_stationary_sigma_after_patch_refresh() -> None:
    config = StateSpaceDecoderConfig(
        mode="imm",
        stationary_sigma_cm=True,
        diffusion_sigma_cm_sqrt_s=1.0,
        momentum_sigma_cm_sqrt_s=1.0,
        momentum_initial_sigma_cm_sqrt_s=1.0,
        momentum_candidate_top_k=0,
        momentum_predicted_candidate_top_k=0,
    )
    model = StateSpaceReplayModel(mode="imm", config=config)
    centers = np.array([[0.0], [1.0]], dtype=float)

    hipporeplayimm.apply_runtime_patches()
    hipporeplayimm.apply_runtime_patches()

    with pytest.raises(TypeError, match="sigma_cm.*not boolean"):
        model.score(_uniform_emissions(), centers)
