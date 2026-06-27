from __future__ import annotations

import numpy as np
import pytest

import hipporeplayimm
from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.state_space import StateSpaceDecoderConfig, StateSpaceReplayModel, _displacement_lattice


def _tiny_emissions() -> LogEmissionTensor:
    return LogEmissionTensor(
        log_likelihood=np.log(np.array([[0.6, 0.4], [0.4, 0.6]], dtype=float)),
        spike_counts=np.zeros((2, 1), dtype=int),
        times=np.array([0.0, 0.003], dtype=float),
        dt=0.003,
        cell_ids=np.array([1], dtype=int),
        n_spikes=0,
    )


def _centers() -> np.ndarray:
    return np.array([[0.0], [1.0]], dtype=float)


@pytest.mark.parametrize("radius", [True, np.bool_(True)])
def test_displacement_lattice_rejects_boolean_radius(radius: object) -> None:
    hipporeplayimm.apply_runtime_patches()

    with pytest.raises(TypeError, match="displacement_radius_bins"):
        _displacement_lattice(_centers(), radius_bins=radius)


@pytest.mark.parametrize("mode", ["displacement-momentum", "displacement-imm"])
def test_displacement_models_reject_boolean_radius(mode: str) -> None:
    hipporeplayimm.apply_runtime_patches()
    config = StateSpaceDecoderConfig(
        mode=mode,
        displacement_radius_bins=True,  # type: ignore[arg-type]
    )
    model = StateSpaceReplayModel(mode=mode, config=config)

    with pytest.raises(TypeError, match="displacement_radius_bins"):
        model.score(_tiny_emissions(), _centers(), return_trajectory=False)


@pytest.mark.parametrize("mode", ["displacement-momentum", "displacement-imm"])
@pytest.mark.parametrize(
    "field",
    [
        "displacement_position_sigma_cm",
        "displacement_prior_sigma_cm",
        "displacement_transition_sigma_cm_sqrt_s",
    ],
)
def test_displacement_models_reject_boolean_explicit_scales(mode: str, field: str) -> None:
    hipporeplayimm.apply_runtime_patches()
    config = StateSpaceDecoderConfig(
        mode=mode,
        displacement_radius_bins=0,
        **{field: True},  # type: ignore[arg-type]
    )
    model = StateSpaceReplayModel(mode=mode, config=config)

    with pytest.raises(TypeError, match=field):
        model.score(_tiny_emissions(), _centers(), return_trajectory=False)


def test_displacement_default_transition_sigma_rejects_boolean_fallback() -> None:
    hipporeplayimm.apply_runtime_patches()
    config = StateSpaceDecoderConfig(
        mode="displacement-momentum",
        displacement_radius_bins=0,
        displacement_transition_sigma_cm_sqrt_s=0.0,
        momentum_sigma_cm_sqrt_s=True,  # type: ignore[arg-type]
    )
    model = StateSpaceReplayModel(mode="displacement-momentum", config=config)

    with pytest.raises(TypeError, match="momentum_sigma_cm_sqrt_s"):
        model.score(_tiny_emissions(), _centers(), return_trajectory=False)
