from __future__ import annotations

import numpy as np
import pytest

import hipporeplayimm
from hipporeplayimm import state_space_displacement_imm as displacement_imm
from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.state_space_model import StateSpaceDecoderConfig, StateSpaceReplayModel


def _toy_emissions() -> LogEmissionTensor:
    return LogEmissionTensor(
        log_likelihood=np.zeros((2, 4), dtype=float),
        spike_counts=np.zeros((2, 1), dtype=int),
        times=np.array([0.0, 0.02], dtype=float),
        dt=0.02,
        cell_ids=np.array([1], dtype=int),
        n_spikes=0,
    )


def _toy_bin_centers() -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=float,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"imm_mode_stickiness": True},
        {"imm_mode_stickiness": np.bool_(False)},
        {"imm_switch_tau_s": True},
        {"imm_switch_tau_s": np.bool_(False)},
    ],
)
def test_displacement_imm_score_rejects_boolean_mode_transition_parameters(kwargs: dict[str, object]) -> None:
    hipporeplayimm.apply_runtime_patches()
    config = StateSpaceDecoderConfig(mode="displacement-imm", **kwargs)  # type: ignore[arg-type]
    model = StateSpaceReplayModel(mode="displacement-imm", config=config)

    with pytest.raises(TypeError, match="not boolean"):
        model.score(_toy_emissions(), _toy_bin_centers())


@pytest.mark.parametrize(
    ("mode_stickiness", "switch_tau_s"),
    [
        (True, 0.0),
        (np.bool_(False), 0.0),
        (0.95, True),
        (0.95, np.bool_(False)),
    ],
)
def test_displacement_imm_mode_transition_helper_rejects_boolean_parameters(
    mode_stickiness: object,
    switch_tau_s: object,
) -> None:
    hipporeplayimm.apply_runtime_patches()

    with pytest.raises(TypeError, match="not boolean"):
        displacement_imm._mode_transition_matrices(
            4,
            mode_stickiness,
            switch_tau_s,
            np.array([0.02], dtype=float),
        )
