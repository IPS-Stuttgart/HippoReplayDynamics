from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.displacement_imm_decay_validation import apply_displacement_imm_decay_validation_patch
from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.state_space_model import StateSpaceDecoderConfig, StateSpaceReplayModel


def _tiny_emissions() -> LogEmissionTensor:
    return LogEmissionTensor(
        log_likelihood=np.zeros((2, 3), dtype=float),
        spike_counts=np.zeros((2, 1), dtype=int),
        times=np.asarray([0.0, 0.02], dtype=float),
        dt=0.02,
        cell_ids=np.asarray([1], dtype=int),
        n_spikes=0,
    )


def _score_displacement_imm_with_decay(decay: object) -> None:
    config = StateSpaceDecoderConfig(
        mode="displacement-imm",
        momentum_velocity_decay=decay,  # type: ignore[arg-type]
        momentum_velocity_decay_tau_s=0.0,
        displacement_radius_bins=1,
    )
    model = StateSpaceReplayModel(mode="displacement-imm", config=config)
    model.score(
        _tiny_emissions(),
        np.asarray([0.0, 1.0, 2.0], dtype=float),
        return_trajectory=False,
    )


def _stale_unvalidated_decay_helper():
    import hipporeplayimm.state_space_displacement_momentum as displacement_momentum

    helper = displacement_momentum._duration_adjusted_decays
    while hasattr(helper, "__wrapped__"):
        helper = helper.__wrapped__
    return helper


def _stale_unvalidated_sparse_decay_helper():
    import hipporeplayimm.state_space_sparse_momentum as sparse_momentum

    helper = sparse_momentum._duration_adjusted_decays
    while hasattr(helper, "__wrapped__"):
        helper = helper.__wrapped__
    return helper


@pytest.mark.parametrize("decay", [True, np.bool_(True), 1.1])
def test_displacement_imm_decay_validation_refreshes_stale_alias(
    monkeypatch: pytest.MonkeyPatch,
    decay: object,
) -> None:
    import hipporeplayimm.state_space_displacement_imm as displacement_imm
    import hipporeplayimm.state_space_displacement_momentum as displacement_momentum

    monkeypatch.setattr(
        displacement_imm,
        "_duration_adjusted_decays",
        _stale_unvalidated_decay_helper(),
    )

    apply_displacement_imm_decay_validation_patch()

    assert displacement_imm._duration_adjusted_decays is displacement_momentum._duration_adjusted_decays
    with pytest.raises((TypeError, ValueError), match="momentum_velocity_decay"):
        _score_displacement_imm_with_decay(decay)


@pytest.mark.parametrize("decay", [True, np.bool_(True), 1.1])
def test_trajectory_imm_decay_validation_refreshes_stale_sparse_alias(
    monkeypatch: pytest.MonkeyPatch,
    decay: object,
) -> None:
    import hipporeplayimm.state_space_sparse_momentum as sparse_momentum
    import hipporeplayimm.state_space_trajectory_imm as trajectory_imm

    monkeypatch.setattr(
        trajectory_imm,
        "_duration_adjusted_decays",
        _stale_unvalidated_sparse_decay_helper(),
    )

    apply_displacement_imm_decay_validation_patch()

    assert trajectory_imm._duration_adjusted_decays is sparse_momentum._duration_adjusted_decays
    config = StateSpaceDecoderConfig(
        momentum_velocity_decay=decay,  # type: ignore[arg-type]
        momentum_velocity_decay_tau_s=0.0,
    )
    with pytest.raises((TypeError, ValueError), match="momentum_velocity_decay"):
        trajectory_imm._duration_adjusted_decays(
            config,
            np.asarray([0.02], dtype=float),
            0.02,
        )
