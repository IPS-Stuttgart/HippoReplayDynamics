from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from hipporeplayimm.displacement_imm_decay_validation import apply_displacement_imm_decay_validation_patch


def _raw_duration_decay_helper(module):
    helper = module._duration_adjusted_decays
    while hasattr(helper, "__wrapped__"):
        helper = helper.__wrapped__
    return helper


def _invalid_decay_config():
    return SimpleNamespace(momentum_velocity_decay=1.1, momentum_velocity_decay_tau_s=0.0)


def test_displacement_decay_patch_refreshes_replaced_source_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    import hipporeplayimm.state_space_displacement_imm as displacement_imm
    import hipporeplayimm.state_space_displacement_momentum as displacement_momentum
    from hipporeplayimm.model_parameter_validation import _DISPLACEMENT_MOMENTUM_DECAY_PATCHED_FLAG

    monkeypatch.setattr(
        displacement_momentum,
        "_duration_adjusted_decays",
        _raw_duration_decay_helper(displacement_momentum),
    )
    monkeypatch.setattr(displacement_momentum, _DISPLACEMENT_MOMENTUM_DECAY_PATCHED_FLAG, True, raising=False)

    apply_displacement_imm_decay_validation_patch()

    assert displacement_imm._duration_adjusted_decays is displacement_momentum._duration_adjusted_decays
    assert getattr(displacement_momentum._duration_adjusted_decays, _DISPLACEMENT_MOMENTUM_DECAY_PATCHED_FLAG, False)
    with pytest.raises((TypeError, ValueError), match="momentum_velocity_decay"):
        displacement_momentum._duration_adjusted_decays(_invalid_decay_config(), np.asarray([0.02]), 0.02)


def test_sparse_decay_patch_refreshes_replaced_source_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    import hipporeplayimm.state_space_sparse_momentum as sparse_momentum
    import hipporeplayimm.state_space_trajectory_imm as trajectory_imm
    from hipporeplayimm.model_parameter_validation import _SPARSE_MOMENTUM_DECAY_PATCHED_FLAG

    monkeypatch.setattr(
        sparse_momentum,
        "_duration_adjusted_decays",
        _raw_duration_decay_helper(sparse_momentum),
    )
    monkeypatch.setattr(sparse_momentum, _SPARSE_MOMENTUM_DECAY_PATCHED_FLAG, True, raising=False)

    apply_displacement_imm_decay_validation_patch()

    assert trajectory_imm._duration_adjusted_decays is sparse_momentum._duration_adjusted_decays
    assert getattr(sparse_momentum._duration_adjusted_decays, _SPARSE_MOMENTUM_DECAY_PATCHED_FLAG, False)
    with pytest.raises((TypeError, ValueError), match="momentum_velocity_decay"):
        sparse_momentum._duration_adjusted_decays(_invalid_decay_config(), np.asarray([0.02]), 0.02)
