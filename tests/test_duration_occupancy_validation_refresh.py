from __future__ import annotations

import numpy as np
import pytest

import hipporeplayimm
from hipporeplayimm import duration_occupancy
from hipporeplayimm.duration_occupancy_mode_transition_validation import _ORIGINAL_ATTR
from hipporeplayimm.state_space_model import StateSpaceDecoderConfig


def _unwrap_helper(helper):
    seen: set[int] = set()
    current = helper
    while id(current) not in seen:
        seen.add(id(current))
        wrapped = getattr(current, "__wrapped__", None)
        if wrapped is not None:
            current = wrapped
            continue
        original = getattr(current, _ORIGINAL_ATTR, None)
        if original is not None:
            current = original
            continue
        return current
    return current


def test_duration_decay_validation_refreshes_replaced_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    hipporeplayimm.apply_runtime_patches()
    raw_helper = _unwrap_helper(duration_occupancy._duration_adjusted_decays)
    monkeypatch.setattr(duration_occupancy, "_duration_adjusted_decays", raw_helper)

    hipporeplayimm.apply_runtime_patches()

    config = StateSpaceDecoderConfig(
        momentum_velocity_decay=0.5,
        momentum_velocity_decay_tau_s=True,  # type: ignore[arg-type]
    )
    with pytest.raises(TypeError, match="momentum_velocity_decay_tau_s"):
        duration_occupancy._duration_adjusted_decays(
            config,
            np.array([0.01, 0.02], dtype=float),
            0.01,
        )


def test_mode_transition_parameter_validation_refreshes_replaced_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    import hipporeplayimm.state_space as ss

    hipporeplayimm.apply_runtime_patches()
    raw_helper = _unwrap_helper(duration_occupancy._mode_transition_matrices)
    monkeypatch.setattr(duration_occupancy, "_mode_transition_matrices", raw_helper)

    hipporeplayimm.apply_runtime_patches()

    with pytest.raises(TypeError, match="imm_switch_tau_s"):
        duration_occupancy._mode_transition_matrices(
            ss,
            3,
            0.95,
            True,  # type: ignore[arg-type]
            np.array([0.01], dtype=float),
        )
