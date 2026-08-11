from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm import apply_runtime_patches
from hipporeplayimm.accuracy_upgrades import ReplayGainConfig, valid_grid_graph_transition


def _nested_object_scalar(value: object) -> np.ndarray:
    inner = np.empty((), dtype=object)
    inner[()] = value
    outer = np.empty((), dtype=object)
    outer[()] = inner
    return outer


def test_accuracy_grid_rejects_nested_boolean_dimension() -> None:
    apply_runtime_patches()

    with pytest.raises(ValueError, match="grid_shape must contain positive integer dimensions"):
        valid_grid_graph_transition(
            (_nested_object_scalar(np.bool_(True)), 2),
            np.ones(2, dtype=bool),
        )


def test_accuracy_grid_rejects_nested_boolean_stay_probability() -> None:
    apply_runtime_patches()

    with pytest.raises(TypeError, match="stay_probability must be numeric, not boolean"):
        valid_grid_graph_transition(
            (2, 2),
            np.ones(4, dtype=bool),
            stay_probability=_nested_object_scalar(np.bool_(False)),
        )


def test_replay_gain_rejects_nested_text_scalar() -> None:
    apply_runtime_patches()

    with pytest.raises(ValueError, match="max_gain must be numeric, not text"):
        ReplayGainConfig(max_gain=_nested_object_scalar("2.0"))


def test_replay_gain_accepts_nested_real_scalar() -> None:
    apply_runtime_patches()

    config = ReplayGainConfig(max_gain=_nested_object_scalar(np.float32(2.0)))

    assert config.max_gain == 2.0
    assert isinstance(config.max_gain, float)
