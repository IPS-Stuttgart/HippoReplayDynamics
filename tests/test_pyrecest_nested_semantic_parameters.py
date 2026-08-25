from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.pyrecest_models import (
    PyRecEstGoalParticleIMMModel,
    PyRecEstGoalParticleModel,
)


def _nested_object_scalar(value: object, *, depth: int = 2) -> np.ndarray:
    current: object = value
    for _ in range(depth):
        wrapper = np.empty((), dtype=object)
        wrapper[()] = current
        current = wrapper
    return current  # type: ignore[return-value]


@pytest.mark.parametrize(
    ("field", "leaf", "match"),
    [
        ("alpha", "0.8", "alpha must be finite and positive"),
        ("beta", np.bytes_(b"1.0"), "beta must be finite and positive"),
        (
            "initial_velocity_sigma_cm_s",
            bytearray(b"120.0"),
            "initial_velocity_sigma_cm_s must be finite and positive",
        ),
        (
            "jump_probability",
            np.bool_(True),
            r"jump_probability must lie in \[0, 1\]",
        ),
        (
            "goal_reset_probability",
            True,
            r"goal_reset_probability must lie in \[0, 1\]",
        ),
        (
            "position_proposal_probability",
            memoryview(b"0.5"),
            r"position_proposal_probability must lie in \[0, 1\]",
        ),
        (
            "alpha",
            np.complex128(0.8 + 0.4j),
            "alpha must be finite and positive",
        ),
    ],
)
def test_pyrecest_goal_particle_rejects_nested_semantic_scalar_aliases(
    field: str,
    leaf: object,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        PyRecEstGoalParticleModel(**{field: _nested_object_scalar(leaf)})


def test_pyrecest_imm_rejects_nested_boolean_velocity_decay() -> None:
    with pytest.raises(TypeError, match="momentum_velocity_decay must be numeric, not boolean"):
        PyRecEstGoalParticleIMMModel(
            momentum_velocity_decay=_nested_object_scalar(np.bool_(True))
        )


def test_pyrecest_rejects_cyclic_zero_dimensional_scalar_wrapper() -> None:
    cyclic = np.empty((), dtype=object)
    cyclic[()] = cyclic

    with pytest.raises(ValueError, match="alpha must be finite and positive"):
        PyRecEstGoalParticleModel(alpha=cyclic)
