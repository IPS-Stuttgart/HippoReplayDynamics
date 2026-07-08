from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import hipporeplayimm
from hipporeplayimm import state_space_trajectory_imm as trajectory_imm

_TEXT_NUMERIC_VALUES = ("0.9", b"0.9", np.str_("0.9"), np.bytes_(b"0.9"))


@pytest.mark.parametrize("value", [True, np.bool_(False)])
def test_trajectory_imm_rejects_boolean_mode_stickiness(value: object) -> None:
    hipporeplayimm.apply_runtime_patches()

    with pytest.raises(TypeError, match="not boolean"):
        trajectory_imm._trajectory_imm_mode_stickiness(
            SimpleNamespace(trajectory_imm_mode_stickiness=value)
        )


@pytest.mark.parametrize("value", [True, np.bool_(False)])
def test_trajectory_imm_rejects_boolean_fallback_mode_stickiness(value: object) -> None:
    hipporeplayimm.apply_runtime_patches()

    with pytest.raises(TypeError, match="not boolean"):
        trajectory_imm._trajectory_imm_mode_stickiness(
            SimpleNamespace(
                trajectory_imm_mode_stickiness=None,
                imm_mode_stickiness=value,
            )
        )


@pytest.mark.parametrize("value", [True, np.bool_(False)])
def test_trajectory_imm_rejects_boolean_initial_momentum_probability(value: object) -> None:
    hipporeplayimm.apply_runtime_patches()

    with pytest.raises(TypeError, match="not boolean"):
        trajectory_imm._trajectory_imm_mode_prior(
            SimpleNamespace(trajectory_imm_momentum_initial_probability=value)
        )


@pytest.mark.parametrize("value", [True, np.bool_(False)])
def test_trajectory_imm_rejects_boolean_momentum_switch_probability(value: object) -> None:
    hipporeplayimm.apply_runtime_patches()

    with pytest.raises(TypeError, match="not boolean"):
        trajectory_imm._trajectory_imm_mode_transition_matrix(
            SimpleNamespace(trajectory_imm_momentum_switch_probability=value),
            0.9,
        )


@pytest.mark.parametrize("value", [True, np.bool_(False)])
def test_trajectory_imm_rejects_boolean_switch_tau(value: object) -> None:
    hipporeplayimm.apply_runtime_patches()

    with pytest.raises(TypeError, match="not boolean"):
        trajectory_imm._trajectory_imm_mode_transition_matrices(
            SimpleNamespace(imm_switch_tau_s=value),
            0.9,
            np.array([0.002, 0.003]),
        )


@pytest.mark.parametrize("value", _TEXT_NUMERIC_VALUES)
def test_trajectory_imm_rejects_text_mode_stickiness(value: object) -> None:
    hipporeplayimm.apply_runtime_patches()

    with pytest.raises(TypeError, match="not string"):
        trajectory_imm._trajectory_imm_mode_stickiness(
            SimpleNamespace(trajectory_imm_mode_stickiness=value)
        )


@pytest.mark.parametrize("value", _TEXT_NUMERIC_VALUES)
def test_trajectory_imm_rejects_text_fallback_mode_stickiness(value: object) -> None:
    hipporeplayimm.apply_runtime_patches()

    with pytest.raises(TypeError, match="not string"):
        trajectory_imm._trajectory_imm_mode_stickiness(
            SimpleNamespace(
                trajectory_imm_mode_stickiness=None,
                imm_mode_stickiness=value,
            )
        )


@pytest.mark.parametrize("value", _TEXT_NUMERIC_VALUES)
def test_trajectory_imm_rejects_text_initial_momentum_probability(value: object) -> None:
    hipporeplayimm.apply_runtime_patches()

    with pytest.raises(TypeError, match="not string"):
        trajectory_imm._trajectory_imm_mode_prior(
            SimpleNamespace(trajectory_imm_momentum_initial_probability=value)
        )


@pytest.mark.parametrize("value", _TEXT_NUMERIC_VALUES)
def test_trajectory_imm_rejects_text_momentum_switch_probability(value: object) -> None:
    hipporeplayimm.apply_runtime_patches()

    with pytest.raises(TypeError, match="not string"):
        trajectory_imm._trajectory_imm_mode_transition_matrix(
            SimpleNamespace(trajectory_imm_momentum_switch_probability=value),
            0.9,
        )


@pytest.mark.parametrize("value", _TEXT_NUMERIC_VALUES)
def test_trajectory_imm_rejects_text_switch_tau(value: object) -> None:
    hipporeplayimm.apply_runtime_patches()

    with pytest.raises(TypeError, match="not string"):
        trajectory_imm._trajectory_imm_mode_transition_matrices(
            SimpleNamespace(imm_switch_tau_s=value),
            0.9,
            np.array([0.002, 0.003]),
        )
