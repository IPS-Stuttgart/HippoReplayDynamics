from __future__ import annotations

from decimal import Decimal
import warnings

import numpy as np
import pytest

import hipporeplayimm.state_space as state_space
from hipporeplayimm import duration_occupancy


def _nested_object_scalar(value: object, *, depth: int = 2) -> np.ndarray:
    current = value
    for _ in range(depth):
        wrapper = np.empty((), dtype=object)
        wrapper[()] = current
        current = wrapper
    return current


@pytest.mark.parametrize("value", [True, False, np.bool_(True), np.array([85.0])])
def test_state_space_per_bin_sigma_rejects_boolean_or_array_sigma(value):
    with pytest.raises(TypeError, match="sigma_cm_sqrt_s"):
        state_space._per_bin_sigma(value, 0.003)


@pytest.mark.parametrize("value", [True, False, np.bool_(True), np.array([0.003])])
def test_state_space_per_bin_sigma_rejects_boolean_or_array_dt(value):
    with pytest.raises(TypeError, match="dt_s"):
        state_space._per_bin_sigma(85.0, value)


@pytest.mark.parametrize("value", ["85.0", b"85.0", np.str_("85.0"), np.asarray("85.0")])
def test_state_space_per_bin_sigma_rejects_string_sigma(value):
    with pytest.raises(TypeError, match="sigma_cm_sqrt_s"):
        state_space._per_bin_sigma(value, 0.003)


@pytest.mark.parametrize("value", ["0.003", b"0.003", np.str_("0.003"), np.asarray("0.003")])
def test_state_space_per_bin_sigma_rejects_string_dt(value):
    with pytest.raises(TypeError, match="dt_s"):
        state_space._per_bin_sigma(85.0, value)


@pytest.mark.parametrize(
    "value",
    [
        85.0 + 1.0j,
        np.complex128(85.0 + 1.0j),
        np.asarray(85.0 + 1.0j),
        np.array(85.0 + 1.0j, dtype=object),
    ],
)
def test_state_space_per_bin_sigma_rejects_complex_sigma(value):
    with pytest.raises(TypeError, match="sigma_cm_sqrt_s.*complex"):
        state_space._per_bin_sigma(value, 0.003)


@pytest.mark.parametrize(
    "value",
    [
        0.003 + 0.001j,
        np.complex128(0.003 + 0.001j),
        np.asarray(0.003 + 0.001j),
        np.array(0.003 + 0.001j, dtype=object),
    ],
)
def test_state_space_per_bin_sigma_rejects_complex_dt(value):
    with pytest.raises(TypeError, match="dt_s.*complex"):
        state_space._per_bin_sigma(85.0, value)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (_nested_object_scalar(True), "boolean"),
        (_nested_object_scalar(np.bool_(False)), "boolean"),
        (_nested_object_scalar("85.0"), "string"),
        (_nested_object_scalar(np.asarray("85.0")), "string"),
        (_nested_object_scalar(85.0 + 1.0j), "complex"),
        (_nested_object_scalar(np.array([85.0])), "numeric scalar"),
    ],
)
def test_state_space_per_bin_sigma_rejects_nested_lossy_sigma_wrappers(value, message):
    with pytest.raises(TypeError, match=rf"sigma_cm_sqrt_s.*{message}"):
        state_space._per_bin_sigma(value, 0.003)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (_nested_object_scalar(True), "boolean"),
        (_nested_object_scalar("0.003"), "string"),
        (_nested_object_scalar(0.003 + 0.001j), "complex"),
        (_nested_object_scalar(np.array([0.003])), "numeric scalar"),
    ],
)
def test_state_space_per_bin_sigma_rejects_nested_lossy_dt_wrappers(value, message):
    with pytest.raises(TypeError, match=rf"dt_s.*{message}"):
        state_space._per_bin_sigma(85.0, value)


def test_state_space_per_bin_sigma_keeps_valid_scalar_behavior():
    assert state_space._per_bin_sigma(85.0, 0.003) == pytest.approx(85.0 * np.sqrt(0.003))


def test_state_space_per_bin_sigma_accepts_nested_numeric_scalars():
    sigma = _nested_object_scalar(np.float64(85.0))
    dt = _nested_object_scalar(np.float64(0.003))

    assert state_space._per_bin_sigma(sigma, dt) == pytest.approx(85.0 * np.sqrt(0.003))


def test_state_space_per_bin_sigma_preserves_decimal_scalar_support():
    assert state_space._per_bin_sigma(Decimal("85"), Decimal("0.003")) == pytest.approx(
        85.0 * np.sqrt(0.003)
    )


def test_duration_occupancy_per_bin_sigma_uses_same_scalar_validation():
    with pytest.raises(TypeError, match="dt_s"):
        duration_occupancy._per_bin_sigma(85.0, True)


def test_duration_occupancy_per_bin_sigma_rejects_string_values():
    with pytest.raises(TypeError, match="sigma_cm_sqrt_s"):
        duration_occupancy._per_bin_sigma("85.0", 0.003)


def test_duration_occupancy_per_bin_sigma_rejects_complex_values():
    with pytest.raises(TypeError, match="sigma_cm_sqrt_s.*complex"):
        duration_occupancy._per_bin_sigma(np.complex128(85.0 + 1.0j), 0.003)


def test_duration_occupancy_per_bin_sigma_rejects_nested_boolean_values():
    with pytest.raises(TypeError, match="sigma_cm_sqrt_s.*boolean"):
        duration_occupancy._per_bin_sigma(_nested_object_scalar(True), 0.003)


@pytest.mark.parametrize(
    "helper",
    [state_space._per_bin_sigma, duration_occupancy._per_bin_sigma],
    ids=["state-space", "duration-occupancy"],
)
def test_per_bin_sigma_rejects_derived_overflow_without_runtime_warning(helper):
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(ValueError, match="finite per-bin sigma"):
            helper(np.finfo(float).max, 4.0)


@pytest.mark.parametrize(
    "helper",
    [state_space._per_bin_sigma, duration_occupancy._per_bin_sigma],
    ids=["state-space", "duration-occupancy"],
)
def test_per_bin_sigma_preserves_large_representable_result(helper):
    sigma = np.finfo(float).max / 4.0

    result = helper(sigma, 4.0)

    assert np.isfinite(result)
    assert result == pytest.approx(np.finfo(float).max / 2.0)


def test_mode_transition_matrix_rejects_complex_stickiness():
    with pytest.raises(ValueError, match=r"mode_stickiness.*\[0, 1\]"):
        state_space._mode_transition_matrix(2, np.complex128(0.9 + 0.1j))


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (_nested_object_scalar(True), "boolean"),
        (_nested_object_scalar("0.9"), "string"),
        (_nested_object_scalar(np.array([0.9])), "numeric scalar"),
    ],
)
def test_mode_transition_matrix_rejects_nested_lossy_stickiness(value, message):
    with pytest.raises(TypeError, match=rf"mode_stickiness.*{message}"):
        state_space._mode_transition_matrix(2, value)


def test_mode_transition_matrix_rejects_nested_complex_stickiness():
    with pytest.raises(ValueError, match=r"mode_stickiness.*\[0, 1\]"):
        state_space._mode_transition_matrix(
            2,
            _nested_object_scalar(0.9 + 0.1j),
        )


def test_mode_transition_matrix_accepts_nested_numeric_stickiness():
    matrix = state_space._mode_transition_matrix(
        2,
        _nested_object_scalar(np.float64(0.9)),
    )

    assert matrix.shape == (2, 2)
    assert np.allclose(matrix.sum(axis=1), 1.0)


@pytest.mark.parametrize(
    ("name", "mode_stickiness", "imm_switch_tau_s"),
    [
        ("mode_stickiness", "0.9", 0.0),
        ("mode_stickiness", np.asarray("0.9"), 0.0),
        ("imm_switch_tau_s", 0.9, "0.05"),
        ("imm_switch_tau_s", 0.9, np.asarray("0.05")),
    ],
)
def test_duration_occupancy_mode_transition_rejects_string_scalars(
    name,
    mode_stickiness,
    imm_switch_tau_s,
):
    with pytest.raises(TypeError, match=name):
        duration_occupancy._mode_transition_matrices(
            state_space,
            3,
            mode_stickiness,
            imm_switch_tau_s,
            np.array([0.02, 0.03]),
        )


@pytest.mark.parametrize(
    ("name", "mode_stickiness", "imm_switch_tau_s", "message"),
    [
        (
            "mode_stickiness",
            _nested_object_scalar(True),
            0.0,
            "boolean",
        ),
        (
            "mode_stickiness",
            _nested_object_scalar("0.9"),
            0.0,
            "string",
        ),
    ],
)
def test_duration_occupancy_mode_transition_rejects_nested_lossy_scalars(
    name,
    mode_stickiness,
    imm_switch_tau_s,
    message,
):
    with pytest.raises(TypeError, match=rf"{name}.*{message}"):
        duration_occupancy._mode_transition_matrices(
            state_space,
            3,
            mode_stickiness,
            imm_switch_tau_s,
            np.array([0.02, 0.03]),
        )


def test_duration_occupancy_mode_transition_rejects_nested_complex_switch_tau():
    with pytest.raises(
        ValueError,
        match=r"imm_switch_tau_s.*finite and nonnegative",
    ):
        duration_occupancy._mode_transition_matrices(
            state_space,
            3,
            0.9,
            _nested_object_scalar(0.05 + 0.01j),
            np.array([0.02, 0.03]),
        )


def test_duration_occupancy_mode_transition_keeps_numeric_scalar_behavior():
    transitions = duration_occupancy._mode_transition_matrices(
        state_space,
        3,
        np.float64(0.9),
        np.asarray(0.0),
        np.array([0.02, 0.03]),
    )

    assert len(transitions) == 2
    for transition in transitions:
        np.testing.assert_allclose(transition.sum(axis=1), np.ones(3))
