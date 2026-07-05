from __future__ import annotations

import numpy as np
import pytest

import hipporeplayimm.state_space as state_space
from hipporeplayimm import duration_occupancy


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


def test_state_space_per_bin_sigma_keeps_valid_scalar_behavior():
    assert state_space._per_bin_sigma(85.0, 0.003) == pytest.approx(85.0 * np.sqrt(0.003))


def test_duration_occupancy_per_bin_sigma_uses_same_scalar_validation():
    with pytest.raises(TypeError, match="dt_s"):
        duration_occupancy._per_bin_sigma(85.0, True)


def test_duration_occupancy_per_bin_sigma_rejects_string_values():
    with pytest.raises(TypeError, match="sigma_cm_sqrt_s"):
        duration_occupancy._per_bin_sigma("85.0", 0.003)
