from __future__ import annotations

import numpy as np
import pytest

import hipporeplayimm
import hipporeplayimm.state_space as state_space
from hipporeplayimm import state_space_model, state_space_utils


@pytest.mark.parametrize(
    "threshold",
    [
        True,
        False,
        np.bool_(True),
        np.array([0.1]),
        "0.1",
        b"0.1",
        np.str_("0.1"),
        np.asarray("0.1"),
    ],
)
def test_valid_occupancy_threshold_rejects_boolean_string_or_array(threshold: object) -> None:
    with pytest.raises(TypeError, match="min_occupancy_s"):
        state_space._valid_bin_mask_from_occupancy(np.ones(3, dtype=float), threshold, 3)


@pytest.mark.parametrize(
    "occupancy_s",
    [
        np.array([True, False, True]),
        np.array(["1.0", "0.5", "0.0"]),
        np.array([1.0, True, 0.0], dtype=object),
        np.array([1.0, "0.5", 0.0], dtype=object),
    ],
)
def test_valid_occupancy_rejects_boolean_or_string_values(occupancy_s: object) -> None:
    with pytest.raises(TypeError, match="occupancy_s"):
        state_space._valid_bin_mask_from_occupancy(occupancy_s, 0.1, 3)


def test_state_space_model_occupancy_helper_alias_is_patched() -> None:
    hipporeplayimm.apply_runtime_patches()

    assert state_space_model._valid_bin_mask_from_occupancy is state_space_utils._valid_bin_mask_from_occupancy
    with pytest.raises(TypeError, match="min_occupancy_s"):
        state_space_model._valid_bin_mask_from_occupancy(np.ones(3, dtype=float), True, 3)


def test_valid_occupancy_threshold_keeps_numeric_behavior() -> None:
    mask = state_space._valid_bin_mask_from_occupancy(
        np.array([0.0, 0.25, 0.5], dtype=float),
        0.25,
        3,
    )

    np.testing.assert_array_equal(mask, np.array([False, True, True]))
