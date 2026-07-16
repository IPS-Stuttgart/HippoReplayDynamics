from __future__ import annotations

import numpy as np
import pytest

import hipporeplayimm
import hipporeplayimm.state_space as state_space
from hipporeplayimm import state_space_model, state_space_utils
from hipporeplayimm.encoding import LogEmissionTensor


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


@pytest.mark.parametrize("threshold", [0.0, 0.1])
@pytest.mark.parametrize(
    "occupancy_s",
    [
        np.array([0.5, -0.1, 0.2], dtype=float),
        np.array([0.5, -0.1, 0.2], dtype=object),
    ],
)
def test_valid_occupancy_rejects_negative_seconds(occupancy_s: object, threshold: float) -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        state_space._valid_bin_mask_from_occupancy(occupancy_s, threshold, 3)


@pytest.mark.parametrize(
    "occupancy_s",
    [
        np.array([0.5, np.nan, 0.2]),
        np.array([0.5, np.inf, 0.2]),
        np.array([0.5, -np.inf, 0.2]),
        np.array([10**400, 1.0, 0.0], dtype=object),
    ],
)
def test_valid_occupancy_rejects_nonfinite_or_unrepresentable_seconds(occupancy_s: object) -> None:
    with pytest.raises(ValueError, match="finite occupancy seconds"):
        state_space._valid_bin_mask_from_occupancy(occupancy_s, 0.1, 3)


@pytest.mark.parametrize("threshold", [10**400, np.asarray(10**400, dtype=object)])
def test_valid_occupancy_threshold_normalizes_numeric_overflow(threshold: object) -> None:
    with pytest.raises(ValueError, match="min_occupancy_s must be finite and nonnegative"):
        state_space._valid_bin_mask_from_occupancy(np.ones(3, dtype=float), threshold, 3)


def test_state_space_model_occupancy_helper_alias_is_patched() -> None:
    hipporeplayimm.apply_runtime_patches()

    assert state_space_model._valid_bin_mask_from_occupancy is state_space_utils._valid_bin_mask_from_occupancy
    with pytest.raises(TypeError, match="min_occupancy_s"):
        state_space_model._valid_bin_mask_from_occupancy(np.ones(3, dtype=float), True, 3)


def test_state_space_score_rejects_nonfinite_occupancy_seconds() -> None:
    emissions = LogEmissionTensor(
        log_likelihood=np.zeros((2, 2), dtype=float),
        spike_counts=np.zeros((2, 1), dtype=int),
        times=np.array([0.0, 0.02]),
        dt=0.02,
        cell_ids=np.array([1]),
        n_spikes=0,
    )
    model = state_space.StateSpaceReplayModel(
        mode="stationary",
        config=state_space.StateSpaceDecoderConfig(
            mode="stationary",
            valid_occupancy_threshold_s=0.1,
        ),
    )

    with pytest.raises(ValueError, match="finite occupancy seconds"):
        model.score(
            emissions,
            np.array([[0.0, 0.0], [1.0, 0.0]]),
            occupancy_s=np.array([1.0, np.inf]),
        )


def test_valid_occupancy_threshold_keeps_numeric_behavior() -> None:
    mask = state_space._valid_bin_mask_from_occupancy(
        np.array([0.0, 0.25, 0.5], dtype=float),
        0.25,
        3,
    )

    np.testing.assert_array_equal(mask, np.array([False, True, True]))
