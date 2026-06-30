from __future__ import annotations

import numpy as np
import pytest

import hipporeplayimm
from hipporeplayimm.accuracy_upgrades import ValidStateConfig, valid_state_mask_from_encoding
from hipporeplayimm.encoding import EncodingConfig, EncodingModel


def _encoding() -> EncodingModel:
    return EncodingModel(
        x_edges=np.array([0.0, 1.0, 2.0, 3.0], dtype=float),
        y_edges=np.array([0.0, 1.0], dtype=float),
        bin_centers=np.array(
            [
                [0.5, 0.5],
                [1.5, 0.5],
                [2.5, 0.5],
            ],
            dtype=float,
        ),
        rates_hz=np.array([[1.0, 2.0, 3.0]], dtype=float),
        occupancy_s=np.array([0.0, 0.03, 0.10], dtype=float),
        cell_ids=np.array([1], dtype=int),
        config=EncodingConfig(),
    )


@pytest.mark.parametrize("value", [True, np.bool_(True), np.asarray(True, dtype=object)])
def test_valid_state_mask_rejects_boolean_min_occupancy(value: object) -> None:
    hipporeplayimm.apply_runtime_patches()

    with pytest.raises(TypeError, match="min_occupancy_s"):
        valid_state_mask_from_encoding(_encoding(), ValidStateConfig(min_occupancy_s=value))


@pytest.mark.parametrize("value", [float("nan"), -0.01])
def test_valid_state_mask_rejects_invalid_min_occupancy(value: float) -> None:
    hipporeplayimm.apply_runtime_patches()

    with pytest.raises(ValueError, match="min_occupancy_s"):
        valid_state_mask_from_encoding(_encoding(), ValidStateConfig(min_occupancy_s=value))


@pytest.mark.parametrize("value", [np.array([0.02]), np.array([[0.02]])])
def test_valid_state_mask_rejects_array_min_occupancy(value: object) -> None:
    hipporeplayimm.apply_runtime_patches()

    with pytest.raises(ValueError, match="min_occupancy_s"):
        valid_state_mask_from_encoding(_encoding(), ValidStateConfig(min_occupancy_s=value))


@pytest.mark.parametrize("value", [True, np.bool_(False), np.asarray(True, dtype=object)])
def test_valid_state_mask_rejects_boolean_top_occupancy_fraction(value: object) -> None:
    hipporeplayimm.apply_runtime_patches()

    with pytest.raises(TypeError, match="keep_top_occupancy_fraction"):
        valid_state_mask_from_encoding(
            _encoding(),
            ValidStateConfig(keep_top_occupancy_fraction=value),
        )


@pytest.mark.parametrize("value", [np.array([0.5]), np.array([[0.5]])])
def test_valid_state_mask_rejects_array_top_occupancy_fraction(value: object) -> None:
    hipporeplayimm.apply_runtime_patches()

    with pytest.raises(ValueError, match="keep_top_occupancy_fraction"):
        valid_state_mask_from_encoding(
            _encoding(),
            ValidStateConfig(keep_top_occupancy_fraction=value),
        )


def test_valid_state_mask_rejects_non_boolean_require_finite_rates() -> None:
    hipporeplayimm.apply_runtime_patches()

    with pytest.raises(TypeError, match="require_finite_rates"):
        valid_state_mask_from_encoding(
            _encoding(),
            ValidStateConfig(require_finite_rates="False"),
        )
