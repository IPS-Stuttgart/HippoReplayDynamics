from __future__ import annotations

import numpy as np
import pytest

import hipporeplayimm
import hipporeplayimm.ground_truth as gt


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"well_arrival_window_s": 0.0}, "well_arrival_window_s"),
        ({"visit_radius_cm": -1.0}, "visit_radius_cm"),
        ({"min_dwell_s": -0.01}, "min_dwell_s"),
        ({"future_horizon_s": float("nan")}, "future_horizon_s"),
    ],
)
def test_ground_truth_config_rejects_invalid_numeric_values(kwargs, match) -> None:
    hipporeplayimm.apply_runtime_patches()

    with pytest.raises(ValueError, match=match):
        gt.GroundTruthConfig(**kwargs)


def test_ground_truth_config_rejects_boolean_window() -> None:
    hipporeplayimm.apply_runtime_patches()

    with pytest.raises(TypeError, match="well_arrival_window_s"):
        gt.GroundTruthConfig(well_arrival_window_s=True)


@pytest.mark.parametrize(
    "value",
    [
        "10.0",
        b"10.0",
        np.str_("10.0"),
        np.bytes_(b"10.0"),
        np.array("10.0"),
        np.array("10.0", dtype=object),
    ],
)
def test_ground_truth_config_rejects_text_numeric_values(value) -> None:
    hipporeplayimm.apply_runtime_patches()

    with pytest.raises(TypeError, match="visit_radius_cm"):
        gt.GroundTruthConfig(visit_radius_cm=value)


@pytest.mark.parametrize(
    "values",
    [
        ("7.5", 10.0),
        (b"7.5", 10.0),
        np.array(["7.5", "10.0"]),
        np.array(["7.5", 10.0], dtype=object),
    ],
)
def test_ground_truth_sensitivity_config_rejects_text_numeric_grid_values(values) -> None:
    hipporeplayimm.apply_runtime_patches()

    with pytest.raises(TypeError, match="visit_radii_cm"):
        gt.GroundTruthSensitivityConfig(visit_radii_cm=values)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"visit_radii_cm": ()}, "visit_radii_cm"),
        ({"visit_radii_cm": (10.0, 0.0)}, "visit_radii_cm"),
        ({"min_dwells_s": (0.1, -0.1)}, "min_dwells_s"),
        ({"future_horizons_s": (15.0, float("inf"))}, "future_horizons_s"),
        ({"well_arrival_window_s": np.array([1.0])}, "well_arrival_window_s"),
    ],
)
def test_ground_truth_sensitivity_config_rejects_invalid_grid_values(kwargs, match) -> None:
    hipporeplayimm.apply_runtime_patches()

    with pytest.raises((TypeError, ValueError), match=match):
        gt.GroundTruthSensitivityConfig(**kwargs)


def test_ground_truth_sensitivity_config_validates_expansion_after_mutation() -> None:
    hipporeplayimm.apply_runtime_patches()
    config = gt.GroundTruthSensitivityConfig()
    object.__setattr__(config, "future_horizons_s", (-1.0,))

    with pytest.raises(ValueError, match="future_horizons_s"):
        config.ground_truth_configs()


def test_ground_truth_sensitivity_config_accepts_numpy_grids() -> None:
    hipporeplayimm.apply_runtime_patches()
    config = gt.GroundTruthSensitivityConfig(
        visit_radii_cm=np.array([7.5, 10.0]),
        min_dwells_s=np.array([0.0]),
        future_horizons_s=np.array([15.0]),
    )

    configs = config.ground_truth_configs()

    assert len(configs) == 2
    assert [curr.visit_radius_cm for curr in configs] == [7.5, 10.0]
