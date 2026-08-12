import numpy as np
import pytest

from hipporeplayimm import ground_truth_float_metadata
from hipporeplayimm.ground_truth import GroundTruthConfig
from hipporeplayimm.ground_truth_cell_id_metadata import apply_ground_truth_cell_id_metadata_patch


def _nested_object_scalar(value: object, depth: int = 2) -> np.ndarray:
    wrapped = value
    for _ in range(depth):
        outer = np.empty((), dtype=object)
        outer[()] = wrapped
        wrapped = outer
    return wrapped


def test_ground_truth_config_rejects_nested_boolean_scalar():
    apply_ground_truth_cell_id_metadata_patch()

    with pytest.raises(TypeError, match="visit_radius_cm must be numeric, not boolean"):
        GroundTruthConfig(visit_radius_cm=_nested_object_scalar(np.bool_(True)))


def test_direct_ground_truth_scalar_parser_rejects_hidden_non_scalar_array():
    apply_ground_truth_cell_id_metadata_patch()
    value = _nested_object_scalar(np.array([1.0]))

    with pytest.raises(TypeError, match="future_horizon_s must be a scalar"):
        ground_truth_float_metadata._parse_config_scalar("future_horizon_s", value)


def test_ground_truth_scalar_parser_rejects_cyclic_object_wrapper():
    apply_ground_truth_cell_id_metadata_patch()
    value = np.empty((), dtype=object)
    value[()] = value

    with pytest.raises(TypeError, match="time_s must be a scalar"):
        ground_truth_float_metadata._parse_config_scalar("time_s", value)


def test_ground_truth_scalar_parser_preserves_nested_real_scalar():
    apply_ground_truth_cell_id_metadata_patch()
    value = _nested_object_scalar(np.float64(2.5), depth=3)

    assert ground_truth_float_metadata._parse_config_scalar("visit_radius_cm", value) == 2.5
