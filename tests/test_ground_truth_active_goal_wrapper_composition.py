from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from hipporeplayimm import ground_truth_float_metadata as float_metadata


def _ground_truth_stub() -> SimpleNamespace:
    return SimpleNamespace(
        active_goal_at_time=lambda session, time_s: int(session.well_sequence[0, 1]),
        first_post_ripple_well_visit=lambda *args, **kwargs: None,
        infer_well_locations_from_arrays=lambda *args, **kwargs: None,
    )


def _session(well_id: object) -> SimpleNamespace:
    return SimpleNamespace(
        well_sequence=np.array([[0.0, well_id]], dtype=object),
    )


def test_direct_helper_composes_active_goal_time_and_well_id_validation() -> None:
    ground_truth = _ground_truth_stub()

    float_metadata._patch_direct_ground_truth_numeric_helpers(ground_truth)

    assert ground_truth.active_goal_at_time(_session(2), 0.5) == 2
    with pytest.raises(TypeError, match="time_s must be numeric"):
        ground_truth.active_goal_at_time(_session(2), "0.5")
    with pytest.raises(ValueError, match="well IDs"):
        ground_truth.active_goal_at_time(_session(1.5), 0.5)

    active = ground_truth.active_goal_at_time
    float_metadata._patch_direct_ground_truth_numeric_helpers(ground_truth)
    assert ground_truth.active_goal_at_time is active
