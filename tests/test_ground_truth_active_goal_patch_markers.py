from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from hipporeplayimm.ground_truth_float_metadata import (
    _patch_direct_ground_truth_numeric_helpers,
)


def test_active_goal_installs_numeric_and_well_id_validation_independently() -> None:
    ground_truth = SimpleNamespace(
        active_goal_at_time=lambda session, time_s: (session, time_s),
        first_post_ripple_well_visit=lambda *args, **kwargs: None,
        infer_well_locations_from_arrays=lambda *args, **kwargs: None,
    )

    _patch_direct_ground_truth_numeric_helpers(ground_truth)

    invalid_ids = SimpleNamespace(
        well_sequence=np.array([[0.0, 1.5]], dtype=float),
    )
    with pytest.raises((TypeError, ValueError), match="well IDs"):
        ground_truth.active_goal_at_time(invalid_ids, 0.0)

    valid_ids = SimpleNamespace(
        well_sequence=np.array([[0.0, 1.0]], dtype=float),
    )
    with pytest.raises(TypeError, match="time_s must be numeric, not boolean"):
        ground_truth.active_goal_at_time(valid_ids, True)
