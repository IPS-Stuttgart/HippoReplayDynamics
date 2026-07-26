from types import SimpleNamespace

import numpy as np

import hipporeplayimm
from hipporeplayimm import ground_truth
from hipporeplayimm import ground_truth_float_metadata


def _session() -> SimpleNamespace:
    return SimpleNamespace(
        well_sequence=np.array(
            [
                [1.0, 1.0],
                [2.0, 2.0],
            ]
        )
    )


def test_active_goal_validation_survives_patch_reapplication():
    for _ in range(3):
        ground_truth_float_metadata.apply_ground_truth_float_metadata_patch()
        hipporeplayimm.apply_runtime_patches()

    assert ground_truth.active_goal_at_time(_session(), np.float32(1.5)) == 1
