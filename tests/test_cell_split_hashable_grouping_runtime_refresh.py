from __future__ import annotations

import importlib

import numpy as np
import pandas as pd

import hipporeplayimm
import hipporeplayimm.benchmark_cell_split_metadata as cell_split_metadata
from hipporeplayimm import cell_split_hashable_grouping as hashable_grouping


def _shape_distinct_cell_split_scores() -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "session": ["rat1", "rat1"],
            "heldout_log_likelihood": [-1.0, -2.0],
            "train_log_likelihood": [-3.0, -4.0],
            "joint_log_likelihood": [-4.0, -6.0],
        }
    )
    cell_ids = np.empty(2, dtype=object)
    cell_ids[0] = np.array([[1, 2], [3, 4]])
    cell_ids[1] = np.array([1, 2, 3, 4])
    frame["train_cell_ids"] = cell_ids
    return frame


def test_cell_split_scope_detection_uses_shape_preserving_hashable_keys() -> None:
    hipporeplayimm.apply_runtime_patches()
    frame = _shape_distinct_cell_split_scores()

    assert cell_split_metadata._score_table_needs_cell_split_scoped_decode(frame)
    assert hashable_grouping._scope_key_column("train_cell_ids") in (
        cell_split_metadata._cell_split_decode_group_columns(frame)
    )


def test_runtime_patches_restore_hashable_grouping_after_metadata_reload() -> None:
    module = importlib.reload(cell_split_metadata)

    assert getattr(module, hashable_grouping._HASHABLE_GROUPING_PATCH_FLAG, False)
    assert not hashable_grouping._metadata_grouping_patch_is_current(module)

    hipporeplayimm.apply_runtime_patches()

    assert hashable_grouping._metadata_grouping_patch_is_current(module)
    assert module._score_table_needs_cell_split_scoped_decode(
        _shape_distinct_cell_split_scores()
    )
