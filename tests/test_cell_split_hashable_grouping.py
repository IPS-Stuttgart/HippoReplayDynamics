from __future__ import annotations

import numpy as np
import pandas as pd

from hipporeplayimm import benchmark_cell_split_metadata as metadata


def test_cell_split_scope_grouping_handles_in_memory_explicit_cell_id_arrays() -> None:
    scores = pd.DataFrame(
        {
            "session": ["Rat1/Open1", "Rat1/Open1"],
            "event_index": [0, 0],
            "model": ["state-space-imm", "state-space-imm"],
            "log_likelihood": [-1.0, -2.0],
            "heldout_log_likelihood": [-1.5, -2.5],
            "train_log_likelihood": [-0.5, -0.75],
            "joint_log_likelihood": [-2.0, -3.0],
            "benchmark_cell_split_index": [0, 1],
            "benchmark_test_cell_fraction": [0.5, 0.5],
            "benchmark_random_seed": [11, 11],
            "benchmark_cell_split_seed": [11, 12],
            "benchmark_cell_split_strategy": [
                "spike_count_stratified",
                "spike_count_stratified",
            ],
            "benchmark_cell_split_strata": [3, 3],
            "train_cell_ids": [
                np.array([1, 2], dtype=int),
                np.array([1, 3], dtype=int),
            ],
            "test_cell_ids": [[3], [2]],
        }
    )

    scores_frame = metadata._scores_frame_for_cell_split_metadata(scores)

    assert metadata._score_table_needs_cell_split_scoped_decode(scores_frame)
    group_columns = metadata._cell_split_decode_group_columns(scores_frame)
    assert "__cell_split_scope_key__train_cell_ids" in group_columns
    assert "__cell_split_scope_key__test_cell_ids" in group_columns

    grouped = list(scores_frame.groupby(group_columns, sort=False, dropna=False))

    assert len(grouped) == 2
    assert all("train_cell_ids" in split_scores.columns for _, split_scores in grouped)


def test_cell_split_scope_grouping_distinguishes_missing_and_present_metadata() -> None:
    scores = pd.DataFrame(
        {
            "session": ["Rat1/Open1", "Rat1/Open1"],
            "event_index": [0, 1],
            "model": ["state-space-imm", "state-space-imm"],
            "heldout_log_likelihood": [-1.5, -2.5],
            "train_log_likelihood": [-0.5, -0.75],
            "joint_log_likelihood": [-2.0, -3.0],
            "benchmark_test_cell_fraction": [0.5, 0.5],
            "benchmark_random_seed": [11, 11],
            "benchmark_cell_split_seed": [11, 11],
            "benchmark_cell_split_strategy": [np.nan, "peak-rate"],
            "benchmark_cell_split_strata": [np.nan, 6],
        }
    )

    scores_frame = metadata._scores_frame_for_cell_split_metadata(scores)

    assert metadata._score_table_needs_cell_split_scoped_decode(scores_frame)
    group_columns = metadata._cell_split_decode_group_columns(scores_frame)
    assert "__cell_split_scope_key__benchmark_cell_split_strategy" in group_columns
    assert "__cell_split_scope_key__benchmark_cell_split_strata" in group_columns

    groups = [group for _, group in scores_frame.groupby(group_columns, sort=False, dropna=False)]

    assert len(groups) == 2
    assert pd.isna(groups[0]["benchmark_cell_split_strategy"].iloc[0])
    assert groups[1]["benchmark_cell_split_strategy"].iloc[0] == "peak-rate"
