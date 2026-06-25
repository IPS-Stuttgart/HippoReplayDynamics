from __future__ import annotations

import pandas as pd

from hipporeplayimm import ground_truth as gt


def test_benchmark_decode_group_columns_include_random_seed_split_metadata() -> None:
    scores = pd.DataFrame(
        {
            "session": ["Rat1/Open1", "Rat1/Open1"],
            "benchmark_random_seed": [1, 2],
            "benchmark_cell_split_seed": [1, 2],
            "benchmark_cell_split_index": [0, 0],
            "heldout_log_likelihood": [0.0, 0.0],
            "train_log_likelihood": [0.0, 0.0],
            "joint_log_likelihood": [0.0, 0.0],
        }
    )

    columns = gt._decode_group_columns(scores, benchmark_decode=True)

    assert columns == [
        "session",
        "benchmark_random_seed",
        "benchmark_cell_split_seed",
        "benchmark_cell_split_index",
    ]


def test_decoded_merge_columns_keep_random_seed_decodes_aligned() -> None:
    scores = pd.DataFrame(
        {
            "session": ["Rat1/Open1"],
            "event_index": [0],
            "model": ["state-space-imm"],
            "benchmark_random_seed": [7],
            "benchmark_cell_split_seed": [11],
            "benchmark_cell_split_index": [3],
            "heldout_log_likelihood": [0.0],
            "train_log_likelihood": [0.0],
            "joint_log_likelihood": [0.0],
        }
    )
    decoded = pd.DataFrame(
        {
            "session": ["Rat1/Open1"],
            "event_index": [0],
            "model": ["state-space-imm"],
            "benchmark_random_seed": [7],
            "benchmark_cell_split_seed": [11],
            "benchmark_cell_split_index": [3],
        }
    )

    columns = gt._decoded_merge_columns(scores, decoded, benchmark_decode=True)

    assert columns == [
        "session",
        "event_index",
        "model",
        "benchmark_random_seed",
        "benchmark_cell_split_seed",
        "benchmark_cell_split_index",
    ]
