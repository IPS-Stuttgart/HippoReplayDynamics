import pandas as pd

from hipporeplayimm import ground_truth as gt


def test_ground_truth_decode_group_columns_include_cell_split_index():
    scores = pd.DataFrame(
        {
            "session": ["Rat1/Open1", "Rat1/Open1"],
            "event_index": [0, 0],
            "model": ["split-sensitive", "split-sensitive"],
            "heldout_log_likelihood": [0.0, 0.0],
            "train_log_likelihood": [0.0, 0.0],
            "joint_log_likelihood": [0.0, 0.0],
            "cell_split_index": [0, 1],
            "train_cell_ids": ["1", "3"],
            "test_cell_ids": ["2", "4"],
        }
    )

    columns = gt._decode_group_columns(scores, benchmark_decode=True)

    assert columns[:2] == ["session", "cell_split_index"]
