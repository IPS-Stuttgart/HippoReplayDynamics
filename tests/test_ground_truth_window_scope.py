import pandas as pd

from hipporeplayimm.ground_truth_window_scope import _window_decode_groups


def test_window_decode_groups_handles_non_default_index():
    scores = pd.DataFrame(
        {
            "session": ["rat1", "rat1"],
            "event_index": [0, 0],
            "event_window_variant": ["contracted", "expanded"],
            "window_start_s": [1.0, 1.1],
            "window_end_s": [1.2, 1.4],
        },
        index=[10, 20],
    )

    groups = list(_window_decode_groups(scores))

    assert [list(group.index) for group in groups] == [[10], [20]]
    assert [float(group["window_start_s"].iloc[0]) for group in groups] == [1.0, 1.1]
