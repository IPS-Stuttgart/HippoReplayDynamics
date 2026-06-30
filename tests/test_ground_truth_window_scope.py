from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from hipporeplayimm.data import RippleEvent
from hipporeplayimm.ground_truth_window_scope import _compare_scores_for_replay_window, _window_decode_groups


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


def test_window_decode_groups_handles_mixed_type_set_metadata():
    scores = pd.DataFrame(
        {
            "session": ["rat1", "rat1"],
            "event_index": [0, 0],
            "event_window_variant": [{1, "contracted"}, {"contracted", 1}],
            "window_start_s": [1.0, 1.0],
            "window_end_s": [1.2, 1.2],
        }
    )

    groups = list(_window_decode_groups(scores))

    assert len(groups) == 1
    assert groups[0]["window_start_s"].tolist() == [1.0, 1.0]


def test_compare_scores_for_replay_window_preserves_emission_call_extras():
    calls = []

    class Session:
        def ripple(self, index: int) -> RippleEvent:
            assert index == 0
            return RippleEvent(0.0, 10.0, 5.0, 1.0, 2.0, 3.0)

    def original_build_emissions(session, encoding, ripple, *args, **kwargs):
        calls.append(("standard", ripple, args, kwargs))
        return object()

    def original_build_clusterless_mark_emissions(session, encoding, ripple, *args, **kwargs):
        calls.append(("clusterless", ripple, args, kwargs))
        return object()

    gt = SimpleNamespace(
        build_emissions=original_build_emissions,
        build_clusterless_mark_emissions=original_build_clusterless_mark_emissions,
    )

    def base_compare(root, scores, *args, **kwargs):
        session = Session()
        gt.build_emissions(session, "encoding", 0, config="cfg", calibration="kept")
        gt.build_clusterless_mark_emissions(session, "encoding", 0, "cluster_cfg", calibration="cluster_kept")
        return pd.DataFrame({"ok": [True]})

    result = _compare_scores_for_replay_window(
        gt,
        base_compare,
        "root",
        pd.DataFrame({"model": ["stationary"]}),
        RippleEvent(1.0, 2.0, 1.5, np.nan, np.nan, np.nan),
        (),
        {},
    )

    assert result["ok"].tolist() == [True]
    assert gt.build_emissions is original_build_emissions
    assert gt.build_clusterless_mark_emissions is original_build_clusterless_mark_emissions
    assert [call[0] for call in calls] == ["standard", "clusterless"]
    assert all(isinstance(call[1], RippleEvent) for call in calls)
    assert [calls[0][1].start, calls[0][1].end, calls[0][1].peak] == [1.0, 2.0, 1.5]
    assert calls[0][2] == ()
    assert calls[0][3] == {"config": "cfg", "calibration": "kept"}
    assert calls[1][2] == ("cluster_cfg",)
    assert calls[1][3] == {"calibration": "cluster_kept"}
