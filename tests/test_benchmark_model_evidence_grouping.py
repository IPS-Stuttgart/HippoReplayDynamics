from __future__ import annotations

import pandas as pd

from scripts.benchmark_model_evidence import _add_evidence_columns


def test_evidence_columns_are_grouped_by_window_index() -> None:
    rows = pd.DataFrame(
        {
            "status": ["success", "success", "success", "success"],
            "session": ["s1", "s1", "s1", "s1"],
            "event_index": [0, 0, 0, 0],
            "window_index": [0, 0, 1, 1],
            "model": ["a", "b", "a", "b"],
            "requested_model": ["a", "b", "a", "b"],
            "model_family": ["trajectory", "trajectory", "trajectory", "trajectory"],
            "log_evidence": [10.0, 9.0, 0.0, 1.0],
            "n_time": [2, 2, 2, 2],
            "n_spikes": [10, 10, 10, 10],
            "runtime_s": [0.0, 0.0, 0.0, 0.0],
            "error": ["", "", "", ""],
        }
    )

    out = _add_evidence_columns(rows)

    best_by_window = (
        out[out["is_best_model"]]
        .sort_values("window_index")
        .set_index("window_index")["model"]
        .to_dict()
    )
    assert best_by_window == {0: "a", 1: "b"}
