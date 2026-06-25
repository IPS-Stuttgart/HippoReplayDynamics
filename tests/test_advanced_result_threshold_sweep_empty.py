from __future__ import annotations

import pandas as pd

from hipporeplayimm.advanced_result_diagnostics import paired_model_margin_threshold_sweep


def test_paired_model_margin_threshold_sweep_preserves_metadata_for_empty_decisions() -> None:
    scores = pd.DataFrame(
        {
            "session": ["Rat1/Open1"],
            "event_index": [0],
            "model": ["diffusion"],
            "log_evidence": [0.0],
            "status": ["success"],
            "evidence_comparable": [True],
        }
    )

    sweep = paired_model_margin_threshold_sweep(
        scores,
        positive_model="momentum",
        reference_model="diffusion",
        thresholds=(0.0, 5.0),
    )

    assert sweep["events"].tolist() == [0, 0]
    assert sweep["positive_model"].tolist() == ["momentum", "momentum"]
    assert sweep["reference_model"].tolist() == ["diffusion", "diffusion"]
    assert sweep["margin_threshold"].tolist() == [0.0, 5.0]
    assert sweep["group_cols"].tolist() == ["session,event_index", "session,event_index"]
