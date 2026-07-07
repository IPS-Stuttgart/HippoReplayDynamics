from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hipporeplayimm import advanced_result_diagnostics as diagnostics


def _paired_scores() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "session": ["Rat1/Open1", "Rat1/Open1"],
            "event_index": [0, 0],
            "model": ["momentum", "diffusion"],
            "log_evidence": [1.0, 0.0],
            "status": ["success", "success"],
            "evidence_comparable": [True, True],
        }
    )


@pytest.mark.parametrize(
    "threshold",
    [
        "1.0",
        b"1.0",
        np.str_("1.0"),
        np.array("1.0"),
        np.array("1.0", dtype=object),
    ],
)
def test_paired_model_margin_decisions_rejects_text_margin_thresholds(threshold: object) -> None:
    with pytest.raises(ValueError, match="margin_threshold"):
        diagnostics.paired_model_margin_decisions(
            _paired_scores(),
            positive_model="momentum",
            reference_model="diffusion",
            margin_threshold=threshold,
        )


def test_paired_model_margin_threshold_sweep_rejects_text_thresholds() -> None:
    with pytest.raises(ValueError, match="thresholds"):
        diagnostics.paired_model_margin_threshold_sweep(
            _paired_scores(),
            positive_model="momentum",
            reference_model="diffusion",
            thresholds=("0.0",),
        )


def _valid_events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_index": [0],
            "start": [1.0],
            "end": [1.02],
        }
    )


def test_event_window_variants_rejects_text_padding_and_min_duration() -> None:
    with pytest.raises(ValueError, match="paddings_s"):
        diagnostics.event_window_variants(_valid_events(), paddings_s=("0.01",))

    with pytest.raises(ValueError, match="min_duration_s"):
        diagnostics.event_window_variants(_valid_events(), min_duration_s=np.array("0.003"))


def test_event_window_variants_rejects_text_event_times() -> None:
    events = pd.DataFrame(
        {
            "event_index": [0],
            "start": ["1.0"],
            "end": [1.02],
        }
    )

    with pytest.raises(ValueError, match="start.*finite scalar"):
        diagnostics.event_window_variants(events)
