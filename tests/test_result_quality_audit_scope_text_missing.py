from __future__ import annotations

import numpy as np
import pandas as pd

from hipporeplayimm import result_quality_audit
from hipporeplayimm.result_quality_audit_scope_patch import (
    _first_influence_value_column,
    _heldout_aware_influence_summary,
    _scoped_event_group_columns,
)


def test_textual_missing_optional_metadata_does_not_scope_result_quality_events() -> None:
    scores = pd.DataFrame(
        {
            "session": ["Rat1/Open1", "Rat1/Open1"],
            "event_index": [3, 3],
            "model": ["diffusion", "momentum"],
            "window_index": ["nan", " <NA> "],
            "benchmark_random_seed": ["", "None"],
        }
    )

    assert _scoped_event_group_columns(scores) == ["session", "event_index"]


def test_observed_optional_metadata_still_scopes_result_quality_events() -> None:
    scores = pd.DataFrame(
        {
            "session": ["Rat1/Open1", "Rat1/Open1"],
            "event_index": [3, 3],
            "model": ["diffusion", "momentum"],
            "window_index": ["nan", "0"],
        }
    )

    assert _scoped_event_group_columns(scores) == ["session", "event_index", "window_index"]


def test_heldout_influence_uses_finite_heldout_scores_when_relative_is_missing() -> None:
    scores = pd.DataFrame(
        {
            "session": ["Rat1/Open1", "Rat2/Open1", "Rat1/Open1", "Rat2/Open1"],
            "model": ["diffusion", "diffusion", "momentum", "momentum"],
            "status": ["success", "success", "success", "success"],
            "relative_log_evidence": [np.nan, np.nan, np.nan, np.nan],
            "log_evidence": [np.nan, np.nan, np.nan, np.nan],
            "heldout_log_likelihood": [-1.0, -3.0, -0.5, -1.0],
        }
    )

    assert _first_influence_value_column(scores) == "heldout_log_likelihood"

    influence = _heldout_aware_influence_summary(result_quality_audit, scores)

    assert not influence.empty
    assert set(influence["left_out_group_col"]) == {"session", "rat"}
    assert np.isfinite(influence["full_mean"].to_numpy(dtype=float)).all()
