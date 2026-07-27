from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hipporeplayimm.shuffle_controls import _scope_label, add_shuffle_p_values


def test_shuffle_scope_labels_preserve_large_integer_identity() -> None:
    lower = 2**53
    upper = lower + 1

    assert _scope_label(lower) != _scope_label(upper)
    assert _scope_label(np.uint64(lower)) != _scope_label(np.uint64(upper))


def test_shuffle_p_values_do_not_merge_large_integer_event_scopes() -> None:
    lower = 2**53
    upper = lower + 1
    real_scores = pd.DataFrame(
        {
            "session": ["Rat1/Open1", "Rat1/Open1"],
            "event_index": np.array([lower, upper], dtype=np.uint64),
            "model": ["diffusion", "diffusion"],
            "log_evidence": [5.0, 5.0],
        }
    )
    control_scores = pd.DataFrame(
        {
            "session": ["Rat1/Open1"] * 4,
            "event_index": np.array([lower, lower, upper, upper], dtype=np.uint64),
            "model": ["diffusion"] * 4,
            "log_evidence": [0.0, 10.0, -10.0, -5.0],
        }
    )

    scored = add_shuffle_p_values(real_scores, control_scores)

    assert scored["shuffle_count"].tolist() == [2, 2]
    assert scored["shuffle_log_evidence_median"].tolist() == pytest.approx([5.0, -7.5])
    assert scored["shuffle_p_value"].tolist() == pytest.approx([2.0 / 3.0, 1.0 / 3.0])
