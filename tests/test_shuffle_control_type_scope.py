from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hipporeplayimm.shuffle_controls import add_shuffle_p_values


def _control_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "session": ["Rat1/Open1"] * 4,
            "event_index": [3] * 4,
            "model": ["state-space-imm"] * 4,
            "control_type": [
                "spatial-roll",
                "spatial-roll",
                "cell-permutation",
                "cell-permutation",
            ],
            "log_evidence": [0.0, 10.0, 0.0, 1.0],
        }
    )


def test_shuffle_p_values_reject_ambiguous_mixed_control_families() -> None:
    real_scores = pd.DataFrame(
        {
            "session": ["Rat1/Open1"],
            "event_index": [3],
            "model": ["state-space-imm"],
            "log_evidence": [5.0],
        }
    )

    with pytest.raises(ValueError, match="multiple control_type values"):
        add_shuffle_p_values(real_scores, _control_rows())


def test_shuffle_p_values_match_each_requested_control_family() -> None:
    real_scores = pd.DataFrame(
        {
            "session": ["Rat1/Open1", "Rat1/Open1"],
            "event_index": [3, 3],
            "model": ["state-space-imm", "state-space-imm"],
            "control_type": ["spatial-roll", "cell-permutation"],
            "log_evidence": [5.0, 5.0],
        }
    )

    out = add_shuffle_p_values(real_scores, _control_rows())

    np.testing.assert_allclose(out["shuffle_p_value"], [2.0 / 3.0, 1.0 / 3.0])
    np.testing.assert_array_equal(out["shuffle_count"], [2, 2])
    np.testing.assert_allclose(out["shuffle_log_evidence_mean"], [5.0, 0.5])
