from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hipporeplayimm.shuffle_controls import add_shuffle_p_values


def test_shuffle_p_values_reject_missing_real_scope_discriminator() -> None:
    real_scores = pd.DataFrame(
        {
            "session": ["Rat1/Open1"],
            "event_index": [3],
            "model": ["state-space-imm"],
            "log_evidence": [5.0],
        }
    )
    control_scores = pd.DataFrame(
        {
            "session": ["Rat1/Open1"] * 4,
            "event_index": [3] * 4,
            "model": ["state-space-imm"] * 4,
            "benchmark_random_seed": [11, 11, 22, 22],
            "log_evidence": [0.0, 1.0, 10.0, 11.0],
        }
    )

    with pytest.raises(
        ValueError,
        match=r"control_scores contains multiple benchmark_random_seed values.*real_scores is missing",
    ):
        add_shuffle_p_values(real_scores, control_scores)


def test_shuffle_p_values_reject_missing_control_scope_discriminator() -> None:
    real_scores = pd.DataFrame(
        {
            "session": ["Rat1/Open1", "Rat1/Open1"],
            "event_index": [3, 3],
            "model": ["state-space-imm", "state-space-imm"],
            "matrix_id": ["matrix-a", "matrix-b"],
            "log_evidence": [5.0, 5.0],
        }
    )
    control_scores = pd.DataFrame(
        {
            "session": ["Rat1/Open1", "Rat1/Open1"],
            "event_index": [3, 3],
            "model": ["state-space-imm", "state-space-imm"],
            "log_evidence": [0.0, 10.0],
        }
    )

    with pytest.raises(
        ValueError,
        match=r"real_scores contains multiple matrix_id values.*control_scores is missing",
    ):
        add_shuffle_p_values(real_scores, control_scores)


def test_shuffle_p_values_allow_constant_one_sided_scope_metadata() -> None:
    real_scores = pd.DataFrame(
        {
            "session": ["Rat1/Open1"],
            "event_index": [3],
            "model": ["state-space-imm"],
            "log_evidence": [5.0],
        }
    )
    control_scores = pd.DataFrame(
        {
            "session": ["Rat1/Open1", "Rat1/Open1"],
            "event_index": [3, 3],
            "model": ["state-space-imm", "state-space-imm"],
            "benchmark_random_seed": [11, 11],
            "log_evidence": [0.0, 10.0],
        }
    )

    result = add_shuffle_p_values(real_scores, control_scores)

    np.testing.assert_array_equal(result["shuffle_count"].to_numpy(), np.array([2]))
    np.testing.assert_allclose(result["shuffle_p_value"].to_numpy(), np.array([2.0 / 3.0]))
