from __future__ import annotations

import numpy as np
import pandas as pd

from hipporeplayimm.shuffle_controls import add_shuffle_p_values


def test_shuffle_p_values_keep_parameter_matrix_cells_separate() -> None:
    real_scores = pd.DataFrame(
        {
            "session": ["Rat1/Open1", "Rat1/Open1"],
            "event_index": [0, 0],
            "model": ["sorted-spike-state-space-diffusion"] * 2,
            "matrix_id": ["matrix-a", "matrix-b"],
            "log_evidence": [10.0, 100.5],
        }
    )
    control_scores = pd.DataFrame(
        {
            "session": ["Rat1/Open1"] * 4,
            "event_index": [0] * 4,
            "model": ["sorted-spike-state-space-diffusion"] * 4,
            "matrix_id": ["matrix-a", "matrix-a", "matrix-b", "matrix-b"],
            "log_evidence": [9.0, 11.0, 100.0, 101.0],
        }
    )

    result = add_shuffle_p_values(real_scores, control_scores)

    np.testing.assert_array_equal(result["shuffle_count"].to_numpy(), np.array([2, 2]))
    np.testing.assert_allclose(
        result["shuffle_log_evidence_median"].to_numpy(),
        np.array([10.0, 100.5]),
    )
    np.testing.assert_allclose(
        result["shuffle_p_value"].to_numpy(),
        np.array([2.0 / 3.0, 2.0 / 3.0]),
    )
