from __future__ import annotations

import numpy as np
import pandas as pd

from hipporeplayimm.advanced_result_diagnostics import (
    wrong_map_absolute_evidence_deltas,
    wrong_map_delta_summary,
)


def test_wrong_map_delta_summary_coerces_csv_string_evidence() -> None:
    current = pd.DataFrame(
        {
            "session": ["Rat1/Open1", "Rat1/Open1"],
            "event_index": [0, 0],
            "model": ["stationary", "diffusion"],
            "log_evidence": ["9.5", "10.0"],
            "status": ["success", "success"],
        }
    )
    wrong = pd.DataFrame(
        {
            "session": ["Rat1/Open1", "Rat1/Open1"],
            "event_index": [0, 0],
            "model": ["stationary", "diffusion"],
            "log_evidence": ["8.0", "9.0"],
            "status": ["success", "success"],
        }
    )

    out = wrong_map_delta_summary(current, wrong).set_index("model")

    assert np.isclose(out.loc["stationary", "delta_vs_wrong_environment_map"], 1.5)
    assert np.isclose(out.loc["diffusion", "delta_vs_wrong_environment_map"], 1.0)
    assert out["wrong_map_best_model"].unique().tolist() == ["diffusion"]


def test_wrong_map_absolute_deltas_ignore_nonfinite_csv_evidence() -> None:
    current = pd.DataFrame(
        {
            "session": ["Rat1/Open1", "Rat1/Open1"],
            "event_index": [0, 0],
            "model": [
                "sorted-spike-state-space-stationary",
                "sorted-spike-state-space-first-order-imm",
            ],
            "log_evidence": ["1.0", "12.0"],
            "status": ["success", "success"],
        }
    )
    wrong = pd.DataFrame(
        {
            "session": ["Rat1/Open1", "Rat1/Open1"],
            "event_index": [0, 0],
            "map_session": ["Rat1/Open2", "Rat1/Open2"],
            "model": [
                "sorted-spike-state-space-stationary",
                "sorted-spike-state-space-first-order-imm",
            ],
            "log_evidence": ["-1.0", "not-a-number"],
            "status": ["success", "success"],
        }
    )

    deltas = wrong_map_absolute_evidence_deltas(
        current,
        wrong,
        fixed_models=("sorted-spike-state-space-stationary",),
        exact_core_models=(
            "sorted-spike-state-space-stationary",
            "sorted-spike-state-space-first-order-imm",
        ),
        exact_trajectory_models=("sorted-spike-state-space-first-order-imm",),
    ).set_index("statistic")

    assert np.isclose(
        deltas.loc[
            "sorted-spike-state-space-stationary",
            "delta_map_log_evidence",
        ],
        2.0,
    )
    assert deltas.loc[
        "best_exact_core_model_real_map",
        "selected_model",
    ] == "sorted-spike-state-space-stationary"
    assert "best_exact_trajectory_model_real_map" not in deltas.index
