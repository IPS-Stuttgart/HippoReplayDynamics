import numpy as np
import pandas as pd

import hipporeplayimm
from hipporeplayimm import advanced_result_diagnostics as diagnostics


def test_window_sensitivity_keeps_missing_scope_metadata_groups() -> None:
    hipporeplayimm.apply_runtime_patches()
    scores = pd.DataFrame(
        [
            {
                "session": "Rat1/Open1",
                "event_index": 0,
                "benchmark_cell_split_strata": np.nan,
                "model": "state-space-first-order-imm",
                "window_variant": "core",
                "log_evidence": 1.0,
                "status": "success",
            },
            {
                "session": "Rat1/Open1",
                "event_index": 0,
                "benchmark_cell_split_strata": np.nan,
                "model": "state-space-first-order-imm",
                "window_variant": "expanded",
                "log_evidence": 3.0,
                "status": "success",
            },
        ]
    )

    summary = diagnostics.summarize_window_sensitivity(
        scores,
        group_cols=("session", "event_index", "benchmark_cell_split_strata"),
        variant_col="window_variant",
    )

    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["window_variants"] == 2
    assert row["evidence_window_mean"] == 2.0
    assert row["evidence_window_range"] == 2.0
    assert pd.isna(row["benchmark_cell_split_strata"])
