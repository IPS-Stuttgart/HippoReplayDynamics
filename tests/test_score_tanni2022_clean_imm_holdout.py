from __future__ import annotations

import pandas as pd

from scripts.score_tanni2022_clean_imm_holdout import MODELS, holdout_decisions


def test_holdout_decisions_require_imm_best_for_strict_clean_imm() -> None:
    rows = []
    examples = {
        1: {
            "stationary": 0.0,
            "diffusion": 4.0,
            "fragmented": 1.0,
            "first_order_imm": 8.0,
            "exact_sparse_momentum": 3.0,
        },
        2: {
            "stationary": 0.0,
            "diffusion": 9.0,
            "fragmented": 1.0,
            "first_order_imm": 8.0,
            "exact_sparse_momentum": 2.0,
        },
    }
    for event_index, values in examples.items():
        for model in MODELS:
            rows.append(
                {
                    "animal": "RatA",
                    "session": "RatA_session",
                    "event_index": event_index,
                    "model": model,
                    "log_evidence": values[model],
                    "status": "success",
                    "evidence_comparable": True,
                }
            )

    decisions = holdout_decisions(pd.DataFrame(rows), margin_threshold=5.5)

    assert bool(decisions.loc[decisions["event_index"].eq(1), "strict_clean_imm"].iloc[0])
    assert not bool(decisions.loc[decisions["event_index"].eq(2), "strict_clean_imm"].iloc[0])
    assert decisions.loc[decisions["event_index"].eq(2), "best_model"].iloc[0] == "diffusion"
