from pathlib import Path

import pandas as pd

from scripts.audit_second_order_lower_bound_gap import (
    EXACT_SUPPORT,
    TRUNCATED_SUPPORT,
    build_lower_bound_gap_tables,
    load_score_table,
)


MODEL = "sorted-spike-state-space-momentum"


def test_load_score_table_preserves_large_nullable_scope_identifiers(tmp_path: Path) -> None:
    scores = tmp_path / "event_scores.csv"
    scores.write_text(
        "session,event_index,benchmark_cell_split_index,model,evidence_support,log_evidence\n"
        f"Rat1/Open1,0,{2**53},{MODEL},{EXACT_SUPPORT},-1.0\n"
        f"Rat1/Open1,0,{2**53 + 1},{MODEL},{TRUNCATED_SUPPORT},-2.0\n"
        f"Rat1/Open1,1,,{MODEL},{EXACT_SUPPORT},-3.0\n",
        encoding="utf-8",
    )

    frame = load_score_table(scores)

    assert frame.loc[0, "benchmark_cell_split_index"] == 2**53
    assert frame.loc[1, "benchmark_cell_split_index"] == 2**53 + 1
    assert frame.loc[0, "benchmark_cell_split_index"] != frame.loc[1, "benchmark_cell_split_index"]
    assert pd.isna(frame.loc[2, "benchmark_cell_split_index"])


def test_lower_bound_gap_pairing_respects_event_window_scope() -> None:
    scores = pd.DataFrame(
        [
            {
                "session": "Rat1/Open1",
                "event_index": 0,
                "window_index": 0,
                "model": MODEL,
                "evidence_support": EXACT_SUPPORT,
                "log_evidence": -10.0,
            },
            {
                "session": "Rat1/Open1",
                "event_index": 0,
                "window_index": 0,
                "model": MODEL,
                "evidence_support": TRUNCATED_SUPPORT,
                "log_evidence": -12.0,
            },
            {
                "session": "Rat1/Open1",
                "event_index": 0,
                "window_index": 1,
                "model": MODEL,
                "evidence_support": EXACT_SUPPORT,
                "log_evidence": -100.0,
            },
            {
                "session": "Rat1/Open1",
                "event_index": 0,
                "window_index": 1,
                "model": MODEL,
                "evidence_support": TRUNCATED_SUPPORT,
                "log_evidence": -105.0,
            },
        ]
    )

    tables = build_lower_bound_gap_tables(
        scores,
        models=(MODEL,),
        value_column="log_evidence",
    )
    event_gaps = tables.event_gaps.sort_values("window_index").reset_index(drop=True)

    assert event_gaps["window_index"].tolist() == [0, 1]
    assert event_gaps["lower_bound_gap_log_evidence"].tolist() == [2.0, 5.0]
    assert event_gaps["exact_source_rows"].tolist() == [1, 1]
