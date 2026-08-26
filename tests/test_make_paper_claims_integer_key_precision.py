from pathlib import Path

import pandas as pd

from scripts.make_paper_claims import load_score_tables


def test_load_score_tables_preserves_large_nullable_split_identifiers(tmp_path: Path) -> None:
    scores = tmp_path / "event_scores.csv"
    scores.write_text(
        "session,event_index,benchmark_cell_split_index,model,heldout_log_likelihood,evidence_support,status\n"
        "Rat1/Open1,0,9007199254740992,diffusion,-1.0,exact_full_grid,success\n"
        "Rat1/Open1,0,9007199254740993,momentum,-2.0,exact_full_grid,success\n"
        "Rat1/Open1,1,,diffusion,-3.0,exact_full_grid,success\n",
        encoding="utf-8",
    )

    frame = load_score_tables([scores])

    assert frame.loc[0, "benchmark_cell_split_index"] == 2**53
    assert frame.loc[1, "benchmark_cell_split_index"] == 2**53 + 1
    assert frame.loc[0, "benchmark_cell_split_index"] != frame.loc[1, "benchmark_cell_split_index"]
    assert pd.isna(frame.loc[2, "benchmark_cell_split_index"])
