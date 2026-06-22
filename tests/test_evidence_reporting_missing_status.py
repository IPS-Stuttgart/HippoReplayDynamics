import pandas as pd

from hipporeplayimm.evidence_reporting import (
    EXACT_EVIDENCE_SUPPORT,
    simulation_add_evidence_columns,
    simulation_event_best_rows,
)


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "session": ["s", "s"],
            "event_index": [0, 0],
            "model": ["a", "b"],
            "log_evidence": [1.0, 3.0],
            "diagnostic_state_space_imm_evidence_support": [
                EXACT_EVIDENCE_SUPPORT,
                EXACT_EVIDENCE_SUPPORT,
            ],
        }
    )


def test_simulation_add_evidence_columns_without_status_column():
    out = simulation_add_evidence_columns(_rows())

    assert out["evidence_comparable"].tolist() == [True, True]
    assert out["best_model"].unique().tolist() == ["b"]
    assert out.loc[out["model"].eq("b"), "is_best_model"].item()
    assert out.loc[out["model"].eq("b"), "relative_log_evidence"].item() == 0.0


def test_simulation_event_best_rows_without_status_column():
    best = simulation_event_best_rows(_rows())

    assert best["model"].tolist() == ["b"]
    assert best["log_evidence"].tolist() == [3.0]
