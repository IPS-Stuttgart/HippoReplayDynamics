from pathlib import Path

import pandas as pd

from scripts.audit_imm_fragmented_hypotheses import (
    DIFFUSION,
    FIRST_ORDER_IMM,
    FRAGMENTED,
    MOMENTUM_EXACT,
    STATIONARY,
    _read_evidence,
    build_event_table,
)


def test_blank_legacy_status_rows_are_kept(tmp_path: Path):
    rows = pd.DataFrame(
        [
            _row(0, STATIONARY, 0.0),
            _row(0, DIFFUSION, 20.0),
            _row(0, FRAGMENTED, 30.0),
            _row(0, FIRST_ORDER_IMM, 80.0),
            _row(0, MOMENTUM_EXACT, 40.0),
            _row(1, FIRST_ORDER_IMM, 999.0, status="error"),
        ]
    )
    rows.loc[rows["event_index"].eq(0), "status"] = ""
    path = tmp_path / "event_model_evidence.csv"
    rows.to_csv(path, index=False)

    loaded = _read_evidence(path)
    table = build_event_table(loaded)

    assert set(loaded["event_index"].astype(int)) == {0}
    assert len(table) == 1
    assert table.iloc[0]["within_family_classification"] == "clean_imm_switching_candidate"


def _row(event_index: int, model: str, log_evidence: float, *, status: str = "success") -> dict[str, object]:
    return {
        "status": status,
        "session": "Rat1/Open1",
        "event_index": event_index,
        "model": model,
        "log_evidence": log_evidence,
        "evidence_comparable": True,
    }
