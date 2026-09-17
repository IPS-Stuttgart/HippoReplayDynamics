from __future__ import annotations

from pathlib import Path

from scripts.audit_imm_fragmented_hypotheses import STATIONARY, _read_evidence


def test_read_evidence_preserves_large_event_id_before_status_filter(tmp_path: Path) -> None:
    event_index = 2**53 + 1
    path = tmp_path / "event_model_evidence.csv"
    path.write_text(
        "status,session,event_index,model,log_evidence,evidence_comparable\n"
        f"success,Rat1/Open1,{event_index},{STATIONARY},1.0,True\n"
        f"failed,Rat1/Open1,,{STATIONARY},,False\n",
        encoding="utf-8",
    )

    loaded = _read_evidence(path)

    assert loaded["event_index"].tolist() == [event_index]
