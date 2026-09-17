from pathlib import Path

import pytest

from scripts.audit_imm_fragmented_hypotheses import _load_labels, _read_evidence


def test_imm_hypothesis_audit_preserves_adjacent_large_nullable_event_ids(tmp_path: Path) -> None:
    lower = 2**53
    upper = lower + 1
    path = tmp_path / "event_model_evidence.csv"
    path.write_text(
        "session,event_index,model,log_evidence,status,evidence_comparable\n"
        f"Rat1/Open1,{lower},model-a,1.0,success,true\n"
        f"Rat1/Open1,{upper},model-a,2.0,success,true\n"
        "Rat1/Open1,,model-a,0.0,failed,true\n",
        encoding="utf-8",
    )

    loaded = _read_evidence(path)

    assert loaded["event_index"].tolist() == [lower, upper]
    assert loaded["event_index"].nunique() == 2


def test_imm_hypothesis_audit_rejects_fractional_extended_precision_event_id(tmp_path: Path) -> None:
    path = tmp_path / "event_model_evidence.csv"
    path.write_text(
        "session,event_index,model,log_evidence,status,evidence_comparable\n"
        "Rat1/Open1,9007199254740993.5,model-a,1.0,success,true\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="event_index must contain integer-valued identifiers"):
        _read_evidence(path)


def test_imm_hypothesis_labels_preserve_decimal_form_large_integer_ids(tmp_path: Path) -> None:
    lower = 2**53
    upper = lower + 1
    path = tmp_path / "labels.csv"
    path.write_text(
        "session,event_index,label\n"
        f"Rat1/Open1,{lower}.0,momentum\n"
        f"Rat1/Open1,{upper}.0,fragmented\n",
        encoding="utf-8",
    )

    labels = _load_labels(path)

    assert labels["event_index"].tolist() == [lower, upper]
