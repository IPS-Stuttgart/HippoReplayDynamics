import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))

from audit_first_order_imm_mode_usage import (  # noqa: E402
    DIFFUSION,
    FIRST_ORDER_IMM,
    FRAGMENTED,
    MOMENTUM_EXACT,
    STATIONARY,
    _read_event_model_evidence,
    build_first_order_imm_mode_usage_event_table,
)


def _event_rows(event_index: object) -> list[dict[str, object]]:
    return [
        {
            "status": "success",
            "session": "Rat1/Open1",
            "event_index": event_index,
            "model": model,
            "log_evidence": log_evidence,
            "evidence_comparable": True,
        }
        for model, log_evidence in (
            (STATIONARY, 0.0),
            (DIFFUSION, 10.0),
            (FRAGMENTED, 20.0),
            (FIRST_ORDER_IMM, 80.0),
            (MOMENTUM_EXACT, 30.0),
        )
    ]


def test_mode_usage_audit_preserves_adjacent_decimal_event_ids_above_float_precision(
    tmp_path: Path,
) -> None:
    base = 2**53
    evidence = pd.DataFrame(
        [
            *_event_rows(f"{base}.0"),
            *_event_rows(f"{base + 1}.0"),
        ]
    )
    path = tmp_path / "event_model_evidence.csv"
    evidence.to_csv(path, index=False)

    loaded = _read_event_model_evidence(path)
    event_table = build_first_order_imm_mode_usage_event_table(loaded)

    assert loaded["event_index"].drop_duplicates().tolist() == [base, base + 1]
    assert event_table["event_index"].tolist() == [base, base + 1]


def test_mode_usage_audit_rejects_fractional_event_ids() -> None:
    evidence = pd.DataFrame(_event_rows(1.5))

    with pytest.raises(ValueError, match="integer-valued"):
        build_first_order_imm_mode_usage_event_table(evidence)


def test_mode_usage_audit_rejects_unsafe_preparsed_float_event_ids() -> None:
    evidence = pd.DataFrame(_event_rows(float(2**53)))

    with pytest.raises(ValueError, match="outside the exact integer range"):
        build_first_order_imm_mode_usage_event_table(evidence)
