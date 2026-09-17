from pathlib import Path

import numpy as np
import pandas as pd

from scripts.audit_imm_fragmented_hypotheses import (
    DIFFUSION,
    FIRST_ORDER_IMM,
    FRAGMENTED,
    MOMENTUM_EXACT,
    STATIONARY,
    _finite_difference,
    _load_labels,
    _read_evidence,
    build_event_table,
)


def test_imm_fragmented_csv_readers_preserve_decimal_form_event_ids_above_2_to_53(tmp_path: Path) -> None:
    event_ids = [2**53, 2**53 + 1]
    evidence_path = tmp_path / "event_model_evidence.csv"
    evidence_path.write_text(
        "session,event_index,model,log_evidence\n"
        + "\n".join(
            f"Rat1/Open1,{event_index}.0,{STATIONARY},{offset}.0"
            for offset, event_index in enumerate(event_ids)
        )
        + "\n",
        encoding="utf-8",
    )
    labels_path = tmp_path / "labels.csv"
    labels_path.write_text(
        "session,event_index,original_algorithm_label\n"
        + "\n".join(
            f"Rat1/Open1,{event_index}.0,momentum"
            for event_index in event_ids
        )
        + "\n",
        encoding="utf-8",
    )

    evidence = _read_evidence(evidence_path)
    labels = _load_labels(labels_path)

    assert evidence["event_index"].tolist() == event_ids
    assert labels["event_index"].tolist() == event_ids


def test_imm_fragmented_audit_treats_negative_infinite_competitor_as_decisive_evidence() -> None:
    evidence = pd.DataFrame(
        [
            _score(0, STATIONARY, 0.0),
            _score(0, DIFFUSION, 10.0),
            _score(0, FRAGMENTED, float("-inf")),
            _score(0, FIRST_ORDER_IMM, 20.0),
            _score(0, MOMENTUM_EXACT, 15.0),
            _score(1, STATIONARY, 0.0),
            _score(1, DIFFUSION, 10.0),
            _score(1, FRAGMENTED, 30.0),
            _score(1, FIRST_ORDER_IMM, float("-inf")),
            _score(1, MOMENTUM_EXACT, 15.0),
        ]
    )

    table = build_event_table(evidence, threshold=5.5).set_index("event_index")

    assert np.isposinf(table.loc[0, "delta_imm_minus_fragmented"])
    assert table.loc[0, "within_family_classification"] == "clean_imm_switching_candidate"
    assert np.isneginf(table.loc[1, "delta_imm_minus_fragmented"])
    assert table.loc[1, "within_family_classification"] == "fragmented_candidate"


def test_imm_fragmented_margin_keeps_undefined_infinite_comparisons_as_nan() -> None:
    assert np.isnan(_finite_difference(float("-inf"), float("-inf")))
    assert np.isnan(_finite_difference(float("inf"), 0.0))
    assert np.isnan(_finite_difference(0.0, float("inf")))


def _score(event_index: int, model: str, log_evidence: float) -> dict[str, object]:
    return {
        "session": "Rat1/Open1",
        "event_index": event_index,
        "model": model,
        "log_evidence": log_evidence,
        "evidence_comparable": True,
    }
