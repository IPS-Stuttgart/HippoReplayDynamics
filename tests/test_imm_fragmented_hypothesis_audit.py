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
    write_outputs,
)


def test_imm_fragmented_audit_separates_clean_imm_fragmented_and_momentum(tmp_path: Path):
    evidence = pd.DataFrame(
        [
            _score("Rat1/Open1", 0, STATIONARY, 0.0),
            _score("Rat1/Open1", 0, DIFFUSION, 20.0),
            _score("Rat1/Open1", 0, FRAGMENTED, 30.0),
            _score("Rat1/Open1", 0, FIRST_ORDER_IMM, 80.0),
            _score("Rat1/Open1", 0, MOMENTUM_EXACT, 40.0),
            _score("Rat1/Open1", 1, STATIONARY, 0.0),
            _score("Rat1/Open1", 1, DIFFUSION, 20.0),
            _score("Rat1/Open1", 1, FRAGMENTED, 70.0),
            _score("Rat1/Open1", 1, FIRST_ORDER_IMM, 30.0),
            _score("Rat1/Open1", 1, MOMENTUM_EXACT, 40.0),
            _score("Rat2/Open1", 2, STATIONARY, 0.0),
            _score("Rat2/Open1", 2, DIFFUSION, 20.0),
            _score("Rat2/Open1", 2, FRAGMENTED, 45.0),
            _score("Rat2/Open1", 2, FIRST_ORDER_IMM, 48.0),
            _score("Rat2/Open1", 2, MOMENTUM_EXACT, 70.0),
        ]
    )

    event_table = build_event_table(evidence, threshold=5.5)
    assert (
        event_table.loc[event_table["event_index"].eq(0), "within_family_classification"].iloc[0]
        == "clean_imm_switching_candidate"
    )
    assert (
        event_table.loc[event_table["event_index"].eq(1), "within_family_classification"].iloc[0]
        == "fragmented_candidate"
    )
    assert (
        event_table.loc[event_table["event_index"].eq(2), "within_family_classification"].iloc[0]
        == "momentum_like_candidate"
    )

    labels = pd.DataFrame(
        [
            {"session": "Rat1/Open1", "event_index": 0, "original_algorithm_label": "momentum"},
            {"session": "Rat1/Open1", "event_index": 1, "original_algorithm_label": "momentum"},
            {"session": "Rat2/Open1", "event_index": 2, "original_algorithm_label": "momentum"},
        ]
    )
    outputs = write_outputs(evidence, tmp_path, labels=labels, threshold=5.5)
    assert set(outputs) == {
        "imm_fragmented_head_to_head_event_table.csv",
        "imm_fragmented_head_to_head_summary.csv",
        "trajectory_taxonomy_event_table.csv",
        "trajectory_taxonomy_summary.csv",
        "original_momentum_reassignment_event_table.csv",
        "original_momentum_reassignment_summary.csv",
        "imm_fragmented_hypothesis_gate_summary.csv",
    }
    reassignment = pd.read_csv(tmp_path / "original_momentum_reassignment_event_table.csv")
    assert len(reassignment) == 3
    assert set(reassignment["within_family_classification"]) == {
        "clean_imm_switching_candidate",
        "fragmented_candidate",
        "momentum_like_candidate",
    }


def test_imm_fragmented_audit_uses_only_comparable_exact_core_rows():
    evidence = pd.DataFrame(
        [
            _score("Rat1/Open1", 0, STATIONARY, 0.0),
            _score("Rat1/Open1", 0, DIFFUSION, 20.0),
            _score("Rat1/Open1", 0, FRAGMENTED, 30.0, evidence_comparable=False),
            _score("Rat1/Open1", 0, FIRST_ORDER_IMM, 80.0),
            _score("Rat1/Open1", 0, MOMENTUM_EXACT, 40.0),
        ]
    )

    event = build_event_table(evidence, threshold=5.5).iloc[0]

    assert not bool(event["exact_core_complete"])
    assert FRAGMENTED in event["missing_required_exact_core_models"]
    assert pd.isna(event["logZ_fragmented"])
    assert pd.isna(event["delta_imm_minus_fragmented"])
    assert event["within_family_classification"] == "trajectory_family_ambiguous"


def test_imm_fragmented_audit_parses_numeric_comparable_flags_from_csv(tmp_path: Path):
    evidence = pd.DataFrame(
        [
            _score("Rat1/Open1", 0, STATIONARY, 0.0, evidence_comparable=1.0),
            _score("Rat1/Open1", 0, DIFFUSION, 20.0, evidence_comparable=1.0),
            _score("Rat1/Open1", 0, FRAGMENTED, 100.0, evidence_comparable=0.0),
            _score("Rat1/Open1", 0, FRAGMENTED, 30.0, evidence_comparable=1.0),
            _score("Rat1/Open1", 0, FIRST_ORDER_IMM, 80.0, evidence_comparable=1.0),
            _score("Rat1/Open1", 0, MOMENTUM_EXACT, 40.0, evidence_comparable=1.0),
        ]
    )
    path = tmp_path / "event_model_evidence.csv"
    evidence.to_csv(path, index=False)

    loaded = _read_evidence(path)
    event = build_event_table(loaded, threshold=5.5).iloc[0]

    assert bool(event["exact_core_complete"])
    assert event["logZ_fragmented"] == 30.0
    assert event["best_exact_core_model"] == FIRST_ORDER_IMM
    assert event["within_family_classification"] == "clean_imm_switching_candidate"


def _score(session: str, event_index: int, model: str, log_evidence: float, *, evidence_comparable: object = True) -> dict[str, object]:
    return {
        "status": "success",
        "session": session,
        "event_index": event_index,
        "model": model,
        "log_evidence": log_evidence,
        "evidence_comparable": evidence_comparable,
    }
