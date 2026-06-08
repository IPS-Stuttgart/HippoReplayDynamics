from pathlib import Path

import pandas as pd

from scripts.build_sota_comparator_pack import (
    DIFFUSION,
    FIRST_ORDER_IMM,
    FRAGMENTED,
    IMM_CANDIDATE,
    MOMENTUM_CANDIDATE,
    MOMENTUM_EXACT,
    STATIONARY,
    build_sota_comparator_event_table,
    build_sota_comparator_family_summary,
    build_sota_comparator_gate_summary,
    build_sota_comparator_lower_bound_audit,
    build_sota_comparator_claim_delta_summary,
    build_sota_comparator_model_summary,
    write_sota_comparator_pack,
)


def test_sota_comparator_pack_refines_momentum_story(tmp_path: Path):
    evidence = pd.DataFrame(
        [
            # Event 0: momentum beats diffusion, but first-order IMM is the exact-core winner.
            _score("Rat1/Open1", 0, STATIONARY, 0.0),
            _score("Rat1/Open1", 0, DIFFUSION, 10.0),
            _score("Rat1/Open1", 0, FRAGMENTED, 20.0),
            _score("Rat1/Open1", 0, FIRST_ORDER_IMM, 80.0),
            _score("Rat1/Open1", 0, MOMENTUM_EXACT, 40.0),
            _score("Rat1/Open1", 0, MOMENTUM_CANDIDATE, 35.0, comparable=False),
            _score("Rat1/Open1", 0, IMM_CANDIDATE, 70.0, comparable=False),
            # Event 1: same pattern.
            _score("Rat1/Open1", 1, STATIONARY, 0.0),
            _score("Rat1/Open1", 1, DIFFUSION, 20.0),
            _score("Rat1/Open1", 1, FRAGMENTED, 5.0),
            _score("Rat1/Open1", 1, FIRST_ORDER_IMM, 60.0),
            _score("Rat1/Open1", 1, MOMENTUM_EXACT, 50.0),
            _score("Rat1/Open1", 1, MOMENTUM_CANDIDATE, 45.0, comparable=False),
            _score("Rat1/Open1", 1, IMM_CANDIDATE, 55.0, comparable=False),
            # Event 2: exact-sparse momentum is the exact-core winner.
            _score("Rat2/Open1", 2, STATIONARY, 0.0),
            _score("Rat2/Open1", 2, DIFFUSION, 20.0),
            _score("Rat2/Open1", 2, FRAGMENTED, 5.0),
            _score("Rat2/Open1", 2, FIRST_ORDER_IMM, 10.0),
            _score("Rat2/Open1", 2, MOMENTUM_EXACT, 70.0),
            _score("Rat2/Open1", 2, MOMENTUM_CANDIDATE, 65.0, comparable=False),
            _score("Rat2/Open1", 2, IMM_CANDIDATE, 60.0, comparable=False),
        ]
    )

    event_table = build_sota_comparator_event_table(evidence, margin_threshold=5.5)
    claim = build_sota_comparator_claim_delta_summary(event_table, margin_threshold=5.5)
    model = build_sota_comparator_model_summary(event_table, margin_threshold=5.5)
    family = build_sota_comparator_family_summary(event_table)
    audit = build_sota_comparator_lower_bound_audit(event_table)
    gate = build_sota_comparator_gate_summary(event_table, margin_threshold=5.5)

    prior = _claim_row(claim, "prior_momentum_vs_diffusion_recovered")
    assert prior["raw_positive_events"] == 3
    assert prior["confident_positive_events"] == 3
    assert prior["confident_reference_events"] == 0

    refined = _claim_row(claim, "full_core_refines_momentum_story")
    assert refined["raw_positive_events"] == 2
    assert refined["confident_positive_events"] == 2

    dominance = _claim_row(claim, "exact_sparse_momentum_full_core_dominance")
    assert dominance["raw_positive_events"] == 1
    assert dominance["confident_positive_events"] == 1

    family_row = family.iloc[0]
    assert family_row["trajectory_raw_wins"] == 3
    assert family_row["trajectory_confident_claims"] == 3
    assert family_row["nontrajectory_confident_claims"] == 0

    first_order_summary = model[model["model"].eq(FIRST_ORDER_IMM)].iloc[0]
    assert first_order_summary["raw_best_events"] == 2
    assert first_order_summary["confident_exact_core_claims"] == 2

    momentum_audit = audit[audit["audit"].eq("candidate_pruned_momentum_lower_bound")].iloc[0]
    assert momentum_audit["matched_events"] == 3
    assert momentum_audit["violations_candidate_above_exact"] == 0
    assert momentum_audit["min_exact_minus_candidate"] == 5.0

    assert gate[gate["gate"].eq("overall")]["passed"].iloc[0]

    outputs = write_sota_comparator_pack(evidence, tmp_path, margin_threshold=5.5)
    assert set(outputs) == {
        "sota_comparator_event_table.csv",
        "sota_comparator_model_summary.csv",
        "sota_comparator_family_summary.csv",
        "sota_comparator_lower_bound_audit.csv",
        "sota_comparator_claim_delta_summary.csv",
        "sota_comparator_gate_summary.csv",
    }
    for name in outputs:
        assert (tmp_path / name).is_file()


def _score(
    session: str,
    event_index: int,
    model: str,
    log_evidence: float,
    *,
    comparable: bool = True,
) -> dict[str, object]:
    return {
        "status": "success",
        "session": session,
        "event_index": event_index,
        "model": model,
        "log_evidence": log_evidence,
        "evidence_comparable": comparable,
        "evidence_support": "exact_full_grid" if comparable else "truncated_lower_bound",
    }


def _claim_row(claim: pd.DataFrame, axis: str) -> pd.Series:
    return claim[claim["claim_axis"].eq(axis)].iloc[0]
