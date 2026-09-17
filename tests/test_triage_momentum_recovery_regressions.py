from __future__ import annotations

import math

import pandas as pd
import pytest

from scripts.triage_momentum_recovery import (
    DEFAULT_EXPECTED_MOMENTUM_MODEL,
    build_momentum_recovery_triage,
    load_scores,
)


def _row(
    event_index: object,
    model: str,
    log_evidence: float,
    *,
    support: str = "exact_full_grid",
    comparable: bool = True,
) -> dict[str, object]:
    return {
        "session": "Rat1/Open1",
        "event_index": event_index,
        "true_model": "momentum",
        "expected_model": DEFAULT_EXPECTED_MOMENTUM_MODEL,
        "model": model,
        "status": "success",
        "log_evidence": log_evidence,
        "evidence_support": support,
        "evidence_comparable": comparable,
        "candidate_true_bin_coverage": 1.0,
        "candidate_true_pair_coverage": 1.0,
        "candidate_true_triplet_coverage": 1.0,
        "candidate_true_path_fully_supported": 1,
        "candidate_true_path_missing_bins": 0,
    }


def test_successful_negative_infinite_exact_evidence_is_not_dropped() -> None:
    scores = pd.DataFrame(
        [_row(0, DEFAULT_EXPECTED_MOMENTUM_MODEL, float("-inf"))]
    )

    event = build_momentum_recovery_triage(scores).event_table.iloc[0]

    assert event["triage_category"] == "strict_exact_recovery"
    assert bool(event["strict_exact_recovery"])
    assert event["expected_model_log_evidence"] == float("-inf")


def test_finite_truncated_lower_bound_beats_negative_infinite_exact_reference() -> None:
    scores = pd.DataFrame(
        [
            _row(0, "sorted-spike-state-space-diffusion", float("-inf")),
            _row(
                0,
                DEFAULT_EXPECTED_MOMENTUM_MODEL,
                -10.0,
                support="truncated_full_grid",
                comparable=False,
            ),
        ]
    )

    event = build_momentum_recovery_triage(scores).event_table.iloc[0]

    assert event["triage_category"] == "lower_bound_certified_recovery"
    assert bool(event["lower_bound_certified_recovery"])
    assert event["best_comparable_log_evidence"] == float("-inf")
    assert math.isinf(event["expected_minus_best_comparable_log_evidence"])
    assert event["expected_minus_best_comparable_log_evidence"] > 0.0


def test_load_scores_preserves_adjacent_decimal_event_ids_above_float_precision(
    tmp_path,
) -> None:
    path = tmp_path / "simulation_recovery_event_scores.csv"
    path.write_text(
        "session,event_index,true_model,expected_model,model,status,log_evidence,evidence_support,evidence_comparable\n"
        "Rat1/Open1,9007199254740992.0,momentum,sorted-spike-state-space-momentum,sorted-spike-state-space-momentum,success,0.0,exact_full_grid,True\n"
        "Rat1/Open1,9007199254740993.0,momentum,sorted-spike-state-space-momentum,sorted-spike-state-space-momentum,success,0.0,exact_full_grid,True\n",
        encoding="utf-8",
    )

    scores = load_scores(path)
    tables = build_momentum_recovery_triage(scores)

    assert scores["event_index"].tolist() == [2**53, 2**53 + 1]
    assert tables.event_table["event_index"].tolist() == [2**53, 2**53 + 1]
    assert len(tables.event_table) == 2


def test_in_memory_unsafe_float_event_id_is_rejected() -> None:
    scores = pd.DataFrame(
        [_row(float(2**53), DEFAULT_EXPECTED_MOMENTUM_MODEL, 0.0)]
    )

    with pytest.raises(ValueError, match="unsafe floating-point identifier"):
        build_momentum_recovery_triage(scores)
