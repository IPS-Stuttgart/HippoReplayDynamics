import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))

from deduplicate_off_swr_candidates import (  # noqa: E402
    FIRST_ORDER_IMM,
    MOMENTUM_EXACT,
    build_cluster_robustness_gate_summary,
    build_one_per_source_group_decisions,
    build_one_per_source_group_summary,
    build_source_event_group_summary,
    write_off_swr_candidate_dedup_outputs,
)


def test_off_swr_candidate_dedup_collapses_duplicate_source_groups(tmp_path):
    validation = pd.DataFrame(
        [
            _candidate("Rat1/Open1", 10, 0, exact=60.0, discovery=55.0, model=FIRST_ORDER_IMM, start=1.0),
            _candidate("Rat1/Open1", 10, 1, exact=90.0, discovery=80.0, model=FIRST_ORDER_IMM, start=2.0),
            _candidate("Rat1/Open1", 11, 0, exact=70.0, discovery=65.0, model=MOMENTUM_EXACT, start=3.0),
            _candidate("Rat2/Open1", 20, 0, exact=58.0, discovery=58.0, model=FIRST_ORDER_IMM, start=4.0),
            _candidate("Rat2/Open1", 20, 1, exact=64.0, discovery=62.0, model=FIRST_ORDER_IMM, start=5.0),
            _candidate("Rat3/Open1", 30, 0, exact=75.0, discovery=70.0, model=FIRST_ORDER_IMM, start=6.0),
            _candidate("Rat4/Open1", 40, 0, exact=65.0, discovery=60.0, model=MOMENTUM_EXACT, start=7.0),
        ]
    )
    candidate_table = validation.rename(columns={"trajectory_minus_nontrajectory_log_evidence": "unused_exact"}).copy()
    high_specificity = candidate_table.copy()
    high_specificity["passes_high_specificity_promotion_filter"] = True

    source = build_source_event_group_summary(
        candidate_table=candidate_table,
        high_specificity=high_specificity,
        validation_decisions=validation,
    )
    decisions = build_one_per_source_group_decisions(validation)
    summary = build_one_per_source_group_summary(decisions)
    gates = build_cluster_robustness_gate_summary(
        validation_decisions=validation,
        source_groups=source,
        one_per_summary=summary,
        margin_threshold=5.5,
    ).set_index("gate")

    assert len(source) == 5
    assert int(source["exact_validated_windows"].sum()) == 7

    strongest = summary[summary["selection_rule"].eq("strongest_exact_margin")].iloc[0]
    assert int(strongest["source_event_groups"]) == 5
    assert int(strongest["selected_candidates"]) == 5
    assert int(strongest["trajectory_confident_candidates"]) == 5
    assert int(strongest["nontrajectory_confident_candidates"]) == 0
    assert float(strongest["min_exact_margin"]) == 64.0
    assert int(strongest["first_order_imm_best_candidates"]) == 3

    earliest = decisions[
        decisions["selection_rule"].eq("earliest_window")
        & decisions["session"].eq("Rat1/Open1")
        & decisions["event_index"].eq(10)
    ].iloc[0]
    assert int(earliest["null_index"]) == 0

    assert bool(gates.loc["deduplicated_source_groups_nontrivial", "passed"])
    assert bool(gates.loc["strongest_exact_rule_trajectory_confident", "passed"])
    assert bool(gates.loc["strongest_exact_rule_no_nontrajectory", "passed"])
    assert bool(gates.loc["all_selection_rules_preserve_trajectory_claim", "passed"])
    assert bool(gates.loc["overall", "passed"])

    outputs = write_off_swr_candidate_dedup_outputs(
        validation_decisions=validation,
        candidate_table=candidate_table,
        high_specificity=high_specificity,
        output=tmp_path,
    )
    assert set(outputs) == {
        "off_swr_candidate_source_event_group_summary.csv",
        "off_swr_candidate_one_per_source_group_decisions.csv",
        "off_swr_candidate_one_per_source_group_summary.csv",
        "off_swr_candidate_cluster_robustness_gate_summary.csv",
    }
    for filename in outputs:
        assert (tmp_path / filename).exists()


def test_off_swr_candidate_dedup_accepts_validation_decisions_only(tmp_path):
    validation = pd.DataFrame(
        [
            _candidate("Rat1/Open1", 10, 0, exact=60.0, discovery=55.0, model=FIRST_ORDER_IMM, start=1.0),
            _candidate("Rat1/Open1", 10, 1, exact=90.0, discovery=80.0, model=FIRST_ORDER_IMM, start=2.0),
        ]
    )

    outputs = write_off_swr_candidate_dedup_outputs(
        validation_decisions=validation,
        output=tmp_path,
    )

    source = outputs["off_swr_candidate_source_event_group_summary.csv"]
    decisions = outputs["off_swr_candidate_one_per_source_group_decisions.csv"]
    gates = outputs["off_swr_candidate_cluster_robustness_gate_summary.csv"].set_index("gate")

    assert len(source) == 1
    assert int(source.iloc[0]["candidate_windows"]) == 0
    assert int(source.iloc[0]["exact_validated_windows"]) == 2
    assert int(decisions["source_event_group_id"].nunique()) == 1
    assert int(gates.loc["deduplicated_source_groups_nontrivial", "observed"]) == 1


def _candidate(
    session: str,
    event_index: int,
    null_index: int,
    *,
    exact: float,
    discovery: float,
    model: str,
    start: float,
) -> dict[str, object]:
    return {
        "session": session,
        "rat": session.split("/", 1)[0],
        "event_index": event_index,
        "null_index": null_index,
        "candidate_rank": null_index + 1,
        "window_start_s": start,
        "window_end_s": start + 0.12,
        "window_duration_s": 0.12,
        "n_spikes": 40 + event_index,
        "active_cell_count": 12,
        "run_or_immobility_state": "immobile",
        "animal_speed_mean": 1.5,
        "distance_to_nearest_swr_s": 10.0 + event_index,
        "trajectory_family_margin": discovery,
        "trajectory_minus_nontrajectory_log_evidence": exact,
        "best_trajectory_model": model,
        "trajectory_confident_claim": True,
        "nontrajectory_confident_claim": False,
        "margin_decision": "trajectory_confident",
    }



def test_off_swr_candidate_dedup_rejects_fractional_event_index():
    validation = pd.DataFrame(
        [{"session": "Rat1/Open1", "event_index": 10.5, "null_index": 0}]
    )

    with pytest.raises(ValueError, match="event_index must contain integer identifiers"):
        build_one_per_source_group_decisions(validation)


def test_off_swr_candidate_dedup_accepts_integer_valued_float_event_index():
    validation = pd.DataFrame(
        [{"session": "Rat1/Open1", "event_index": 10.0, "null_index": 0}]
    )

    decisions = build_one_per_source_group_decisions(validation)

    assert decisions["event_index"].eq(10).all()
    assert decisions["source_event_group_id"].eq("Rat1/Open1|event=10").all()
