import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))

from audit_first_order_imm_mode_usage import (  # noqa: E402
    DIFFUSION,
    FIRST_ORDER_IMM,
    FRAGMENTED,
    MOMENTUM_EXACT,
    STATIONARY,
    build_first_order_imm_mode_usage_event_table,
    build_first_order_imm_mode_usage_gate_summary,
    build_first_order_imm_mode_usage_summary,
    write_first_order_imm_mode_usage_comparison_audit,
    write_first_order_imm_mode_usage_audit,
)


def test_first_order_imm_audit_supports_content_when_event_mean_modes_are_present(tmp_path):
    evidence = pd.DataFrame(
        [
            *_event("Rat1/Open1", 0, first_order=80.0, event_nonstationary=0.8, terminal_nonstationary=0.7),
            *_event("Rat1/Open1", 1, first_order=70.0, event_nonstationary=0.65, terminal_nonstationary=0.6),
            *_event("Rat2/Open1", 2, first_order=10.0, momentum=90.0, event_nonstationary=0.2, terminal_nonstationary=0.3),
        ]
    )

    event_table = build_first_order_imm_mode_usage_event_table(evidence, margin_threshold=5.5)
    summary = build_first_order_imm_mode_usage_summary(event_table).set_index("scope")
    gates = build_first_order_imm_mode_usage_gate_summary(event_table, margin_threshold=5.5).set_index("gate")

    first_order = summary.loc["first_order_imm_exact_core_best_events"]
    assert int(first_order["events"]) == 2
    assert int(first_order["event_mean_mode_diagnostic_events"]) == 2
    assert int(first_order["event_nonstationary_majority_events"]) == 2
    assert first_order["posterior_content_status"] == "event_mean_mode_mass_available"
    assert bool(gates.loc["posterior_content_claim_supported", "passed"])
    assert bool(gates.loc["overall", "passed"])

    outputs = write_first_order_imm_mode_usage_audit(evidence, tmp_path, margin_threshold=5.5)
    assert set(outputs) == {
        "first_order_imm_mode_usage_event_table.csv",
        "first_order_imm_mode_usage_summary.csv",
        "first_order_imm_mode_usage_rat_summary.csv",
        "first_order_imm_mode_usage_gate_summary.csv",
    }
    for filename in outputs:
        assert (tmp_path / filename).exists()


def test_first_order_imm_audit_blocks_content_claim_for_terminal_only_artifact():
    evidence = pd.DataFrame(
        [
            *_event(
                "Rat1/Open1",
                0,
                first_order=80.0,
                terminal_nonstationary=0.7,
                include_event_mean_modes=False,
            ),
            *_event(
                "Rat1/Open1",
                1,
                first_order=70.0,
                terminal_nonstationary=0.6,
                include_event_mean_modes=False,
            ),
        ]
    )

    event_table = build_first_order_imm_mode_usage_event_table(evidence, margin_threshold=5.5)
    summary = build_first_order_imm_mode_usage_summary(event_table).set_index("scope")
    gates = build_first_order_imm_mode_usage_gate_summary(event_table, margin_threshold=5.5).set_index("gate")

    first_order = summary.loc["first_order_imm_exact_core_best_events"]
    assert int(first_order["terminal_mode_diagnostic_events"]) == 2
    assert int(first_order["terminal_nonstationary_majority_events"]) == 2
    assert int(first_order["event_mean_mode_diagnostic_events"]) == 0
    assert first_order["posterior_content_status"] == "terminal_only_mode_audit"
    assert not bool(gates.loc["first_order_imm_event_mean_mode_diagnostics_present", "passed"])
    assert not bool(gates.loc["posterior_content_claim_supported", "passed"])
    assert bool(gates.loc["overall", "passed"])


def test_first_order_imm_summary_parses_string_false_event_flags():
    event_table = pd.DataFrame(
        [
            {
                "session": "Rat1/Open1",
                "rat": "Rat1",
                "event_index": 0,
                "exact_core_complete": "True",
                "first_order_imm_is_best_exact_core": "False",
                "first_order_imm_confident_exact_core_best": "False",
                "trajectory_capable_confident_vs_stationary": "False",
                "terminal_mode_diagnostics_present": "False",
                "event_mean_mode_diagnostics_present": "False",
                "terminal_nonstationary_majority": "False",
                "event_nonstationary_majority": "False",
            },
            {
                "session": "Rat1/Open1",
                "rat": "Rat1",
                "event_index": 1,
                "exact_core_complete": "False",
                "first_order_imm_is_best_exact_core": "True",
                "first_order_imm_confident_exact_core_best": "True",
                "trajectory_capable_confident_vs_stationary": "True",
                "terminal_mode_diagnostics_present": "True",
                "event_mean_mode_diagnostics_present": "True",
                "terminal_nonstationary_majority": "True",
                "event_nonstationary_majority": "True",
            },
        ]
    )

    summary = build_first_order_imm_mode_usage_summary(event_table).set_index("scope")
    gates = build_first_order_imm_mode_usage_gate_summary(event_table).set_index("gate")

    assert int(summary.loc["complete_exact_core_events", "events"]) == 1
    assert int(summary.loc["first_order_imm_exact_core_best_events", "events"]) == 0
    assert int(summary.loc["complete_exact_core_events", "first_order_imm_best_events"]) == 0
    assert not bool(gates.loc["first_order_imm_best_rows_present", "passed"])


def test_off_swr_audit_keeps_same_source_event_candidates_separate():
    evidence = pd.DataFrame(
        [
            *_tag_candidate(
                _event("Rat2/Open1", 1093, first_order=80.0, event_nonstationary=0.8),
                null_index=2,
            ),
            *_tag_candidate(
                _event("Rat2/Open1", 1093, first_order=70.0, event_nonstationary=0.7),
                null_index=4,
            ),
        ]
    )

    event_table = build_first_order_imm_mode_usage_event_table(
        evidence,
        event_class="promoted_off_swr",
        group_columns=("session", "event_index", "null_index"),
        margin_threshold=5.5,
    )

    assert len(event_table) == 2
    assert set(event_table["event_class"]) == {"promoted_off_swr"}
    assert set(event_table["null_index"].astype(int)) == {2, 4}
    assert event_table["first_order_imm_is_best_exact_core"].all()


def test_comparison_audit_adds_promoted_and_one_per_source_scopes(tmp_path):
    detected = pd.DataFrame(
        _event("Rat1/Open1", 0, first_order=80.0, event_nonstationary=0.8)
    )
    promoted = pd.DataFrame(
        [
            *_tag_candidate(
                _event("Rat2/Open1", 1093, first_order=90.0, event_nonstationary=0.75),
                null_index=2,
                source_event_group_id="Rat2/Open1|event=1093",
            ),
            *_tag_candidate(
                _event("Rat2/Open1", 1093, first_order=70.0, event_nonstationary=0.65),
                null_index=4,
                source_event_group_id="Rat2/Open1|event=1093",
            ),
        ]
    )
    decisions = pd.DataFrame(
        [
            {
                "selection_rule": "strongest_exact_margin",
                "source_event_group_id": "Rat2/Open1|event=1093",
                "session": "Rat2/Open1",
                "event_index": 1093,
                "null_index": 2,
            }
        ]
    )

    outputs = write_first_order_imm_mode_usage_comparison_audit(
        detected,
        tmp_path,
        promoted_off_swr_event_model_evidence=promoted,
        one_per_source_decisions=decisions,
        margin_threshold=5.5,
    )
    gates = outputs["first_order_imm_mode_usage_gate_summary.csv"].set_index(
        ["event_class", "gate"]
    )

    complete_scope = outputs["swr_off_swr_first_order_imm_mode_usage_comparison.csv"]
    complete_scope = complete_scope[
        complete_scope["scope"].eq("complete_exact_core_events")
    ].set_index("event_class")

    assert int(complete_scope.loc["detected_replay_or_swr", "events"]) == 1
    assert int(complete_scope.loc["promoted_off_swr", "events"]) == 2
    assert int(complete_scope.loc["promoted_off_swr_one_per_source", "events"]) == 1
    assert bool(
        gates.loc[
            ("promoted_off_swr_one_per_source", "posterior_content_claim_supported"),
            "passed",
        ]
    )
    for filename in (
        "first_order_imm_mode_usage_event_summary.csv",
        "swr_off_swr_first_order_imm_mode_usage_comparison.csv",
        "rat_first_order_imm_mode_usage_summary.csv",
        "off_swr_one_per_source_group_posterior_content_gate.csv",
    ):
        assert (tmp_path / filename).exists()


def _event(
    session: str,
    event_index: int,
    *,
    first_order: float,
    stationary: float = 0.0,
    diffusion: float = 10.0,
    fragmented: float = 20.0,
    momentum: float = 30.0,
    event_nonstationary: float = 0.8,
    terminal_nonstationary: float = 0.7,
    include_event_mean_modes: bool = True,
) -> list[dict[str, object]]:
    return [
        _score(session, event_index, STATIONARY, stationary),
        _score(session, event_index, DIFFUSION, diffusion),
        _score(session, event_index, FRAGMENTED, fragmented),
        _score(
            session,
            event_index,
            FIRST_ORDER_IMM,
            first_order,
            terminal_nonstationary=terminal_nonstationary,
            event_nonstationary=event_nonstationary,
            include_event_mean_modes=include_event_mean_modes,
        ),
        _score(session, event_index, MOMENTUM_EXACT, momentum),
    ]


def _tag_candidate(
    rows: list[dict[str, object]],
    *,
    null_index: int,
    source_event_group_id: str = "",
) -> list[dict[str, object]]:
    tagged = []
    for row in rows:
        tagged_row = row.copy()
        tagged_row.update(
            {
                "window_role": "promoted_off_swr_candidate",
                "null_index": null_index,
                "source_event_group_id": source_event_group_id,
            }
        )
        tagged.append(tagged_row)
    return tagged


def _score(
    session: str,
    event_index: int,
    model: str,
    log_evidence: float,
    *,
    terminal_nonstationary: float | None = None,
    event_nonstationary: float | None = None,
    include_event_mean_modes: bool = True,
) -> dict[str, object]:
    row = {
        "status": "success",
        "session": session,
        "event_index": event_index,
        "model": model,
        "log_evidence": log_evidence,
        "evidence_comparable": True,
    }
    if model == FIRST_ORDER_IMM:
        terminal = float(terminal_nonstationary)
        row.update(
            {
                "diagnostic_state_space_mode_stationary_terminal_probability": 1.0 - terminal,
                "diagnostic_state_space_mode_diffusion_terminal_probability": terminal * 0.75,
                "diagnostic_state_space_mode_fragmented_terminal_probability": terminal * 0.25,
            }
        )
        if include_event_mean_modes:
            event = float(event_nonstationary)
            row.update(
                {
                    "diagnostic_state_space_mode_stationary_event_probability": 1.0 - event,
                    "diagnostic_state_space_mode_diffusion_event_probability": event * 0.75,
                    "diagnostic_state_space_mode_fragmented_event_probability": event * 0.25,
                }
            )
    return row
