import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))

from audit_first_order_imm_event_mean_mode_usage import (  # noqa: E402
    DIFFUSION,
    FIRST_ORDER_IMM,
    FRAGMENTED,
    MOMENTUM_EXACT,
    STATIONARY,
    _read_event_model_evidence,
    build_event_mean_mode_usage_event_summary,
    build_mode_usage_gate_summary,
    write_event_mean_mode_usage_audit,
)


def test_event_mean_mode_usage_audit_passes_with_content_diagnostics(tmp_path):
    evidence = pd.DataFrame(
        [
            *_event("Rat1/Open1", 0, first_order=80.0, mean_nonstationary=0.8, map_nonstationary=0.7, path=25.0),
            *_event("Rat1/Open1", 1, first_order=70.0, mean_nonstationary=0.75, map_nonstationary=0.6, path=18.0),
            *_event("Rat2/Open1", 2, first_order=10.0, momentum=90.0, mean_nonstationary=0.2, map_nonstationary=0.2, path=5.0),
        ]
    )

    event_summary = build_event_mean_mode_usage_event_summary(
        evidence,
        event_class="detected_replay_or_swr",
    )
    gates = build_mode_usage_gate_summary(event_summary).set_index("gate")

    assert int(event_summary["trajectory_content_gate_passed"].sum()) == 2
    assert bool(gates.loc["moderate_content_majority", "passed"])
    assert bool(gates.loc["overall", "passed"])

    outputs = write_event_mean_mode_usage_audit(
        event_model_evidence=evidence,
        output=tmp_path,
    )
    assert set(outputs) == {
        "first_order_imm_mode_usage_event_summary.csv",
        "first_order_imm_mode_usage_gate_summary.csv",
        "rat_first_order_imm_mode_usage_summary.csv",
        "session_first_order_imm_mode_usage_summary.csv",
        "swr_off_swr_first_order_imm_mode_usage_comparison.csv",
        "off_swr_one_per_source_group_mode_usage_summary.csv",
    }
    for filename in outputs:
        assert (tmp_path / filename).exists()


def test_event_mean_mode_usage_audit_fails_terminal_only_artifact():
    evidence = pd.DataFrame(
        [
            *_event(
                "Rat1/Open1",
                0,
                first_order=80.0,
                include_event_content=False,
            ),
            *_event(
                "Rat1/Open1",
                1,
                first_order=70.0,
                include_event_content=False,
            ),
        ]
    )

    event_summary = build_event_mean_mode_usage_event_summary(
        evidence,
        event_class="detected_replay_or_swr",
    )
    gates = build_mode_usage_gate_summary(event_summary).set_index("gate")

    assert int(event_summary["event_mean_mode_diagnostics_present"].sum()) == 0
    assert int(event_summary["trajectory_content_gate_passed"].sum()) == 0
    assert not bool(gates.loc["event_mean_mode_diagnostics_complete", "passed"])
    assert not bool(gates.loc["overall", "passed"])


def test_event_mean_gate_parses_string_false_first_order_flags():
    event_summary = pd.DataFrame(
        [
            {
                "event_class": "detected_replay_or_swr",
                "selection_rule": "",
                "first_order_imm_is_best_exact_core": "False",
                "event_mean_mode_diagnostics_present": "True",
                "map_mode_diagnostics_present": "True",
                "spatial_content_diagnostics_present": "True",
                "trajectory_content_gate_passed": "True",
                "strong_trajectory_content_gate_passed": "True",
            }
        ]
    )

    gates = build_mode_usage_gate_summary(event_summary).set_index("gate")

    assert not bool(gates.loc["first_order_imm_best_rows_present", "passed"])
    assert not bool(gates.loc["moderate_content_majority", "passed"])
    assert not bool(gates.loc["overall", "passed"])


def test_event_mean_evidence_reader_keeps_blank_status_rows(tmp_path):
    legacy_rows = pd.DataFrame(_event("Rat1/Open1", 0, first_order=80.0, mean_nonstationary=0.8, map_nonstationary=0.7, path=25.0))
    legacy_rows["status"] = ""
    failed_row = _score("Rat1/Open1", 1, FIRST_ORDER_IMM, 999.0)
    failed_row["status"] = "failed"
    evidence = pd.concat([legacy_rows, pd.DataFrame([failed_row])], ignore_index=True)
    path = tmp_path / "event_model_evidence.csv"
    evidence.to_csv(path, index=False)

    loaded = _read_event_model_evidence(path)

    assert len(loaded) == len(legacy_rows)
    assert set(loaded["event_index"].astype(int)) == {0}
    assert set(loaded["model"]) == set(legacy_rows["model"])


def test_event_mean_mode_usage_audit_keeps_off_swr_candidates_distinct(tmp_path):
    promoted = pd.DataFrame(
        [
            *_tag_candidate(
                _event("Rat2/Open1", 1093, first_order=90.0, mean_nonstationary=0.8, map_nonstationary=0.7, path=25.0),
                null_index=2,
            ),
            *_tag_candidate(
                _event("Rat2/Open1", 1093, first_order=85.0, mean_nonstationary=0.75, map_nonstationary=0.65, path=22.0),
                null_index=4,
            ),
        ]
    )
    detected = pd.DataFrame(
        _event("Rat1/Open1", 0, first_order=80.0, mean_nonstationary=0.8, map_nonstationary=0.7, path=25.0)
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

    outputs = write_event_mean_mode_usage_audit(
        event_model_evidence=detected,
        promoted_off_swr_event_model_evidence=promoted,
        one_per_source_decisions=decisions,
        output=tmp_path,
    )
    event_summary = outputs["first_order_imm_mode_usage_event_summary.csv"]
    promoted_rows = event_summary[event_summary["event_class"].eq("promoted_off_swr")]
    one_per_rows = event_summary[event_summary["event_class"].eq("promoted_off_swr_one_per_source")]

    assert len(promoted_rows) == 2
    assert set(promoted_rows["null_index"].astype(int)) == {2, 4}
    assert len(one_per_rows) == 1
    assert int(one_per_rows.iloc[0]["null_index"]) == 2


def _event(
    session: str,
    event_index: int,
    *,
    first_order: float,
    stationary: float = 0.0,
    diffusion: float = 10.0,
    fragmented: float = 20.0,
    momentum: float = 30.0,
    mean_nonstationary: float = 0.8,
    map_nonstationary: float = 0.7,
    path: float = 20.0,
    include_event_content: bool = True,
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
            mean_nonstationary=mean_nonstationary,
            map_nonstationary=map_nonstationary,
            path=path,
            include_event_content=include_event_content,
        ),
        _score(session, event_index, MOMENTUM_EXACT, momentum),
    ]


def _tag_candidate(rows: list[dict[str, object]], *, null_index: int) -> list[dict[str, object]]:
    tagged = []
    for row in rows:
        copy = row.copy()
        copy.update(
            {
                "window_role": "promoted_off_swr_candidate",
                "null_index": null_index,
                "source_event_group_id": "Rat2/Open1|event=1093",
            }
        )
        tagged.append(copy)
    return tagged


def _score(
    session: str,
    event_index: int,
    model: str,
    log_evidence: float,
    *,
    mean_nonstationary: float = 0.8,
    map_nonstationary: float = 0.7,
    path: float = 20.0,
    include_event_content: bool = True,
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
        row.update(
            {
                "diagnostic_state_space_mode_stationary_terminal_probability": 0.25,
                "diagnostic_state_space_mode_diffusion_terminal_probability": 0.5,
                "diagnostic_state_space_mode_fragmented_terminal_probability": 0.25,
            }
        )
        if include_event_content:
            row.update(
                {
                    "diagnostic_state_space_mode_stationary_event_probability": 1.0 - mean_nonstationary,
                    "diagnostic_state_space_mode_diffusion_event_probability": mean_nonstationary * 0.75,
                    "diagnostic_state_space_mode_fragmented_event_probability": mean_nonstationary * 0.25,
                    "diagnostic_state_space_imm_fraction_time_map_stationary": 1.0 - map_nonstationary,
                    "diagnostic_state_space_imm_fraction_time_map_nonstationary": map_nonstationary,
                    "diagnostic_state_space_imm_nonstationary_bout_count": 1,
                    "diagnostic_state_space_imm_longest_nonstationary_bout_s": 0.05,
                    "diagnostic_state_space_imm_posterior_expected_path_length_cm": path,
                    "diagnostic_state_space_imm_posterior_net_displacement_cm": path / 2.0,
                    "diagnostic_state_space_imm_posterior_path_speed_cm_s": path * 10.0,
                }
            )
    return row
