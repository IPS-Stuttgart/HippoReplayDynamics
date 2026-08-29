from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))

from audit_first_order_imm_event_mean_mode_usage import (  # noqa: E402
    FIRST_ORDER_IMM,
    build_event_mean_mode_usage_event_summary,
    build_mode_usage_gate_summary,
)


def test_incomplete_exact_core_cannot_be_reported_as_first_order_imm_winner() -> None:
    evidence = pd.DataFrame(
        [
            {
                "status": "success",
                "session": "Rat1/Open1",
                "event_index": 7,
                "model": FIRST_ORDER_IMM,
                "log_evidence": 100.0,
                "evidence_comparable": True,
                "diagnostic_state_space_mode_stationary_event_probability": 0.1,
                "diagnostic_state_space_mode_diffusion_event_probability": 0.7,
                "diagnostic_state_space_mode_fragmented_event_probability": 0.2,
                "diagnostic_state_space_mode_stationary_terminal_probability": 0.1,
                "diagnostic_state_space_mode_diffusion_terminal_probability": 0.7,
                "diagnostic_state_space_mode_fragmented_terminal_probability": 0.2,
                "diagnostic_state_space_imm_fraction_time_map_stationary": 0.1,
                "diagnostic_state_space_imm_fraction_time_map_nonstationary": 0.9,
                "diagnostic_state_space_imm_nonstationary_bout_count": 1,
                "diagnostic_state_space_imm_longest_nonstationary_bout_s": 0.05,
                "diagnostic_state_space_imm_posterior_expected_path_length_cm": 25.0,
                "diagnostic_state_space_imm_posterior_net_displacement_cm": 15.0,
                "diagnostic_state_space_imm_posterior_path_speed_cm_s": 250.0,
            }
        ]
    )

    event_summary = build_event_mean_mode_usage_event_summary(
        evidence,
        event_class="detected_replay_or_swr",
    )
    row = event_summary.iloc[0]

    assert not bool(row["first_order_imm_is_best_exact_core"])
    assert row["best_exact_core_model"] == ""
    assert not bool(row["trajectory_content_gate_passed"])

    gates = build_mode_usage_gate_summary(event_summary).set_index("gate")
    assert not bool(gates.loc["first_order_imm_best_rows_present", "passed"])
    assert not bool(gates.loc["overall", "passed"])
