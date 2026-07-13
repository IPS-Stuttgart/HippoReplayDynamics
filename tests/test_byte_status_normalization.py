from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import hipporeplayimm
from hipporeplayimm import accuracy_model_probability_status_patch
from hipporeplayimm import accuracy_upgrades
from hipporeplayimm import advanced_result_diagnostics
from hipporeplayimm import evidence_reliability
from hipporeplayimm import evidence_reporting
from hipporeplayimm import evidence_status_coercion
from hipporeplayimm import recovery_diagnostics_bool_patch
from hipporeplayimm import result_quality_gates
from hipporeplayimm import simulation_best_row_flags


@pytest.mark.parametrize(
    "status",
    [
        b"success",
        np.bytes_("success"),
        bytearray(b"success"),
        b"",
        np.bytes_("NA"),
    ],
)
def test_byte_encoded_success_and_missing_statuses_are_accepted(status: object) -> None:
    hipporeplayimm.apply_runtime_patches()

    helpers = (
        evidence_status_coercion._status_is_success_or_missing,
        evidence_reporting._status_is_success_or_missing,
        result_quality_gates._status_is_success_or_missing,
        simulation_best_row_flags._status_is_success,
        recovery_diagnostics_bool_patch._status_is_success_or_missing,
        evidence_reliability._status_is_success_or_missing,
    )
    assert all(helper(status) for helper in helpers)
    assert accuracy_model_probability_status_patch._normalize_status_value(status) == "success"


@pytest.mark.parametrize("status", [b"failed", np.bytes_("failed"), b"\xff"])
def test_byte_encoded_failure_statuses_remain_excluded(status: object) -> None:
    hipporeplayimm.apply_runtime_patches()

    helpers = (
        evidence_status_coercion._status_is_success_or_missing,
        evidence_reporting._status_is_success_or_missing,
        result_quality_gates._status_is_success_or_missing,
        simulation_best_row_flags._status_is_success,
        recovery_diagnostics_bool_patch._status_is_success_or_missing,
        evidence_reliability._status_is_success_or_missing,
    )
    assert not any(helper(status) for helper in helpers)
    assert accuracy_model_probability_status_patch._normalize_status_value(status) is status


def test_byte_encoded_success_rows_reach_evidence_summaries() -> None:
    hipporeplayimm.apply_runtime_patches()
    rows = pd.DataFrame(
        {
            "session": ["session", "session"],
            "event_index": [0, 0],
            "model": ["winner", "runner_up"],
            "status": [np.bytes_("success"), b"success"],
            "log_evidence": [2.0, 1.0],
            "evidence_support": [
                evidence_reporting.EXACT_EVIDENCE_SUPPORT,
                evidence_reporting.EXACT_EVIDENCE_SUPPORT,
            ],
            "n_spikes": [5, 5],
            "n_time": [3, 3],
        }
    )

    scored = evidence_reporting.ensure_evidence_support_columns(rows)
    assert scored["status"].tolist() == ["success", "success"]
    assert scored["evidence_comparable"].tolist() == [True, True]
    assert simulation_best_row_flags._status_success_mask(rows).all()
    assert advanced_result_diagnostics._successful_rows(rows).shape[0] == 2
    assert evidence_reliability.add_event_reliability_flags(rows)["event_reliable"].all()

    probabilities = accuracy_upgrades.model_probability_diagnostics(rows)
    assert probabilities.shape[0] == 1
    assert probabilities.loc[0, "best_model"] == "winner"

    gates = result_quality_gates.quality_gate_summary(
        rows,
        min_exact_models_per_event=2,
    ).set_index("gate")
    assert gates.loc["no_failed_rows", "value"] == 0
    assert gates.loc["exact_comparable_rows", "value"] == 2
    assert gates.loc["events_with_min_exact_models", "value"] == 1.0
