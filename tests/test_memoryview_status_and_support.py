from __future__ import annotations

import pandas as pd
import pytest

import hipporeplayimm
from hipporeplayimm import accuracy_model_probability_status_patch
from hipporeplayimm import advanced_result_diagnostics
from hipporeplayimm import evidence_reliability
from hipporeplayimm import evidence_reporting
from hipporeplayimm import evidence_status_coercion
from hipporeplayimm import recovery_diagnostics_bool_patch
from hipporeplayimm import result_quality_gates
from hipporeplayimm import simulation_best_row_flags
from hipporeplayimm.evidence_reliability import add_event_reliability_flags


@pytest.mark.parametrize(
    "status",
    [
        memoryview(b"success"),
        memoryview(b""),
        memoryview(b"NA"),
    ],
)
def test_memoryview_success_and_missing_statuses_are_accepted(status: memoryview) -> None:
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


def test_memoryview_failure_status_remains_excluded() -> None:
    hipporeplayimm.apply_runtime_patches()
    status = memoryview(b"failed")

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


def test_memoryview_success_row_reaches_advanced_diagnostics() -> None:
    hipporeplayimm.apply_runtime_patches()
    rows = pd.DataFrame(
        {
            "status": [memoryview(b"success"), memoryview(b"failed")],
            "log_evidence": [2.0, 1.0],
        }
    )

    successful = advanced_result_diagnostics._successful_rows(rows)

    assert successful.index.tolist() == [0]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (memoryview(b"exact_full_grid"), evidence_reporting.EVIDENCE_COMPARISON_EXACT),
        (
            memoryview(b"truncated_full_grid"),
            evidence_reporting.EVIDENCE_COMPARISON_LOWER_BOUND,
        ),
        (
            memoryview(b"particle_approximation"),
            evidence_reporting.EVIDENCE_COMPARISON_PARTICLE_APPROXIMATION,
        ),
    ],
)
def test_memoryview_support_scalars_preserve_comparison_scope(
    value: memoryview,
    expected: str,
) -> None:
    hipporeplayimm.apply_runtime_patches()

    assert evidence_reporting.evidence_comparison_from_support(value) == expected


def test_memoryview_degenerate_support_marks_event_unreliable() -> None:
    hipporeplayimm.apply_runtime_patches()
    scores = pd.DataFrame(
        [
            {
                "status": "success",
                "n_spikes": 4,
                "n_time": 3,
                "mean_candidate_log_mass": 0.0,
                "diagnostic_candidate_evidence_support": memoryview(
                    b"degenerate_single_bin"
                ),
            }
        ]
    )

    flagged = add_event_reliability_flags(scores)

    assert not bool(flagged.loc[0, "event_reliable"])
    assert flagged.loc[0, "event_reliability_reasons"] == "degenerate_single_bin"
