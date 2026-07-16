from __future__ import annotations

import pandas as pd

import hipporeplayimm
import hipporeplayimm.evidence_reporting as evidence_reporting
import hipporeplayimm.simulation_recovery as simulation_recovery


_FLOAT_COLUMNS = (
    "relative_log_evidence",
    "model_probability",
    "truncated_relative_log_evidence",
    "exact_surrogate_log_evidence",
    "exact_surrogate_minus_best_comparable_log_evidence",
)
_BOOL_COLUMNS = (
    "is_best_model",
    "is_best_truncated_lower_bound",
    "exact_surrogate_recovered_expected_model",
    "recovered_expected_model",
    "lower_bound_recovered_expected_model",
)
_OBJECT_COLUMNS = (
    "best_model",
    "best_truncated_lower_bound_model",
    "exact_surrogate_best_model",
)


def _empty_scores() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "session": pd.Series(dtype=object),
            "event_index": pd.Series(dtype=int),
            "model": pd.Series(dtype=object),
            "log_evidence": pd.Series(dtype=float),
            "expected_model": pd.Series(dtype=object),
        }
    )


def _assert_simulation_schema(frame: pd.DataFrame) -> None:
    assert frame.empty
    for column in _FLOAT_COLUMNS:
        assert frame.columns.tolist().count(column) == 1
        assert frame[column].dtype == float
    for column in _BOOL_COLUMNS:
        assert frame.columns.tolist().count(column) == 1
        assert frame[column].dtype == bool
    for column in _OBJECT_COLUMNS:
        assert frame.columns.tolist().count(column) == 1
        assert frame[column].dtype == object


def test_empty_simulation_evidence_annotation_preserves_schema_on_rerun() -> None:
    scores = _empty_scores()

    once = evidence_reporting.simulation_add_evidence_columns(scores)
    twice = evidence_reporting.simulation_add_evidence_columns(once)
    through_recovery = simulation_recovery.add_evidence_columns(scores)

    _assert_simulation_schema(once)
    _assert_simulation_schema(twice)
    _assert_simulation_schema(through_recovery)


def test_empty_simulation_evidence_patch_is_runtime_idempotent() -> None:
    patched = evidence_reporting.simulation_add_evidence_columns

    hipporeplayimm.apply_runtime_patches()

    assert evidence_reporting.simulation_add_evidence_columns is patched
    assert simulation_recovery.add_evidence_columns is patched
