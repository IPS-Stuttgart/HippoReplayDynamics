"""Preserve schemas for empty threshold sweeps and result annotations.

Advanced paired-model threshold sweeps may be run on filtered diagnostic tables
where every group is missing either the positive or reference model.  The
underlying decision table is then empty but still has a meaningful threshold and
model-pair context.  Keep those columns present so downstream threshold
selection reports a visible fallback instead of failing with a missing-column
error.

Reliability annotation has the same schema obligation: an empty score table
must retain the standard reliability columns so it can be concatenated with
annotated nonempty tables and safely annotated again.

Simulation evidence annotation also has a stable public schema.  Empty recovery
tables must retain the probability, best-model, surrogate, and recovery columns
that are emitted for nonempty simulations.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import wraps

import numpy as np
import pandas as pd

_PATCHED_FLAG = "_advanced_result_empty_threshold_patch_applied"
_RELIABILITY_PATCHED_FLAG = "_empty_reliability_schema_patch_applied"
_RELIABILITY_WRAPPER_FLAG = "_empty_reliability_schema_wrapper"
_SIMULATION_PATCHED_FLAG = "_empty_simulation_evidence_schema_patch_applied"
_SIMULATION_WRAPPER_FLAG = "_empty_simulation_evidence_schema_wrapper"
_EVIDENCE_STATUS_CORE_WRAPPER_FLAG = "_evidence_status_coercion_core_wrapper"
_SIMULATION_EVIDENCE_COLUMNS = (
    ("relative_log_evidence", float),
    ("model_probability", float),
    ("is_best_model", bool),
    ("best_model", object),
    ("truncated_relative_log_evidence", float),
    ("is_best_truncated_lower_bound", bool),
    ("best_truncated_lower_bound_model", object),
    ("exact_surrogate_best_model", object),
    ("exact_surrogate_recovered_expected_model", bool),
    ("exact_surrogate_log_evidence", float),
    ("exact_surrogate_minus_best_comparable_log_evidence", float),
)
_SIMULATION_EXPECTED_MODEL_COLUMNS = (
    ("recovered_expected_model", bool),
    ("lower_bound_recovered_expected_model", bool),
)


def _normalize_group_cols(group_cols: Sequence[str] | str | None, scores: pd.DataFrame) -> tuple[str, ...]:
    if group_cols is None:
        from . import advanced_result_diagnostics as diagnostics

        return tuple(diagnostics.infer_paired_model_group_cols(scores))
    if isinstance(group_cols, str):
        return (group_cols,)
    return tuple(group_cols)


def _with_threshold_context(
    summary: pd.DataFrame,
    *,
    positive_model: str,
    reference_model: str,
    threshold: float,
    true_model_col: str | None,
) -> pd.DataFrame:
    """Attach model-pair and gate columns to one threshold-summary row."""

    out = summary.copy()
    out["positive_model"] = str(positive_model)
    out["reference_model"] = str(reference_model)
    out["margin_threshold"] = float(threshold)

    if true_model_col:
        defaults: dict[str, float | int] = {
            "thresholded_binary_accuracy": np.nan,
            "positive_true_events": 0,
            "reference_true_events": 0,
            "positive_true_claimed_events": 0,
            "reference_true_rejected_events": 0,
            "positive_claim_recall": np.nan,
            "reference_specificity": np.nan,
            "false_positive_claims": 0,
            "false_negative_claims": 0,
        }
        for column, value in defaults.items():
            if column not in out:
                out[column] = value
    return out


def _with_empty_reliability_schema(frame: pd.DataFrame, reliability) -> pd.DataFrame:
    """Return an empty frame with every public reliability column present."""

    if not frame.empty:
        return frame
    out = frame.copy()
    for column in reliability.RELIABILITY_FLAG_COLUMNS:
        if column in out.columns:
            continue
        dtype = object if column == "event_reliability_reasons" else bool
        out[column] = pd.Series(index=out.index, dtype=dtype)
    return out


def _with_empty_simulation_evidence_schema(frame: pd.DataFrame) -> pd.DataFrame:
    """Return an empty simulation table with all derived evidence columns."""

    if not frame.empty:
        return frame
    out = frame.copy()
    columns = list(_SIMULATION_EVIDENCE_COLUMNS)
    if "expected_model" in out.columns:
        columns.extend(_SIMULATION_EXPECTED_MODEL_COLUMNS)
    for column, dtype in columns:
        if column not in out.columns:
            out[column] = pd.Series(index=out.index, dtype=dtype)
    return out


def _apply_event_reliability_empty_schema_patch() -> None:
    """Keep empty reliability outputs schema-stable and idempotent."""

    from . import evidence_reliability as reliability

    current = reliability.add_event_reliability_flags
    if getattr(current, _RELIABILITY_WRAPPER_FLAG, False):
        setattr(reliability, _RELIABILITY_PATCHED_FLAG, True)
        return

    @wraps(current)
    def add_event_reliability_flags(
        df: pd.DataFrame,
        *,
        min_spikes: int = reliability.DEFAULT_MIN_SPIKES,
        min_time_bins: int = reliability.DEFAULT_MIN_TIME_BINS,
        min_candidate_log_mass: float = reliability.DEFAULT_MIN_CANDIDATE_LOG_MASS,
        max_terminal_entropy: float = reliability.DEFAULT_MAX_TERMINAL_ENTROPY,
    ) -> pd.DataFrame:
        annotated = current(
            df,
            min_spikes=min_spikes,
            min_time_bins=min_time_bins,
            min_candidate_log_mass=min_candidate_log_mass,
            max_terminal_entropy=max_terminal_entropy,
        )
        return _with_empty_reliability_schema(annotated, reliability)

    setattr(add_event_reliability_flags, _RELIABILITY_WRAPPER_FLAG, True)
    reliability.add_event_reliability_flags = add_event_reliability_flags
    setattr(reliability, _RELIABILITY_PATCHED_FLAG, True)


def _apply_simulation_evidence_empty_schema_patch() -> None:
    """Keep empty simulation evidence outputs schema-stable and idempotent."""

    from . import evidence_reporting as reporting
    from . import simulation_recovery as recovery

    current = reporting.simulation_add_evidence_columns
    if getattr(current, _SIMULATION_WRAPPER_FLAG, False):
        recovery.add_evidence_columns = current
        setattr(reporting, _SIMULATION_PATCHED_FLAG, True)
        setattr(recovery, _SIMULATION_PATCHED_FLAG, True)
        return

    @wraps(current)
    def simulation_add_evidence_columns(df: pd.DataFrame) -> pd.DataFrame:
        annotated = current(df)
        return _with_empty_simulation_evidence_schema(annotated)

    setattr(simulation_add_evidence_columns, _SIMULATION_WRAPPER_FLAG, True)
    if getattr(current, _EVIDENCE_STATUS_CORE_WRAPPER_FLAG, False):
        setattr(simulation_add_evidence_columns, _EVIDENCE_STATUS_CORE_WRAPPER_FLAG, True)
    reporting.simulation_add_evidence_columns = simulation_add_evidence_columns
    recovery.add_evidence_columns = simulation_add_evidence_columns
    setattr(reporting, _SIMULATION_PATCHED_FLAG, True)
    setattr(recovery, _SIMULATION_PATCHED_FLAG, True)


def _apply_distinct_model_margin_patch() -> None:
    """Install the event-margin duplicate-model fix from package startup."""

    from . import advanced_result_margin_duplicate_patch

    advanced_result_margin_duplicate_patch.apply_advanced_result_margin_duplicate_patch()


def apply_advanced_result_empty_threshold_patch() -> None:
    """Install empty-result schema preservation patches."""

    from . import advanced_result_diagnostics as diagnostics

    _apply_distinct_model_margin_patch()
    _apply_event_reliability_empty_schema_patch()
    _apply_simulation_evidence_empty_schema_patch()
    if getattr(diagnostics, _PATCHED_FLAG, False):
        return

    def paired_model_margin_threshold_sweep(
        scores: pd.DataFrame,
        *,
        positive_model: str,
        reference_model: str,
        thresholds: Sequence[float],
        group_cols: Sequence[str] | str | None = None,
        evidence_col: str = "log_evidence",
        model_col: str = "model",
        true_model_col: str | None = None,
        positive_true_label: str | None = None,
    ) -> pd.DataFrame:
        """Summarize paired margin decisions over candidate thresholds."""

        paired_group_cols = _normalize_group_cols(group_cols, scores)
        rows: list[pd.DataFrame] = []
        for threshold in thresholds:
            threshold_value = float(threshold)
            decisions = diagnostics.paired_model_margin_decisions(
                scores,
                positive_model=positive_model,
                reference_model=reference_model,
                margin_threshold=threshold_value,
                group_cols=paired_group_cols,
                evidence_col=evidence_col,
                model_col=model_col,
                true_model_col=true_model_col,
                positive_true_label=positive_true_label,
            )
            summary = diagnostics.paired_model_margin_summary(
                decisions,
                true_model_col=true_model_col,
            )
            summary = _with_threshold_context(
                summary,
                positive_model=positive_model,
                reference_model=reference_model,
                threshold=threshold_value,
                true_model_col=true_model_col,
            )
            summary["group_cols"] = ",".join(paired_group_cols)
            rows.append(summary)
        if not rows:
            summary = diagnostics.paired_model_margin_summary(
                pd.DataFrame(),
                true_model_col=true_model_col,
            )
            summary = _with_threshold_context(
                summary,
                positive_model=positive_model,
                reference_model=reference_model,
                threshold=np.nan,
                true_model_col=true_model_col,
            )
            summary["group_cols"] = ",".join(paired_group_cols)
            return summary.iloc[0:0].copy()
        out = pd.concat(rows, ignore_index=True)
        return out.sort_values("margin_threshold", kind="stable").reset_index(drop=True)

    diagnostics.paired_model_margin_threshold_sweep = paired_model_margin_threshold_sweep
    setattr(diagnostics, _PATCHED_FLAG, True)


__all__ = ["apply_advanced_result_empty_threshold_patch"]
