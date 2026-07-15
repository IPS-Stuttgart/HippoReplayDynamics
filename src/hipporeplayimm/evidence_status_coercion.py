"""Normalize legacy simulation-recovery status and evidence-comparison values."""

from __future__ import annotations

from functools import wraps
from typing import Any

import numpy as np
import pandas as pd


_PATCHED_FLAG = "_evidence_status_coercion_patch_applied"
_CORE_WRAPPER_FLAG = "_evidence_status_coercion_core_wrapper"
_CERTIFIED_RECOVERY_PATCHED_FLAG = "_certified_recovery_status_coercion_patch_applied"
_CERTIFIED_EVENT_WRAPPER_FLAG = "_certified_recovery_status_coercion_event_wrapper"
_CERTIFIED_SUMMARY_WRAPPER_FLAG = "_certified_recovery_status_coercion_summary_wrapper"
_CERTIFIED_ORIGINAL_ATTR = "_certified_recovery_status_coercion_original"
_RECOVERY_DIAGNOSTICS_PATCHED_FLAG = "_recovery_diagnostics_status_coercion_patch_applied"
_RECOVERY_DIAGNOSTICS_WRAPPER_FLAG = "_recovery_diagnostics_status_coercion_successful_finite_scores_wrapper"
_MISSING_STATUS_VALUES = {"", "nan", "na", "n/a", "none", "null", "<na>"}
_EXPLICIT_FALSE_BOOL_VALUES = {"0", "0.0", "false", "f", "no", "n", "off"}
_SIMULATION_EVENT_GROUP_COLUMNS = (
    "session",
    "simulation_random_seed",
    "random_seed",
    "benchmark_random_seed",
    "simulation_event_index",
    "event_index",
    "window_index",
    "benchmark_cell_split_index",
    "event_window_variant",
)


def apply_evidence_status_coercion_patch() -> None:
    """Treat missing legacy status values as successful, but keep failures excluded."""

    from . import evidence_reporting as reporting

    if getattr(reporting, _PATCHED_FLAG, False) and _core_reporting_patch_current(reporting):
        _patch_optional_recovery_modules(reporting)
        return

    original_simulation_add_evidence_columns = reporting.simulation_add_evidence_columns
    original_simulation_event_best_rows = reporting.simulation_event_best_rows

    def evidence_support_from_row(row: pd.Series) -> str:
        """Infer evidence support with legacy-missing statuses treated as success."""

        status = row.get("status", "success")
        if not _status_is_success_or_missing(status):
            return "not_scored"

        labels: list[str] = []
        for column in reporting.EVIDENCE_SUPPORT_DIAGNOSTIC_COLUMNS:
            value = row.get(column)
            if _is_missing_scalar(value):
                continue
            labels.extend(reporting._evidence_support_labels(value))

        for non_exact_support in (
            reporting.TRUNCATED_EVIDENCE_SUPPORT,
            reporting.DEGENERATE_SINGLE_BIN_EVIDENCE_SUPPORT,
            reporting.PYRECEST_PARTICLE_EVIDENCE_SUPPORT,
        ):
            if non_exact_support in labels:
                return non_exact_support
        if reporting.EXACT_EVIDENCE_SUPPORT in labels:
            return reporting.EXACT_EVIDENCE_SUPPORT
        if labels:
            return reporting.EVIDENCE_COMPARISON_UNKNOWN
        return reporting.EXACT_EVIDENCE_SUPPORT

    def ensure_evidence_support_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Add comparable-evidence flags with consistent legacy-status handling.

        Older score tables can contain an explicit ``evidence_comparable=False``
        flag without the newer ``evidence_support`` label.  Do not infer exact
        full-grid support for those rows: the explicit non-comparable flag is the
        only trustworthy provenance, so keep the row out of exact-evidence model
        normalization until a support label is supplied.
        """

        out = df.copy()
        if out.empty:
            return _empty_evidence_columns(out, reporting)
        if "status" in out.columns:
            out["status"] = out["status"].map(_normalize_status_value)
        inferred = out.apply(evidence_support_from_row, axis=1)
        if "evidence_support" in out:
            existing = out["evidence_support"].astype(object)
            missing_support = existing.map(reporting._is_missing_evidence_support)
            out["evidence_support"] = existing.where(~missing_support, inferred)
        else:
            missing_support = pd.Series(True, index=out.index)
            out["evidence_support"] = inferred

        explicit_noncomparable_without_support = _explicit_noncomparable_without_support_mask(
            out,
            missing_support=missing_support,
        )
        if explicit_noncomparable_without_support.any():
            out.loc[
                explicit_noncomparable_without_support,
                "evidence_support",
            ] = reporting.EVIDENCE_COMPARISON_UNKNOWN

        status_ok = _status_success_mask(out)
        failed_status = ~status_ok
        if failed_status.any():
            out.loc[failed_status, "evidence_support"] = reporting.EVIDENCE_COMPARISON_NOT_SCORED
        finite_evidence = reporting._finite_evidence_series(out)
        out["evidence_comparison"] = out["evidence_support"].map(
            reporting.evidence_comparison_from_support
        )
        out["evidence_comparison_note"] = out["evidence_comparison"].map(
            reporting.EVIDENCE_COMPARISON_DESCRIPTIONS
        ).fillna(
            reporting.EVIDENCE_COMPARISON_DESCRIPTIONS[reporting.EVIDENCE_COMPARISON_UNKNOWN]
        )
        out["evidence_comparable"] = (
            status_ok
            & finite_evidence
            & out["evidence_support"].eq(reporting.EXACT_EVIDENCE_SUPPORT)
        )
        return reporting.add_candidate_support_quality_columns(out)

    @wraps(original_simulation_add_evidence_columns)
    def simulation_add_evidence_columns(df: pd.DataFrame) -> pd.DataFrame:
        scored = original_simulation_add_evidence_columns(_normalize_status_frame(df))
        if scored.empty:
            scored = _empty_evidence_columns(scored, reporting)
        return _normalize_lower_bound_recovery_flags(scored, reporting)

    @wraps(original_simulation_event_best_rows)
    def simulation_event_best_rows(event_scores: pd.DataFrame) -> pd.DataFrame:
        return original_simulation_event_best_rows(_normalize_status_frame(event_scores))

    for function in (
        evidence_support_from_row,
        ensure_evidence_support_columns,
        simulation_add_evidence_columns,
        simulation_event_best_rows,
    ):
        setattr(function, _CORE_WRAPPER_FLAG, True)

    reporting.evidence_support_from_row = evidence_support_from_row
    reporting.ensure_evidence_support_columns = ensure_evidence_support_columns
    reporting.simulation_add_evidence_columns = simulation_add_evidence_columns
    reporting.simulation_event_best_rows = simulation_event_best_rows
    setattr(reporting, _PATCHED_FLAG, True)

    _patch_optional_recovery_modules(reporting)


def _core_reporting_patch_current(reporting: Any) -> bool:
    """Return whether the core reporting aliases still point to this patch."""

    return all(
        getattr(getattr(reporting, name, None), _CORE_WRAPPER_FLAG, False)
        for name in (
            "evidence_support_from_row",
            "ensure_evidence_support_columns",
            "simulation_add_evidence_columns",
            "simulation_event_best_rows",
        )
    )


def _empty_evidence_columns(frame: pd.DataFrame, reporting: Any) -> pd.DataFrame:
    """Return an empty frame with the standard evidence-reporting columns present."""

    out = frame.copy()
    if "evidence_support" not in out.columns:
        out["evidence_support"] = pd.Series(dtype=object)
    if "evidence_comparison" not in out.columns:
        out["evidence_comparison"] = pd.Series(dtype=object)
    if "evidence_comparison_note" not in out.columns:
        out["evidence_comparison_note"] = pd.Series(dtype=object)
    if "evidence_comparable" not in out.columns:
        out["evidence_comparable"] = pd.Series(dtype=bool)
    return reporting.add_candidate_support_quality_columns(out)


def _patch_optional_recovery_modules(reporting: Any) -> None:
    """Refresh optional recovery-module aliases whenever they are importable.

    ``apply_evidence_status_coercion_patch`` can be called after the core reporting
    patch is already installed. Optional recovery modules that were imported later
    or reloaded still need their local aliases synchronized with the patched
    reporting/recovery functions.
    """

    try:
        from . import simulation_recovery as recovery
    except ImportError:
        recovery = None
    else:
        reporting.patch_simulation_recovery_module(recovery)
        _patch_certified_recovery(recovery)

    try:
        from . import recovery_diagnostics as diagnostics
    except ImportError:
        return
    _patch_recovery_diagnostics(diagnostics, recovery)


def _patch_certified_recovery(recovery: Any) -> None:
    """Normalize legacy status values before certified-vs-exact recovery views."""

    if getattr(recovery, _CERTIFIED_RECOVERY_PATCHED_FLAG, False) and _certified_recovery_patch_current(recovery):
        return

    original_certified_events = _unwrap_certified_recovery(recovery.certified_vs_exact_event_recovery)
    original_certified_summary = _unwrap_certified_recovery(recovery.certified_vs_exact_recovery_summary)

    @wraps(original_certified_events)
    def certified_vs_exact_event_recovery(event_scores: pd.DataFrame) -> pd.DataFrame:
        return original_certified_events(_normalize_status_frame(event_scores))

    @wraps(original_certified_summary)
    def certified_vs_exact_recovery_summary(event_scores: pd.DataFrame) -> pd.DataFrame:
        return original_certified_summary(_normalize_status_frame(event_scores))

    setattr(certified_vs_exact_event_recovery, _CERTIFIED_EVENT_WRAPPER_FLAG, True)
    setattr(certified_vs_exact_event_recovery, _CERTIFIED_ORIGINAL_ATTR, original_certified_events)
    setattr(certified_vs_exact_recovery_summary, _CERTIFIED_SUMMARY_WRAPPER_FLAG, True)
    setattr(certified_vs_exact_recovery_summary, _CERTIFIED_ORIGINAL_ATTR, original_certified_summary)
    recovery.certified_vs_exact_event_recovery = certified_vs_exact_event_recovery
    recovery.certified_vs_exact_recovery_summary = certified_vs_exact_recovery_summary
    setattr(recovery, _CERTIFIED_RECOVERY_PATCHED_FLAG, True)


def _certified_recovery_patch_current(recovery: Any) -> bool:
    """Return whether certified-recovery aliases still point to this patch."""

    return bool(
        getattr(getattr(recovery, "certified_vs_exact_event_recovery", None), _CERTIFIED_EVENT_WRAPPER_FLAG, False)
        and getattr(getattr(recovery, "certified_vs_exact_recovery_summary", None), _CERTIFIED_SUMMARY_WRAPPER_FLAG, False)
    )


def _unwrap_certified_recovery(function: Any) -> Any:
    return getattr(function, _CERTIFIED_ORIGINAL_ATTR, function)


def _normalize_lower_bound_recovery_flags(frame: pd.DataFrame, reporting: Any) -> pd.DataFrame:
    """Make lower-bound recovery flags event-scoped and surrogate-aware."""

    required_columns = {
        "best_truncated_lower_bound_model",
        "expected_model",
        "lower_bound_recovered_expected_model",
    }
    if frame.empty or not required_columns.issubset(frame.columns):
        return frame

    out = frame.copy()
    group_columns = _simulation_event_group_columns(out)
    if not group_columns:
        _set_lower_bound_recovery_flag(out, out, reporting)
        return out

    for _, group in out.groupby(group_columns, sort=False, dropna=False):
        _set_lower_bound_recovery_flag(out, group, reporting)
    return out


def _simulation_event_group_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in _SIMULATION_EVENT_GROUP_COLUMNS if column in frame.columns]


def _set_lower_bound_recovery_flag(out: pd.DataFrame, group: pd.DataFrame, reporting: Any) -> None:
    acceptable_models = {
        str(model).strip()
        for model in reporting._simulation_acceptable_recovery_models(group)
        if str(model).strip()
    }
    best_model = _first_nonmissing_text(group["best_truncated_lower_bound_model"])
    out.loc[group.index, "lower_bound_recovered_expected_model"] = bool(
        best_model and best_model in acceptable_models
    )


def _first_nonmissing_text(values: pd.Series) -> str:
    for value in values:
        if _is_missing_scalar(value):
            continue
        text = str(value).strip()
        if text and text.lower() not in _MISSING_STATUS_VALUES:
            return text
    return ""


def _normalize_status_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if "status" not in frame.columns:
        return frame
    out = frame.copy()
    out["status"] = out["status"].map(_normalize_status_value)
    return out


def _patch_recovery_diagnostics(diagnostics: Any, recovery: Any | None = None) -> None:
    """Keep recovery-diagnostic score filtering aligned with evidence reporting."""

    if recovery is not None:
        diagnostics.certified_vs_exact_event_recovery = recovery.certified_vs_exact_event_recovery
        diagnostics.certified_vs_exact_recovery_summary = recovery.certified_vs_exact_recovery_summary

    if getattr(diagnostics, _RECOVERY_DIAGNOSTICS_PATCHED_FLAG, False) and _recovery_diagnostics_patch_current(diagnostics):
        return

    original_successful_finite_scores = diagnostics._successful_finite_scores

    @wraps(original_successful_finite_scores)
    def successful_finite_scores(group: pd.DataFrame) -> pd.DataFrame:
        status_ok = _status_success_mask(group)
        finite = _finite_log_evidence_mask(group)
        return group[status_ok & finite].copy()

    setattr(successful_finite_scores, _RECOVERY_DIAGNOSTICS_WRAPPER_FLAG, True)
    diagnostics._successful_finite_scores = successful_finite_scores
    setattr(diagnostics, _RECOVERY_DIAGNOSTICS_PATCHED_FLAG, True)


def _recovery_diagnostics_patch_current(diagnostics: Any) -> bool:
    """Return whether recovery diagnostics still use the status-coercion helper."""

    return bool(
        getattr(
            getattr(diagnostics, "_successful_finite_scores", None),
            _RECOVERY_DIAGNOSTICS_WRAPPER_FLAG,
            False,
        )
    )


def _normalize_status_value(value: object) -> object:
    return "success" if _status_is_success_or_missing(value) else value


def _status_success_mask(frame: pd.DataFrame) -> pd.Series:
    if "status" not in frame.columns:
        return pd.Series(True, index=frame.index)
    return frame["status"].map(_status_is_success_or_missing).astype(bool)


def _finite_log_evidence_mask(frame: pd.DataFrame) -> pd.Series:
    if "log_evidence" not in frame.columns:
        return pd.Series(True, index=frame.index)
    values = pd.to_numeric(frame["log_evidence"], errors="coerce")
    return pd.Series(np.isfinite(values.to_numpy(dtype=float)), index=frame.index)


def _explicit_noncomparable_without_support_mask(
    frame: pd.DataFrame,
    *,
    missing_support: pd.Series,
) -> pd.Series:
    """Return rows with explicit non-comparable flags but no support label."""

    if "evidence_comparable" not in frame.columns:
        return pd.Series(False, index=frame.index)
    missing = pd.Series(missing_support, index=frame.index).astype(bool)
    explicit_false = frame["evidence_comparable"].map(_is_explicit_false_value).astype(bool)
    return missing & explicit_false


def _is_explicit_false_value(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return not bool(value)
    if _is_missing_scalar(value):
        return False
    if isinstance(value, (int, np.integer)):
        return int(value) == 0
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        return bool(np.isfinite(numeric) and numeric == 0.0)
    return str(value).strip().lower() in _EXPLICIT_FALSE_BOOL_VALUES


def _status_is_success_or_missing(value: object) -> bool:
    if _is_missing_scalar(value):
        return True
    return str(value).strip().lower() == "success"


def _is_missing_scalar(value: object) -> bool:
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, (bool, np.bool_)) and bool(missing):
        return True
    return str(value).strip().lower() in _MISSING_STATUS_VALUES


__all__ = ["apply_evidence_status_coercion_patch"]
