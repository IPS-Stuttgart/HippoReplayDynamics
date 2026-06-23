"""Normalize legacy simulation-recovery status values in evidence reporting."""

from __future__ import annotations

from functools import wraps

import numpy as np
import pandas as pd


_PATCHED_FLAG = "_evidence_status_coercion_patch_applied"
_MISSING_STATUS_VALUES = {"", "nan", "na", "n/a", "none", "null", "<na>"}


def apply_evidence_status_coercion_patch() -> None:
    """Treat missing legacy status values as successful, but keep failures excluded."""

    from . import evidence_reporting as reporting

    if getattr(reporting, _PATCHED_FLAG, False):
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
            text = str(value).strip()
            if text:
                labels.append(text)

        for non_exact_support in (
            reporting.TRUNCATED_EVIDENCE_SUPPORT,
            reporting.DEGENERATE_SINGLE_BIN_EVIDENCE_SUPPORT,
            reporting.PYRECEST_PARTICLE_EVIDENCE_SUPPORT,
        ):
            if non_exact_support in labels:
                return non_exact_support
        if reporting.EXACT_EVIDENCE_SUPPORT in labels:
            return reporting.EXACT_EVIDENCE_SUPPORT
        return reporting.EXACT_EVIDENCE_SUPPORT

    def ensure_evidence_support_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Add comparable-evidence flags with consistent legacy-status handling."""

        out = df.copy()
        if out.empty:
            return out
        if "status" in out.columns:
            out["status"] = out["status"].map(_normalize_status_value)
        inferred = out.apply(evidence_support_from_row, axis=1)
        if "evidence_support" in out:
            existing = out["evidence_support"].astype(object)
            missing = existing.map(reporting._is_missing_evidence_support)
            out["evidence_support"] = existing.where(~missing, inferred)
        else:
            out["evidence_support"] = inferred
        status_ok = _status_success_mask(out)
        finite_log_evidence = _finite_log_evidence_mask(out)
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
            & finite_log_evidence
            & out["evidence_support"].eq(reporting.EXACT_EVIDENCE_SUPPORT)
        )
        return reporting.add_candidate_support_quality_columns(out)

    @wraps(original_simulation_add_evidence_columns)
    def simulation_add_evidence_columns(df: pd.DataFrame) -> pd.DataFrame:
        return original_simulation_add_evidence_columns(_normalize_status_frame(df))

    @wraps(original_simulation_event_best_rows)
    def simulation_event_best_rows(event_scores: pd.DataFrame) -> pd.DataFrame:
        return original_simulation_event_best_rows(_normalize_status_frame(event_scores))

    reporting.evidence_support_from_row = evidence_support_from_row
    reporting.ensure_evidence_support_columns = ensure_evidence_support_columns
    reporting.simulation_add_evidence_columns = simulation_add_evidence_columns
    reporting.simulation_event_best_rows = simulation_event_best_rows
    setattr(reporting, _PATCHED_FLAG, True)

    try:
        from . import simulation_recovery as recovery
    except ImportError:
        return
    reporting.patch_simulation_recovery_module(recovery)


def _normalize_status_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if "status" not in frame.columns:
        return frame
    out = frame.copy()
    out["status"] = out["status"].map(_normalize_status_value)
    return out


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
