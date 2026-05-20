"""Utilities for keeping exact evidences separate from truncated lower bounds."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import logsumexp

EXACT_EVIDENCE_SUPPORT = "exact_full_grid"
TRUNCATED_EVIDENCE_SUPPORT = "truncated_full_grid"
DEGENERATE_SINGLE_BIN_EVIDENCE_SUPPORT = "degenerate_single_bin"
PYRECEST_PARTICLE_EVIDENCE_SUPPORT = "particle_approximation"
EVIDENCE_SUPPORT_DIAGNOSTIC_COLUMNS = (
    "diagnostic_candidate_evidence_support",
    "diagnostic_state_space_momentum_evidence_support",
    "diagnostic_state_space_imm_evidence_support",
    "diagnostic_pyrecest_evidence_support",
)

EVIDENCE_COMPARISON_EXACT = "exact_model_evidence"
EVIDENCE_COMPARISON_LOWER_BOUND = "truncated_lower_bound"
EVIDENCE_COMPARISON_DEGENERATE = "degenerate_single_bin"
EVIDENCE_COMPARISON_NOT_SCORED = "not_scored"
EVIDENCE_COMPARISON_UNKNOWN = "unknown_noncomparable"

EVIDENCE_COMPARISON_DESCRIPTIONS = {
    EVIDENCE_COMPARISON_EXACT: "Exact full-grid model evidences: safe to normalize into posterior model probabilities within the event.",
    EVIDENCE_COMPARISON_LOWER_BOUND: "Truncated candidate-support evidences: lower-bound diagnostics only; do not rank directly against exact full-grid evidences.",
    EVIDENCE_COMPARISON_DEGENERATE: "Degenerate single-bin evidence: exact for a collapsed state support, but not directly comparable to full-grid state supports.",
    EVIDENCE_COMPARISON_NOT_SCORED: "Model was not scored successfully for this event.",
    EVIDENCE_COMPARISON_UNKNOWN: "Evidence support is missing or unknown; treat as non-comparable until classified explicitly.",
}


def evidence_support_from_row(row: pd.Series) -> str:
    """Infer whether a score is exact evidence, a lower bound, or non-comparable."""

    status = row.get("status", "success")
    if pd.notna(status) and str(status) != "success":
        return "not_scored"
    for column in EVIDENCE_SUPPORT_DIAGNOSTIC_COLUMNS:
        value = row.get(column)
        if pd.isna(value):
            continue
        text = str(value)
        if text == TRUNCATED_EVIDENCE_SUPPORT:
            return TRUNCATED_EVIDENCE_SUPPORT
        if text == EXACT_EVIDENCE_SUPPORT:
            return EXACT_EVIDENCE_SUPPORT
        if text == DEGENERATE_SINGLE_BIN_EVIDENCE_SUPPORT:
            return DEGENERATE_SINGLE_BIN_EVIDENCE_SUPPORT
        if text == PYRECEST_PARTICLE_EVIDENCE_SUPPORT:
            return PYRECEST_PARTICLE_EVIDENCE_SUPPORT
    return EXACT_EVIDENCE_SUPPORT


def evidence_comparison_from_support(support: object) -> str:
    """Return the comparison scope implied by an evidence-support label."""

    if support is None:
        return EVIDENCE_COMPARISON_UNKNOWN
    try:
        if pd.isna(support):
            return EVIDENCE_COMPARISON_UNKNOWN
    except (TypeError, ValueError):
        pass
    text = str(support)
    if text == EXACT_EVIDENCE_SUPPORT:
        return EVIDENCE_COMPARISON_EXACT
    if text == TRUNCATED_EVIDENCE_SUPPORT:
        return EVIDENCE_COMPARISON_LOWER_BOUND
    if text == DEGENERATE_SINGLE_BIN_EVIDENCE_SUPPORT:
        return EVIDENCE_COMPARISON_DEGENERATE
    if text == "not_scored":
        return EVIDENCE_COMPARISON_NOT_SCORED
    return EVIDENCE_COMPARISON_UNKNOWN


def ensure_evidence_support_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add comparable-evidence flags used by reporting and aggregation."""

    out = df.copy()
    if out.empty:
        return out
    inferred = out.apply(evidence_support_from_row, axis=1)
    if "evidence_support" in out:
        existing = out["evidence_support"].astype(object)
        missing = existing.isna() | existing.astype(str).str.len().eq(0)
        out["evidence_support"] = existing.where(~missing, inferred)
    else:
        out["evidence_support"] = inferred
    status_ok = out["status"].eq("success") if "status" in out else pd.Series(True, index=out.index)
    out["evidence_comparison"] = out["evidence_support"].map(evidence_comparison_from_support)
    out["evidence_comparison_note"] = out["evidence_comparison"].map(EVIDENCE_COMPARISON_DESCRIPTIONS).fillna(EVIDENCE_COMPARISON_DESCRIPTIONS[EVIDENCE_COMPARISON_UNKNOWN])
    out["evidence_comparable"] = status_ok & out["evidence_support"].eq(EXACT_EVIDENCE_SUPPORT)
    return out


def simulation_add_evidence_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add evidence summaries to simulation-recovery rows without mixing supports."""

    if df.empty:
        return df
    df = ensure_evidence_support_columns(df)
    groups = []
    for _, group in df.groupby(["session", "event_index"], sort=False):
        group = group.copy()
        scored = group[group["status"] == "success"]
        group["relative_log_evidence"] = np.nan
        group["model_probability"] = np.nan
        group["is_best_model"] = False
        group["best_model"] = ""
        group["truncated_relative_log_evidence"] = np.nan
        group["is_best_truncated_lower_bound"] = False
        group["best_truncated_lower_bound_model"] = ""
        if scored.empty:
            if "expected_model" in group:
                group["recovered_expected_model"] = False
                group["lower_bound_recovered_expected_model"] = False
            groups.append(group)
            continue

        exact = scored[scored["evidence_comparable"].fillna(False).astype(bool)]
        if not exact.empty:
            values = exact["log_evidence"].to_numpy(float)
            max_value = float(np.max(values))
            probabilities = np.exp(values - logsumexp(values))
            best_index = exact.index[int(np.argmax(values))]
            best = str(group.loc[best_index, "model"])
            group.loc[exact.index, "relative_log_evidence"] = values - max_value
            group.loc[exact.index, "model_probability"] = probabilities
            group.loc[best_index, "is_best_model"] = True
            group["best_model"] = best

        truncated = scored[scored["evidence_support"].eq(TRUNCATED_EVIDENCE_SUPPORT)]
        if not truncated.empty:
            lower_bounds = truncated["log_evidence"].to_numpy(float)
            max_lower_bound = float(np.max(lower_bounds))
            best_truncated_index = truncated.index[int(np.argmax(lower_bounds))]
            best_truncated = str(group.loc[best_truncated_index, "model"])
            group.loc[truncated.index, "truncated_relative_log_evidence"] = lower_bounds - max_lower_bound
            group.loc[best_truncated_index, "is_best_truncated_lower_bound"] = True
            group["best_truncated_lower_bound_model"] = best_truncated

        if "expected_model" in group:
            group["recovered_expected_model"] = group["best_model"] == group["expected_model"]
            group["lower_bound_recovered_expected_model"] = (
                group["best_truncated_lower_bound_model"] == group["expected_model"]
            )
        groups.append(group)
    return pd.concat(groups, ignore_index=True).sort_values(["event_index", "model"]).reset_index(drop=True)


def simulation_event_best_rows(event_scores: pd.DataFrame) -> pd.DataFrame:
    """Return one exact-comparable best row per simulated event."""

    event_scores = ensure_evidence_support_columns(event_scores)
    ok = event_scores[(event_scores["status"] == "success") & event_scores["evidence_comparable"]]
    if ok.empty:
        return pd.DataFrame()
    if "is_best_model" in ok:
        best = ok[ok["is_best_model"].fillna(False).astype(bool)]
        if not best.empty:
            return best.reset_index(drop=True)
    best = ok.sort_values(["session", "event_index", "log_evidence"], ascending=[True, True, False])
    return best.drop_duplicates(["session", "event_index"], keep="first").reset_index(drop=True)


def patch_simulation_recovery_module(module: object) -> None:
    """Patch simulation recovery reporting to separate exact and truncated evidence."""

    setattr(module, "_ensure_evidence_support_columns", ensure_evidence_support_columns)
    setattr(module, "add_evidence_columns", simulation_add_evidence_columns)
    setattr(module, "_event_best_rows", simulation_event_best_rows)
