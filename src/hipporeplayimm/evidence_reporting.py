"""Utilities for keeping exact evidences separate from truncated lower bounds."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from .result_improvements import add_candidate_support_quality_columns

EXACT_EVIDENCE_SUPPORT = "exact_full_grid"
TRUNCATED_EVIDENCE_SUPPORT = "truncated_full_grid"
DEGENERATE_SINGLE_BIN_EVIDENCE_SUPPORT = "degenerate_single_bin"
PYRECEST_PARTICLE_EVIDENCE_SUPPORT = "particle_approximation"
EVIDENCE_SUPPORT_DIAGNOSTIC_COLUMNS = (
    "diagnostic_candidate_evidence_support",
    # Check specific state-space model labels before generic component labels.
    # Exact-sparse and trajectory-IMM rows also emit state_space_momentum_*
    # diagnostics for shared momentum parameters; those component diagnostics
    # must not override a more specific non-comparable support label.
    "diagnostic_state_space_sparse_momentum_evidence_support",
    "diagnostic_state_space_trajectory_imm_evidence_support",
    "diagnostic_state_space_displacement_momentum_evidence_support",
    "diagnostic_state_space_displacement_imm_evidence_support",
    "diagnostic_state_space_momentum_evidence_support",
    "diagnostic_state_space_imm_evidence_support",
    "diagnostic_goal_state_space_evidence_support",
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
_MOMENTUM_EXACT_SURROGATE_MODELS = (
    "sorted-spike-state-space-momentum-exact-sparse",
    "state-space-momentum-exact-sparse",
    "sorted-spike-state-space-displacement-momentum",
    "state-space-displacement-momentum",
    "sorted-spike-state-space-velocity-momentum",
    "state-space-velocity-momentum",
)


_FALSE_BOOL_STRINGS = {"", "0", "0.0", "false", "f", "no", "n", "off", "nan", "none", "null"}
_TRUE_BOOL_STRINGS = {"1", "1.0", "true", "t", "yes", "y", "on"}


def _coerce_bool_series(values: pd.Series, *, default: bool = False) -> pd.Series:
    """Coerce bool-like scalars without treating every non-empty string as true.

    Pandas ``Series.astype(bool)`` treats any non-empty object string as
    ``True``.  Score tables are often read back from CSV, where false flags may
    appear as strings such as ``"False"`` or ``"0"``.  Keep unrecognised and
    missing values on the conservative default side so reporting code does not
    accidentally admit non-comparable rows or stale best-model markers.
    """

    def coerce(value: object) -> bool:
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        try:
            if pd.isna(value):
                return bool(default)
        except (TypeError, ValueError):
            return bool(default)
        if isinstance(value, (int, float, np.integer, np.floating)):
            numeric = float(value)
            return bool(np.isfinite(numeric) and numeric != 0.0)
        text = str(value).strip().lower()
        if text in _TRUE_BOOL_STRINGS:
            return True
        if text in _FALSE_BOOL_STRINGS:
            return False
        try:
            numeric = float(text)
        except ValueError:
            return bool(default)
        return bool(np.isfinite(numeric) and numeric != 0.0)

    return values.map(coerce).astype(bool)


def evidence_support_from_row(row: pd.Series) -> str:
    """Infer whether a score is exact evidence, a lower bound, or non-comparable."""

    status = row.get("status", "success")
    if pd.notna(status) and str(status) != "success":
        return "not_scored"

    labels: list[str] = []
    for column in EVIDENCE_SUPPORT_DIAGNOSTIC_COLUMNS:
        value = row.get(column)
        if pd.isna(value):
            continue
        text = str(value).strip()
        if text:
            labels.append(text)

    for non_exact_support in (
        TRUNCATED_EVIDENCE_SUPPORT,
        DEGENERATE_SINGLE_BIN_EVIDENCE_SUPPORT,
        PYRECEST_PARTICLE_EVIDENCE_SUPPORT,
    ):
        if non_exact_support in labels:
            return non_exact_support
    if EXACT_EVIDENCE_SUPPORT in labels:
        return EXACT_EVIDENCE_SUPPORT
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
    return add_candidate_support_quality_columns(out)


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
        group["exact_surrogate_best_model"] = ""
        group["exact_surrogate_recovered_expected_model"] = False
        group["exact_surrogate_log_evidence"] = np.nan
        group["exact_surrogate_minus_best_comparable_log_evidence"] = np.nan

        if not scored.empty:
            finite_log_evidence = pd.Series(
                np.isfinite(scored["log_evidence"].to_numpy(float)),
                index=scored.index,
            )
            nonfinite_index = finite_log_evidence.index[~finite_log_evidence.to_numpy()]
            if len(nonfinite_index):
                group.loc[nonfinite_index, "evidence_comparable"] = False
            scored = scored.loc[finite_log_evidence]

        if scored.empty:
            if "expected_model" in group:
                group["recovered_expected_model"] = False
                group["lower_bound_recovered_expected_model"] = False
            groups.append(group)
            continue

        best = ""
        exact = scored[_coerce_bool_series(scored["evidence_comparable"])]
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
            surrogate_models = _simulation_exact_surrogate_models(group)
            surrogate_rows = exact[exact["model"].astype(str).isin(surrogate_models)]
            if not surrogate_rows.empty:
                surrogate = _best_log_evidence_row(surrogate_rows)
                surrogate_log_evidence = float(surrogate["log_evidence"])
                group["exact_surrogate_best_model"] = str(surrogate["model"])
                group["exact_surrogate_log_evidence"] = surrogate_log_evidence
                group["exact_surrogate_minus_best_comparable_log_evidence"] = (
                    surrogate_log_evidence - max_value
                )
                group["exact_surrogate_recovered_expected_model"] = bool(
                    str(surrogate["model"]) == best
                )

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
            group["recovered_expected_model"] = best in _simulation_acceptable_recovery_models(
                group
            )
            group["lower_bound_recovered_expected_model"] = (
                group["best_truncated_lower_bound_model"] == group["expected_model"]
            )
        groups.append(group)
    return pd.concat(groups, ignore_index=True).sort_values(["event_index", "model"]).reset_index(drop=True)


def _simulation_acceptable_recovery_models(group: pd.DataFrame) -> tuple[str, ...]:
    models: list[str] = []
    expected = _simulation_event_text(group, "expected_model")
    if expected:
        models.append(expected)
    models.extend(_simulation_exact_surrogate_models(group))
    return tuple(dict.fromkeys(models))


def _simulation_exact_surrogate_models(group: pd.DataFrame) -> tuple[str, ...]:
    models: list[str] = []
    true_model = _simulation_event_text(group, "true_model").lower()
    if true_model == "momentum":
        models.extend(_MOMENTUM_EXACT_SURROGATE_MODELS)
    models.extend(_simulation_model_values(group, "expected_exact_surrogate_model"))
    return tuple(dict.fromkeys(models))


def _simulation_event_text(group: pd.DataFrame, column: str) -> str:
    if column not in group.columns:
        return ""
    values = group[column].dropna().astype(str)
    if values.empty:
        return ""
    return str(values.iloc[0]).strip()


def _simulation_model_values(group: pd.DataFrame, column: str) -> list[str]:
    values: list[str] = []
    if column not in group.columns:
        return values
    for value in group[column].dropna().astype(str):
        for model in value.replace(",", " ").split():
            if model:
                values.append(model)
    return list(dict.fromkeys(values))


def _best_log_evidence_row(frame: pd.DataFrame) -> pd.Series:
    values = frame["log_evidence"].to_numpy(float)
    return frame.iloc[int(np.argmax(values))]


def simulation_event_best_rows(event_scores: pd.DataFrame) -> pd.DataFrame:
    """Return one exact-comparable best row per simulated event."""

    event_scores = ensure_evidence_support_columns(event_scores)
    comparable = _coerce_bool_series(event_scores["evidence_comparable"])
    ok = event_scores[(event_scores["status"] == "success") & comparable]
    if ok.empty:
        return pd.DataFrame()
    if "is_best_model" in ok:
        best = ok[_coerce_bool_series(ok["is_best_model"])]
        if not best.empty:
            return best.reset_index(drop=True)
    best = ok.sort_values(["session", "event_index", "log_evidence"], ascending=[True, True, False])
    return best.drop_duplicates(["session", "event_index"], keep="first").reset_index(drop=True)


def patch_simulation_recovery_module(module: object) -> None:
    """Patch simulation recovery reporting to separate exact and truncated evidence."""

    setattr(module, "_MOMENTUM_EXACT_SURROGATE_MODELS", _MOMENTUM_EXACT_SURROGATE_MODELS)
    setattr(module, "_ensure_evidence_support_columns", ensure_evidence_support_columns)
    setattr(module, "add_evidence_columns", simulation_add_evidence_columns)
    setattr(module, "_event_best_rows", simulation_event_best_rows)
