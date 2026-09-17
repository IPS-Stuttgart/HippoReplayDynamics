"""Preserve valid negative-infinite evidence in simulation recovery reports.

A log evidence of ``-inf`` represents zero path probability.  It is still a
valid model-evidence value for recovery ranking: any finite evidence beats it,
and a finite truncated lower bound is certified against an exact ``-inf``
reference. NaN and ``+inf`` remain invalid for these recovery comparisons.

Generic evidence-support reporting intentionally keeps its stricter finiteness
contract. The exception in this module is scoped to simulation recovery only.
"""

from __future__ import annotations

from functools import wraps
from typing import Any

import numpy as np
import pandas as pd

_ANNOTATION_WRAPPER_FLAG = "_negative_infinite_recovery_annotation_wrapper"
_ROW_ID_BASE = "__hipporeplayimm_negative_infinite_row_id__"


def _temporary_row_id_column(frame: pd.DataFrame) -> str:
    column = _ROW_ID_BASE
    while column in frame.columns:
        column += "_"
    return column


def _real_log_evidence_values(values: pd.Series) -> np.ndarray:
    """Coerce real scalar evidence without warning on nested complex objects."""

    from . import evidence_complex_validation as complex_validation

    return complex_validation._real_numeric_series(values).to_numpy(float)


def _usable_log_evidence_mask(values: object) -> np.ndarray:
    numeric = np.asarray(values, dtype=float)
    return ~(np.isnan(numeric) | np.isposinf(numeric))


def _log_evidence_margin(left: float, right: float) -> float:
    if np.isnan(left) or np.isnan(right):
        return float("nan")
    if np.isneginf(left) and np.isneginf(right):
        return float("nan")
    return float(left - right)


def _patch_evidence_annotation(best_row_flags: Any, reporting: Any) -> None:
    """Restore recovery comparability for successful exact ``-inf`` rows."""

    current = best_row_flags._simulation_add_evidence_columns
    if getattr(current, _ANNOTATION_WRAPPER_FLAG, False):
        return

    @wraps(current)
    def simulation_add_evidence_columns_preserving_negative_infinity(
        event_scores: pd.DataFrame,
        reporting_module: Any,
    ) -> pd.DataFrame:
        if event_scores.empty or "log_evidence" not in event_scores.columns:
            return current(event_scores, reporting_module)

        prepared = event_scores.copy()
        row_id_column = _temporary_row_id_column(prepared)
        prepared[row_id_column] = np.arange(len(prepared), dtype=np.int64)
        classified = reporting_module.ensure_evidence_support_columns(prepared)
        source_values = _real_log_evidence_values(classified["log_evidence"])
        source_negative_infinite = np.isneginf(source_values)
        source_success = reporting_module._status_success_series(classified).to_numpy(
            bool
        )
        source_exact = (
            classified["evidence_support"]
            .astype(str)
            .eq(reporting_module.EXACT_EVIDENCE_SUPPORT)
            .to_numpy(bool)
        )
        recoverable_negative_infinite = (
            source_negative_infinite & source_success & source_exact
        )
        comparable_row_ids = classified.loc[
            recoverable_negative_infinite, row_id_column
        ].tolist()

        annotated = current(prepared, reporting_module)
        if row_id_column not in annotated.columns:
            return annotated

        if comparable_row_ids:
            negative_ids = annotated[row_id_column].isin(comparable_row_ids)
            annotated.loc[negative_ids, "evidence_comparable"] = True

            # When at least one exact model has finite evidence, ``-inf`` exact
            # rows have exactly zero posterior model probability and ``-inf``
            # relative evidence. The finite-model probabilities already sum to
            # one because zero-mass models contribute nothing to the normalizer.
            for _, group in best_row_flags._iter_event_groups(annotated):
                log_values = _real_log_evidence_values(group["log_evidence"])
                comparable = reporting_module._coerce_bool_series(
                    group["evidence_comparable"]
                ).to_numpy(bool)
                negative_exact = comparable & np.isneginf(log_values)
                finite_exact = comparable & np.isfinite(log_values)
                if np.any(negative_exact) and np.any(finite_exact):
                    indices = group.index[negative_exact]
                    annotated.loc[indices, "relative_log_evidence"] = -np.inf
                    annotated.loc[indices, "model_probability"] = 0.0

                if "evidence_support" in group.columns:
                    truncated = group["evidence_support"].eq(
                        reporting_module.TRUNCATED_EVIDENCE_SUPPORT
                    ).to_numpy(bool)
                    negative_truncated = truncated & np.isneginf(log_values)
                    finite_truncated = truncated & np.isfinite(log_values)
                    if np.any(negative_truncated) and np.any(finite_truncated):
                        annotated.loc[
                            group.index[negative_truncated],
                            "truncated_relative_log_evidence",
                        ] = -np.inf

        return annotated.drop(columns=[row_id_column])

    setattr(
        simulation_add_evidence_columns_preserving_negative_infinity,
        _ANNOTATION_WRAPPER_FLAG,
        True,
    )
    best_row_flags._simulation_add_evidence_columns = (
        simulation_add_evidence_columns_preserving_negative_infinity
    )


def _certified_vs_exact_event_recovery(
    event_scores: pd.DataFrame,
    reporting: Any,
    recovery: Any,
    best_row_flags: Any,
) -> pd.DataFrame:
    """Certified recovery that treats exact ``-inf`` as valid zero evidence."""

    if event_scores.empty:
        return pd.DataFrame()

    event_scores = reporting._coerce_log_evidence_column(
        reporting.ensure_evidence_support_columns(event_scores)
    )
    rows: list[dict[str, object]] = []
    for _, group in best_row_flags._iter_event_groups(event_scores):
        group = group.copy()
        first = group.iloc[0]
        expected_model = str(first.get("expected_model", ""))
        base: dict[str, object] = {
            "session": best_row_flags._event_scalar(group, "session"),
            "event_index": best_row_flags._event_scalar(group, "event_index"),
            "true_model": str(first.get("true_model", "")),
            "expected_model": expected_model,
            "n_time": best_row_flags._event_scalar(group, "n_time"),
            "n_spikes": best_row_flags._event_scalar(group, "n_spikes"),
        }
        for column in best_row_flags._event_group_columns(group):
            if column not in base:
                base[column] = best_row_flags._event_scalar(group, column)

        scored = group[reporting._status_success_series(group)].copy()
        if scored.empty:
            rows.append(
                {
                    **base,
                    "certified_vs_exact_recovered_expected_model": False,
                    "certified_vs_exact_reason": "no_successful_scores",
                    "expected_model_log_evidence": np.nan,
                    "expected_model_evidence_support": "",
                    "expected_model_evidence_comparable": False,
                    "best_comparable_model": "",
                    "best_comparable_log_evidence": np.nan,
                    "expected_minus_best_comparable_log_evidence": np.nan,
                }
            )
            continue

        log_values = _real_log_evidence_values(scored["log_evidence"])
        usable_scores = _usable_log_evidence_mask(log_values)
        scored = scored.loc[usable_scores].copy()
        log_values = log_values[usable_scores]
        if scored.empty:
            rows.append(
                {
                    **base,
                    "certified_vs_exact_recovered_expected_model": False,
                    "certified_vs_exact_reason": "no_finite_scores",
                    "expected_model_log_evidence": np.nan,
                    "expected_model_evidence_support": "",
                    "expected_model_evidence_comparable": False,
                    "best_comparable_model": "",
                    "best_comparable_log_evidence": np.nan,
                    "expected_minus_best_comparable_log_evidence": np.nan,
                }
            )
            continue

        comparable_mask = np.asarray(recovery._comparable_mask(scored), dtype=bool)
        exact_negative_infinite = (
            scored["evidence_support"]
            .astype(str)
            .eq(reporting.EXACT_EVIDENCE_SUPPORT)
            .to_numpy(bool)
            & np.isneginf(log_values)
        )
        comparable_mask |= exact_negative_infinite
        comparable_rows = scored.loc[comparable_mask].copy()
        best_comparable_row: pd.Series | None = None
        best_comparable_model = ""
        best_comparable_log_evidence = np.nan
        if not comparable_rows.empty:
            best_comparable_row = best_row_flags._best_log_evidence_row(
                comparable_rows
            )
            best_comparable_model = str(best_comparable_row["model"])
            best_comparable_log_evidence = float(
                best_comparable_row["log_evidence"]
            )

        acceptable_models = recovery._event_acceptable_recovery_models(group)
        acceptable_rows = scored[
            scored["model"].astype(str).isin(acceptable_models)
        ].copy()
        expected_rows = scored[scored["model"].astype(str) == expected_model]
        if acceptable_rows.empty:
            rows.append(
                {
                    **base,
                    "certified_vs_exact_recovered_expected_model": False,
                    "certified_vs_exact_reason": "expected_model_not_scored",
                    "certified_reference_model": "",
                    "expected_model_log_evidence": np.nan,
                    "expected_model_evidence_support": "",
                    "expected_model_evidence_comparable": False,
                    "best_comparable_model": best_comparable_model,
                    "best_comparable_log_evidence": best_comparable_log_evidence,
                    "expected_minus_best_comparable_log_evidence": np.nan,
                }
            )
            continue

        if (
            best_comparable_row is not None
            and best_comparable_model in acceptable_models
        ):
            expected = best_comparable_row
        elif not expected_rows.empty:
            expected = best_row_flags._best_log_evidence_row(expected_rows)
        else:
            expected = best_row_flags._best_log_evidence_row(acceptable_rows)

        certified_reference_model = str(expected["model"])
        expected_log_evidence = float(expected["log_evidence"])
        expected_support = str(expected.get("evidence_support", ""))
        raw_expected_comparable = expected.get(
            "evidence_comparable",
            recovery._evidence_is_comparable(expected_support),
        )
        expected_comparable = (
            recovery._evidence_is_comparable(expected_support)
            if pd.isna(raw_expected_comparable)
            else bool(
                reporting._coerce_bool_series(
                    pd.Series([raw_expected_comparable])
                ).iloc[0]
            )
        )
        if (
            expected_support == reporting.EXACT_EVIDENCE_SUPPORT
            and np.isneginf(expected_log_evidence)
        ):
            expected_comparable = True
        margin = _log_evidence_margin(
            expected_log_evidence,
            best_comparable_log_evidence,
        )

        if (
            best_comparable_row is not None
            and best_comparable_model in acceptable_models
        ):
            recovered = True
            reason = (
                "expected_comparable_best"
                if best_comparable_model == expected_model
                else "exact_surrogate_comparable_best"
            )
        elif expected_comparable:
            recovered = False
            reason = (
                "expected_comparable_not_best"
                if certified_reference_model == expected_model
                else "exact_surrogate_comparable_not_best"
            )
        elif expected_support != reporting.TRUNCATED_EVIDENCE_SUPPORT:
            recovered = False
            reason = "expected_noncomparable_not_certified"
        elif np.isnan(best_comparable_log_evidence) or np.isposinf(
            best_comparable_log_evidence
        ):
            recovered = False
            reason = "no_comparable_exact_reference"
        else:
            recovered = bool(margin > 0.0)
            reason = (
                "expected_lower_bound_beats_best_comparable"
                if recovered
                else "expected_lower_bound_not_above_best_comparable"
            )

        rows.append(
            {
                **base,
                "certified_vs_exact_recovered_expected_model": recovered,
                "certified_vs_exact_reason": reason,
                "certified_reference_model": certified_reference_model,
                "expected_model_log_evidence": expected_log_evidence,
                "expected_model_evidence_support": expected_support,
                "expected_model_evidence_comparable": expected_comparable,
                "best_comparable_model": best_comparable_model,
                "best_comparable_log_evidence": best_comparable_log_evidence,
                "expected_minus_best_comparable_log_evidence": float(margin),
            }
        )
    return pd.DataFrame(rows)


def apply_simulation_recovery_negative_infinity_patch() -> None:
    """Install ``-inf``-aware recovery annotation and certification helpers."""

    from . import evidence_reporting as reporting
    from . import simulation_best_row_flags as best_row_flags
    from . import simulation_recovery_event_count_impl as event_count_impl

    _patch_evidence_annotation(best_row_flags, reporting)
    event_count_impl._certified_vs_exact_event_recovery = (
        _certified_vs_exact_event_recovery
    )


__all__ = ["apply_simulation_recovery_negative_infinity_patch"]
