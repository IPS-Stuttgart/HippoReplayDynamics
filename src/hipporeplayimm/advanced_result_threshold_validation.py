"""Validate advanced result thresholds and event-window inputs."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

_PATCHED_FLAG = "_advanced_result_threshold_validation_patch_applied"
_BASE_DECISIONS_ATTR = "_advanced_result_threshold_validation_base_decisions"
_BASE_DECISIONS_DUPLICATE_WRAPPER_FLAG = "_advanced_result_threshold_validation_base_duplicate_wrapper"
_PATCHED_DECISIONS_ATTR = "_advanced_result_threshold_validation_patched_decisions"
_PATCHED_SWEEP_ATTR = "_advanced_result_threshold_validation_patched_sweep"
_EVENT_WINDOW_WRAPPER_FLAG = "_advanced_result_event_window_validation_wrapper"
_PATCHED_EVENT_WINDOW_ATTR = "_advanced_result_event_window_validation_patched_event_window_variants"
_TEXT_SCALAR_TYPES = (str, bytes, np.str_, np.bytes_)


def _is_text_scalar(value: object) -> bool:
    """Return True for text-backed scalar values that float(...) would coerce."""

    if isinstance(value, _TEXT_SCALAR_TYPES):
        return True
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return False
    if array.ndim != 0:
        return False
    if np.issubdtype(array.dtype, np.str_) or np.issubdtype(array.dtype, np.bytes_):
        return True
    if array.dtype == object:
        try:
            return isinstance(array.item(), _TEXT_SCALAR_TYPES)
        except ValueError:
            return False
    return False


def _validated_threshold(
    threshold: float,
    *,
    message: str = "margin_threshold must be a finite nonnegative value",
) -> float:
    if isinstance(threshold, (bool, np.bool_)):
        raise ValueError(message)
    try:
        scalar = np.asarray(threshold)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    if scalar.ndim != 0:
        raise ValueError(message)
    if np.issubdtype(scalar.dtype, np.bool_) or _is_text_scalar(threshold):
        raise ValueError(message)
    try:
        value = float(scalar.item())
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(message)
    return value


def _validated_thresholds(thresholds: Sequence[float]) -> tuple[float, ...]:
    return tuple(
        _validated_threshold(
            threshold,
            message="thresholds must contain finite nonnegative values",
        )
        for threshold in thresholds
    )


def _validated_finite_scalar(value: object, *, message: str) -> float:
    """Return a finite scalar float while rejecting booleans, text, and arrays."""

    if isinstance(value, (bool, np.bool_)):
        raise ValueError(message)
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    if array.ndim != 0:
        raise ValueError(message)
    if np.issubdtype(array.dtype, np.bool_) or _is_text_scalar(value):
        raise ValueError(message)
    try:
        numeric = float(array.item())
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if not np.isfinite(numeric):
        raise ValueError(message)
    return numeric


def _validated_positive_scalar(value: object, *, message: str) -> float:
    numeric = _validated_finite_scalar(value, message=message)
    if numeric <= 0.0:
        raise ValueError(message)
    return numeric


def _validated_nonnegative_sequence(values: Sequence[float], *, message: str) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise ValueError(message)
    out: list[float] = []
    for value in values:
        numeric = _validated_finite_scalar(value, message=message)
        if numeric < 0.0:
            raise ValueError(message)
        out.append(numeric)
    return tuple(out)


def _validate_event_window_inputs(
    events: pd.DataFrame,
    *,
    start_col: str,
    end_col: str,
    event_id_col: str,
    paddings_s: Sequence[float],
    min_duration_s: float,
) -> tuple[tuple[float, ...], float]:
    """Validate event-window generation inputs before windows are materialized."""

    missing = [column for column in (start_col, end_col, event_id_col) if column not in events.columns]
    if missing:
        raise KeyError(f"events is missing required columns: {missing}")

    validated_paddings = _validated_nonnegative_sequence(
        paddings_s,
        message="paddings_s must contain finite nonnegative scalar values",
    )
    validated_min_duration = _validated_positive_scalar(
        min_duration_s,
        message="min_duration_s must be a finite positive scalar",
    )

    for row_label, event in events.iterrows():
        start = _validated_finite_scalar(
            event[start_col],
            message=f"{start_col} must contain finite scalar event times",
        )
        end = _validated_finite_scalar(
            event[end_col],
            message=f"{end_col} must contain finite scalar event times",
        )
        if end <= start:
            raise ValueError(f"{end_col} must be greater than {start_col} for event-window row {row_label!r}")
    return validated_paddings, validated_min_duration


def _normalize_group_cols(group_cols: Sequence[str] | str | None, scores: pd.DataFrame) -> tuple[str, ...]:
    """Return grouping columns without expanding a single column name into characters."""

    if group_cols is None:
        from . import advanced_result_diagnostics as diagnostics

        return tuple(diagnostics.infer_paired_model_group_cols(scores))
    if isinstance(group_cols, str):
        return (group_cols,)
    return tuple(group_cols)


def _coerce_nonfinite_evidence_to_missing(scores: pd.DataFrame, evidence_col: str) -> pd.DataFrame:
    """Ensure paired comparisons ignore NaN/+inf/-inf evidence rows uniformly."""

    if scores.empty or evidence_col not in scores.columns:
        return scores
    out = scores.copy()
    numeric = pd.to_numeric(out[evidence_col], errors="coerce")
    out[evidence_col] = numeric.where(np.isfinite(numeric), np.nan)
    return out


def _sort_scores_for_duplicate_model_evidence(
    scores: pd.DataFrame,
    group_cols: Sequence[str],
    evidence_col: str,
    model_col: str,
) -> pd.DataFrame:
    """Order duplicate model rows so keep-last reducers use the best finite evidence."""

    if scores.empty or evidence_col not in scores.columns or model_col not in scores.columns:
        return scores

    out = _coerce_nonfinite_evidence_to_missing(scores, evidence_col)
    evidence_key = "__paired_model_numeric_evidence"
    while evidence_key in out.columns:
        evidence_key = f"_{evidence_key}"
    out[evidence_key] = pd.to_numeric(out[evidence_col], errors="coerce")

    sort_columns = [model_col, evidence_key]
    if group_cols:
        try:
            grouped = out.groupby(list(group_cols), sort=False, dropna=False)
        except (TypeError, ValueError):
            out = out.sort_values(sort_columns, ascending=[True, True], kind="stable", na_position="first")
        else:
            pieces = [
                group.sort_values(sort_columns, ascending=[True, True], kind="stable", na_position="first")
                for _, group in grouped
            ]
            out = pd.concat(pieces, axis=0) if pieces else out
    else:
        out = out.sort_values(sort_columns, ascending=[True, True], kind="stable", na_position="first")
    return out.drop(columns=[evidence_key])


def _wrap_base_decisions_for_duplicate_model_evidence(base_decisions):
    """Ensure stored paired-decision bases also prefer best duplicate evidence."""

    if getattr(base_decisions, _BASE_DECISIONS_DUPLICATE_WRAPPER_FLAG, False):
        return base_decisions

    def paired_model_margin_decisions(
        scores: pd.DataFrame,
        *,
        positive_model: str,
        reference_model: str,
        margin_threshold: float = 0.0,
        group_cols: Sequence[str] | str | None = ("session", "event_index"),
        evidence_col: str = "log_evidence",
        model_col: str = "model",
        true_model_col: str | None = None,
        positive_true_label: str | None = None,
    ) -> pd.DataFrame:
        paired_group_cols = _normalize_group_cols(group_cols, scores)
        prepared_scores = _sort_scores_for_duplicate_model_evidence(
            scores,
            paired_group_cols,
            evidence_col,
            model_col,
        )
        return base_decisions(
            prepared_scores,
            positive_model=positive_model,
            reference_model=reference_model,
            margin_threshold=margin_threshold,
            group_cols=paired_group_cols,
            evidence_col=evidence_col,
            model_col=model_col,
            true_model_col=true_model_col,
            positive_true_label=positive_true_label,
        )

    paired_model_margin_decisions.__name__ = getattr(base_decisions, "__name__", "paired_model_margin_decisions")
    paired_model_margin_decisions.__doc__ = getattr(base_decisions, "__doc__", None)
    setattr(paired_model_margin_decisions, _BASE_DECISIONS_DUPLICATE_WRAPPER_FLAG, True)
    setattr(paired_model_margin_decisions, "__hipporeplayimm_original__", base_decisions)
    return paired_model_margin_decisions


def _ensure_true_model_summary_columns(summary: pd.DataFrame) -> pd.DataFrame:
    """Keep threshold-selection columns present even when no paired events exist."""

    defaults: dict[str, object] = {
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
    out = summary.copy()
    for column, value in defaults.items():
        if column not in out.columns:
            out[column] = value
    return out


def _threshold_patch_current(diagnostics) -> bool:
    return (
        getattr(diagnostics, _PATCHED_FLAG, False)
        and getattr(diagnostics, "paired_model_margin_decisions", None) is getattr(diagnostics, _PATCHED_DECISIONS_ATTR, None)
        and getattr(diagnostics, "paired_model_margin_threshold_sweep", None) is getattr(diagnostics, _PATCHED_SWEEP_ATTR, None)
    )


def _event_window_patch_current(diagnostics) -> bool:
    return getattr(getattr(diagnostics, "event_window_variants", None), _EVENT_WINDOW_WRAPPER_FLAG, False)


def apply_advanced_result_threshold_validation_patch() -> None:
    """Install paired-threshold validation and event-window input validation."""

    from . import advanced_result_diagnostics as diagnostics

    if not _threshold_patch_current(diagnostics):
        base_decisions = getattr(diagnostics, _BASE_DECISIONS_ATTR, None)
        if base_decisions is None:
            base_decisions = diagnostics.paired_model_margin_decisions
        base_decisions = _wrap_base_decisions_for_duplicate_model_evidence(base_decisions)
        setattr(diagnostics, _BASE_DECISIONS_ATTR, base_decisions)

        def paired_model_margin_decisions(
            scores: pd.DataFrame,
            *,
            positive_model: str,
            reference_model: str,
            margin_threshold: float = 0.0,
            group_cols: Sequence[str] | str = ("session", "event_index"),
            evidence_col: str = "log_evidence",
            model_col: str = "model",
            true_model_col: str | None = None,
            positive_true_label: str | None = None,
        ) -> pd.DataFrame:
            threshold = _validated_threshold(margin_threshold)
            paired_group_cols = _normalize_group_cols(group_cols, scores)
            prepared_scores = _sort_scores_for_duplicate_model_evidence(
                scores,
                paired_group_cols,
                evidence_col,
                model_col,
            )
            out = base_decisions(
                prepared_scores,
                positive_model=positive_model,
                reference_model=reference_model,
                margin_threshold=threshold,
                group_cols=paired_group_cols,
                evidence_col=evidence_col,
                model_col=model_col,
                true_model_col=true_model_col,
                positive_true_label=positive_true_label,
            )
            if out.empty:
                return out
            zero_tie = np.isclose(
                pd.to_numeric(out["margin_threshold"], errors="coerce"),
                0.0,
            ) & np.isclose(
                pd.to_numeric(out["positive_minus_reference_log_evidence"], errors="coerce"),
                0.0,
            )
            if not bool(zero_tie.any()):
                return out
            out = out.copy()
            out.loc[zero_tie, "margin_decision"] = "ambiguous"
            out.loc[zero_tie, "positive_model_claimed"] = False
            if "margin_binary_correct" in out and "true_is_positive" in out:
                true_is_positive = diagnostics._bool_column(out, "true_is_positive")
                out.loc[zero_tie, "margin_binary_correct"] = (~true_is_positive.loc[zero_tie]).astype(bool)
            return out

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
            validated_thresholds = _validated_thresholds(thresholds)
            paired_group_cols = _normalize_group_cols(group_cols, scores)
            rows: list[pd.DataFrame] = []
            for threshold in validated_thresholds:
                decisions = paired_model_margin_decisions(
                    scores,
                    positive_model=positive_model,
                    reference_model=reference_model,
                    margin_threshold=threshold,
                    group_cols=paired_group_cols,
                    evidence_col=evidence_col,
                    model_col=model_col,
                    true_model_col=true_model_col,
                    positive_true_label=positive_true_label,
                )
                summary = diagnostics.paired_model_margin_summary(decisions, true_model_col=true_model_col).copy()
                if true_model_col:
                    summary = _ensure_true_model_summary_columns(summary)
                summary["positive_model"] = str(positive_model)
                summary["reference_model"] = str(reference_model)
                summary["margin_threshold"] = float(threshold)
                summary["group_cols"] = ",".join(paired_group_cols)
                rows.append(summary)
            if not rows:
                return pd.DataFrame()
            out = pd.concat(rows, ignore_index=True)
            return out.sort_values("margin_threshold", kind="stable").reset_index(drop=True)

        diagnostics.paired_model_margin_decisions = paired_model_margin_decisions
        diagnostics.paired_model_margin_threshold_sweep = paired_model_margin_threshold_sweep
        setattr(diagnostics, _PATCHED_DECISIONS_ATTR, paired_model_margin_decisions)
        setattr(diagnostics, _PATCHED_SWEEP_ATTR, paired_model_margin_threshold_sweep)
        setattr(diagnostics, _PATCHED_FLAG, True)

    if not _event_window_patch_current(diagnostics):
        base_event_window_variants = diagnostics.event_window_variants

        def event_window_variants(
            events: pd.DataFrame,
            *,
            start_col: str = "start",
            end_col: str = "end",
            event_id_col: str = "event_index",
            paddings_s: Sequence[float] = (0.0, 0.01, 0.02),
            min_duration_s: float = 0.003,
        ) -> pd.DataFrame:
            validated_paddings, validated_min_duration = _validate_event_window_inputs(
                events,
                start_col=start_col,
                end_col=end_col,
                event_id_col=event_id_col,
                paddings_s=paddings_s,
                min_duration_s=min_duration_s,
            )
            return base_event_window_variants(
                events,
                start_col=start_col,
                end_col=end_col,
                event_id_col=event_id_col,
                paddings_s=validated_paddings,
                min_duration_s=validated_min_duration,
            )

        setattr(event_window_variants, _EVENT_WINDOW_WRAPPER_FLAG, True)
        diagnostics.event_window_variants = event_window_variants
        setattr(diagnostics, _PATCHED_EVENT_WINDOW_ATTR, event_window_variants)


__all__ = ["apply_advanced_result_threshold_validation_patch"]
