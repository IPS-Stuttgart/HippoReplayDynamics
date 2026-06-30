"""Keep posterior-calibration summary denominators aligned."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

_PATCHED_FLAG = "_posterior_calibration_summary_patch_applied"
_GATES_SCOPE_PATCHED_FLAG = "_result_quality_gates_scope_patch_applied"
_PAIRED_GROUP_PATCHED_FLAG = "_paired_model_missing_group_patch_applied"
_PAIRED_SWEEP_GROUP_PATCHED_FLAG = "_paired_model_sweep_missing_group_patch_applied"
_ORIGINAL_PAIRED_ATTR = "_paired_model_missing_group_original"
_ORIGINAL_SWEEP_ATTR = "_paired_model_sweep_missing_group_original"
_MISSING_GROUP_SENTINEL = "__hipporeplayimm_missing_group__"
_ADDITIONAL_RESULT_QUALITY_GATE_SCOPE_COLUMNS = (
    "window_role",
    "window_index",
    "event_window_variant",
    "window_variant",
    "window_start_s",
    "window_end_s",
    "window_duration_s",
    "null_index",
    "matched_null_rank",
    "template_event_index",
    "benchmark_random_seed",
    "benchmark_cell_split_index",
    "benchmark_cell_split_seed",
    "benchmark_event_subset_seed",
    "benchmark_event_subset_base_seed",
    "benchmark_test_cell_fraction",
    "benchmark_cell_split_strategy",
    "benchmark_cell_split_strata",
)


def _apply_result_quality_gates_scope_patch() -> None:
    """Keep result-quality gate event grouping aligned with audit window scope."""

    from . import result_quality_gates

    existing = tuple(getattr(result_quality_gates, "_EVENT_GROUP_SCOPE_COLUMNS", ()))
    updated = tuple(
        dict.fromkeys(
            (
                *existing,
                *_ADDITIONAL_RESULT_QUALITY_GATE_SCOPE_COLUMNS,
            )
        )
    )
    if getattr(result_quality_gates, _GATES_SCOPE_PATCHED_FLAG, False) and existing == updated:
        return
    result_quality_gates._EVENT_GROUP_SCOPE_COLUMNS = updated
    setattr(result_quality_gates, _GATES_SCOPE_PATCHED_FLAG, True)


def _apply_paired_model_missing_group_patch() -> None:
    """Keep paired model-margin diagnostics for rows with missing optional group keys."""

    from . import advanced_result_diagnostics as diagnostics

    current = diagnostics.paired_model_margin_decisions
    if not getattr(current, _PAIRED_GROUP_PATCHED_FLAG, False):
        original = getattr(current, _ORIGINAL_PAIRED_ATTR, current)

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
            groups = _normalize_group_cols(group_cols)
            result = original(
                _fill_missing_group_metadata(scores, groups),
                positive_model=positive_model,
                reference_model=reference_model,
                margin_threshold=margin_threshold,
                group_cols=groups,
                evidence_col=evidence_col,
                model_col=model_col,
                true_model_col=true_model_col,
                positive_true_label=positive_true_label,
            )
            return _restore_missing_group_metadata(result, groups)

        setattr(paired_model_margin_decisions, _PAIRED_GROUP_PATCHED_FLAG, True)
        setattr(paired_model_margin_decisions, _ORIGINAL_PAIRED_ATTR, original)
        diagnostics.paired_model_margin_decisions = paired_model_margin_decisions

    current_sweep = diagnostics.paired_model_margin_threshold_sweep
    if getattr(current_sweep, _PAIRED_SWEEP_GROUP_PATCHED_FLAG, False):
        return
    original_sweep = getattr(current_sweep, _ORIGINAL_SWEEP_ATTR, current_sweep)

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
        groups = _normalize_optional_group_cols(group_cols)
        sweep_scores = scores if groups is None else _fill_missing_group_metadata(scores, groups)
        result = original_sweep(
            sweep_scores,
            positive_model=positive_model,
            reference_model=reference_model,
            thresholds=thresholds,
            group_cols=groups,
            evidence_col=evidence_col,
            model_col=model_col,
            true_model_col=true_model_col,
            positive_true_label=positive_true_label,
        )
        return _restore_missing_group_metadata(result, groups or ())

    setattr(paired_model_margin_threshold_sweep, _PAIRED_SWEEP_GROUP_PATCHED_FLAG, True)
    setattr(paired_model_margin_threshold_sweep, _ORIGINAL_SWEEP_ATTR, original_sweep)
    diagnostics.paired_model_margin_threshold_sweep = paired_model_margin_threshold_sweep


def _normalize_group_cols(group_cols: Sequence[str] | str) -> tuple[str, ...]:
    if isinstance(group_cols, str):
        return (group_cols,)
    return tuple(group_cols)


def _normalize_optional_group_cols(group_cols: Sequence[str] | str | None) -> tuple[str, ...] | None:
    if group_cols is None:
        return None
    return _normalize_group_cols(group_cols)


def _fill_missing_group_metadata(frame: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    if frame.empty or not group_cols:
        return frame.copy()
    out = frame.copy()
    for column in group_cols:
        if column not in out.columns:
            continue
        missing = out[column].isna()
        if missing.any():
            out[column] = out[column].astype(object)
            out.loc[missing, column] = _MISSING_GROUP_SENTINEL
    return out


def _restore_missing_group_metadata(frame: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    if frame.empty or not group_cols:
        return frame
    out = frame.copy()
    for column in group_cols:
        if column not in out.columns:
            continue
        missing = out[column].astype(object).eq(_MISSING_GROUP_SENTINEL)
        if missing.any():
            out.loc[missing, column] = pd.NA
    return out


def _to_float_numpy(values: pd.Series) -> np.ndarray:
    """Convert pandas numeric data to float, preserving nullable missing values."""

    return values.to_numpy(dtype=float, na_value=np.nan)


def _rank_fraction(rank: pd.Series, n_bins: pd.Series) -> pd.Series:
    """Return finite rank fractions only for possible 1-based rank/bin pairs."""

    rank_values = _to_float_numpy(pd.to_numeric(rank, errors="coerce"))
    n_bin_values = _to_float_numpy(pd.to_numeric(n_bins, errors="coerce"))
    integer_like = np.isclose(rank_values, np.rint(rank_values), rtol=0.0, atol=0.0) & np.isclose(
        n_bin_values,
        np.rint(n_bin_values),
        rtol=0.0,
        atol=0.0,
    )
    valid = (
        np.isfinite(rank_values)
        & np.isfinite(n_bin_values)
        & integer_like
        & (rank_values >= 1.0)
        & (n_bin_values > 0.0)
        & (rank_values <= n_bin_values)
    )
    fractions = np.full(rank_values.shape, np.nan, dtype=float)
    fractions[valid] = rank_values[valid] / n_bin_values[valid]
    return pd.Series(fractions, index=rank.index)


def _rank_coverage(rank_fraction: pd.Series, threshold: float) -> pd.Series:
    """Return nullable rank-coverage indicators for finite rank fractions only."""

    values = pd.to_numeric(rank_fraction, errors="coerce")
    numeric = _to_float_numpy(values)
    valid = np.isfinite(numeric)
    coverage = pd.Series(pd.NA, index=values.index, dtype="boolean")
    if np.any(valid):
        coverage.loc[valid] = numeric[valid] <= float(threshold)
    return coverage


def apply_posterior_calibration_summary_patch() -> None:
    """Patch calibration summaries to drop invalid probability rows consistently."""

    from . import result_improvements
    from . import result_quality_audit_scope_patch

    result_quality_audit_scope_patch.apply_result_quality_audit_scope_patch()
    _apply_result_quality_gates_scope_patch()
    _apply_paired_model_missing_group_patch()

    if getattr(result_improvements, _PATCHED_FLAG, False):
        return

    def posterior_calibration_summary(
        samples: pd.DataFrame,
        *,
        probability_column: str = "true_bin_probability",
        rank_column: str = "true_bin_rank",
        n_bins_column: str = "n_position_bins",
    ) -> pd.DataFrame:
        if samples.empty or probability_column not in samples:
            return pd.DataFrame()
        group_columns = [column for column in ("session", "model") if column in samples]
        if not group_columns:
            group_columns = ["_all"]
            samples = samples.copy()
            samples["_all"] = "all"

        frame = samples.copy()
        raw_probabilities = pd.to_numeric(frame[probability_column], errors="coerce")
        raw_probability_values = _to_float_numpy(raw_probabilities)
        valid_probability = pd.Series(
            np.isfinite(raw_probability_values)
            & (raw_probability_values >= 0.0)
            & (raw_probability_values <= 1.0),
            index=frame.index,
        )
        frame = frame.loc[valid_probability].copy()
        if frame.empty:
            return pd.DataFrame()

        probabilities = pd.to_numeric(frame[probability_column], errors="coerce").clip(
            lower=np.finfo(float).tiny,
            upper=1.0,
        )
        frame[probability_column] = probabilities
        frame["true_negative_log_probability"] = -np.log(probabilities)
        if rank_column in frame and n_bins_column in frame:
            frame["rank_fraction"] = _rank_fraction(frame[rank_column], frame[n_bins_column])
            frame["coverage_50_rank"] = _rank_coverage(frame["rank_fraction"], 0.50)
            frame["coverage_80_rank"] = _rank_coverage(frame["rank_fraction"], 0.80)
            frame["coverage_95_rank"] = _rank_coverage(frame["rank_fraction"], 0.95)
        else:
            frame["rank_fraction"] = np.nan
            frame["coverage_50_rank"] = np.nan
            frame["coverage_80_rank"] = np.nan
            frame["coverage_95_rank"] = np.nan
        return (
            frame.groupby(group_columns, as_index=False)
            .agg(
                rows=(probability_column, "count"),
                mean_true_probability=(probability_column, "mean"),
                median_true_probability=(probability_column, "median"),
                mean_true_negative_log_probability=("true_negative_log_probability", "mean"),
                median_rank_fraction=("rank_fraction", "median"),
                coverage_50_rank=("coverage_50_rank", "mean"),
                coverage_80_rank=("coverage_80_rank", "mean"),
                coverage_95_rank=("coverage_95_rank", "mean"),
            )
            .reset_index(drop=True)
        )

    result_improvements.posterior_calibration_summary = posterior_calibration_summary
    setattr(result_improvements, _PATCHED_FLAG, True)


__all__ = ["apply_posterior_calibration_summary_patch"]
