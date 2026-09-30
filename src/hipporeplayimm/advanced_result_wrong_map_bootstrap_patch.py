"""Patch wrong-map post-hoc diagnostics for robust grouped summaries."""

from __future__ import annotations

import operator
from collections.abc import Sequence
from functools import wraps

import numpy as np
import pandas as pd

_PATCHED_FLAG = "_wrong_map_rat_bootstrap_patch_applied"
_RAT_BOOTSTRAP_WRAPPER_FLAG = "_wrong_map_rat_bootstrap_wrapper"
_NUMERIC_DELTA_WRAPPER_FLAG = "_wrong_map_numeric_delta_summary_wrapper"
_NUMERIC_ABSOLUTE_WRAPPER_FLAG = "_wrong_map_numeric_absolute_deltas_wrapper"
_NUMERIC_ORIGINAL_ATTR = "_wrong_map_numeric_evidence_original"
_TEXT_SCALAR_TYPES = (str, bytes, np.str_, np.bytes_)


def _bootstrap_summary_columns() -> list[str]:
    return [
        "bootstrap_unit",
        "bootstrap_replicates",
        "random_seed",
        "statistic",
        "observed_events",
        "observed_rats",
        "observed_positive_delta_fraction",
        "positive_delta_fraction_ci95_low",
        "positive_delta_fraction_ci95_high",
        "observed_mean_delta_map_log_evidence",
        "mean_delta_ci95_low",
        "mean_delta_ci95_high",
        "probability_mean_delta_gt_0",
        "observed_median_delta_map_log_evidence",
        "median_delta_ci95_low",
        "median_delta_ci95_high",
        "probability_median_delta_gt_0",
        "most_common_selected_model",
    ]


def _integer_scalar(value: object, name: str, *, minimum: int) -> int:
    """Return an exact integer scalar without bool, text, or array coercion."""

    qualifier = "positive" if minimum == 1 else "finite nonnegative"
    message = f"{name} must be a {qualifier} integer"
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    if array.ndim != 0:
        raise ValueError(message)
    scalar = array.item()
    if isinstance(scalar, (bool, np.bool_, *_TEXT_SCALAR_TYPES)):
        raise ValueError(message)

    try:
        integer = operator.index(scalar)
    except TypeError:
        try:
            integer = int(scalar)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(message) from exc
        try:
            exact = scalar == integer
            if not isinstance(exact, (bool, np.bool_)) or not bool(exact):
                raise ValueError(message)
        except (TypeError, ValueError) as exc:
            raise ValueError(message) from exc

    if integer < minimum:
        raise ValueError(message)
    return int(integer)


def _positive_integer(value: object, name: str) -> int:
    """Return a positive integer argument or raise a clear validation error."""

    return _integer_scalar(value, name, minimum=1)


def _nonnegative_integer(value: object, name: str) -> int:
    """Return a nonnegative integer argument without boolean or float truncation."""

    return _integer_scalar(value, name, minimum=0)


def apply_wrong_map_rat_bootstrap_patch() -> None:
    """Patch wrong-map diagnostics after CSV round-trips and rat resampling."""

    from . import advanced_result_diagnostics as diagnostics

    if wrong_map_rat_bootstrap_patch_current(diagnostics):
        return

    if not getattr(
        diagnostics.rat_bootstrap_wrong_map_absolute_evidence_summary,
        _RAT_BOOTSTRAP_WRAPPER_FLAG,
        False,
    ):
        _apply_rat_bootstrap_wrapper(diagnostics)
    _apply_numeric_evidence_wrappers(diagnostics)
    setattr(diagnostics, _PATCHED_FLAG, True)


def _apply_rat_bootstrap_wrapper(diagnostics) -> None:
    original = diagnostics.rat_bootstrap_wrong_map_absolute_evidence_summary

    @wraps(original)
    def rat_bootstrap_wrong_map_absolute_evidence_summary(
        deltas: pd.DataFrame,
        *,
        n_bootstrap: int = 2000,
        random_seed: int = 1,
    ) -> pd.DataFrame:
        """Rat-cluster bootstrap uncertainty for absolute map sensitivity."""

        replicates = _positive_integer(n_bootstrap, "n_bootstrap")
        seed = _nonnegative_integer(random_seed, "random_seed")
        columns = _bootstrap_summary_columns()
        if deltas.empty or "session" not in deltas.columns:
            return pd.DataFrame(columns=columns)
        frame = diagnostics._with_rat(_coerce_numeric_delta_evidence(deltas))
        if frame.empty:
            return pd.DataFrame(columns=columns)
        rng = np.random.default_rng(seed)
        rows: list[dict[str, object]] = []
        for statistic, group in frame.groupby("statistic", sort=False):
            statistic_rats = sorted(group["rat"].dropna().astype(str).unique())
            if not statistic_rats:
                continue
            observed = diagnostics._wrong_map_delta_summary(group).iloc[0]
            positive_fractions: list[float] = []
            means: list[float] = []
            medians: list[float] = []
            by_rat = {
                rat: group[group["rat"].astype(str) == rat]
                for rat in statistic_rats
            }
            for _ in range(replicates):
                sampled = rng.choice(statistic_rats, size=len(statistic_rats), replace=True)
                sample = pd.concat([by_rat[rat] for rat in sampled], ignore_index=True)
                values = sample["delta_map_log_evidence"].to_numpy(float)
                positive_fractions.append(float(np.mean(values > 0.0)))
                means.append(float(np.mean(values)))
                medians.append(float(np.median(values)))
            rows.append(
                {
                    "bootstrap_unit": "rat",
                    "bootstrap_replicates": replicates,
                    "random_seed": seed,
                    "statistic": str(statistic),
                    "observed_events": int(observed["events"]),
                    "observed_rats": int(len(statistic_rats)),
                    "observed_positive_delta_fraction": float(observed["positive_delta_fraction"]),
                    "positive_delta_fraction_ci95_low": _quantile_preserving_infinity(diagnostics, positive_fractions, 0.025),
                    "positive_delta_fraction_ci95_high": _quantile_preserving_infinity(diagnostics, positive_fractions, 0.975),
                    "observed_mean_delta_map_log_evidence": float(observed["mean_delta_map_log_evidence"]),
                    "mean_delta_ci95_low": _quantile_preserving_infinity(diagnostics, means, 0.025),
                    "mean_delta_ci95_high": _quantile_preserving_infinity(diagnostics, means, 0.975),
                    "probability_mean_delta_gt_0": float(np.mean(np.asarray(means) > 0.0)),
                    "observed_median_delta_map_log_evidence": float(observed["median_delta_map_log_evidence"]),
                    "median_delta_ci95_low": _quantile_preserving_infinity(diagnostics, medians, 0.025),
                    "median_delta_ci95_high": _quantile_preserving_infinity(diagnostics, medians, 0.975),
                    "probability_median_delta_gt_0": float(np.mean(np.asarray(medians) > 0.0)),
                    "most_common_selected_model": str(observed["most_common_selected_model"]),
                }
            )
        return pd.DataFrame(rows, columns=columns)

    setattr(
        rat_bootstrap_wrong_map_absolute_evidence_summary,
        _RAT_BOOTSTRAP_WRAPPER_FLAG,
        True,
    )
    diagnostics.rat_bootstrap_wrong_map_absolute_evidence_summary = (
        rat_bootstrap_wrong_map_absolute_evidence_summary
    )


def _coerce_numeric_delta_evidence(frame: pd.DataFrame) -> pd.DataFrame:
    """Return rows with numeric wrong-map evidence deltas, including infinities."""

    out = frame.copy()
    if out.empty or "delta_map_log_evidence" not in out.columns:
        return out
    out["delta_map_log_evidence"] = pd.to_numeric(
        out["delta_map_log_evidence"],
        errors="coerce",
    )
    values = out["delta_map_log_evidence"].to_numpy(dtype=float)
    return out.loc[~np.isnan(values)].copy()


def _quantile_preserving_infinity(
    diagnostics,
    values: Sequence[float],
    q: float,
) -> float:
    """Use ordinary quantiles unless extended-real values require no interpolation."""

    if not values:
        return float("nan")
    array = np.asarray(values, dtype=float)
    if np.any(np.isinf(array)):
        return float(np.quantile(array, q, method="inverted_cdf"))
    return diagnostics._quantile(values, q)


def _apply_numeric_evidence_wrappers(diagnostics) -> None:
    if not getattr(diagnostics.wrong_map_delta_summary, _NUMERIC_DELTA_WRAPPER_FLAG, False):
        original_delta = _unwrap_numeric(diagnostics.wrong_map_delta_summary)

        @wraps(original_delta)
        def wrong_map_delta_summary(
            current_map_scores: pd.DataFrame,
            wrong_map_scores: pd.DataFrame,
            *,
            key_cols: Sequence[str] = ("session", "event_index", "model"),
            evidence_col: str = "log_evidence",
        ) -> pd.DataFrame:
            key_columns = tuple(key_cols)
            result = original_delta(
                _best_duplicate_key_evidence(
                    _coerce_numeric_evidence(current_map_scores, evidence_col),
                    key_columns,
                    evidence_col,
                ),
                _best_duplicate_key_evidence(
                    _coerce_numeric_evidence(wrong_map_scores, evidence_col),
                    key_columns,
                    evidence_col,
                ),
                key_cols=key_cols,
                evidence_col=evidence_col,
            )
            return _normalize_wrong_map_delta_output(
                result,
                key_cols=key_columns,
                evidence_col=evidence_col,
            )

        _mark_numeric(wrong_map_delta_summary, original_delta, _NUMERIC_DELTA_WRAPPER_FLAG)
        diagnostics.wrong_map_delta_summary = wrong_map_delta_summary

    if not getattr(diagnostics.wrong_map_absolute_evidence_deltas, _NUMERIC_ABSOLUTE_WRAPPER_FLAG, False):
        original_absolute = _unwrap_numeric(diagnostics.wrong_map_absolute_evidence_deltas)

        @wraps(original_absolute)
        def wrong_map_absolute_evidence_deltas(
            current_map_scores: pd.DataFrame,
            wrong_map_scores: pd.DataFrame,
            *,
            group_cols: Sequence[str] = ("session", "event_index"),
            fixed_models: Sequence[str] = diagnostics.DEFAULT_WRONG_MAP_FIXED_MODELS,
            exact_core_models: Sequence[str] = diagnostics.DEFAULT_WRONG_MAP_EXACT_CORE_MODELS,
            exact_trajectory_models: Sequence[str] = diagnostics.DEFAULT_WRONG_MAP_EXACT_TRAJECTORY_MODELS,
            evidence_col: str = "log_evidence",
            model_col: str = "model",
        ) -> pd.DataFrame:
            key_columns = tuple(group_cols) + (model_col,)
            result = original_absolute(
                _best_duplicate_key_evidence(
                    _coerce_numeric_evidence(current_map_scores, evidence_col),
                    key_columns,
                    evidence_col,
                ),
                _best_duplicate_key_evidence(
                    _coerce_numeric_evidence(wrong_map_scores, evidence_col),
                    key_columns,
                    evidence_col,
                ),
                group_cols=group_cols,
                fixed_models=fixed_models,
                exact_core_models=exact_core_models,
                exact_trajectory_models=exact_trajectory_models,
                evidence_col=evidence_col,
                model_col=model_col,
            )
            return _normalize_wrong_map_absolute_output(
                result,
                evidence_col=evidence_col,
            )

        _mark_numeric(
            wrong_map_absolute_evidence_deltas,
            original_absolute,
            _NUMERIC_ABSOLUTE_WRAPPER_FLAG,
        )
        diagnostics.wrong_map_absolute_evidence_deltas = wrong_map_absolute_evidence_deltas


def wrong_map_rat_bootstrap_patch_current(diagnostics) -> bool:
    return all(
        getattr(getattr(diagnostics, name, None), flag, False)
        for name, flag in (
            (
                "rat_bootstrap_wrong_map_absolute_evidence_summary",
                _RAT_BOOTSTRAP_WRAPPER_FLAG,
            ),
            ("wrong_map_delta_summary", _NUMERIC_DELTA_WRAPPER_FLAG),
            ("wrong_map_absolute_evidence_deltas", _NUMERIC_ABSOLUTE_WRAPPER_FLAG),
        )
    )


def _coerce_numeric_evidence(frame: pd.DataFrame, evidence_col: str) -> pd.DataFrame:
    """Return numeric evidence rows while preserving valid exact zero evidence."""

    out = frame.copy()
    if out.empty or evidence_col not in out.columns:
        return out
    out[evidence_col] = pd.to_numeric(out[evidence_col], errors="coerce")
    values = out[evidence_col].to_numpy(dtype=float)
    if np.any(np.isposinf(values)):
        raise ValueError("wrong-map log_evidence must be finite or -inf")
    return out.loc[~np.isnan(values)].copy()


def _normalize_wrong_map_delta_output(
    frame: pd.DataFrame,
    *,
    key_cols: Sequence[str],
    evidence_col: str,
) -> pd.DataFrame:
    """Restore exact-zero evidence semantics after pairwise wrong-map matching."""

    if frame.empty:
        return frame
    out = frame.copy()
    current_col = f"{evidence_col}_current_map"
    wrong_col = f"{evidence_col}_wrong_map"
    if current_col not in out.columns or wrong_col not in out.columns:
        return out

    current = pd.to_numeric(out[current_col], errors="coerce").to_numpy(dtype=float)
    wrong = pd.to_numeric(out[wrong_col], errors="coerce").to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        delta = current - wrong
    both_impossible = np.isneginf(current) & np.isneginf(wrong)
    delta[both_impossible] = 0.0
    out["delta_vs_wrong_environment_map"] = delta

    if "wrong_map_best_model" in out.columns:
        event_columns = [
            column for column in key_cols if column != "model" and column in out.columns
        ]
        finite_wrong = np.isfinite(wrong)
        if event_columns:
            marker = "__wrong_map_has_finite_evidence"
            grouped = out[event_columns].copy()
            grouped[marker] = finite_wrong
            has_finite = (
                grouped.groupby(
                    event_columns,
                    sort=False,
                    dropna=False,
                )[marker]
                .transform("any")
                .to_numpy(dtype=bool)
            )
        else:
            has_finite = np.full(len(out), bool(np.any(finite_wrong)), dtype=bool)
        out.loc[~has_finite, "wrong_map_best_model"] = ""
    return out


def _normalize_wrong_map_absolute_output(
    frame: pd.DataFrame,
    *,
    evidence_col: str,
) -> pd.DataFrame:
    """Preserve infinite attenuation and suppress all-impossible pseudo-winners."""

    if frame.empty:
        return frame
    out = frame.copy()
    real_col = f"{evidence_col}_real_map"
    wrong_col = f"{evidence_col}_wrong_map"
    if real_col not in out.columns or wrong_col not in out.columns:
        return out

    real = pd.to_numeric(out[real_col], errors="coerce").to_numpy(dtype=float)
    wrong = pd.to_numeric(out[wrong_col], errors="coerce").to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        delta = real - wrong
    both_impossible = np.isneginf(real) & np.isneginf(wrong)
    delta[both_impossible] = 0.0
    out["delta_map_log_evidence"] = delta

    if "statistic_type" in out.columns:
        selected = out["statistic_type"].astype(str).eq("real_map_selected_model")
        fabricated = selected.to_numpy(dtype=bool) & np.isneginf(real)
        if np.any(fabricated):
            out = out.loc[~fabricated].copy()
    return out


def _best_duplicate_key_evidence(
    frame: pd.DataFrame,
    key_cols: Sequence[str],
    evidence_col: str,
) -> pd.DataFrame:
    """Keep the highest finite evidence row for each wrong-map comparison key."""

    if frame.empty or evidence_col not in frame.columns:
        return frame
    key_columns = tuple(key_cols)
    if any(column not in frame.columns for column in key_columns):
        return frame

    out = frame.copy()
    evidence_key = "__wrong_map_duplicate_numeric_evidence"
    while evidence_key in out.columns:
        evidence_key = f"_{evidence_key}"
    out[evidence_key] = pd.to_numeric(out[evidence_col], errors="coerce")
    out = out.sort_values(
        evidence_key,
        ascending=True,
        kind="stable",
        na_position="first",
    )
    out = out.drop_duplicates(list(key_columns), keep="last")
    return out.drop(columns=[evidence_key])


def _unwrap_numeric(function):
    return getattr(function, _NUMERIC_ORIGINAL_ATTR, function)


def _mark_numeric(function, original, flag: str) -> None:
    setattr(function, flag, True)
    setattr(function, _NUMERIC_ORIGINAL_ATTR, original)


__all__ = ["apply_wrong_map_rat_bootstrap_patch", "wrong_map_rat_bootstrap_patch_current"]
