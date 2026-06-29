"""Result-quality, null-control, and reporting helpers for replay experiments.

The helpers in this module are intentionally lightweight and independent of the
benchmark runner.  They make it easier to keep scientific comparisons honest:
candidate-pruned evidences are labelled by support quality, confidence intervals
can respect session nesting, and benchmark runs can emit reproducible settings
metadata alongside CSV results.
"""

from __future__ import annotations

import platform
import sys
from dataclasses import asdict, is_dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:  # Python 3.10 compatibility for older local environments.
    from importlib import metadata as importlib_metadata
except ImportError:  # pragma: no cover
    import importlib_metadata  # type: ignore[no-redef]


CANDIDATE_SUPPORT_EXACT = "exact_or_not_pruned"
CANDIDATE_SUPPORT_GOOD = "conservative_good"
CANDIDATE_SUPPORT_WARNING = "conservative_warning"
CANDIDATE_SUPPORT_POOR = "conservative_poor"
CANDIDATE_SUPPORT_UNKNOWN = "conservative_unknown"

DEFAULT_GOOD_LOG_MASS_THRESHOLD = -0.01
DEFAULT_WARNING_LOG_MASS_THRESHOLD = -0.10

_EXACT_EVIDENCE_SUPPORT = "exact_full_grid"
_TRUNCATED_EVIDENCE_SUPPORT = "truncated_full_grid"
_NONCOMPARABLE_EVIDENCE_SUPPORTS = {
    "degenerate_single_bin",
    "not_scored",
    "particle_approximation",
    "unknown",
    "unknown_noncomparable",
}
_MISSING_SUPPORT_STRINGS = {"", "nan", "na", "n/a", "none", "null", "<na>"}
_MISSING_STATUS_VALUES = {"", "nan", "na", "n/a", "none", "null", "<na>"}
_CANDIDATE_MIN_LOG_MASS_COLUMNS = (
    "min_candidate_log_mass",
    "diagnostic_min_candidate_log_mass",
    "diagnostic_state_space_sparse_momentum_min_candidate_log_mass",
    "diagnostic_state_space_trajectory_imm_min_candidate_log_mass",
    "diagnostic_state_space_displacement_momentum_min_candidate_log_mass",
    "diagnostic_state_space_displacement_imm_min_candidate_log_mass",
    "diagnostic_state_space_momentum_min_candidate_log_mass",
    "diagnostic_state_space_imm_min_candidate_log_mass",
    "diagnostic_goal_state_space_min_candidate_log_mass",
)


def add_candidate_support_quality_columns(
    frame: pd.DataFrame,
    *,
    good_threshold: float = DEFAULT_GOOD_LOG_MASS_THRESHOLD,
    warning_threshold: float = DEFAULT_WARNING_LOG_MASS_THRESHOLD,
) -> pd.DataFrame:
    """Add candidate-support quality labels to a score table.

    Candidate-pruned momentum/IMM rows are lower bounds on full-grid evidence.
    The support quality label is based on the minimum log candidate mass when it
    is available in diagnostics.  Non-pruned rows are labelled as exact/not
    pruned so downstream tables can filter conservatively without losing exact
    baselines.
    """

    if frame.empty:
        return frame.copy()
    out = frame.copy()
    labels = []
    min_masses = []
    for _, row in out.iterrows():
        mass = _candidate_min_log_mass(row)
        min_masses.append(mass)
        labels.append(
            candidate_support_quality(
                row,
                min_log_mass=mass,
                good_threshold=good_threshold,
                warning_threshold=warning_threshold,
            )
        )
    out["candidate_min_log_mass"] = min_masses
    out["candidate_support_quality"] = labels
    out["candidate_support_quality_good"] = out["candidate_support_quality"].isin(
        {CANDIDATE_SUPPORT_EXACT, CANDIDATE_SUPPORT_GOOD}
    )
    return out


def candidate_support_quality(
    row: pd.Series,
    *,
    min_log_mass: float | None = None,
    good_threshold: float = DEFAULT_GOOD_LOG_MASS_THRESHOLD,
    warning_threshold: float = DEFAULT_WARNING_LOG_MASS_THRESHOLD,
) -> str:
    """Return a conservative quality label for one score row."""

    if not _status_is_success_or_missing(row.get("status", "success")):
        return CANDIDATE_SUPPORT_UNKNOWN

    evidence_supports = _row_evidence_support_labels(row)
    if _has_any_support(evidence_supports, _NONCOMPARABLE_EVIDENCE_SUPPORTS):
        return CANDIDATE_SUPPORT_UNKNOWN
    if not _has_support(evidence_supports, _TRUNCATED_EVIDENCE_SUPPORT):
        return CANDIDATE_SUPPORT_EXACT
    if min_log_mass is None or not np.isfinite(min_log_mass):
        return CANDIDATE_SUPPORT_UNKNOWN
    if min_log_mass >= good_threshold:
        return CANDIDATE_SUPPORT_GOOD
    if min_log_mass >= warning_threshold:
        return CANDIDATE_SUPPORT_WARNING
    return CANDIDATE_SUPPORT_POOR


def _row_evidence_support_labels(row: pd.Series) -> list[str]:
    labels: list[str] = []
    for column in row.index:
        name = str(column)
        if name == "evidence_support" or (name.startswith("diagnostic_") and name.endswith("_evidence_support")):
            labels.extend(_evidence_support_labels(row.get(column)))
    return labels


def _evidence_support_labels(value: object) -> list[str]:
    labels: list[str] = []
    for item in _flatten_value(value):
        if _is_missing_scalar(item):
            continue
        text = str(item).strip().lower()
        if text in _MISSING_SUPPORT_STRINGS:
            continue
        labels.append(text)
    return labels


def _flatten_value(value: object) -> list[object]:
    if _is_missing_scalar(value):
        return []
    if isinstance(value, (str, bytes)):
        return [value]
    try:
        array = np.asarray(value, dtype=object)
    except (TypeError, ValueError):
        return [value]
    if array.ndim == 0:
        try:
            return [array.item()]
        except ValueError:
            return []
    if array.size == 0:
        return []
    return list(array.ravel())


def _has_support(labels: list[str], support: str) -> bool:
    return any(label == support or support in label for label in labels)


def _has_any_support(labels: list[str], supports: set[str]) -> bool:
    return any(_has_support(labels, support) for support in supports)


def _status_is_success_or_missing(value: object) -> bool:
    if _is_missing_scalar(value):
        return True
    text = str(value).strip().lower()
    if text in _MISSING_STATUS_VALUES:
        return True
    return text == "success"


def _is_missing_scalar(value: object) -> bool:
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return isinstance(missing, (bool, np.bool_)) and bool(missing)


def _candidate_min_log_mass(row: pd.Series) -> float:
    columns: list[object] = list(_CANDIDATE_MIN_LOG_MASS_COLUMNS)
    seen = {str(column) for column in columns}
    for column in row.index:
        name = str(column)
        if name.endswith("_min_candidate_log_mass") and name not in seen:
            columns.append(column)
            seen.add(name)
    for column in columns:
        value = row.get(column)
        scalar = _first_finite_numeric_value(value)
        if scalar is not None:
            return scalar
    return float("nan")


def _first_finite_numeric_value(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        return _finite_numeric_scalar(text)
    try:
        array = np.asarray(value, dtype=object)
    except (TypeError, ValueError):
        return _finite_numeric_scalar(value)
    if array.ndim == 0:
        return _finite_numeric_scalar(array.item())
    for item in array.ravel():
        number = _finite_numeric_scalar(item)
        if number is not None:
            return number
    return None


def _finite_numeric_scalar(value: object) -> float | None:
    if isinstance(value, (bool, np.bool_)) or _is_missing_scalar(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _positive_integer_count(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if not np.isfinite(numeric) or numeric < 1.0 or numeric != np.floor(numeric):
        raise ValueError(f"{name} must be a positive integer")
    return int(numeric)


def hierarchical_bootstrap_ci(
    rows: pd.DataFrame,
    *,
    model: str,
    value_column: str = "delta_vs_best_static",
    group_columns: tuple[str, ...] = ("session",),
    n_bootstrap: int = 5000,
    random_seed: int = 1,
) -> tuple[float, float]:
    """Return a nested bootstrap confidence interval for a model-level mean.

    Groups (sessions by default) are resampled first, and rows within each group
    are resampled second.  This is deliberately more conservative than a pooled
    event bootstrap for session-nested replay events.
    """

    n_bootstrap = _positive_integer_count(n_bootstrap, "n_bootstrap")
    values = _model_metric_rows(rows, model, value_column, group_columns)
    if values.empty:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(random_seed)
    if group_columns:
        groupby_keys = group_columns[0] if len(group_columns) == 1 else list(group_columns)
        grouped_values = [
            group[value_column].to_numpy(dtype=float)
            for _, group in values.groupby(groupby_keys, sort=False)
        ]
    else:
        grouped_values = [values[value_column].to_numpy(dtype=float)]
    if not grouped_values:
        return (float("nan"), float("nan"))
    bootstrap_means = np.empty(n_bootstrap, dtype=float)
    for index in range(n_bootstrap):
        sampled_groups = rng.choice(
            np.arange(len(grouped_values)),
            size=len(grouped_values),
            replace=True,
        )
        sampled_values: list[np.ndarray] = []
        for group_index in sampled_groups:
            curr = grouped_values[int(group_index)]
            sampled_values.append(rng.choice(curr, size=curr.size, replace=True))
        merged = np.concatenate(sampled_values) if sampled_values else np.array([], dtype=float)
        bootstrap_means[index] = float(np.mean(merged)) if merged.size else np.nan
    finite = bootstrap_means[np.isfinite(bootstrap_means)]
    if finite.size == 0:
        return (float("nan"), float("nan"))
    return (float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975)))


def paired_sign_flip_p_value(
    rows: pd.DataFrame,
    *,
    model: str,
    value_column: str = "delta_vs_best_static",
    n_permutations: int = 10000,
    random_seed: int = 1,
) -> float:
    """Two-sided paired sign-flip p-value for event-level deltas."""

    n_permutations = _positive_integer_count(n_permutations, "n_permutations")
    values = rows.loc[rows["model"].astype(str).eq(str(model)), value_column]
    values = values.dropna().to_numpy(dtype=float)
    if values.size == 0:
        return float("nan")
    observed = abs(float(np.mean(values)))
    rng = np.random.default_rng(random_seed)
    count = 1
    for _ in range(n_permutations):
        signs = rng.choice(np.array([-1.0, 1.0]), size=values.size, replace=True)
        if abs(float(np.mean(values * signs))) >= observed:
            count += 1
    return float(count / (n_permutations + 1))


def _model_metric_rows(
    rows: pd.DataFrame,
    model: str,
    value_column: str,
    group_columns: tuple[str, ...],
) -> pd.DataFrame:
    if rows.empty or value_column not in rows or "model" not in rows:
        return pd.DataFrame()
    required = ["model", value_column, *group_columns]
    missing = [column for column in required if column not in rows]
    if missing:
        raise KeyError(f"required columns missing from score table: {missing}")
    values = rows.loc[rows["model"].astype(str).eq(str(model)), required].copy()
    values[value_column] = pd.to_numeric(values[value_column], errors="coerce")
    return values.dropna(subset=[value_column])


def summarize_grouped_model_metrics(
    rows: pd.DataFrame,
    group_columns: tuple[str, ...],
    *,
    value_columns: tuple[str, ...] = (
        "heldout_log_likelihood",
        "delta_vs_best_static",
        "bits_per_spike_vs_best_static",
        "lower_bound_delta_vs_best_static",
        "lower_bound_bits_per_spike_vs_best_static",
    ),
) -> pd.DataFrame:
    """Summarize model metrics by session, rat, split, or any grouping."""

    if rows.empty:
        return pd.DataFrame()
    missing = [column for column in ("model", *group_columns) if column not in rows]
    if missing:
        raise KeyError(f"required columns missing from score table: {missing}")
    available = [column for column in value_columns if column in rows]
    if not available:
        return pd.DataFrame()
    frame = add_candidate_support_quality_columns(rows)
    agg: dict[str, tuple[str, str]] = {"events": (available[0], "count")}
    for column in available:
        agg[f"mean_{column}"] = (column, "mean")
        agg[f"median_{column}"] = (column, "median")
    if "candidate_support_quality_good" in frame:
        agg["candidate_good_fraction"] = ("candidate_support_quality_good", "mean")
    return (
        frame.groupby([*group_columns, "model"], as_index=False)
        .agg(**agg)
        .sort_values([*group_columns, "model"])
    )


def stratified_cell_split(
    cell_ids: np.ndarray,
    stratum_values: np.ndarray,
    test_fraction: float,
    random_seed: int,
    *,
    n_strata: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Split cells while balancing a scalar cell-quality/rate score."""

    if not 0.0 < float(test_fraction) < 1.0:
        raise ValueError("test_fraction must be in (0, 1)")
    ids = np.asarray(cell_ids, dtype=int)
    scores = np.asarray(stratum_values, dtype=float)
    if ids.ndim != 1 or scores.shape != ids.shape:
        raise ValueError("cell_ids and stratum_values must be one-dimensional arrays with matching shape")
    if ids.size <= 1:
        return ids.copy(), np.array([], dtype=int)
    finite_scores = np.where(np.isfinite(scores), scores, np.nanmedian(scores[np.isfinite(scores)]) if np.any(np.isfinite(scores)) else 0.0)
    order = np.argsort(finite_scores, kind="mergesort")
    strata = np.array_split(order, max(1, min(int(n_strata), ids.size)))
    rng = np.random.default_rng(random_seed)
    test_indices: list[int] = []
    for stratum in strata:
        if stratum.size == 0:
            continue
        shuffled = np.asarray(stratum, dtype=int).copy()
        rng.shuffle(shuffled)
        n_test = int(round(shuffled.size * float(test_fraction)))
        if n_test == 0 and len(test_indices) == 0:
            n_test = 1
        test_indices.extend(int(index) for index in shuffled[:n_test])
    n_test_total = max(1, int(round(ids.size * float(test_fraction))))
    n_test_total = min(n_test_total, ids.size - 1)
    selected = np.asarray(sorted(set(test_indices)), dtype=int)
    if selected.size < n_test_total:
        remaining = np.setdiff1d(np.arange(ids.size), selected, assume_unique=False)
        rng.shuffle(remaining)
        selected = np.concatenate(
            [selected, remaining[: n_test_total - selected.size]]
        )
    elif selected.size > n_test_total:
        selected = rng.choice(selected, size=n_test_total, replace=False)
    test_indices = np.sort(selected.astype(int))
    test = np.sort(ids[test_indices])
    train = np.sort(np.setdiff1d(ids, test, assume_unique=False))
    return train, test


def posterior_calibration_summary(
    samples: pd.DataFrame,
    *,
    probability_column: str = "true_bin_probability",
    rank_column: str = "true_bin_rank",
    n_bins_column: str = "n_position_bins",
) -> pd.DataFrame:
    """Return simple posterior calibration diagnostics for validation samples."""

    if samples.empty or probability_column not in samples:
        return pd.DataFrame()
    group_columns = [column for column in ("session", "model") if column in samples]
    if not group_columns:
        group_columns = ["_all"]
        samples = samples.copy()
        samples["_all"] = "all"
    frame = samples.copy()
    probabilities = pd.to_numeric(frame[probability_column], errors="coerce").clip(lower=np.finfo(float).tiny, upper=1.0)
    frame[probability_column] = probabilities
    frame["true_negative_log_probability"] = -np.log(probabilities)
    if rank_column in frame and n_bins_column in frame:
        rank = pd.to_numeric(frame[rank_column], errors="coerce")
        n_bins = pd.to_numeric(frame[n_bins_column], errors="coerce")
        frame["rank_fraction"] = rank / n_bins
        frame["coverage_50_rank"] = frame["rank_fraction"] <= 0.50
        frame["coverage_80_rank"] = frame["rank_fraction"] <= 0.80
        frame["coverage_95_rank"] = frame["rank_fraction"] <= 0.95
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
