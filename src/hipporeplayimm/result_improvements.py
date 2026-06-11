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

    evidence_support = str(row.get("evidence_support", ""))
    if evidence_support and evidence_support not in {"truncated_full_grid", "nan"}:
        return CANDIDATE_SUPPORT_EXACT
    diagnostic_support = " ".join(
        str(row.get(column, ""))
        for column in (
            "diagnostic_candidate_evidence_support",
            "diagnostic_state_space_momentum_evidence_support",
            "diagnostic_state_space_imm_evidence_support",
        )
    )
    if "truncated_full_grid" not in diagnostic_support and evidence_support != "truncated_full_grid":
        return CANDIDATE_SUPPORT_EXACT
    if min_log_mass is None or not np.isfinite(min_log_mass):
        return CANDIDATE_SUPPORT_UNKNOWN
    if min_log_mass >= good_threshold:
        return CANDIDATE_SUPPORT_GOOD
    if min_log_mass >= warning_threshold:
        return CANDIDATE_SUPPORT_WARNING
    return CANDIDATE_SUPPORT_POOR


def _candidate_min_log_mass(row: pd.Series) -> float:
    for column in (
        "min_candidate_log_mass",
        "diagnostic_min_candidate_log_mass",
        "diagnostic_state_space_momentum_min_candidate_log_mass",
        "diagnostic_state_space_imm_min_candidate_log_mass",
    ):
        value = row.get(column)
        if value is None or pd.isna(value):
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return float("nan")


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
    bootstrap_means = np.empty(int(n_bootstrap), dtype=float)
    for index in range(int(n_bootstrap)):
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

    values = rows.loc[rows["model"].astype(str).eq(str(model)), value_column]
    values = values.dropna().to_numpy(dtype=float)
    if values.size == 0:
        return float("nan")
    observed = abs(float(np.mean(values)))
    rng = np.random.default_rng(random_seed)
    count = 1
    for _ in range(int(n_permutations)):
        signs = rng.choice(np.array([-1.0, 1.0]), size=values.size, replace=True)
        if abs(float(np.mean(values * signs))) >= observed:
            count += 1
    return float(count / (int(n_permutations) + 1))


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


def shuffle_spike_times_session(session, random_seed: int = 1):
    """Return a session with spike times permuted across spikes.

    Clusterless mark rows are stored in the same row order as ``session.spikes``.
    Keep mark timestamps in that row order as well; otherwise downstream marked
    point-process emissions can pair a spike waveform with a different spike time.
    """

    rng = np.random.default_rng(random_seed)
    spikes = np.asarray(session.spikes, dtype=float).copy()
    if spikes.size:
        spikes[:, 0] = rng.permutation(spikes[:, 0])
    marks = session.spike_marks
    if marks is not None:
        mark_times = np.asarray(marks.times, dtype=float).copy()
        if spikes.ndim == 2 and mark_times.shape[0] == spikes.shape[0]:
            mark_times = spikes[:, 0].copy()
        elif mark_times.size:
            mark_times = rng.permutation(mark_times)
        marks = _replace_spike_mark_rows(marks, times=mark_times)
    return replace(session, spikes=spikes, spike_marks=marks)


def circular_shift_spikes_session(session, shift_s: float | None = None, random_seed: int = 1):
    """Return a session with spike times circularly shifted within session bounds.

    The returned ``spikes`` array is time-sorted.  Reorder clusterless mark rows
    with the same permutation so mark features, mark cell IDs, and spike rows stay
    aligned.
    """

    spikes = np.asarray(session.spikes, dtype=float).copy()
    if spikes.size == 0:
        return session
    start = float(np.nanmin(spikes[:, 0]))
    end = float(np.nanmax(spikes[:, 0]))
    duration = max(end - start, np.finfo(float).eps)
    if shift_s is None:
        rng = np.random.default_rng(random_seed)
        shift_s = float(rng.uniform(0.1 * duration, 0.9 * duration))
    shifted_times = ((spikes[:, 0] - start + float(shift_s)) % duration) + start
    order = np.argsort(shifted_times, kind="mergesort")
    spikes[:, 0] = shifted_times
    spikes = spikes[order]
    marks = session.spike_marks
    if marks is not None:
        mark_times = ((np.asarray(marks.times, dtype=float) - start + float(shift_s)) % duration) + start
        if mark_times.shape[0] == order.shape[0]:
            marks = _replace_spike_mark_rows(marks, times=mark_times[order], order=order)
        else:
            marks = _replace_spike_mark_rows(marks, times=mark_times)
    return replace(session, spikes=spikes, spike_marks=marks)


def _replace_spike_mark_rows(
    marks,
    *,
    times: np.ndarray | None = None,
    order: np.ndarray | None = None,
):
    """Return spike marks with row-aligned arrays replaced or reordered."""

    updates: dict[str, np.ndarray] = {}
    if times is not None:
        updates["times"] = np.asarray(times, dtype=float).copy()
    if order is not None:
        row_order = np.asarray(order, dtype=int)
        if marks.marks.shape[0] == row_order.shape[0]:
            updates["marks"] = np.asarray(marks.marks).copy()[row_order]
        if marks.cell_ids is not None:
            cell_ids = np.asarray(marks.cell_ids)
            if cell_ids.shape[0] == row_order.shape[0]:
                updates["cell_ids"] = cell_ids.copy()[row_order]
        if marks.group_ids is not None:
            group_ids = np.asarray(marks.group_ids)
            if group_ids.shape[0] == row_order.shape[0]:
                updates["group_ids"] = group_ids.copy()[row_order]
    return replace(marks, **updates)


def shuffle_cell_identities_session(session, random_seed: int = 1):
    """Return a session with cell IDs randomly remapped."""

    rng = np.random.default_rng(random_seed)
    spikes = np.asarray(session.spikes, dtype=float).copy()
    if spikes.size == 0:
        return session
    cells = np.unique(spikes[:, 1].astype(int))
    shuffled = rng.permutation(cells)
    mapping = {int(src): int(dst) for src, dst in zip(cells, shuffled, strict=True)}
    spikes[:, 1] = [mapping[int(cell)] for cell in spikes[:, 1].astype(int)]
    marks = session.spike_marks
    if marks is not None and marks.cell_ids is not None:
        mark_cell_ids = np.asarray([mapping.get(int(cell), int(cell)) for cell in marks.cell_ids], dtype=int)
        marks = replace(marks, cell_ids=mark_cell_ids)
    return replace(session, spikes=spikes, spike_marks=marks)


def shuffle_mark_features_session(session, random_seed: int = 1):
    """Return a session with mark-feature rows independently permuted."""

    marks = session.spike_marks
    if marks is None or marks.n_features == 0:
        return session
    rng = np.random.default_rng(random_seed)
    values = np.asarray(marks.marks, dtype=float).copy()
    for column in range(values.shape[1]):
        values[:, column] = rng.permutation(values[:, column])
    return replace(session, spike_marks=replace(marks, marks=values))


def shuffle_well_labels(frame: pd.DataFrame, random_seed: int = 1) -> pd.DataFrame:
    """Shuffle behavioral well labels in a ground-truth/score comparison table."""

    if frame.empty or "true_well_id" not in frame:
        return frame.copy()
    rng = np.random.default_rng(random_seed)
    out = frame.copy()
    label_columns = [column for column in ("true_well_id", "true_well_x", "true_well_y") if column in out]
    for column in label_columns:
        values = out[column].to_numpy(copy=True)
        finite = pd.notna(values)
        values[finite] = rng.permutation(values[finite])
        out[column] = values
    return out


def benchmark_settings_dict(config, args: dict[str, object] | None = None, rows: pd.DataFrame | None = None) -> dict[str, object]:
    """Build reproducibility metadata for a benchmark run."""

    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "packages": _package_versions(("numpy", "pandas", "scipy", "hipporeplayimm")),
        "config": _as_serializable(config),
    }
    if args is not None:
        payload["cli_args"] = _as_serializable(args)
    if rows is not None and not rows.empty:
        payload["rows"] = {
            "count": int(rows.shape[0]),
            "sessions": sorted(str(value) for value in rows.get("session", pd.Series(dtype=object)).dropna().unique()),
            "models": sorted(str(value) for value in rows.get("model", pd.Series(dtype=object)).dropna().unique()),
        }
    return payload


def write_benchmark_settings(path: str | Path, config, args: dict[str, object] | None = None, rows: pd.DataFrame | None = None) -> None:
    """Write benchmark settings metadata as a small YAML file."""

    Path(path).write_text(_yaml_lines(benchmark_settings_dict(config, args, rows)), encoding="utf-8")


def _package_versions(names: Iterable[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def _as_serializable(value):
    if is_dataclass(value):
        return _as_serializable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _as_serializable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_as_serializable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _yaml_lines(value: object, indent: int = 0) -> str:
    prefix = " " * indent
    if isinstance(value, dict):
        lines = []
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}{key}:")
                lines.append(_yaml_lines(item, indent + 2).rstrip())
            else:
                lines.append(f"{prefix}{key}: {_yaml_scalar(item)}")
        return "\n".join(lines) + "\n"
    if isinstance(value, list):
        lines = []
        for item in value:
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}-")
                lines.append(_yaml_lines(item, indent + 2).rstrip())
            else:
                lines.append(f"{prefix}- {_yaml_scalar(item)}")
        return "\n".join(lines) + "\n"
    return f"{prefix}{_yaml_scalar(value)}\n"


def _yaml_scalar(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float, np.integer, np.floating)):
        return str(value)
    text = str(value)
    if not text or any(ch in text for ch in ":#[]{}&*!|>'\"%@`"):
        return repr(text)
    return text


