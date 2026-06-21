"""Compatibility patch for benchmark cell-split options.

The score-table metadata compatibility layer replaces ``benchmarks.BenchmarkConfig``
with a local dataclass so post-hoc decoding can reconstruct old score tables.
When new benchmark fields are added, that replacement class has to stay in sync
with the canonical benchmark configuration.  This patch keeps the stratified
cell-split knobs available even when the compatibility layer is active.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import importlib
import numpy as np
import pandas as pd


_DEFAULT_CELL_SPLIT_STRATEGY = "random"
_DEFAULT_CELL_SPLIT_STRATA = 4
_DEFAULT_TEST_CELL_FRACTION = 0.25
_DEFAULT_RANDOM_SEED = 1
_MISSING_METADATA_STRINGS = {"", "nan", "na", "n/a", "none", "null", "<na>"}
_HELDOUT_BENCHMARK_COLUMNS = {
    "heldout_log_likelihood",
    "train_log_likelihood",
    "joint_log_likelihood",
}
_CELL_SPLIT_SCOPE_COLUMNS = (
    "benchmark_cell_split_index",
    "benchmark_test_cell_fraction",
    "benchmark_random_seed",
    "benchmark_cell_split_seed",
    "benchmark_cell_split_strategy",
    "benchmark_cell_split_strata",
    "train_cell_ids",
    "test_cell_ids",
)


def apply_benchmark_cell_split_metadata_patch() -> None:
    """Preserve benchmark cell-split options after metadata monkey-patching."""

    from . import benchmarks as bench
    from . import ground_truth as gt

    benchmark_config = bench.BenchmarkConfig
    field_names = _dataclass_field_names(benchmark_config)
    if (
        "cell_split_strategy" not in field_names
        or "cell_split_strata" not in field_names
    ):

        @dataclass(frozen=True)
        class BenchmarkConfigWithCellSplit(benchmark_config):  # type: ignore[misc, valid-type]
            cell_split_strategy: str = _DEFAULT_CELL_SPLIT_STRATEGY
            cell_split_strata: int = _DEFAULT_CELL_SPLIT_STRATA

        bench.BenchmarkConfig = BenchmarkConfigWithCellSplit
        gt.BenchmarkConfig = BenchmarkConfigWithCellSplit
    else:
        # Keep ground_truth's imported alias synchronized with benchmarks after
        # score_metadata.apply_model_hyperparam_patch() replaces both modules.
        gt.BenchmarkConfig = benchmark_config

    metadata = bench._benchmark_config_metadata
    if not getattr(metadata, "_cell_split_metadata_wrapped", False):

        def benchmark_config_metadata_with_cell_split(config: Any) -> dict[str, object]:
            out = dict(metadata(config))
            out["benchmark_cell_split_strategy"] = str(
                getattr(config, "cell_split_strategy", _DEFAULT_CELL_SPLIT_STRATEGY)
            )
            out["benchmark_cell_split_strata"] = int(
                getattr(config, "cell_split_strata", _DEFAULT_CELL_SPLIT_STRATA)
            )
            return out

        benchmark_config_metadata_with_cell_split._cell_split_metadata_wrapped = True  # type: ignore[attr-defined]
        bench._benchmark_config_metadata = benchmark_config_metadata_with_cell_split

    _wrap_ground_truth_compare_scores(bench, gt)
    _wrap_late_ground_truth_patches(bench, gt)
    bench._benchmark_cell_split_metadata_patch_applied = True


def _wrap_late_ground_truth_patches(bench: Any, gt: Any) -> None:
    """Re-apply this wrapper after later modules replace ground-truth decoding."""

    try:
        module = importlib.import_module(f"{__package__}.clusterless_ground_truth")
    except Exception:
        return
    apply_patch = getattr(module, "apply_clusterless_ground_truth_patch", None)
    if apply_patch is None or getattr(apply_patch, "_cell_split_metadata_rewraps_compare", False):
        return

    def apply_patch_with_cell_split_metadata() -> Any:
        result = apply_patch()
        _wrap_ground_truth_compare_scores(bench, gt)
        return result

    apply_patch_with_cell_split_metadata._cell_split_metadata_rewraps_compare = True  # type: ignore[attr-defined]
    module.apply_clusterless_ground_truth_patch = apply_patch_with_cell_split_metadata


def _wrap_ground_truth_compare_scores(bench: Any, gt: Any) -> None:
    """Let patched ground-truth decoding consume saved cell-split metadata."""

    compare_scores = gt.compare_scores_to_ground_truth
    if getattr(compare_scores, "_cell_split_metadata_wrapped", False):
        return

    def compare_scores_to_ground_truth_with_cell_split(
        root: str | Path,
        scores: str | Path | pd.DataFrame,
        *,
        cell_split_strategy: str = _DEFAULT_CELL_SPLIT_STRATEGY,
        cell_split_strata: int = _DEFAULT_CELL_SPLIT_STRATA,
        **kwargs: Any,
    ) -> pd.DataFrame:
        scores_frame = _scores_frame_for_cell_split_metadata(scores)
        if _score_table_needs_cell_split_scoped_decode(scores_frame):
            comparisons = [
                _compare_scores_with_cell_split_metadata(
                    compare_scores,
                    bench,
                    gt,
                    root,
                    split_scores.copy(),
                    split_scores.copy(),
                    cell_split_strategy,
                    cell_split_strata,
                    kwargs,
                )
                for split_scores in _cell_split_decode_groups(scores_frame)
            ]
            if comparisons:
                return pd.concat(comparisons, ignore_index=True, sort=False)
            return _compare_scores_with_cell_split_metadata(
                compare_scores,
                bench,
                gt,
                root,
                scores_frame,
                scores_frame,
                cell_split_strategy,
                cell_split_strata,
                kwargs,
            )
        return _compare_scores_with_cell_split_metadata(
            compare_scores,
            bench,
            gt,
            root,
            scores,
            scores_frame,
            cell_split_strategy,
            cell_split_strata,
            kwargs,
        )

    compare_scores_to_ground_truth_with_cell_split._cell_split_metadata_wrapped = True  # type: ignore[attr-defined]
    gt.compare_scores_to_ground_truth = compare_scores_to_ground_truth_with_cell_split


def _compare_scores_with_cell_split_metadata(
    compare_scores: Any,
    bench: Any,
    gt: Any,
    root: str | Path,
    scores: str | Path | pd.DataFrame,
    scores_frame: pd.DataFrame,
    default_strategy: str,
    default_strata: int,
    kwargs: dict[str, Any],
) -> pd.DataFrame:
    strategy = _cell_split_strategy_from_scores(scores_frame, default_strategy)
    strata = _cell_split_strata_from_scores(scores_frame, default_strata)
    original_build_models = gt._build_models
    original_cell_split_for_score_rows = gt._cell_split_for_score_rows

    def with_cell_split_config(config: Any) -> SimpleNamespace:
        return _config_with_cell_split_metadata(config, strategy, strata)

    def build_models_with_cell_split(config: Any, *args: Any, **build_kwargs: Any) -> Any:
        return original_build_models(with_cell_split_config(config), *args, **build_kwargs)

    def cell_split_for_score_rows_with_metadata(session_scores: Any, encoding: Any, config: Any) -> Any:
        config_with_metadata = with_cell_split_config(config)
        if _score_rows_have_explicit_cell_ids(gt, session_scores):
            return original_cell_split_for_score_rows(
                session_scores,
                encoding,
                config_with_metadata,
            )

        test_cell_fraction = gt._unique_float_from_column(
            session_scores,
            "benchmark_test_cell_fraction",
            getattr(
                config_with_metadata,
                "test_cell_fraction",
                _DEFAULT_TEST_CELL_FRACTION,
            ),
        )
        benchmark_random_seed = gt._unique_int_from_column(
            session_scores,
            "benchmark_random_seed",
            getattr(config_with_metadata, "random_seed", _DEFAULT_RANDOM_SEED),
        )
        random_seed = gt._unique_int_from_column(
            session_scores,
            "benchmark_cell_split_seed",
            benchmark_random_seed,
        )
        split_config = _config_with_cell_split_metadata(
            _config_with_overrides(
                config_with_metadata,
                test_cell_fraction=float(test_cell_fraction),
            ),
            strategy,
            strata,
        )
        return bench._split_cells_from_encoding(encoding, split_config, int(random_seed))

    gt._build_models = build_models_with_cell_split
    gt._cell_split_for_score_rows = cell_split_for_score_rows_with_metadata
    try:
        return compare_scores(root, scores, **kwargs)
    finally:
        gt._build_models = original_build_models
        gt._cell_split_for_score_rows = original_cell_split_for_score_rows


def _score_table_needs_cell_split_scoped_decode(scores_frame: pd.DataFrame) -> bool:
    """Return true when a score table needs per-session/split cell-split metadata.

    The wrapped ground-truth decoders carry one active cell-split strategy/strata
    pair in their temporary config.  Combined score tables can contain different
    metadata for different sessions, test fractions, benchmark random seeds,
    explicit cell IDs, or benchmark splits, so decode those groups separately
    before each group asks ``_cell_split_for_score_rows`` to rebuild held-out
    emissions.
    """

    if not _HELDOUT_BENCHMARK_COLUMNS.issubset(scores_frame.columns):
        return False
    if "session" not in scores_frame.columns:
        return False
    group_columns = _cell_split_decode_group_columns(scores_frame)
    group_count = int(_cell_split_group_labels(scores_frame, group_columns).drop_duplicates().shape[0])
    session_count = int(scores_frame[["session"]].drop_duplicates().shape[0])
    if group_count > session_count:
        return True
    return any(
        len(_metadata_group_values(scores_frame[column])) > 1
        for column in _CELL_SPLIT_SCOPE_COLUMNS
        if column in scores_frame.columns
    )


def _cell_split_decode_groups(scores_frame: pd.DataFrame):
    group_columns = _cell_split_decode_group_columns(scores_frame)
    group_labels = _cell_split_group_labels(scores_frame, group_columns)
    for indices in group_labels.groupby(list(group_labels.columns), sort=False, dropna=False).indices.values():
        yield scores_frame.iloc[np.asarray(indices, dtype=int)]


def _cell_split_group_labels(
    scores_frame: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            f"__cell_split_group_{index}": [
                _metadata_group_label(value) for value in scores_frame[column]
            ]
            for index, column in enumerate(group_columns)
        }
    )


def _cell_split_decode_group_columns(scores_frame: pd.DataFrame) -> list[str]:
    columns = ["session"]
    for column in _CELL_SPLIT_SCOPE_COLUMNS:
        if column not in scores_frame.columns:
            continue
        if column == "benchmark_cell_split_index" or len(_metadata_group_values(scores_frame[column])) > 1:
            columns.append(column)
    return columns


def _score_rows_have_explicit_cell_ids(gt: Any, session_scores: Any) -> bool:
    train_cells = gt._cell_ids_from_score_column(session_scores, "train_cell_ids")
    test_cells = gt._cell_ids_from_score_column(session_scores, "test_cell_ids")
    return train_cells is not None or test_cells is not None


def _config_with_cell_split_metadata(
    config: Any,
    strategy: str,
    strata: int,
) -> SimpleNamespace:
    """Copy config attributes while overriding cell-split metadata fields."""

    values = _config_values(config)
    values["cell_split_strategy"] = strategy
    values["cell_split_strata"] = int(strata)
    return SimpleNamespace(**values)


def _config_with_overrides(config: Any, **overrides: Any) -> SimpleNamespace:
    values = _config_values(config)
    values.update(overrides)
    return SimpleNamespace(**values)


def _config_values(config: Any) -> dict[str, Any]:
    if is_dataclass(config) and not isinstance(config, type):
        return {field.name: getattr(config, field.name) for field in fields(config)}
    return dict(getattr(config, "__dict__", {}))


def _scores_frame_for_cell_split_metadata(scores: str | Path | pd.DataFrame) -> pd.DataFrame:
    if isinstance(scores, pd.DataFrame):
        return scores.copy()
    return pd.read_csv(scores)


def _metadata_group_values(values: pd.Series) -> set[str]:
    out: set[str] = set()
    for value in values:
        label = _metadata_group_label(value)
        if label:
            out.add(label)
    return out


def _metadata_group_label(value: Any) -> str:
    if _is_missing_metadata_value(value):
        return ""
    if isinstance(value, (list, tuple, set, np.ndarray)):
        sequence = list(value) if isinstance(value, set) else value
        array = np.asarray(sequence, dtype=object).reshape(-1)
        return ",".join(str(item).strip() for item in array)
    text = str(value).strip()
    return "" if text.lower() in _MISSING_METADATA_STRINGS else text


def _is_missing_metadata_value(value: Any) -> bool:
    if value is None:
        return True
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return isinstance(missing, (bool, np.bool_)) and bool(missing)


def _cell_split_strategy_from_scores(scores_frame: pd.DataFrame, default: str) -> str:
    values = _string_metadata_values(scores_frame, "benchmark_cell_split_strategy")
    if not values:
        return str(default)
    first = values[0]
    if any(value != first for value in values[1:]):
        raise ValueError("benchmark_cell_split_strategy contains multiple values")
    return first


def _cell_split_strata_from_scores(scores_frame: pd.DataFrame, default: int) -> int:
    values = _numeric_metadata_values(scores_frame, "benchmark_cell_split_strata")
    if not values:
        return int(default)
    integer_values: list[int] = []
    for value in values:
        integer_value = int(round(value))
        if not np.isclose(value, integer_value, rtol=0.0, atol=1e-9):
            raise ValueError("benchmark_cell_split_strata must be an integer")
        integer_values.append(integer_value)
    first = integer_values[0]
    if any(value != first for value in integer_values[1:]):
        raise ValueError("benchmark_cell_split_strata contains multiple values")
    return int(first)


def _string_metadata_values(frame: pd.DataFrame, column: str) -> list[str]:
    if column not in frame.columns:
        return []
    values: list[str] = []
    for value in frame[column]:
        text = _metadata_group_label(value)
        if text:
            values.append(text)
    return values


def _numeric_metadata_values(frame: pd.DataFrame, column: str) -> list[float]:
    values: list[float] = []
    for text in _string_metadata_values(frame, column):
        numeric = float(text)
        if not np.isfinite(numeric):
            raise ValueError(f"{column} must be finite")
        values.append(float(numeric))
    return values


def _dataclass_field_names(cls: type[Any]) -> set[str]:
    if not is_dataclass(cls):
        return set()
    return {field.name for field in fields(cls)}
