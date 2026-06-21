"""Scope post-hoc ground-truth decoding by saved replay-window metadata.

Improved evidence tables can contain several replay-window variants for the same
``(session, event_index, model)`` row key.  The core ground-truth decoder rebuilds
emissions from the integer event index, so without this wrapper it can decode all
variants with the original ripple interval and merge the same decoded posterior
back into every window row.  Split score tables into independent window scopes
and replay each scope with its saved ``window_start_s`` / ``window_end_s`` bounds.
"""

from __future__ import annotations

from functools import wraps
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data import RippleEvent

_WINDOW_SCOPE_COLUMNS = (
    "window_role",
    "window_index",
    "null_index",
    "matched_null_rank",
    "template_event_index",
    "event_window_variant",
    "window_start_s",
    "window_end_s",
    "window_duration_s",
)
_WINDOW_TIME_COLUMNS = ("window_start_s", "window_end_s")
_MISSING_TEXT_VALUES = frozenset({"", "na", "n/a", "nan", "none", "null", "<na>"})
_PATCHED_FLAG = "_ground_truth_window_scope_wrapped"


def apply_ground_truth_window_scope_patch() -> None:
    """Install the replay-window scoped ground-truth comparison wrapper."""

    from . import ground_truth as gt

    base_compare = gt.compare_scores_to_ground_truth
    if getattr(base_compare, _PATCHED_FLAG, False):
        return

    @wraps(base_compare)
    def compare_scores_to_ground_truth_with_window_scope(
        root: str | Path,
        scores: str | Path | pd.DataFrame,
        *args: Any,
        **kwargs: Any,
    ) -> pd.DataFrame:
        scores_frame = _scores_frame(scores)
        if not _score_table_needs_window_scoped_decode(scores_frame):
            return base_compare(root, scores, *args, **kwargs)

        comparisons: list[pd.DataFrame] = []
        for window_scores in _window_decode_groups(scores_frame):
            replay_window = _window_from_score_rows(window_scores)
            if replay_window is None:
                comparisons.append(base_compare(root, window_scores.copy(), *args, **kwargs))
            else:
                comparisons.append(
                    _compare_scores_for_replay_window(
                        gt,
                        base_compare,
                        root,
                        window_scores.copy(),
                        replay_window,
                        args,
                        kwargs,
                    )
                )
        if not comparisons:
            return base_compare(root, scores, *args, **kwargs)
        return pd.concat(comparisons, ignore_index=True, sort=False)

    setattr(compare_scores_to_ground_truth_with_window_scope, _PATCHED_FLAG, True)
    gt.compare_scores_to_ground_truth = compare_scores_to_ground_truth_with_window_scope


def _compare_scores_for_replay_window(
    gt: Any,
    base_compare: Any,
    root: str | Path,
    scores: pd.DataFrame,
    replay_window: RippleEvent,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> pd.DataFrame:
    original_build_emissions = gt.build_emissions
    original_build_clusterless_mark_emissions = gt.build_clusterless_mark_emissions

    @wraps(original_build_emissions)
    def build_emissions_in_saved_window(session: Any, encoding: Any, ripple: Any, config: Any = None):
        return original_build_emissions(
            session,
            encoding,
            _saved_window_for_ripple(session, ripple, replay_window),
            config,
        )

    @wraps(original_build_clusterless_mark_emissions)
    def build_clusterless_mark_emissions_in_saved_window(session: Any, encoding: Any, ripple: Any, config: Any = None):
        return original_build_clusterless_mark_emissions(
            session,
            encoding,
            _saved_window_for_ripple(session, ripple, replay_window),
            config,
        )

    gt.build_emissions = build_emissions_in_saved_window
    gt.build_clusterless_mark_emissions = build_clusterless_mark_emissions_in_saved_window
    try:
        return base_compare(root, scores, *args, **kwargs)
    finally:
        gt.build_emissions = original_build_emissions
        gt.build_clusterless_mark_emissions = original_build_clusterless_mark_emissions


def _score_table_needs_window_scoped_decode(scores_frame: pd.DataFrame) -> bool:
    if scores_frame.empty or "session" not in scores_frame.columns or "event_index" not in scores_frame.columns:
        return False
    if all(column in scores_frame.columns for column in _WINDOW_TIME_COLUMNS):
        return any(_window_from_score_rows(group) is not None for group in _window_decode_groups(scores_frame))
    return any(
        len(_present_metadata_labels(scores_frame[column])) > 1
        for column in _WINDOW_SCOPE_COLUMNS
        if column in scores_frame.columns
    )


def _window_decode_groups(scores_frame: pd.DataFrame):
    columns = _window_group_columns(scores_frame)
    if not columns:
        yield scores_frame
        return
    labels = pd.DataFrame(
        {
            f"__window_scope_{index}": [_metadata_label(value) for value in scores_frame[column]]
            for index, column in enumerate(columns)
        },
        index=scores_frame.index,
    )
    for indices in labels.groupby(list(labels.columns), sort=False, dropna=False).indices.values():
        yield scores_frame.iloc[np.asarray(indices, dtype=int)]


def _window_group_columns(scores_frame: pd.DataFrame) -> list[str]:
    columns = [column for column in ("session", "event_index") if column in scores_frame.columns]
    for column in _WINDOW_SCOPE_COLUMNS:
        if column in scores_frame.columns and column not in columns:
            columns.append(column)
    return columns


def _window_from_score_rows(scores_frame: pd.DataFrame) -> RippleEvent | None:
    if not all(column in scores_frame.columns for column in _WINDOW_TIME_COLUMNS):
        return None
    start = _unique_finite_float(scores_frame, "window_start_s")
    end = _unique_finite_float(scores_frame, "window_end_s")
    if start is None or end is None:
        return None
    if end <= start:
        raise ValueError("window_end_s must be greater than window_start_s")
    peak = _unique_finite_float(scores_frame, "ripple_peak")
    if peak is None or not start <= peak <= end:
        peak = start + 0.5 * (end - start)
    return RippleEvent(float(start), float(end), float(peak), float("nan"), float("nan"), float("nan"))


def _saved_window_for_ripple(session: Any, ripple: Any, replay_window: RippleEvent) -> RippleEvent:
    if not isinstance(ripple, (int, np.integer)):
        return ripple
    base = session.ripple(int(ripple))
    peak = float(base.peak)
    if not replay_window.start <= peak <= replay_window.end:
        peak = float(replay_window.peak)
    return RippleEvent(
        float(replay_window.start),
        float(replay_window.end),
        float(peak),
        float(base.raw_power),
        float(base.z_power_session),
        float(base.z_power_epoch),
    )


def _scores_frame(scores: str | Path | pd.DataFrame) -> pd.DataFrame:
    if isinstance(scores, pd.DataFrame):
        return scores.copy()
    return pd.read_csv(scores)


def _unique_finite_float(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame.columns:
        return None
    values = []
    for value in frame[column]:
        if _is_missing_scalar(value):
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(numeric):
            values.append(float(numeric))
    if not values:
        return None
    first = values[0]
    if any(not np.isclose(value, first, rtol=0.0, atol=1e-12) for value in values[1:]):
        raise ValueError(f"{column} contains multiple values within one replay-window decode group")
    return float(first)


def _present_metadata_labels(values: pd.Series) -> set[str]:
    labels: set[str] = set()
    for value in values:
        label = _metadata_label(value)
        if label:
            labels.add(label)
    return labels


def _metadata_label(value: Any) -> str:
    if _is_missing_scalar(value):
        return ""
    if isinstance(value, np.ndarray):
        return repr(np.asarray(value, dtype=object).reshape(-1).tolist())
    if isinstance(value, (list, tuple)):
        return repr(list(value))
    if isinstance(value, set):
        return repr(sorted(value))
    text = str(value).strip()
    return "" if text.lower() in _MISSING_TEXT_VALUES else text


def _is_missing_scalar(value: Any) -> bool:
    if value is None:
        return True
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return isinstance(missing, (bool, np.bool_)) and bool(missing)
