"""Preserve clusterless decode groups with missing benchmark metadata.

Pandas ``groupby`` drops rows whose group key contains missing values unless
``dropna=False`` is used.  The clusterless ground-truth decoder groups held-out
score rows by the metadata returned from ``ground_truth._decode_group_columns``;
legacy or partially aggregated tables can carry a ``benchmark_cell_split_index``
column whose values are all missing.  Without a guard, those clusterless rows are
never decoded and the comparison silently merges no posterior endpoint columns.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

_PATCH_ATTR = "_clusterless_missing_decode_group_wrapped"
_SENTINEL_PREFIX = "__hipporeplayimm_missing_clusterless_decode_group__"


def apply_clusterless_missing_decode_group_patch() -> None:
    """Keep clusterless ground-truth decode groups when metadata is missing."""

    from . import clusterless_ground_truth as clusterless_gt

    original_compare = clusterless_gt._compare_clusterless_scores_to_ground_truth
    if getattr(original_compare, _PATCH_ATTR, False):
        return

    def compare_clusterless_scores_to_ground_truth_keep_missing(
        gt: Any,
        root: Any,
        scores_frame: pd.DataFrame,
        **kwargs: Any,
    ) -> pd.DataFrame:
        frame = scores_frame.copy()
        sentinels = _install_missing_decode_group_sentinels(gt, frame)
        out = original_compare(gt, root, frame, **kwargs)
        return _restore_missing_decode_group_sentinels(out, sentinels)

    compare_clusterless_scores_to_ground_truth_keep_missing._clusterless_missing_decode_group_wrapped = True  # type: ignore[attr-defined]
    clusterless_gt._compare_clusterless_scores_to_ground_truth = (
        compare_clusterless_scores_to_ground_truth_keep_missing
    )


def _install_missing_decode_group_sentinels(
    gt: Any,
    frame: pd.DataFrame,
) -> dict[str, str]:
    try:
        benchmark_decode = gt._score_table_is_heldout_benchmark(frame)
        group_columns = gt._decode_group_columns(frame, benchmark_decode)
    except Exception:
        return {}

    sentinels: dict[str, str] = {}
    for column in group_columns:
        if column not in frame.columns:
            continue
        missing_mask = frame[column].isna()
        if not bool(missing_mask.any()):
            continue
        sentinel = _missing_decode_group_sentinel(column, frame[column])
        frame[column] = frame[column].astype(object)
        frame.loc[missing_mask, column] = sentinel
        sentinels[column] = sentinel
    return sentinels


def _restore_missing_decode_group_sentinels(
    frame: pd.DataFrame,
    sentinels: dict[str, str],
) -> pd.DataFrame:
    if not sentinels or frame.empty:
        return frame
    out = frame.copy()
    for column, sentinel in sentinels.items():
        if column not in out.columns:
            continue
        out.loc[out[column].eq(sentinel), column] = np.nan
    return out


def _missing_decode_group_sentinel(column: str, values: pd.Series) -> str:
    base = f"{_SENTINEL_PREFIX}{column}__"
    sentinel = base
    text_values = values.astype(str)
    suffix = 0
    while bool(text_values.eq(sentinel).any()):
        suffix += 1
        sentinel = f"{base}{suffix}"
    return sentinel
