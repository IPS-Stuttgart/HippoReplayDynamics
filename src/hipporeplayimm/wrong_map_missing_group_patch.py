"""Keep wrong-map diagnostic groups with missing optional metadata."""
from __future__ import annotations

import pandas as pd

SENTINEL_PREFIX = "__hipporeplayimm_missing_group__"
ABS_FLAG = "_missing_group_metadata_wrong_map_absolute_wrapper"
DID_FLAG = "_missing_group_metadata_wrong_map_family_did_wrapper"
SUMMARY_FLAG = "_missing_group_metadata_wrong_map_delta_summary_wrapper"
DID_SUMMARY_FLAG = "_missing_group_metadata_wrong_map_family_did_summary_wrapper"
ORIGINAL_ATTR = "_missing_group_metadata_original"


def apply_wrong_map_missing_group_patch(diagnostics):
    if wrong_map_missing_group_patch_current(diagnostics):
        return
    original_abs = _unwrap(diagnostics.wrong_map_absolute_evidence_deltas)
    original_did = _unwrap(diagnostics.wrong_map_family_margin_difference_in_differences)
    original_summary = _unwrap(diagnostics._wrong_map_delta_summary)
    original_did_summary = _unwrap(diagnostics.wrong_map_family_margin_difference_in_differences_summary)

    def wrong_map_absolute_evidence_deltas(current_map_scores, wrong_map_scores, *, group_cols=("session", "event_index"), **kwargs):
        groups = tuple(group_cols)
        sentinels = _sentinels((current_map_scores, wrong_map_scores), groups)
        result = original_abs(
            _fill(current_map_scores, groups, sentinels),
            _fill(wrong_map_scores, groups, sentinels),
            group_cols=groups,
            **kwargs,
        )
        return _restore(result, groups, sentinels)

    def wrong_map_family_margin_difference_in_differences(current_map_scores, wrong_map_scores, *, group_cols=("session", "event_index"), **kwargs):
        groups = tuple(group_cols)
        sentinels = _sentinels((current_map_scores, wrong_map_scores), groups)
        result = original_did(
            _fill(current_map_scores, groups, sentinels),
            _fill(wrong_map_scores, groups, sentinels),
            group_cols=groups,
            **kwargs,
        )
        return _restore(result, groups, sentinels)

    def _wrong_map_delta_summary(deltas, *, group_cols=()):
        groups = tuple(group_cols)
        sentinels = _sentinels((deltas,), groups)
        return _restore(original_summary(_fill(deltas, groups, sentinels), group_cols=groups), groups, sentinels)

    def wrong_map_family_margin_difference_in_differences_summary(deltas, *, group_cols=()):
        groups = tuple(group_cols)
        sentinels = _sentinels((deltas,), groups)
        return _restore(original_did_summary(_fill(deltas, groups, sentinels), group_cols=groups), groups, sentinels)

    _mark(wrong_map_absolute_evidence_deltas, original_abs, ABS_FLAG)
    _mark(wrong_map_family_margin_difference_in_differences, original_did, DID_FLAG)
    _mark(_wrong_map_delta_summary, original_summary, SUMMARY_FLAG)
    _mark(wrong_map_family_margin_difference_in_differences_summary, original_did_summary, DID_SUMMARY_FLAG)
    diagnostics.wrong_map_absolute_evidence_deltas = wrong_map_absolute_evidence_deltas
    diagnostics.wrong_map_family_margin_difference_in_differences = wrong_map_family_margin_difference_in_differences
    diagnostics._wrong_map_delta_summary = _wrong_map_delta_summary
    diagnostics.wrong_map_family_margin_difference_in_differences_summary = wrong_map_family_margin_difference_in_differences_summary


def wrong_map_missing_group_patch_current(diagnostics):
    return all(
        getattr(getattr(diagnostics, name, None), flag, False)
        for name, flag in (
            ("wrong_map_absolute_evidence_deltas", ABS_FLAG),
            ("wrong_map_family_margin_difference_in_differences", DID_FLAG),
            ("_wrong_map_delta_summary", SUMMARY_FLAG),
            ("wrong_map_family_margin_difference_in_differences_summary", DID_SUMMARY_FLAG),
        )
    )


def _unwrap(function):
    return getattr(function, ORIGINAL_ATTR, function)


def _mark(function, original, flag):
    setattr(function, flag, True)
    setattr(function, ORIGINAL_ATTR, original)


def _sentinels(frames: tuple[pd.DataFrame, ...], group_cols) -> dict[str, str]:
    sentinels: dict[str, str] = {}
    for column in group_cols:
        existing: set[str] = set()
        for frame in frames:
            if column in frame.columns:
                existing.update(str(value) for value in frame[column].dropna())
        sentinel = SENTINEL_PREFIX
        index = 1
        while sentinel in existing:
            sentinel = f"{SENTINEL_PREFIX}_{index}"
            index += 1
        sentinels[str(column)] = sentinel
    return sentinels


def _fill(frame: pd.DataFrame, group_cols, sentinels: dict[str, str]):
    if frame.empty or not group_cols:
        return frame.copy()
    out = frame.copy()
    for column in group_cols:
        if column not in out.columns:
            continue
        missing = out[column].isna()
        if missing.any():
            out[column] = out[column].astype(object)
            out.loc[missing, column] = sentinels.get(str(column), SENTINEL_PREFIX)
    return out


def _restore(frame: pd.DataFrame, group_cols, sentinels: dict[str, str]):
    if frame.empty or not group_cols:
        return frame
    out = frame.copy()
    for column in group_cols:
        if column in out.columns:
            sentinel = sentinels.get(str(column), SENTINEL_PREFIX)
            mask = out[column].astype(object).eq(sentinel)
            if mask.any():
                out.loc[mask, column] = pd.NA
    return out
