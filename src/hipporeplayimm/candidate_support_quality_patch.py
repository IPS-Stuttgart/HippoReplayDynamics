"""Conservative candidate-support quality labels for non-comparable rows."""

from __future__ import annotations

from functools import wraps
from typing import Any

import numpy as np
import pandas as pd

_MISSING_SUPPORT_VALUES = {"", "nan", "na", "n/a", "none", "null", "<na>"}
_NONCOMPARABLE_SUPPORT_VALUES = {
    "degenerate_single_bin",
    "not_scored",
    "unknown",
    "unknown_noncomparable",
    "particle_approximation",
}
_TRUNCATED_SUPPORT = "truncated_full_grid"
_SUCCESS_STATUS_VALUES = {"", "success", "nan", "na", "n/a", "none", "null", "<na>"}
_QUALITY_WRAPPER_ATTR = "_candidate_support_quality_status_wrapper"
_MIN_LOG_MASS_BOOL_PATCHED_FLAG = "_candidate_min_log_mass_bool_patch_applied"
_MIN_LOG_MASS_BOOL_WRAPPER_ATTR = "_candidate_min_log_mass_bool_wrapper"
_RESTRICT_CANDIDATE_ORDER_PATCHED_FLAG = "_candidate_restriction_order_patch_applied"
_PAIRWISE_OVERFLOW_PATCHED_FLAG = "_candidate_pairwise_overflow_nearest_support_patch_applied"
_PAIRWISE_OVERFLOW_WRAPPER_ATTR = "_candidate_pairwise_overflow_nearest_support_wrapper"


def apply_candidate_support_quality_patch() -> None:
    """Keep failed/non-comparable rows out of good candidate-support counts."""

    from . import result_improvements as ri

    current_quality = getattr(ri, "candidate_support_quality", None)
    if not getattr(current_quality, _QUALITY_WRAPPER_ATTR, False):

        def candidate_support_quality(
            row: pd.Series,
            *,
            min_log_mass: float | None = None,
            good_threshold: float = ri.DEFAULT_GOOD_LOG_MASS_THRESHOLD,
            warning_threshold: float = ri.DEFAULT_WARNING_LOG_MASS_THRESHOLD,
        ) -> str:
            """Return a conservative quality label for one score row.

            Candidate-support quality is meaningful only for successful exact rows or
            candidate-pruned lower-bound rows.  Failed rows and non-comparable
            evidence supports should not be counted as ``exact_or_not_pruned`` merely
            because they are not truncated lower bounds.
            """

            status = _text(row.get("status", "success")).lower()
            if status not in _SUCCESS_STATUS_VALUES:
                return ri.CANDIDATE_SUPPORT_UNKNOWN

            support_values = _evidence_support_values(row)
            if any(value in _NONCOMPARABLE_SUPPORT_VALUES for value in support_values):
                return ri.CANDIDATE_SUPPORT_UNKNOWN
            if _TRUNCATED_SUPPORT not in support_values:
                return ri.CANDIDATE_SUPPORT_EXACT
            mass = _finite_candidate_log_mass(min_log_mass)
            if mass is None:
                return ri.CANDIDATE_SUPPORT_UNKNOWN
            if mass >= good_threshold:
                return ri.CANDIDATE_SUPPORT_GOOD
            if mass >= warning_threshold:
                return ri.CANDIDATE_SUPPORT_WARNING
            return ri.CANDIDATE_SUPPORT_POOR

        setattr(candidate_support_quality, _QUALITY_WRAPPER_ATTR, True)
        ri.candidate_support_quality = candidate_support_quality
    ri._candidate_support_quality_status_patch_applied = True

    _patch_boolean_candidate_log_mass(ri)
    _patch_restricted_candidate_order()
    _patch_overflowed_pairwise_nearest_support()


def _wrapper_chain_has_marker(function: object, marker: str) -> bool:
    """Return whether a runtime wrapper chain already contains ``marker``."""

    current = function
    seen: set[int] = set()
    while callable(current) and id(current) not in seen:
        seen.add(id(current))
        if getattr(current, marker, False):
            return True
        current = getattr(current, "__hipporeplayimm_original__", None)
    return False


def _patch_boolean_candidate_log_mass(ri: Any) -> None:
    """Avoid interpreting boolean diagnostics as finite retained log mass."""

    current = ri._first_finite_numeric_value
    if _wrapper_chain_has_marker(current, _MIN_LOG_MASS_BOOL_WRAPPER_ATTR):
        setattr(ri, _MIN_LOG_MASS_BOOL_PATCHED_FLAG, True)
        return

    original_first_finite_numeric_value = current

    @wraps(original_first_finite_numeric_value)
    def _first_finite_numeric_value(value: object) -> float | None:
        if _contains_boolean(value):
            return None
        return original_first_finite_numeric_value(value)

    setattr(_first_finite_numeric_value, _MIN_LOG_MASS_BOOL_WRAPPER_ATTR, True)
    setattr(_first_finite_numeric_value, "__hipporeplayimm_original__", original_first_finite_numeric_value)
    ri._first_finite_numeric_value = _first_finite_numeric_value
    setattr(ri, _MIN_LOG_MASS_BOOL_PATCHED_FLAG, True)


def _patch_restricted_candidate_order() -> None:
    """Keep valid-bin restriction from sorting ranked candidate supports."""

    import sys

    from . import state_space_utils

    current = state_space_utils._restrict_candidates_to_valid_bins
    if getattr(current, _RESTRICT_CANDIDATE_ORDER_PATCHED_FLAG, False):
        return

    @wraps(current)
    def restrict_candidates_to_valid_bins(candidates, log_likelihood, valid_bin_mask):
        values = np.asarray(log_likelihood)
        n_time, n_bins = values.shape
        validated = state_space_utils._validate_candidate_indices(candidates, n_time, n_bins)
        valid_mask = state_space_utils._coerce_valid_bin_mask(valid_bin_mask, n_bins)
        if valid_mask is None:
            return validated

        valid_indices = np.flatnonzero(valid_mask)
        restricted: list[np.ndarray] = []
        for time_index, arr in enumerate(validated):
            keep = arr[valid_mask[arr]]
            if keep.size == 0:
                valid_scores = values[time_index, valid_indices]
                keep = np.asarray([valid_indices[int(np.argmax(valid_scores))]], dtype=int)
            restricted.append(np.asarray(keep, dtype=int))
        return restricted

    setattr(restrict_candidates_to_valid_bins, _RESTRICT_CANDIDATE_ORDER_PATCHED_FLAG, True)
    setattr(restrict_candidates_to_valid_bins, "__hipporeplayimm_original__", current)
    state_space_utils._restrict_candidates_to_valid_bins = restrict_candidates_to_valid_bins

    for module in list(sys.modules.values()):
        module_name = getattr(module, "__name__", "")
        if module_name.startswith("hipporeplayimm") and getattr(module, "_restrict_candidates_to_valid_bins", None) is current:
            module._restrict_candidates_to_valid_bins = restrict_candidates_to_valid_bins

    setattr(state_space_utils, _RESTRICT_CANDIDATE_ORDER_PATCHED_FLAG, True)


def _patch_overflowed_pairwise_nearest_support() -> None:
    """Preserve nearest-support geometry when every standardized distance overflows."""

    from . import candidate_active_support_validation as active_support
    from . import state_space_utils

    current = state_space_utils._full_grid_normalized_pairwise_gaussian_log_prob
    if getattr(current, _PAIRWISE_OVERFLOW_WRAPPER_ATTR, False):
        setattr(state_space_utils, _PAIRWISE_OVERFLOW_PATCHED_FLAG, True)
        return

    @wraps(current)
    def full_grid_normalized_pairwise_gaussian_log_prob(
        predicted,
        observed,
        all_observed,
        sigma_cm,
        valid_bin_mask=None,
    ):
        output = np.asarray(
            current(
                predicted,
                observed,
                all_observed,
                sigma_cm,
                valid_bin_mask=valid_bin_mask,
            ),
            dtype=float,
        ).copy()
        predicted_points = state_space_utils._as_finite_2d_points(predicted, "predicted")
        observed_points = state_space_utils._as_finite_2d_points(observed, "observed")
        all_observed_points = state_space_utils._as_finite_2d_points(all_observed, "all_observed")
        valid_mask = state_space_utils._coerce_valid_bin_mask(
            valid_bin_mask,
            all_observed_points.shape[0],
        )
        normalizer_support = all_observed_points if valid_mask is None else all_observed_points[valid_mask]
        sigma = float(sigma_cm)

        membership = np.all(
            observed_points[:, None, :] == normalizer_support[None, :, :],
            axis=2,
        )
        if not np.all(np.any(membership, axis=1)):
            return output

        for row, prediction in enumerate(predicted_points):
            standardized = active_support._standardized_euclidean_distances(
                normalizer_support,
                prediction,
                sigma,
            )
            if not np.all(np.isinf(standardized)):
                continue

            log_distances = _stable_log_euclidean_distances(
                normalizer_support,
                prediction,
            )
            minimum = float(np.min(log_distances))
            nearest = log_distances == minimum
            nearest_count = int(np.sum(nearest))
            if nearest_count < 1:
                continue

            nearest_observed = np.any(membership[:, nearest], axis=1)
            fallback = np.full(observed_points.shape[0], -np.inf, dtype=float)
            fallback[nearest_observed] = -np.log(float(nearest_count))
            output[row] = fallback
        return output

    setattr(full_grid_normalized_pairwise_gaussian_log_prob, _PAIRWISE_OVERFLOW_WRAPPER_ATTR, True)
    setattr(
        full_grid_normalized_pairwise_gaussian_log_prob,
        active_support._FULL_GRID_PAIRWISE_GAUSSIAN_WRAPPER_FLAG,
        True,
    )
    setattr(full_grid_normalized_pairwise_gaussian_log_prob, "__hipporeplayimm_original__", current)
    state_space_utils._full_grid_normalized_pairwise_gaussian_log_prob = (
        full_grid_normalized_pairwise_gaussian_log_prob
    )
    active_support._synchronize_transition_aliases(
        "_full_grid_normalized_pairwise_gaussian_log_prob",
        current,
        full_grid_normalized_pairwise_gaussian_log_prob,
    )
    setattr(state_space_utils, _PAIRWISE_OVERFLOW_PATCHED_FLAG, True)


def _stable_log_euclidean_distances(
    points: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    """Return log Euclidean distances without overflowing coordinate subtraction."""

    values = np.asarray(points, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("points must be a nonempty two-dimensional array")
    center = np.asarray(reference, dtype=float).reshape(values.shape[1])
    scales = np.max(
        np.maximum(np.abs(values), np.abs(center)[None, :]),
        axis=1,
    )
    safe_scales = np.where(scales > 0.0, scales, 1.0)
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        normalized_delta = values / safe_scales[:, None] - center[None, :] / safe_scales[:, None]
        normalized_distance = np.hypot.reduce(normalized_delta, axis=1)
        log_distance = np.log(safe_scales) + np.log(normalized_distance)
    log_distance[np.all(values == center[None, :], axis=1)] = -np.inf
    log_distance[np.isnan(log_distance)] = np.inf
    return log_distance


def _evidence_support_values(row: pd.Series) -> list[str]:
    """Return normalized support labels from canonical and diagnostic columns."""

    values: list[str] = []
    values.extend(_support_value_labels(row.get("evidence_support", "")))
    for column in getattr(row, "index", ()):  # pandas Series in production; duck-typed in tests.
        name = str(column)
        if not name.startswith("diagnostic_") or not name.endswith("_evidence_support"):
            continue
        values.extend(_support_value_labels(row.get(column, "")))
    return list(dict.fromkeys(values))


def _support_value_labels(value: object) -> list[str]:
    """Return normalized non-missing support labels from scalar or array-like cells."""

    labels: list[str] = []
    for item in _flatten_support_value(value):
        text = _text(item).lower()
        if text and text not in _MISSING_SUPPORT_VALUES:
            labels.append(text)
    return labels


def _flatten_support_value(value: object) -> list[object]:
    if isinstance(value, (str, bytes, bytearray, memoryview, np.bytes_)):
        return [value]
    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass
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
    return list(array.reshape(-1))


def _finite_candidate_log_mass(value: object) -> float | None:
    if value is None or _contains_boolean(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if np.isfinite(number) else None


def _contains_boolean(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return True
    try:
        array = np.asarray(value, dtype=object)
    except (TypeError, ValueError):
        return False
    if array.ndim == 0:
        try:
            return isinstance(array.item(), (bool, np.bool_))
        except ValueError:
            return False
    return any(isinstance(item, (bool, np.bool_)) for item in array.reshape(-1))


def _text(value: Any) -> str:
    if isinstance(value, (bytes, bytearray, memoryview, np.bytes_)):
        return bytes(value).decode("utf-8", errors="replace").strip()
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


__all__ = ["apply_candidate_support_quality_patch"]
