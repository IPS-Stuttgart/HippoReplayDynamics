"""Validate random seeds and metric values for result-improvement resampling helpers."""

from __future__ import annotations

from dataclasses import replace
from functools import wraps

import numpy as np
import pandas as pd

_PATCHED_FLAG = "_result_improvement_seed_validation_patch_applied"
_BOOTSTRAP_WRAPPER_FLAG = "_hierarchical_bootstrap_seed_validation_wrapper"
_SIGN_FLIP_WRAPPER_FLAG = "_paired_sign_flip_seed_validation_wrapper"
_MARK_FEATURE_WRAPPER_FLAG = "_mark_feature_nonidentity_seed_wrapper"
_ORIGINAL_ATTR = "__hipporeplayimm_seed_validation_original__"
_INVALID_UTF8_MODEL_LABEL_PREFIX = "<invalid-utf8-model-bytes:"
_WRAPPER_VERSION = 3
_MARK_FEATURE_WRAPPER_VERSION = 1


def _nonnegative_integer_seed(value: object, name: str = "random_seed") -> int:
    """Return an exact nonnegative integer seed without bool/text/array coercion."""

    message = f"{name} must be a finite nonnegative integer"
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    if array.ndim != 0:
        raise ValueError(message)
    scalar = array.item()
    if isinstance(scalar, (bool, np.bool_, str, bytes, np.str_, np.bytes_)):
        raise ValueError(message)
    try:
        integer = int(scalar)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    try:
        exact = bool(scalar == integer)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    if integer < 0 or not exact:
        raise ValueError(message)
    return integer


def _normalized_model_label(value: object) -> str | None:
    """Return stable text for scalar model identifiers loaded from persisted data."""

    if isinstance(value, np.ndarray):
        array = np.asarray(value, dtype=object)
        if array.ndim == 0 or array.size == 1:
            return _normalized_model_label(array.reshape(-1)[0])
        normalized = tuple(_normalized_model_label(item) for item in array.reshape(-1))
        return str(normalized)
    if isinstance(value, np.generic):
        return _normalized_model_label(value.item())
    if isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
        try:
            value = raw.decode("utf-8")
        except UnicodeDecodeError:
            return f"{_INVALID_UTF8_MODEL_LABEL_PREFIX}{raw.hex()}>"
    if isinstance(value, (list, tuple)):
        normalized = tuple(_normalized_model_label(item) for item in value)
        return str(normalized)
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, (bool, np.bool_)) and bool(missing):
        return None
    return str(value).strip()


def _normalized_model_rows(rows: object):
    """Copy a score table with semantic model labels normalized to text."""

    if not isinstance(rows, pd.DataFrame) or "model" not in rows.columns:
        return rows
    normalized = rows.copy()
    normalized["model"] = normalized["model"].map(_normalized_model_label)
    return normalized


def _finite_model_metric_rows(
    rows: object,
    *,
    model: str,
    value_column: str,
):
    """Drop malformed target-model metrics before statistical resampling."""

    normalized_rows = _normalized_model_rows(rows)
    if not isinstance(normalized_rows, pd.DataFrame):
        return normalized_rows
    if "model" not in normalized_rows.columns or value_column not in normalized_rows.columns:
        return normalized_rows

    normalized_model = _normalized_model_label(model)
    model_mask = normalized_rows["model"].eq(normalized_model).to_numpy(dtype=bool)
    if not np.any(model_mask):
        return normalized_rows

    numeric = pd.to_numeric(normalized_rows[value_column], errors="coerce")
    numeric_values = numeric.to_numpy(dtype=np.complex128, na_value=np.nan)
    finite_real = np.isfinite(numeric_values) & (np.imag(numeric_values) == 0.0)
    keep = ~model_mask | finite_real
    if np.all(keep):
        return normalized_rows
    return normalized_rows.iloc[np.flatnonzero(keep)].copy()


def _nonidentity_permuted_values(
    values: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Permute a feature column until it changes when a change is possible."""

    original = np.asarray(values)
    if original.size <= 1 or np.unique(original).size <= 1:
        return original.copy()
    while True:
        permuted = rng.permutation(original)
        if not np.array_equal(permuted, original, equal_nan=True):
            return permuted


def apply_result_improvement_seed_validation_patch() -> None:
    """Install strict seed, metric-value, and replay-emission validation."""

    from . import replay_gain_bound_patch
    from . import result_improvement_emission_validation
    from . import result_improvements
    from . import return_trajectory_validation

    replay_gain_bound_patch.apply_replay_gain_bound_patch()
    result_improvement_emission_validation.apply_result_improvement_emission_validation_patch()
    return_trajectory_validation.apply_return_trajectory_validation_patch()

    if _result_improvement_seed_validation_patch_current(result_improvements):
        setattr(result_improvements, _PATCHED_FLAG, True)
        return

    if not _is_seed_validation_wrapper(
        result_improvements.hierarchical_bootstrap_ci,
        _BOOTSTRAP_WRAPPER_FLAG,
        "original_bootstrap",
    ):
        original_bootstrap = result_improvements.hierarchical_bootstrap_ci

        @wraps(original_bootstrap)
        def hierarchical_bootstrap_ci(
            rows,
            *,
            model: str,
            value_column: str = "delta_vs_best_static",
            group_columns: tuple[str, ...] = ("session",),
            n_bootstrap: int = 5000,
            random_seed: int = 1,
        ):
            seed = _nonnegative_integer_seed(random_seed, "random_seed")
            normalized_model = _normalized_model_label(model)
            selected_model = str(model) if normalized_model is None else normalized_model
            validated_rows = _finite_model_metric_rows(
                rows,
                model=selected_model,
                value_column=value_column,
            )
            return original_bootstrap(
                validated_rows,
                model=selected_model,
                value_column=value_column,
                group_columns=group_columns,
                n_bootstrap=n_bootstrap,
                random_seed=seed,
            )

        setattr(hierarchical_bootstrap_ci, _BOOTSTRAP_WRAPPER_FLAG, _WRAPPER_VERSION)
        setattr(hierarchical_bootstrap_ci, _ORIGINAL_ATTR, original_bootstrap)
        result_improvements.hierarchical_bootstrap_ci = hierarchical_bootstrap_ci

    if not _is_seed_validation_wrapper(
        result_improvements.paired_sign_flip_p_value,
        _SIGN_FLIP_WRAPPER_FLAG,
        "original_sign_flip",
    ):
        original_sign_flip = result_improvements.paired_sign_flip_p_value

        @wraps(original_sign_flip)
        def paired_sign_flip_p_value(
            rows,
            *,
            model: str,
            value_column: str = "delta_vs_best_static",
            n_permutations: int = 10000,
            random_seed: int = 1,
        ):
            seed = _nonnegative_integer_seed(random_seed, "random_seed")
            normalized_model = _normalized_model_label(model)
            selected_model = str(model) if normalized_model is None else normalized_model
            validated_rows = _finite_model_metric_rows(
                rows,
                model=selected_model,
                value_column=value_column,
            )
            return original_sign_flip(
                validated_rows,
                model=selected_model,
                value_column=value_column,
                n_permutations=n_permutations,
                random_seed=seed,
            )

        setattr(paired_sign_flip_p_value, _SIGN_FLIP_WRAPPER_FLAG, _WRAPPER_VERSION)
        setattr(paired_sign_flip_p_value, _ORIGINAL_ATTR, original_sign_flip)
        result_improvements.paired_sign_flip_p_value = paired_sign_flip_p_value

    if not _is_mark_feature_wrapper(result_improvements.shuffle_mark_features_session):
        original_mark_feature_shuffle = result_improvements.shuffle_mark_features_session

        @wraps(original_mark_feature_shuffle)
        def shuffle_mark_features_session(session, random_seed: int = 1):
            seed = _nonnegative_integer_seed(random_seed, "random_seed")
            marks = session.spike_marks
            if marks is None or marks.n_features == 0:
                return session
            rng = np.random.default_rng(seed)
            values = np.asarray(marks.marks, dtype=float).copy()
            for column in range(values.shape[1]):
                values[:, column] = _nonidentity_permuted_values(
                    values[:, column],
                    rng,
                )
            return replace(session, spike_marks=replace(marks, marks=values))

        setattr(
            shuffle_mark_features_session,
            _MARK_FEATURE_WRAPPER_FLAG,
            _MARK_FEATURE_WRAPPER_VERSION,
        )
        setattr(
            shuffle_mark_features_session,
            _ORIGINAL_ATTR,
            original_mark_feature_shuffle,
        )
        result_improvements.shuffle_mark_features_session = shuffle_mark_features_session

    setattr(result_improvements, _PATCHED_FLAG, True)


def _result_improvement_seed_validation_patch_current(result_improvements) -> bool:
    return bool(
        getattr(result_improvements, _PATCHED_FLAG, False)
        and _is_seed_validation_wrapper(
            result_improvements.hierarchical_bootstrap_ci,
            _BOOTSTRAP_WRAPPER_FLAG,
            "original_bootstrap",
        )
        and _is_seed_validation_wrapper(
            result_improvements.paired_sign_flip_p_value,
            _SIGN_FLIP_WRAPPER_FLAG,
            "original_sign_flip",
        )
        and _is_mark_feature_wrapper(
            result_improvements.shuffle_mark_features_session
        )
    )


def _is_seed_validation_wrapper(function, flag: str, original_freevar: str) -> bool:
    return bool(
        getattr(function, flag, None) == _WRAPPER_VERSION
        and original_freevar in getattr(function, "__code__").co_freevars
    )


def _is_mark_feature_wrapper(function) -> bool:
    return bool(
        getattr(function, _MARK_FEATURE_WRAPPER_FLAG, None)
        == _MARK_FEATURE_WRAPPER_VERSION
    )


__all__ = [
    "apply_result_improvement_seed_validation_patch",
]
