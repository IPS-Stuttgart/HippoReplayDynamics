"""Validate random seeds and metric values for result-improvement resampling helpers."""

from __future__ import annotations

from functools import wraps

import numpy as np
import pandas as pd

_PATCHED_FLAG = "_result_improvement_seed_validation_patch_applied"
_BOOTSTRAP_WRAPPER_FLAG = "_hierarchical_bootstrap_seed_validation_wrapper"
_SIGN_FLIP_WRAPPER_FLAG = "_paired_sign_flip_seed_validation_wrapper"
_ORIGINAL_ATTR = "__hipporeplayimm_seed_validation_original__"
_WRAPPER_VERSION = 2


def _nonnegative_integer_seed(value: object, name: str = "random_seed") -> int:
    """Return a finite nonnegative integer seed without bool/string/array coercion."""

    message = f"{name} must be a finite nonnegative integer"
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    if array.ndim != 0:
        raise ValueError(message)
    scalar = array.item()
    if isinstance(scalar, (bool, np.bool_)):
        raise ValueError(message)
    if isinstance(scalar, str):
        raise ValueError(message)
    try:
        numeric = float(scalar)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if not np.isfinite(numeric) or numeric < 0.0 or numeric != np.floor(numeric):
        raise ValueError(message)
    return int(numeric)


def _finite_model_metric_rows(
    rows: object,
    *,
    model: str,
    value_column: str,
):
    """Drop malformed target-model metrics before statistical resampling."""

    if not isinstance(rows, pd.DataFrame):
        return rows
    if "model" not in rows.columns or value_column not in rows.columns:
        return rows

    model_mask = rows["model"].astype(str).eq(str(model)).to_numpy(dtype=bool)
    if not np.any(model_mask):
        return rows

    numeric = pd.to_numeric(rows[value_column], errors="coerce")
    numeric_values = numeric.to_numpy(dtype=np.complex128, na_value=np.nan)
    finite_real = np.isfinite(numeric_values) & (np.imag(numeric_values) == 0.0)
    keep = ~model_mask | finite_real
    if np.all(keep):
        return rows
    return rows.iloc[np.flatnonzero(keep)].copy()


def apply_result_improvement_seed_validation_patch() -> None:
    """Install strict seed, metric-value, and replay-emission validation."""

    from . import result_improvement_emission_validation
    from . import result_improvements
    from . import return_trajectory_validation

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
        current_bootstrap = result_improvements.hierarchical_bootstrap_ci
        original_bootstrap = getattr(current_bootstrap, _ORIGINAL_ATTR, current_bootstrap)

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
            validated_rows = _finite_model_metric_rows(
                rows,
                model=model,
                value_column=value_column,
            )
            return original_bootstrap(
                validated_rows,
                model=model,
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
        current_sign_flip = result_improvements.paired_sign_flip_p_value
        original_sign_flip = getattr(current_sign_flip, _ORIGINAL_ATTR, current_sign_flip)

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
            validated_rows = _finite_model_metric_rows(
                rows,
                model=model,
                value_column=value_column,
            )
            return original_sign_flip(
                validated_rows,
                model=model,
                value_column=value_column,
                n_permutations=n_permutations,
                random_seed=seed,
            )

        setattr(paired_sign_flip_p_value, _SIGN_FLIP_WRAPPER_FLAG, _WRAPPER_VERSION)
        setattr(paired_sign_flip_p_value, _ORIGINAL_ATTR, original_sign_flip)
        result_improvements.paired_sign_flip_p_value = paired_sign_flip_p_value

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
    )


def _is_seed_validation_wrapper(function, flag: str, original_freevar: str) -> bool:
    return bool(
        getattr(function, flag, None) == _WRAPPER_VERSION
        and original_freevar in getattr(function, "__code__").co_freevars
    )


__all__ = [
    "apply_result_improvement_seed_validation_patch",
]
