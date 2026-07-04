"""Validate random seeds for result-improvement resampling helpers."""

from __future__ import annotations

from functools import wraps

import numpy as np

_PATCHED_FLAG = "_result_improvement_seed_validation_patch_applied"
_BOOTSTRAP_WRAPPER_FLAG = "_hierarchical_bootstrap_seed_validation_wrapper"
_SIGN_FLIP_WRAPPER_FLAG = "_paired_sign_flip_seed_validation_wrapper"
_ORIGINAL_ATTR = "__hipporeplayimm_seed_validation_original__"


def _nonnegative_integer_seed(value: object, name: str = "random_seed") -> int:
    """Return a finite nonnegative integer seed without bool/array coercion."""

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
    try:
        numeric = float(scalar)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if not np.isfinite(numeric) or numeric < 0.0 or numeric != np.floor(numeric):
        raise ValueError(message)
    return int(numeric)


def apply_result_improvement_seed_validation_patch() -> None:
    """Install strict seed validation for result-improvement resampling helpers."""

    from . import result_improvements

    if _result_improvement_seed_validation_patch_current(result_improvements):
        setattr(result_improvements, _PATCHED_FLAG, True)
        return

    if not getattr(result_improvements.hierarchical_bootstrap_ci, _BOOTSTRAP_WRAPPER_FLAG, False):
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
            return original_bootstrap(
                rows,
                model=model,
                value_column=value_column,
                group_columns=group_columns,
                n_bootstrap=n_bootstrap,
                random_seed=seed,
            )

        setattr(hierarchical_bootstrap_ci, _BOOTSTRAP_WRAPPER_FLAG, True)
        setattr(hierarchical_bootstrap_ci, _ORIGINAL_ATTR, original_bootstrap)
        result_improvements.hierarchical_bootstrap_ci = hierarchical_bootstrap_ci

    if not getattr(result_improvements.paired_sign_flip_p_value, _SIGN_FLIP_WRAPPER_FLAG, False):
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
            return original_sign_flip(
                rows,
                model=model,
                value_column=value_column,
                n_permutations=n_permutations,
                random_seed=seed,
            )

        setattr(paired_sign_flip_p_value, _SIGN_FLIP_WRAPPER_FLAG, True)
        setattr(paired_sign_flip_p_value, _ORIGINAL_ATTR, original_sign_flip)
        result_improvements.paired_sign_flip_p_value = paired_sign_flip_p_value

    setattr(result_improvements, _PATCHED_FLAG, True)


def _result_improvement_seed_validation_patch_current(result_improvements) -> bool:
    return bool(
        getattr(result_improvements, _PATCHED_FLAG, False)
        and getattr(result_improvements.hierarchical_bootstrap_ci, _BOOTSTRAP_WRAPPER_FLAG, False)
        and getattr(result_improvements.paired_sign_flip_p_value, _SIGN_FLIP_WRAPPER_FLAG, False)
    )


__all__ = [
    "apply_result_improvement_seed_validation_patch",
]
