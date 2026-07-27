"""Validate result-improvement split inputs before lossy coercion."""

from __future__ import annotations

from functools import wraps
import sys
from typing import Any

import numpy as np

_PATCHED_FLAG = "_result_improvement_split_validation_patch_applied"
_WRAPPER_FLAG = "_result_improvement_split_validation_wrapper"
_ORIGINAL_ATTR = "_result_improvement_split_validation_original"


def apply_result_improvement_split_validation_patch() -> None:
    """Install strict validation for split cell IDs and count arguments."""

    from . import result_improvements

    current = result_improvements.stratified_cell_split
    if getattr(current, _WRAPPER_FLAG, False):
        setattr(result_improvements, _PATCHED_FLAG, True)
        return

    original = current

    @wraps(original)
    def stratified_cell_split(
        cell_ids,
        stratum_values,
        test_fraction,
        random_seed,
        *,
        n_strata: int = 4,
    ):
        validated_cell_ids = _validated_unique_cell_ids(cell_ids)
        validated_n_strata = _positive_integer_scalar(n_strata, "n_strata")
        return original(
            validated_cell_ids,
            stratum_values,
            test_fraction,
            random_seed,
            n_strata=validated_n_strata,
        )

    setattr(stratified_cell_split, _WRAPPER_FLAG, True)
    setattr(stratified_cell_split, _ORIGINAL_ATTR, original)
    result_improvements.stratified_cell_split = stratified_cell_split
    _synchronize_aliases(original, stratified_cell_split)
    setattr(result_improvements, _PATCHED_FLAG, True)


def _validated_unique_cell_ids(values: object) -> np.ndarray:
    """Return exact cell IDs and reject duplicate identities before splitting."""

    from .data_cell_id_validation import _coerce_integral_ids

    cell_ids = _coerce_integral_ids(values, "cell_ids")
    if cell_ids.ndim == 1 and np.unique(cell_ids).size != cell_ids.size:
        raise ValueError("cell_ids must contain unique identifiers")
    return cell_ids


def _positive_integer_scalar(value: object, name: str) -> int:
    """Return a positive integer scalar without boolean or array coercion."""

    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if array.ndim != 0 or np.issubdtype(array.dtype, np.bool_):
        raise ValueError(f"{name} must be a positive integer")
    try:
        item = array.item()
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if isinstance(item, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer")
    try:
        numeric = float(item)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if not np.isfinite(numeric) or numeric < 1.0 or numeric != np.floor(numeric):
        raise ValueError(f"{name} must be a positive integer")
    return int(numeric)


def _synchronize_aliases(original: Any, patched: Any) -> None:
    """Refresh modules that imported stratified_cell_split by value."""

    for module in list(sys.modules.values()):
        module_name = getattr(module, "__name__", "")
        if not module_name.startswith("hipporeplayimm"):
            continue
        if getattr(module, "stratified_cell_split", None) is original:
            module.stratified_cell_split = patched


__all__ = ["apply_result_improvement_split_validation_patch"]
