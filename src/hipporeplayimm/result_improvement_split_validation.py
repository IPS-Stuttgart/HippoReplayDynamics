"""Validate result-improvement split helper counts before lossy coercion."""

from __future__ import annotations

from functools import wraps
import sys
from typing import Any

_PATCHED_FLAG = "_result_improvement_split_validation_patch_applied"
_WRAPPER_FLAG = "_result_improvement_split_validation_wrapper"
_ORIGINAL_ATTR = "_result_improvement_split_validation_original"


def apply_result_improvement_split_validation_patch() -> None:
    """Install strict validation for split helper count arguments."""

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
        validated_n_strata = result_improvements._positive_integer_count(n_strata, "n_strata")
        return original(
            cell_ids,
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


def _synchronize_aliases(original: Any, patched: Any) -> None:
    """Refresh modules that imported stratified_cell_split by value."""

    for module in list(sys.modules.values()):
        module_name = getattr(module, "__name__", "")
        if not module_name.startswith("hipporeplayimm"):
            continue
        if getattr(module, "stratified_cell_split", None) is original:
            module.stratified_cell_split = patched


__all__ = ["apply_result_improvement_split_validation_patch"]
