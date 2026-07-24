from __future__ import annotations

from functools import wraps
from typing import Any

import numpy as np
import pandas as pd


_DEFAULT_BMA_NAME = "bayesian-model-average"
_DEFAULT_BMA_EVIDENCE_COLUMN = "auto"
_TRUE_BOOL_STRINGS = {"1", "1.0", "true", "t", "yes", "y", "on"}
_FALSE_BOOL_STRINGS = {"", "0", "0.0", "false", "f", "no", "n", "off", "nan", "none", "null", "<na>"}
_BOOL_OPTION_ERROR = "boolean option must be a scalar true/false value"
_BYTE_BACKED_SCALARS = (bytes, bytearray, memoryview)


def apply_bma_options_patch() -> None:
    import hipporeplayimm.clusterless_ground_truth as clusterless_module

    _patch_clusterless_kwarg_filter(clusterless_module)
    _wrap_ground_truth_compare_for_bma_options()
    _wrap_late_compare_patch("hipporeplayimm.clusterless_ground_truth", "apply_clusterless_ground_truth_patch")
    _wrap_late_compare_patch("hipporeplayimm.pyrecest_score_metadata", "apply_pyrecest_score_metadata_patch")


def _patch_clusterless_kwarg_filter(module: Any) -> None:
    if getattr(module, "_bma_options_clusterless_filter_applied", False):
        return
    names = module._CLUSTERLESS_KWARG_NAMES

    def keep_non_clusterless_options(options: dict[str, object]) -> dict[str, object]:
        return {key: value for key, value in options.items() if key not in names}

    module._drop_clusterless_kwargs = keep_non_clusterless_options
    module._bma_options_clusterless_filter_applied = True


def _wrap_late_compare_patch(module_name: str, function_name: str) -> None:
    import importlib

    module = importlib.import_module(module_name)
    apply_patch = getattr(module, function_name, None)
    if apply_patch is None or getattr(apply_patch, "_bma_options_rewraps_compare", False):
        return

    @wraps(apply_patch)
    def apply_patch_with_bma_options(*args: Any, **kwargs: Any) -> Any:
        result = apply_patch(*args, **kwargs)
        if module_name.endswith("clusterless_ground_truth"):
            _patch_clusterless_kwarg_filter(module)
        _wrap_ground_truth_compare_for_bma_options()
        return result

    apply_patch_with_bma_options._bma_options_rewraps_compare = True  # type: ignore[attr-defined]
    setattr(module, function_name, apply_patch_with_bma_options)


def _wrap_ground_truth_compare_for_bma_options() -> None:

    import hipporeplayimm.ground_truth as gt

    base_compare = gt.compare_scores_to_ground_truth
    if getattr(base_compare, "_bma_options_wrapped", False):
        return

    @wraps(base_compare)
    def compare_scores_to_ground_truth_with_bma_options(
        root,
        scores,
        *,
        include_bayesian_model_average: bool = True,
        bayesian_model_average_name: str = _DEFAULT_BMA_NAME,
        bayesian_model_average_evidence_column: str = _DEFAULT_BMA_EVIDENCE_COLUMN,
        **kwargs: Any,
    ) -> pd.DataFrame:
        evidence_column = _option_text(bayesian_model_average_evidence_column)
        original_score_row_log_evidence = gt._score_row_log_evidence
        replace_evidence_column = evidence_column.lower() != _DEFAULT_BMA_EVIDENCE_COLUMN
        if replace_evidence_column:

            def score_row_log_evidence_with_bma_default(
                score_row: object,
                selected_evidence_column: str = _DEFAULT_BMA_EVIDENCE_COLUMN,
            ):
                selected_column = _option_text(selected_evidence_column)
                column = evidence_column if selected_column.lower() == _DEFAULT_BMA_EVIDENCE_COLUMN else selected_column
                return original_score_row_log_evidence(score_row, column)

            gt._score_row_log_evidence = score_row_log_evidence_with_bma_default
        try:
            comparison = base_compare(root, scores, **kwargs)
        finally:
            if replace_evidence_column:
                gt._score_row_log_evidence = original_score_row_log_evidence
        return _apply_bma_output_options(
            comparison,
            include_bma=_coerce_bool_option(include_bayesian_model_average),
            model_name=_option_text(bayesian_model_average_name),
        )

    compare_scores_to_ground_truth_with_bma_options._bma_options_wrapped = True  # type: ignore[attr-defined]
    gt.compare_scores_to_ground_truth = compare_scores_to_ground_truth_with_bma_options


def _option_text(value: object) -> str:
    """Return scalar option text, decoding byte-backed storage values."""

    if isinstance(value, np.ndarray) and value.ndim == 0:
        return _option_text(value.item())
    if isinstance(value, np.generic):
        return _option_text(value.item())
    if isinstance(value, _BYTE_BACKED_SCALARS):
        return bytes(value).decode("utf-8", errors="replace")
    return str(value)


def _coerce_bool_option(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, _BYTE_BACKED_SCALARS):
        scalar = value
    else:
        array = np.asarray(value)
        if array.ndim != 0:
            raise ValueError(_BOOL_OPTION_ERROR)
        scalar = array.item()
    if isinstance(scalar, (bool, np.bool_)):
        return bool(scalar)
    try:
        if pd.isna(scalar):
            return False
    except (TypeError, ValueError):
        pass
    if isinstance(scalar, (int, float, np.integer, np.floating)):
        numeric = float(scalar)
        if not np.isfinite(numeric) or numeric not in (0.0, 1.0):
            raise ValueError(_BOOL_OPTION_ERROR)
        return bool(numeric)
    text = _option_text(scalar).strip().lower()
    if text in _TRUE_BOOL_STRINGS:
        return True
    if text in _FALSE_BOOL_STRINGS:
        return False
    raise ValueError(_BOOL_OPTION_ERROR)


def _apply_bma_output_options(comparison, *, include_bma: bool, model_name: str):
    if not hasattr(comparison, "columns") or "model" not in comparison.columns:
        return comparison
    normalized_model_name = _option_text(model_name)
    bma_mask = comparison["model"].map(_option_text) == _DEFAULT_BMA_NAME
    if not include_bma:
        if not bma_mask.any():
            return comparison
        return comparison.loc[~bma_mask].reset_index(drop=True)
    if normalized_model_name == _DEFAULT_BMA_NAME or not bma_mask.any():
        return comparison
    renamed = comparison.copy()
    renamed.loc[bma_mask, "model"] = normalized_model_name
    if "requested_model" in renamed.columns:
        requested_bma_mask = renamed["requested_model"].map(_option_text) == _DEFAULT_BMA_NAME
        renamed.loc[requested_bma_mask, "requested_model"] = normalized_model_name
    return renamed
