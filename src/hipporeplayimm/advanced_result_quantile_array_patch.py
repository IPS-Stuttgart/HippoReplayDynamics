"""Make advanced diagnostics robust to array quantiles and malformed resampling controls."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from functools import wraps

import numpy as np
import pandas as pd

_PATCHED_FLAG = "_advanced_result_quantile_array_patch_applied"
_WRAPPER_ATTR = "_advanced_result_quantile_array_wrapper"
_BOOTSTRAP_WRAPPER_ATTR = "_advanced_result_hierarchical_bootstrap_validation_wrapper"
_BOOTSTRAP_WRAPPER_VERSION = 1
_ORIGINAL_ATTR = "__hipporeplayimm_original__"
_TEXT_SCALAR_TYPES = (str, bytes, np.str_, np.bytes_)


def _quantile_patch_installed(diagnostics: object) -> bool:
    current = getattr(diagnostics, "_quantile", None)
    return bool(getattr(current, _WRAPPER_ATTR, False))


def _bootstrap_patch_installed(diagnostics: object) -> bool:
    current = getattr(diagnostics, "hierarchical_bootstrap", None)
    return (
        getattr(current, _BOOTSTRAP_WRAPPER_ATTR, None)
        == _BOOTSTRAP_WRAPPER_VERSION
    )


def apply_advanced_result_quantile_array_patch() -> None:
    """Patch advanced quantiles and hierarchical-bootstrap scalar controls."""

    from . import advanced_result_diagnostics as diagnostics

    if not _quantile_patch_installed(diagnostics):

        def _quantile(values: Sequence[float], q: float) -> float:
            raw = _flatten_quantile_values(values)
            arr = pd.to_numeric(pd.Series(raw), errors="coerce").to_numpy(dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                return float("nan")
            return float(np.quantile(arr, q))

        setattr(_quantile, _WRAPPER_ATTR, True)
        diagnostics._quantile = _quantile

    if not _bootstrap_patch_installed(diagnostics):
        original_bootstrap = diagnostics.hierarchical_bootstrap

        @wraps(original_bootstrap)
        def hierarchical_bootstrap(
            scores: pd.DataFrame,
            *,
            model: str,
            value_col: str = "relative_log_evidence",
            level: str = "session",
            n_bootstrap: int = 1000,
            random_seed: int = 1,
        ) -> dict[str, float | str | int]:
            bootstrap_count = _validated_integer_scalar(
                n_bootstrap,
                "n_bootstrap",
                minimum=1,
            )
            seed = _validated_integer_scalar(
                random_seed,
                "random_seed",
                minimum=0,
            )
            return original_bootstrap(
                scores,
                model=model,
                value_col=value_col,
                level=level,
                n_bootstrap=bootstrap_count,
                random_seed=seed,
            )

        setattr(
            hierarchical_bootstrap,
            _BOOTSTRAP_WRAPPER_ATTR,
            _BOOTSTRAP_WRAPPER_VERSION,
        )
        setattr(hierarchical_bootstrap, _ORIGINAL_ATTR, original_bootstrap)
        diagnostics.hierarchical_bootstrap = hierarchical_bootstrap

    setattr(diagnostics, _PATCHED_FLAG, True)


def _validated_integer_scalar(
    value: object,
    name: str,
    *,
    minimum: int,
) -> int:
    """Return an exact integer scalar without bool, text, or array coercion."""

    qualifier = "positive" if minimum == 1 else "nonnegative"
    message = f"{name} must be a finite {qualifier} integer scalar"
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    if array.ndim != 0:
        raise ValueError(message)
    try:
        scalar = array.item()
    except ValueError as exc:
        raise ValueError(message) from exc
    if isinstance(scalar, (bool, np.bool_, *_TEXT_SCALAR_TYPES)):
        raise ValueError(message)
    if isinstance(scalar, (int, np.integer)):
        integer = int(scalar)
    else:
        try:
            numeric = float(scalar)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(message) from exc
        if not np.isfinite(numeric) or not numeric.is_integer():
            raise ValueError(message)
        integer = int(numeric)
    if integer < minimum:
        raise ValueError(message)
    return integer


def _flatten_quantile_values(values: Sequence[float]) -> np.ndarray:
    """Return scalar values from possibly nested array-valued quantile input."""

    if isinstance(values, Mapping):
        raw = np.asarray(list(values.values()), dtype=object)
    elif isinstance(values, Iterable) and not isinstance(values, (str, bytes)):
        raw = np.asarray(list(values), dtype=object)
    else:
        raw = np.asarray(values, dtype=object)
    if raw.ndim == 0:
        raw = raw.reshape(1)
    flattened: list[object] = []
    for value in raw.reshape(-1):
        if isinstance(value, Mapping):
            flattened.extend(_flatten_quantile_values(list(value.values())).tolist())
            continue
        if isinstance(value, (str, bytes)):
            flattened.append(value)
            continue
        current = np.asarray(value, dtype=object)
        if current.ndim == 0:
            flattened.append(value)
        else:
            flattened.extend(current.reshape(-1).tolist())
    return np.asarray(flattened, dtype=object)


__all__ = ["apply_advanced_result_quantile_array_patch"]
