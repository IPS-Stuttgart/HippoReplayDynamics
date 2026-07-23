"""Retry optional score controls for callables with opaque signatures.

Most replay models expose Python signatures that let the compatibility helpers
filter unsupported optional controls before calling ``score``. Some extension
or decorated callables do not expose an inspectable signature. For those
callables, retry after interpreter-style unexpected-keyword errors without
hiding unrelated ``TypeError`` exceptions from the implementation.
"""

from __future__ import annotations

import inspect
from typing import Any

_PATCHED_FLAG = "_iterative_optional_score_kwarg_fallback_patch_applied"


def apply_score_optional_kwargs_fallback_patch() -> None:
    """Install iterative optional-keyword fallback on both scoring helpers."""

    from . import result_improvement_extensions as extensions
    from . import reverse_models

    modules = (extensions, reverse_models)
    if all(
        getattr(
            getattr(module, "_call_score_with_supported_kwargs", None),
            _PATCHED_FLAG,
            False,
        )
        for module in modules
    ):
        return

    setattr(_call_score_with_supported_kwargs, _PATCHED_FLAG, True)
    for module in modules:
        module._call_score_with_supported_kwargs = _call_score_with_supported_kwargs


def _call_score_with_supported_kwargs(
    score: Any,
    emissions: Any,
    bin_centers: Any,
    optional_kwargs: dict[str, Any],
) -> Any:
    """Call ``score``, removing every confirmed unsupported optional keyword."""

    supported_kwargs = _supported_score_kwargs(score, optional_kwargs)
    if supported_kwargs is not None:
        if supported_kwargs:
            return score(emissions, bin_centers, **supported_kwargs)
        return score(emissions, bin_centers)

    remaining = dict(optional_kwargs)
    while True:
        try:
            if remaining:
                return score(emissions, bin_centers, **remaining)
            return score(emissions, bin_centers)
        except TypeError as exc:
            unsupported = _unexpected_optional_keywords(exc, tuple(remaining))
            if not unsupported:
                raise
            for keyword in unsupported:
                remaining.pop(keyword, None)


def _supported_score_kwargs(
    score: Any,
    optional_kwargs: dict[str, Any],
) -> dict[str, Any] | None:
    if not optional_kwargs:
        return {}
    try:
        signature = inspect.signature(score)
    except (TypeError, ValueError):
        return None

    parameters = signature.parameters
    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    ):
        return dict(optional_kwargs)

    supported: dict[str, Any] = {}
    for keyword, value in optional_kwargs.items():
        parameter = parameters.get(keyword)
        if parameter is not None and parameter.kind in (
            inspect.Parameter.KEYWORD_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            supported[keyword] = value
    return supported


def _unexpected_optional_keywords(
    exc: TypeError,
    keywords: tuple[str, ...],
) -> tuple[str, ...]:
    """Return only keywords identified by an interpreter-style call error."""

    text = str(exc)
    named = tuple(
        keyword
        for keyword in keywords
        if keyword in text
        and (
            "unexpected keyword" in text
            or "got an unexpected" in text
            or "invalid keyword" in text
        )
    )
    if named:
        return named
    if keywords and "takes no keyword" in text:
        return keywords
    return ()


__all__ = ["apply_score_optional_kwargs_fallback_patch"]
