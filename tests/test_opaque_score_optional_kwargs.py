from __future__ import annotations

from typing import Any

import pytest

from hipporeplayimm import result_improvement_extensions as extensions
from hipporeplayimm import reverse_models

_HELPERS = (
    pytest.param(
        extensions._call_score_with_supported_kwargs,
        id="result-improvement",
    ),
    pytest.param(
        reverse_models._call_score_with_supported_kwargs,
        id="direct-reverse",
    ),
)
_OPTIONAL_KWARGS = {
    "candidate_indices": [object()],
    "occupancy_s": object(),
    "return_trajectory": False,
}


class _OpaqueSequentialScore:
    __signature__ = object()

    def __init__(self) -> None:
        self.calls = 0
        self.result = object()

    def __call__(self, emissions: Any, bin_centers: Any) -> object:
        del emissions, bin_centers
        self.calls += 1
        return self.result


class _OpaqueNoKeywordScore(dict[object, object]):
    __signature__ = object()
    __call__ = dict.pop


class _OpaqueBrokenScore:
    __signature__ = object()

    def __call__(self, emissions: Any, bin_centers: Any, **kwargs: Any) -> object:
        del emissions, bin_centers, kwargs
        raise TypeError("internal score bug")


class _OpaqueMisleadingInternalScore:
    __signature__ = object()

    def __call__(self, emissions: Any, bin_centers: Any, **kwargs: Any) -> object:
        del emissions, bin_centers, kwargs
        raise TypeError("got an unexpected keyword argument 'return_trajectory'")


@pytest.mark.parametrize("helper", _HELPERS)
def test_opaque_score_fallback_removes_multiple_unsupported_keywords(helper) -> None:
    score = _OpaqueSequentialScore()

    result = helper(score, object(), object(), dict(_OPTIONAL_KWARGS))

    assert result is score.result
    assert score.calls == 1


@pytest.mark.parametrize("helper", _HELPERS)
def test_opaque_score_fallback_handles_no_keyword_callables(helper) -> None:
    key = object()
    result = object()
    score = _OpaqueNoKeywordScore({key: result})

    assert helper(score, key, object(), dict(_OPTIONAL_KWARGS)) is result


@pytest.mark.parametrize("helper", _HELPERS)
def test_opaque_score_fallback_preserves_internal_type_errors(helper) -> None:
    with pytest.raises(TypeError, match="internal score bug"):
        helper(
            _OpaqueBrokenScore(),
            object(),
            object(),
            {"return_trajectory": False},
        )


@pytest.mark.parametrize("helper", _HELPERS)
def test_opaque_score_fallback_preserves_misleading_internal_type_errors(helper) -> None:
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        helper(
            _OpaqueMisleadingInternalScore(),
            object(),
            object(),
            {"return_trajectory": False},
        )
