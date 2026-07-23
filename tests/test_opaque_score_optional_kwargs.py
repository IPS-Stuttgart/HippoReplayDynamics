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
        self.calls: list[tuple[str, ...]] = []
        self.result = object()

    def __call__(self, emissions: Any, bin_centers: Any, **kwargs: Any) -> object:
        del emissions, bin_centers
        self.calls.append(tuple(kwargs))
        for keyword in (
            "candidate_indices",
            "occupancy_s",
            "return_trajectory",
        ):
            if keyword in kwargs:
                raise TypeError(
                    f"got an unexpected keyword argument '{keyword}'"
                )
        return self.result


class _OpaqueNoKeywordScore:
    __signature__ = object()

    def __init__(self) -> None:
        self.calls = 0
        self.result = object()

    def __call__(self, emissions: Any, bin_centers: Any, **kwargs: Any) -> object:
        del emissions, bin_centers
        self.calls += 1
        if kwargs:
            raise TypeError("_OpaqueNoKeywordScore() takes no keyword arguments")
        return self.result


class _OpaqueBrokenScore:
    __signature__ = object()

    def __call__(self, emissions: Any, bin_centers: Any, **kwargs: Any) -> object:
        del emissions, bin_centers, kwargs
        raise TypeError("internal score bug")


@pytest.mark.parametrize("helper", _HELPERS)
def test_opaque_score_fallback_removes_multiple_unsupported_keywords(helper) -> None:
    score = _OpaqueSequentialScore()

    result = helper(score, object(), object(), dict(_OPTIONAL_KWARGS))

    assert result is score.result
    assert score.calls == [
        ("candidate_indices", "occupancy_s", "return_trajectory"),
        ("occupancy_s", "return_trajectory"),
        ("return_trajectory",),
        (),
    ]


@pytest.mark.parametrize("helper", _HELPERS)
def test_opaque_score_fallback_handles_no_keyword_callables(helper) -> None:
    score = _OpaqueNoKeywordScore()

    result = helper(score, object(), object(), dict(_OPTIONAL_KWARGS))

    assert result is score.result
    assert score.calls == 2


@pytest.mark.parametrize("helper", _HELPERS)
def test_opaque_score_fallback_preserves_internal_type_errors(helper) -> None:
    with pytest.raises(TypeError, match="internal score bug"):
        helper(
            _OpaqueBrokenScore(),
            object(),
            object(),
            {"return_trajectory": False},
        )
