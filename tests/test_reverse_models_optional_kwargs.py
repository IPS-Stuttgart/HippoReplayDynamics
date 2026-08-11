from __future__ import annotations

import operator
from typing import Never

import pytest

from hipporeplayimm import reverse_models


def _signature_unavailable(_score: object) -> Never:
    raise ValueError("signature unavailable")


class _NoOptionalKwargsScore:
    def __init__(self) -> None:
        self.result = object()

    def score(self, emissions: object, bin_centers: object) -> object:
        del emissions, bin_centers
        return self.result


class _InternalTypeErrorScore:
    def score(self, emissions: object, bin_centers: object, **kwargs: object) -> object:
        del emissions, bin_centers, kwargs
        raise TypeError("len() takes no keyword arguments")


def test_uninspectable_score_retries_until_all_unsupported_kwargs_are_removed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(reverse_models.inspect, "signature", _signature_unavailable)
    model = _NoOptionalKwargsScore()

    result = reverse_models._call_score_with_supported_kwargs(
        model.score,
        object(),
        object(),
        {
            "occupancy_s": object(),
            "candidate_indices": object(),
            "return_trajectory": False,
        },
    )

    assert result is model.result


def test_uninspectable_c_callable_drops_generic_no_keyword_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(reverse_models.inspect, "signature", _signature_unavailable)

    result = reverse_models._call_score_with_supported_kwargs(
        operator.add,
        1,
        2,
        {"occupancy_s": object(), "candidate_indices": object()},
    )

    assert result == 3


def test_uninspectable_score_does_not_swallow_internal_type_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(reverse_models.inspect, "signature", _signature_unavailable)

    with pytest.raises(TypeError, match=r"len\(\) takes no keyword arguments"):
        reverse_models._call_score_with_supported_kwargs(
            _InternalTypeErrorScore().score,
            object(),
            object(),
            {"occupancy_s": object()},
        )
