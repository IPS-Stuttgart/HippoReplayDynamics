from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from hipporeplayimm.models import EventScore
from hipporeplayimm.reverse_time_terminal_guard import (
    _score_reverse_with_supported_return_trajectory,
)


def _extensions_stub(result: EventScore) -> SimpleNamespace:
    return SimpleNamespace(
        copy_emissions_with_log_likelihood=lambda emissions, log_likelihood, reverse_time: emissions,
        score_replay_model_compat=lambda *args, **kwargs: result,
        _posterior_diagnostics=lambda terminal, bin_centers: {},
    )


def test_reverse_time_guard_uses_base_model_name_when_wrapper_name_is_missing() -> None:
    result = EventScore("diffusion", 0.0, 1, 0, diagnostics={})
    wrapper = SimpleNamespace(base_model=SimpleNamespace(name="diffusion"), name=None)
    emissions = SimpleNamespace(log_likelihood=np.array([[0.0]], dtype=float))

    scored = _score_reverse_with_supported_return_trajectory(
        _extensions_stub(result),
        wrapper,
        emissions,
        np.array([[0.0, 0.0]], dtype=float),
    )

    assert scored.model_name == "diffusion-reverse"


def test_reverse_time_guard_preserves_explicit_wrapper_name() -> None:
    result = EventScore("diffusion", 0.0, 1, 0, diagnostics={})
    wrapper = SimpleNamespace(base_model=SimpleNamespace(name="diffusion"), name="custom-reverse")
    emissions = SimpleNamespace(log_likelihood=np.array([[0.0]], dtype=float))

    scored = _score_reverse_with_supported_return_trajectory(
        _extensions_stub(result),
        wrapper,
        emissions,
        np.array([[0.0, 0.0]], dtype=float),
    )

    assert scored.model_name == "custom-reverse"
